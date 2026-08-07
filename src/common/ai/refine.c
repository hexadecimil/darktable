/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

/* CascadePSP boundary refinement.
 *
 * The segmentation model produces a mask on a coarse grid: the SAM 2.1
 * decoder emits 256x256 whatever the photo, which on a 24 Mpix file means one
 * mask pixel per ~24 image pixels. No resampling recovers a contour that was
 * never sampled. CascadePSP is a dedicated network that re-derives the
 * boundary from the image itself, given the coarse mask as a hint.
 *
 * The upstream implementation (segmentation_refinement/eval_helper.py) runs a
 * global pass followed by a tiled local pass. Only the global pass is used
 * here: on this hardware the local pass costs ~12 s for a 2048^2 region,
 * which no interactive gesture can absorb.
 *
 * The exported graph covers all three call sites of the upstream cascade, so
 * one model file is chained three times:
 *
 *   #  seg   prev_s8    prev_s4   output read
 *   1  seg   seg        seg       tanh_s8
 *   2  seg   tanh_s8    tanh_s8   tanh_s8', tanh_s4
 *   3  seg   tanh_s8'   tanh_s4   pred_224
 *
 * Input order is positional: image, seg, prev_s8, prev_s4.
 */

#include "common/ai/refine.h"

#include "common/ai_models.h"
#include "common/darktable.h"
#include "common/math.h"

#include <math.h>
#include <string.h>

#define TASK_REFINE "refine"

// ImageNet statistics, as used by the upstream im_transform
static const float REFINE_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float REFINE_STD[3] = { 0.229f, 0.224f, 0.225f };

struct dt_refine_context_t
{
  dt_ai_context_t *ai_ctx;
  int side;              // attributes.input_sizes[0], 912
  double warmup_ms;

  float *t_image;        // 3 * side * side, CHW
  float *t_seg;          // side * side
  float *t_p8;
  float *t_p4;

  float *o_t8;           // outputs, side * side each
  float *o_t4;
  float *o_p56;
  float *o_p224;
};


gboolean dt_refine_available(void)
{
  if(!dt_ai_registry_is_enabled()) return FALSE;

  char *model_id = dt_ai_models_get_active_for_task(TASK_REFINE);
  if(!model_id || !model_id[0])
  {
    g_free(model_id);
    return FALSE;
  }
  dt_ai_model_t *model = dt_ai_models_get_by_id(model_id);
  g_free(model_id);

  const gboolean ok = model && model->status == DT_AI_MODEL_DOWNLOADED;
  dt_ai_model_free(model);
  return ok;
}


/* One inference. Buffers are borrowed by the backend, not copied. */
static int _block(dt_refine_context_t *ctx,
                  const float *const restrict seg,
                  const float *const restrict p8,
                  const float *const restrict p4)
{
  const int s = ctx->side;
  int64_t shape_img[4] = { 1, 3, s, s };
  int64_t shape_1c[4] = { 1, 1, s, s };

  dt_ai_tensor_t inputs[4] = {
    { .data = (void *)ctx->t_image, .shape = shape_img, .ndim = 4, .type = DT_AI_FLOAT },
    { .data = (void *)seg,          .shape = shape_1c,  .ndim = 4, .type = DT_AI_FLOAT },
    { .data = (void *)p8,           .shape = shape_1c,  .ndim = 4, .type = DT_AI_FLOAT },
    { .data = (void *)p4,           .shape = shape_1c,  .ndim = 4, .type = DT_AI_FLOAT },
  };
  dt_ai_tensor_t outputs[4] = {
    { .data = ctx->o_t8,   .shape = shape_1c, .ndim = 4, .type = DT_AI_FLOAT },
    { .data = ctx->o_t4,   .shape = shape_1c, .ndim = 4, .type = DT_AI_FLOAT },
    { .data = ctx->o_p56,  .shape = shape_1c, .ndim = 4, .type = DT_AI_FLOAT },
    { .data = ctx->o_p224, .shape = shape_1c, .ndim = 4, .type = DT_AI_FLOAT },
  };
  return dt_ai_run(ctx->ai_ctx, inputs, 4, outputs, 4);
}


static void _free_buffers(dt_refine_context_t *ctx)
{
  g_free(ctx->t_image);
  g_free(ctx->t_seg);
  g_free(ctx->t_p8);
  g_free(ctx->t_p4);
  g_free(ctx->o_t8);
  g_free(ctx->o_t4);
  g_free(ctx->o_p56);
  g_free(ctx->o_p224);
}


dt_refine_context_t *dt_refine_load(dt_ai_environment_t *env)
{
  if(!env) return NULL;

  // A CPU inference measures ~20 s on this class of hardware, so a full
  // refinement would block the UI for a minute. Refuse rather than fall back.
  if(dt_ai_env_get_provider(env) == DT_AI_PROVIDER_CPU)
  {
    dt_print(DT_DEBUG_AI, "[refine] configured provider is CPU, skipping");
    return NULL;
  }

  char *model_id = dt_ai_models_get_active_for_task(TASK_REFINE);
  if(!model_id || !model_id[0])
  {
    g_free(model_id);
    return NULL;
  }

  const dt_ai_model_info_t *info = dt_ai_get_model_info_by_id(env, model_id);
  int n = 0;
  int *sizes = info ? dt_ai_model_attribute_int_array(info, "input_sizes", &n) : NULL;
  const int side = (sizes && n > 0) ? sizes[0] : 0;
  g_free(sizes);

  // The PSP pyramid pools to 1x1, 2x2, 3x3 and 6x6 over a feature map at
  // stride 8, so side/8 must be divisible by 6, i.e. side must be a multiple
  // of 48. 912 = 48 * 19 is what the reference export uses.
  if(side <= 0 || (side % 48) != 0)
  {
    dt_print(DT_DEBUG_AI,
             "[refine] model %s: input_sizes must be a positive multiple of 48 (got %d)",
             model_id, side);
    g_free(model_id);
    return NULL;
  }

  dt_ai_context_t *ai = dt_ai_load_model_ext(env, model_id, "block.onnx",
                                             DT_AI_PROVIDER_CONFIGURED,
                                             DT_AI_OPT_ALL, NULL, 0);
  g_free(model_id);
  if(!ai) return NULL;

  if(dt_ai_get_input_count(ai) != 4 || dt_ai_get_output_count(ai) != 4)
  {
    dt_print(DT_DEBUG_AI, "[refine] expected 4 inputs / 4 outputs, got %d / %d",
             dt_ai_get_input_count(ai), dt_ai_get_output_count(ai));
    dt_ai_unload_model(ai);
    return NULL;
  }

  dt_refine_context_t *ctx = g_malloc0(sizeof(dt_refine_context_t));
  ctx->ai_ctx = ai;
  ctx->side = side;

  const size_t plane = (size_t)side * side;
  ctx->t_image = g_try_malloc0(3 * plane * sizeof(float));
  ctx->t_seg = g_try_malloc0(plane * sizeof(float));
  ctx->t_p8 = g_try_malloc0(plane * sizeof(float));
  ctx->t_p4 = g_try_malloc0(plane * sizeof(float));
  ctx->o_t8 = g_try_malloc0(plane * sizeof(float));
  ctx->o_t4 = g_try_malloc0(plane * sizeof(float));
  ctx->o_p56 = g_try_malloc0(plane * sizeof(float));
  ctx->o_p224 = g_try_malloc0(plane * sizeof(float));

  if(!ctx->t_image || !ctx->t_seg || !ctx->t_p8 || !ctx->t_p4
     || !ctx->o_t8 || !ctx->o_t4 || !ctx->o_p56 || !ctx->o_p224)
  {
    dt_print(DT_DEBUG_AI, "[refine] out of memory allocating %dx%d buffers", side, side);
    dt_refine_free(ctx);
    return NULL;
  }

  // Warm up on zeroed buffers. This also pays the graph compilation, which is
  // due anyway, and gives a measured cost we can compare against the budget.
  const double t0 = dt_get_wtime();
  if(_block(ctx, ctx->t_seg, ctx->t_p8, ctx->t_p4) != 0)
  {
    dt_print(DT_DEBUG_AI, "[refine] warmup inference failed");
    dt_refine_free(ctx);
    return NULL;
  }
  ctx->warmup_ms = (dt_get_wtime() - t0) * 1000.0;
  dt_print(DT_DEBUG_AI, "[refine] warmup %.0f ms at %dx%d", ctx->warmup_ms, side, side);

  return ctx;
}


void dt_refine_free(dt_refine_context_t *ctx)
{
  if(!ctx) return;
  _free_buffers(ctx);
  if(ctx->ai_ctx) dt_ai_unload_model(ctx->ai_ctx);
  g_free(ctx);
}


double dt_refine_warmup_ms(const dt_refine_context_t *ctx)
{
  return ctx ? ctx->warmup_ms : 0.0;
}


int dt_refine_get_side(const dt_refine_context_t *ctx)
{
  return ctx ? ctx->side : 0;
}


/* Bilinear sample of a single-channel plane, pixel-centre convention.
 * `stride` is the row pitch, which differs from `sw` both for a sub-region of
 * a larger buffer and for the padded network output. */
static inline float _sample_1c(const float *const restrict src,
                               const int stride,
                               const int sw,
                               const int sh,
                               const float fx,
                               const float fy)
{
  const float x = CLAMPF(fx, 0.0f, (float)(sw - 1));
  const float y = CLAMPF(fy, 0.0f, (float)(sh - 1));
  const int x0 = (int)x, y0 = (int)y;
  const int x1 = MIN(x0 + 1, sw - 1), y1 = MIN(y0 + 1, sh - 1);
  const float ax = x - (float)x0, ay = y - (float)y0;
  return src[(size_t)y0 * stride + x0] * (1.0f - ax) * (1.0f - ay)
       + src[(size_t)y0 * stride + x1] * ax * (1.0f - ay)
       + src[(size_t)y1 * stride + x0] * (1.0f - ax) * ay
       + src[(size_t)y1 * stride + x1] * ax * ay;
}


gboolean dt_refine_run(dt_refine_context_t *ctx,
                       const uint8_t *const rgb,
                       const int rgb_w,
                       const int rgb_h,
                       float *const mask,
                       const float threshold,
                       const int roi_x,
                       const int roi_y,
                       const int roi_w,
                       const int roi_h)
{
  if(!ctx || !rgb || !mask) return FALSE;
  if(roi_w < 8 || roi_h < 8) return FALSE;
  if(roi_x < 0 || roi_y < 0 || roi_x + roi_w > rgb_w || roi_y + roi_h > rgb_h)
  {
    dt_print(DT_DEBUG_AI, "[refine] region out of bounds, skipping");
    return FALSE;
  }

  const int s = ctx->side;
  const size_t plane = (size_t)s * s;

  // The region keeps its aspect ratio inside the square input; the unused
  // margin is padded, image with zeros and masks with -1, matching the
  // upstream safe_forward().
  const float scale = (float)s / (float)MAX(roi_w, roi_h);
  const int uw = MAX(1, MIN(s, (int)lrintf(roi_w * scale)));
  const int uh = MAX(1, MIN(s, (int)lrintf(roi_h * scale)));

  memset(ctx->t_image, 0, 3 * plane * sizeof(float));
  for(size_t k = 0; k < plane; k++) ctx->t_seg[k] = -1.0f;

  const float inv_x = (float)roi_w / (float)uw;
  const float inv_y = (float)roi_h / (float)uh;

  // local copies: file-scope constants cannot be shared under default(none)
  const dt_aligned_pixel_t mean = { REFINE_MEAN[0], REFINE_MEAN[1], REFINE_MEAN[2], 0.0f };
  const dt_aligned_pixel_t stdv = { REFINE_STD[0], REFINE_STD[1], REFINE_STD[2], 1.0f };

  DT_OMP_FOR(shared(mean, stdv))
  for(int y = 0; y < uh; y++)
  {
    // pixel-centre mapping back into region coordinates
    const float sy = ((float)y + 0.5f) * inv_y - 0.5f;
    for(int x = 0; x < uw; x++)
    {
      const float sx = ((float)x + 0.5f) * inv_x - 0.5f;
      const float cx = CLAMPF(sx, 0.0f, (float)(roi_w - 1));
      const float cy = CLAMPF(sy, 0.0f, (float)(roi_h - 1));
      const int x0 = (int)cx, y0 = (int)cy;
      const int x1 = MIN(x0 + 1, roi_w - 1), y1 = MIN(y0 + 1, roi_h - 1);
      const float ax = cx - (float)x0, ay = cy - (float)y0;

      const size_t i00 = ((size_t)(roi_y + y0) * rgb_w + roi_x + x0) * 3;
      const size_t i01 = ((size_t)(roi_y + y0) * rgb_w + roi_x + x1) * 3;
      const size_t i10 = ((size_t)(roi_y + y1) * rgb_w + roi_x + x0) * 3;
      const size_t i11 = ((size_t)(roi_y + y1) * rgb_w + roi_x + x1) * 3;

      for(int c = 0; c < 3; c++)
      {
        const float v = (rgb[i00 + c] * (1.0f - ax) * (1.0f - ay)
                         + rgb[i01 + c] * ax * (1.0f - ay)
                         + rgb[i10 + c] * (1.0f - ax) * ay
                         + rgb[i11 + c] * ax * ay) / 255.0f;
        ctx->t_image[(size_t)c * plane + (size_t)y * s + x]
          = (v - mean[c]) / stdv[c];
      }

      // the network binarises its hint anyway (upstream main.py:66), so feed
      // it a clean step rather than the soft mask
      const float m = _sample_1c(mask + (size_t)roi_y * rgb_w + roi_x,
                                 rgb_w, roi_w, roi_h, cx, cy);
      ctx->t_seg[(size_t)y * s + x] = (m > threshold) ? 1.0f : -1.0f;
    }
  }

  // --- the three chained calls ---
  if(_block(ctx, ctx->t_seg, ctx->t_seg, ctx->t_seg) != 0) return FALSE;
  memcpy(ctx->t_p8, ctx->o_t8, plane * sizeof(float));

  if(_block(ctx, ctx->t_seg, ctx->t_p8, ctx->t_p8) != 0) return FALSE;
  memcpy(ctx->t_p8, ctx->o_t8, plane * sizeof(float));
  memcpy(ctx->t_p4, ctx->o_t4, plane * sizeof(float));

  if(_block(ctx, ctx->t_seg, ctx->t_p8, ctx->t_p4) != 0) return FALSE;

  // --- write the refined alpha back into the region ---
  const float fx_back = (float)uw / (float)roi_w;
  const float fy_back = (float)uh / (float)roi_h;

  DT_OMP_FOR()
  for(int y = 0; y < roi_h; y++)
  {
    const float sy = ((float)y + 0.5f) * fy_back - 0.5f;
    float *const out = mask + (size_t)(roi_y + y) * rgb_w + roi_x;
    for(int x = 0; x < roi_w; x++)
    {
      const float sx = ((float)x + 0.5f) * fx_back - 0.5f;
      out[x] = CLAMPF(_sample_1c(ctx->o_p224, s, uw, uh, sx, sy), 0.0f, 1.0f);
    }
  }

  return TRUE;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
