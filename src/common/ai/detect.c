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

/* One-shot promptless detection.
 *
 * Third instance of the model-consumer pattern: segmentation.c drives the
 * interactive SAM stack, refine.c the CascadePSP chain, this file the
 * detectors that take the photo alone -- no prompt -- and answer with a
 * soft mask (salient subject today, semantic classes later). One
 * fixed-size network, one inference, the result resampled back onto the
 * caller's grid.
 *
 * Unlike its siblings this consumer is entirely manifest-driven: input
 * side, letterboxing, normalisation statistics and output activation are
 * read from the model's config.json, because the task families it serves
 * share the calling convention and differ only in those constants -- the
 * bench-selected model replaces the development one by packaging alone.
 *
 * Loading is BY ID, never "the active model of the task": the recipe
 * replay must load the recorded model even when the user has since
 * activated another one. dt_refine_load's active-only loading is what
 * forces the replay gates to demand active == recorded for refinement;
 * no new consumer repeats that.
 */

#include "common/ai/detect.h"

#include "common/ai_models.h"
#include "common/darktable.h"
#include "common/math.h"

#include <math.h>
#include <string.h>

// ImageNet statistics on the [0,1] scale, the default of the BiRefNet
// family; a manifest overrides them with "mean" / "std" attributes
static const float DETECT_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float DETECT_STD[3] = { 0.229f, 0.224f, 0.225f };

struct dt_detect_context_t
{
  dt_ai_context_t *ai_ctx;
  int side;             // attributes.input_sizes[0]
  gboolean letterbox;   // attributes.letterbox: keep aspect, pad with 0
  gboolean sigmoid;     // output_activation != "none": logits -> sigmoid
  float mean[3];
  float stdv[3];

  float *t_image;       // 3 * side * side, CHW
  float *o_mask;        // side * side
};


gboolean dt_detect_available(const char *task)
{
  if(!task || !dt_ai_registry_is_enabled()) return FALSE;

  char *model_id = dt_ai_models_get_active_for_task(task);
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


dt_detect_context_t *dt_detect_load(dt_ai_environment_t *env,
                                    const char *model_id,
                                    const char *task)
{
  if(!env || !model_id || !model_id[0]) return NULL;

  const dt_ai_model_info_t *info
    = dt_ai_get_model_info_by_id(env, model_id);
  if(!info)
  {
    dt_print(DT_DEBUG_AI, "[detect] model %s is not installed", model_id);
    return NULL;
  }

  // the id may come from an untrusted recipe blob: refuse a model of
  // another task family before it can be run out of contract
  if(task && g_strcmp0(info->task_type, task) != 0)
  {
    dt_print(DT_DEBUG_AI, "[detect] model %s serves task '%s', not '%s'",
             model_id, info->task_type ? info->task_type : "?", task);
    return NULL;
  }

  int n = 0;
  int *sizes = dt_ai_model_attribute_int_array(info, "input_sizes", &n);
  const int side = (sizes && n > 0) ? sizes[0] : 0;
  g_free(sizes);
  if(side <= 0)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] model %s: attributes.input_sizes must name the"
             " square input side", model_id);
    return NULL;
  }

  // no CPU refusal, unlike dt_refine_load: a detection is one inference
  // on a background job, slow on CPU but never blocking anything. the
  // optimisation level comes from the manifest (DEFAULT), as everything
  // else about this consumer does
  dt_ai_context_t *ai = dt_ai_load_model_ext(env, model_id, NULL,
                                             DT_AI_PROVIDER_CONFIGURED,
                                             DT_AI_OPT_DEFAULT, NULL, 0);
  if(!ai) return NULL;

  if(dt_ai_get_input_count(ai) != 1 || dt_ai_get_output_count(ai) != 1)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] expected 1 input / 1 output, got %d / %d",
             dt_ai_get_input_count(ai), dt_ai_get_output_count(ai));
    dt_ai_unload_model(ai);
    return NULL;
  }

  dt_detect_context_t *ctx = g_malloc0(sizeof(dt_detect_context_t));
  ctx->ai_ctx = ai;
  ctx->side = side;
  ctx->letterbox = dt_ai_model_attribute_bool(info, "letterbox");

  // the models of this family emit logits; an absent attribute means
  // "sigmoid", only an explicit "none" opts out
  char *act = dt_ai_model_attribute_string(info, "output_activation");
  ctx->sigmoid = !act || strcmp(act, "none") != 0;
  g_free(act);

  for(int c = 0; c < 3; c++)
  {
    ctx->mean[c] = DETECT_MEAN[c];
    ctx->stdv[c] = DETECT_STD[c];
  }
  int nv = 0;
  double *v = dt_ai_model_attribute_double_array(info, "mean", &nv);
  if(v && nv == 3)
    for(int c = 0; c < 3; c++) ctx->mean[c] = (float)v[c];
  g_free(v);
  nv = 0;
  v = dt_ai_model_attribute_double_array(info, "std", &nv);
  // the manifest is untrusted input, and a non-numeric JSON element
  // reads back as 0.0: a zero deviation would turn the normalisation
  // into inf/NaN, so only a fully positive triple replaces the default
  if(v && nv == 3 && v[0] > 0.0 && v[1] > 0.0 && v[2] > 0.0)
    for(int c = 0; c < 3; c++) ctx->stdv[c] = (float)v[c];
  else if(v)
    dt_print(DT_DEBUG_AI,
             "[detect] model %s: ignoring unusable std attribute",
             model_id);
  g_free(v);

  const size_t plane = (size_t)side * side;
  ctx->t_image = g_try_malloc0(3 * plane * sizeof(float));
  ctx->o_mask = g_try_malloc0(plane * sizeof(float));
  if(!ctx->t_image || !ctx->o_mask)
  {
    dt_print(DT_DEBUG_AI, "[detect] out of memory allocating %dx%d buffers",
             side, side);
    dt_detect_free(ctx);
    return NULL;
  }

  // no warmup inference, unlike the multi-call consumers: a detection IS
  // a single call, warming up would exactly double the cost of the job
  return ctx;
}


void dt_detect_free(dt_detect_context_t *ctx)
{
  if(!ctx) return;
  g_free(ctx->t_image);
  g_free(ctx->o_mask);
  if(ctx->ai_ctx) dt_ai_unload_model(ctx->ai_ctx);
  g_free(ctx);
}


int dt_detect_get_side(const dt_detect_context_t *ctx)
{
  return ctx ? ctx->side : 0;
}


/* Bilinear sample of a single-channel plane, pixel-centre convention.
 * `stride` is the row pitch, which differs from `sw` for the padded
 * network output of a letterboxed run. */
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


gboolean dt_detect_run(dt_detect_context_t *ctx,
                       const uint8_t *const rgb,
                       const int rgb_w,
                       const int rgb_h,
                       float *const mask)
{
  if(!ctx || !rgb || !mask) return FALSE;
  if(rgb_w < 8 || rgb_h < 8) return FALSE;

  const int s = ctx->side;
  const size_t plane = (size_t)s * s;

  // used area of the square input: the full square when the manifest
  // asks for a plain stretch (the BiRefNet convention), the
  // aspect-preserving letterbox otherwise, its margin padded with the
  // zeros of the memset
  int uw = s, uh = s;
  if(ctx->letterbox)
  {
    const float scale = (float)s / (float)MAX(rgb_w, rgb_h);
    uw = MAX(1, MIN(s, (int)lrintf(rgb_w * scale)));
    uh = MAX(1, MIN(s, (int)lrintf(rgb_h * scale)));
    memset(ctx->t_image, 0, 3 * plane * sizeof(float));
  }

  const float inv_x = (float)rgb_w / (float)uw;
  const float inv_y = (float)rgb_h / (float)uh;

  // aligned local copies: neither the file-scope defaults nor the
  // ctx fields can be shared as arrays under default(none)
  const dt_aligned_pixel_t mean
    = { ctx->mean[0], ctx->mean[1], ctx->mean[2], 0.0f };
  const dt_aligned_pixel_t stdv
    = { ctx->stdv[0], ctx->stdv[1], ctx->stdv[2], 1.0f };

  DT_OMP_FOR(shared(mean, stdv))
  for(int y = 0; y < uh; y++)
  {
    // pixel-centre mapping back into frame coordinates
    const float sy = ((float)y + 0.5f) * inv_y - 0.5f;
    for(int x = 0; x < uw; x++)
    {
      const float sx = ((float)x + 0.5f) * inv_x - 0.5f;
      const float cx = CLAMPF(sx, 0.0f, (float)(rgb_w - 1));
      const float cy = CLAMPF(sy, 0.0f, (float)(rgb_h - 1));
      const int x0 = (int)cx, y0 = (int)cy;
      const int x1 = MIN(x0 + 1, rgb_w - 1), y1 = MIN(y0 + 1, rgb_h - 1);
      const float ax = cx - (float)x0, ay = cy - (float)y0;

      const size_t i00 = ((size_t)y0 * rgb_w + x0) * 3;
      const size_t i01 = ((size_t)y0 * rgb_w + x1) * 3;
      const size_t i10 = ((size_t)y1 * rgb_w + x0) * 3;
      const size_t i11 = ((size_t)y1 * rgb_w + x1) * 3;

      for(int c = 0; c < 3; c++)
      {
        const float v = (rgb[i00 + c] * (1.0f - ax) * (1.0f - ay)
                         + rgb[i01 + c] * ax * (1.0f - ay)
                         + rgb[i10 + c] * (1.0f - ax) * ay
                         + rgb[i11 + c] * ax * ay) / 255.0f;
        ctx->t_image[(size_t)c * plane + (size_t)y * s + x]
          = (v - mean[c]) / stdv[c];
      }
    }
  }

  int64_t shape_img[4] = { 1, 3, s, s };
  int64_t shape_1c[4] = { 1, 1, s, s };
  dt_ai_tensor_t input = {
    .data = (void *)ctx->t_image,
    .shape = shape_img, .ndim = 4, .type = DT_AI_FLOAT
  };
  dt_ai_tensor_t output = {
    .data = ctx->o_mask,
    .shape = shape_1c, .ndim = 4, .type = DT_AI_FLOAT
  };
  if(dt_ai_run(ctx->ai_ctx, &input, 1, &output, 1) != 0)
  {
    dt_print(DT_DEBUG_AI, "[detect] inference failed");
    return FALSE;
  }

  // the backend rewrites the output shape with the dimensions the model
  // REALLY produced and bounds its copy accordingly: another count here
  // means o_mask holds a partial or foreign-geometry result. the one way
  // to it is a manifest whose input_sizes does not describe its model --
  // untrusted input, refuse rather than resample garbage
  int64_t got = 1;
  for(int d = 0; d < output.ndim; d++) got *= output.shape[d];
  if(got != (int64_t)plane)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] model output carries %lld values where %dx%d were"
             " expected -- manifest input_sizes does not match the model",
             (long long)got, s, s);
    return FALSE;
  }

  if(ctx->sigmoid)
  {
    DT_OMP_FOR()
    for(size_t k = 0; k < plane; k++)
      ctx->o_mask[k] = 1.0f / (1.0f + expf(-ctx->o_mask[k]));
  }

  // back onto the caller's grid: sample only the used area, the exact
  // inverse of the mapping above (pixel-centre both ways)
  const float fx_back = (float)uw / (float)rgb_w;
  const float fy_back = (float)uh / (float)rgb_h;

  DT_OMP_FOR()
  for(int y = 0; y < rgb_h; y++)
  {
    const float sy = ((float)y + 0.5f) * fy_back - 0.5f;
    float *const out = mask + (size_t)y * rgb_w;
    for(int x = 0; x < rgb_w; x++)
    {
      const float sx = ((float)x + 0.5f) * fx_back - 0.5f;
      out[x] = CLAMPF(_sample_1c(ctx->o_mask, s, uw, uh, sx, sy),
                      0.0f, 1.0f);
    }
  }

  return TRUE;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
