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

/* ViTMatte -- the trimap-based matting line of the table.
 *
 * PERSONAL BUILDS ONLY. This file exists only in a tree configured with
 * USE_MATTE_VITMATTE, and it must never travel in a darktable package.
 * The reason is not the code -- which is ordinary GPL C -- but the
 * WEIGHTS: ViTMatte's published checkpoints are trained on the Adobe Deep
 * Image Matting dataset, whose agreement defines the trained model as part
 * of the licensed material and confines it to non-commercial use. So no
 * release may ship the file, no downloader may fetch it, and no dialog may
 * suggest it. The model is installed by hand, by the person who accepted
 * that agreement, into their own models directory. Everything below is
 * written to be dead code in every other build.
 *
 * WHAT IT DOES. The finalisation's band is where sub-pixel coverage is
 * re-derived; the guided filter does it with a local affine model, which
 * is a fast approximation of a matting Laplacian and has no notion of a
 * hair crossing a background. ViTMatte is a real matting network: given
 * the image and a trimap it solves the unknown region directly. In the
 * table's terms it replaces `alpha_gf` INSIDE the band (wmatte = 1) and
 * changes nothing outside it -- the composition's `wband` still fades the
 * whole thing back to the soft hint at the band's edge, so the operator
 * never gets authority over pixels the band does not cover.
 *
 * THE REFERENCE. Every geometric and numeric choice here reproduces the
 * bench pile "S0m" (proto/eval_fourrure/trimap.py + trimap_s0.py), which
 * is what measured -27.8% fur error on the bench's hard case: the 928-px
 * static tiling with 64 px of overlap, the 32 px inner crop, the skip of
 * tiles holding no unknown, ImageNet normalisation applied over the whole
 * padded square, a trimap channel padded with plain zeros, the fused
 * average of overlapping tiles, and the forcing of known pixels back to
 * the trimap. Numbers that look arbitrary below are arbitrary in the same
 * way the reference is: they are the reference. Any change to one of them
 * is a new algorithm and owes the table a version bump.
 *
 * The consumer pattern is the third one's: loading BY ID with a task check
 * (detect.h:37-64, "never the active model of the task", so a replay loads
 * what the recipe recorded), and the single-shot CPU retry of the restore_*
 * family for the GPU failure this hardware actually produces -- DirectML
 * returning 8007000E when darktable's own OpenCL pipe already holds the
 * card. A slow path is still a path: DT_MATTE_CPU_OK says so, and the
 * render core has no execution-provider check of its own.
 *
 * ONE THING IS NOT THE PATTERN, deliberately: when the swap happens the
 * region is restarted from its first tile instead of being finished on the
 * CPU with the GPU's earlier tiles still in the accumulator. The restore_*
 * consumers can keep theirs -- their output is a preview or a denoised
 * frame, and a slightly divergent tile is a slightly divergent pixel. This
 * plane is hashed into a provenance recipe that names a cached render, so
 * a half-and-half plane means two renders under one name. It is not a
 * theoretical worry: on the bench's failing adapter the tile accepted just
 * before the refusal was wrong by up to 0.9 alpha over 28k pixels, and the
 * operator reported success over it.
 */

#include "common/matte.h"

#include "common/ai/detect.h"
#include "common/ai_models.h"
#include "common/darktable.h"
#include "common/math.h"
#include "control/control.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "common/rasterfile_recipe.h"

// the recipe records these verbatim in fixed-size fields; a line that did
// not fit would be silently truncated into another line's id
G_STATIC_ASSERT(sizeof("vitmatte-b-912") <= DT_RF_RECIPE_MATTING_ID_LEN);
G_STATIC_ASSERT(sizeof("2") <= DT_RF_RECIPE_MATTING_VERSION_LEN);

// the registry task and the model installed under it. the directory name
// IS the id, following every other model of this tree
// (refine-cascadepsp, mask-subject-birefnet): "<task>-<arch>"
#define TASK_MATTE "matte"
#define MODEL_MATTE "matte-vitmatte-b912"

// ImageNet statistics, as the reference preprocessing uses them
// the normalisation of THIS model's official processor
// (hustvl/vitmatte-base-composition-1k, preprocessor_config.json:
// image_mean = image_std = 0.5, NOT the ImageNet constants the detectors
// use): the export was validated against that processor, and feeding the
// ImageNet ones stretches the input to [-2.1, 2.2] instead of [-1, 1]
static const float MATTE_MEAN[3] = { 0.5f, 0.5f, 0.5f };
static const float MATTE_STD[3] = { 0.5f, 0.5f, 0.5f };

// tiling geometry of the reference (trimap.py:39-40). the overlap exists so
// the inner crop below can throw away the tile border the network resolves
// worst, and the step is what is left of the tile after it
#define MATTE_OVERLAP 64
// the reference crops 32 px and keeps the tiles disjoint: every pixel is
// one tile's answer and the seams show as steps in the alpha. 16 px are
// cropped here, and the 32 px the neighbouring tiles then share are
// fused with a linear ramp (MATTE_RAMP): the seam becomes a crossfade
#define MATTE_CROP 16
#define MATTE_RAMP (MATTE_OVERLAP - 2 * MATTE_CROP)

typedef struct dt_matte_context_t
{
  dt_ai_context_t *ai_ctx;
  dt_ai_environment_t *env;    // borrowed, for the one CPU reload below
  char *model_id;
  gboolean cpu_fallback_done;  // the retry is single-shot
  int side;                    // attributes.input_sizes[0], 928

  float *t_input;              // 1 x 4 x side x side, CHW: R,G,B,trimap
  float *o_alpha;              // side x side
} dt_matte_context_t;


static void dt_matte_free(dt_matte_context_t *ctx)
{
  if(!ctx) return;
  dt_free_align(ctx->t_input);
  dt_free_align(ctx->o_alpha);
  g_free(ctx->model_id);
  if(ctx->ai_ctx) dt_ai_unload_model(ctx->ai_ctx);
  g_free(ctx);
}


/* Load the matting model BY ID -- never "the active model of the task".
 * The signature the design pinned (Q8), and the reason is detect.h's: a
 * recipe replay must load the model the recipe recorded even when the user
 * has since activated another one. dt_refine_load's active-only loading is
 * what forces the refine replay gate to demand active == recorded; no new
 * consumer repeats that.
 *
 * `task` is the registry task the caller expects, and a model of another
 * family is refused: the id reaches this function from a table line today,
 * but the manifest it resolves to is a file on disk that anybody may edit.
 *
 * Local manifests only, by construction rather than by a check: nothing in
 * this build ever puts this id in a download catalogue, so the only way
 * dt_ai_get_model_info_by_id() answers is a directory the user installed
 * themselves. No CPU refusal (unlike dt_refine_load): this runs on a
 * background job where slow is not blocked. No warmup inference either --
 * the tiles that follow are the warmup, and a run costs seconds. */
static dt_matte_context_t *dt_matte_load(dt_ai_environment_t *env,
                                         const char *model_id,
                                         const char *task)
{
  if(!env || !model_id || !model_id[0]) return NULL;

  const dt_ai_model_info_t *info
    = dt_ai_get_model_info_by_id(env, model_id);
  if(!info)
  {
    dt_print(DT_DEBUG_AI, "[matte] model %s is not installed", model_id);
    return NULL;
  }

  if(task && g_strcmp0(info->task_type, task) != 0)
  {
    dt_print(DT_DEBUG_AI, "[matte] model %s serves task '%s', not '%s'",
             model_id, info->task_type ? info->task_type : "?", task);
    return NULL;
  }

  int n = 0;
  int *sizes = dt_ai_model_attribute_int_array(info, "input_sizes", &n);
  const int side = (sizes && n > 0) ? sizes[0] : 0;
  g_free(sizes);
  // the ViT backbone works on 16 px patches and the detail branch halves
  // the resolution three times, so a side that is not a multiple of 16
  // cannot describe this graph. the export the bench validated is 928.
  // the tiling loops below also need a positive step, which the reference
  // overlap makes equivalent to side > MATTE_OVERLAP
  if(side <= MATTE_OVERLAP || (side % 16) != 0)
  {
    dt_print(DT_DEBUG_AI,
             "[matte] model %s: attributes.input_sizes must be a multiple"
             " of 16 above %d (got %d)",
             model_id, MATTE_OVERLAP, side);
    return NULL;
  }

  dt_ai_context_t *ai = dt_ai_load_model_ext(env, model_id, NULL,
                                             DT_AI_PROVIDER_CONFIGURED,
                                             DT_AI_OPT_DEFAULT, NULL, 0);
  if(!ai) return NULL;

  if(dt_ai_get_input_count(ai) != 1 || dt_ai_get_output_count(ai) != 1)
  {
    dt_print(DT_DEBUG_AI,
             "[matte] expected 1 input / 1 output, got %d / %d",
             dt_ai_get_input_count(ai), dt_ai_get_output_count(ai));
    dt_ai_unload_model(ai);
    return NULL;
  }

  dt_matte_context_t *ctx = g_malloc0(sizeof(dt_matte_context_t));
  ctx->ai_ctx = ai;
  ctx->env = env;
  ctx->model_id = g_strdup(model_id);
  ctx->side = side;

  const size_t plane = (size_t)side * side;
  ctx->t_input = dt_alloc_align_float(4 * plane);
  ctx->o_alpha = dt_alloc_align_float(plane);
  if(!ctx->t_input || !ctx->o_alpha)
  {
    dt_print(DT_DEBUG_AI, "[matte] out of memory allocating %dx%d buffers",
             side, side);
    dt_matte_free(ctx);
    return NULL;
  }
  return ctx;
}


/* Swap the failing session for one on the CPU provider: the
 * _reload_session_cpu of detect.c, for the same failure and with the same
 * single-shot discipline. The swapped session lives on the context, so the
 * remaining tiles reuse it instead of rebuilding one each. */
static gboolean _reload_session_cpu(dt_matte_context_t *ctx)
{
  if(!ctx || !ctx->env || !ctx->model_id) return FALSE;

  // nothing to fall back FROM when the configured provider already is the
  // CPU: the swap would rebuild the same session and toast a GPU failure
  // that never happened
  if(dt_ai_env_get_provider(ctx->env) == DT_AI_PROVIDER_CPU)
  {
    dt_print(DT_DEBUG_AI, "[matte] configured provider is CPU, no fallback");
    return FALSE;
  }

  // unload BEFORE creating the new session: on a GPU EP the failing one may
  // still hold the VRAM the CPU session does not need but the driver does
  dt_ai_unload_model(ctx->ai_ctx);
  ctx->ai_ctx = NULL;

  dt_ai_context_t *cpu = dt_ai_load_model_ext(ctx->env, ctx->model_id, NULL,
                                              DT_AI_PROVIDER_CPU,
                                              DT_AI_OPT_DEFAULT, NULL, 0);
  if(!cpu)
  {
    dt_print(DT_DEBUG_AI, "[matte] CPU fallback session load failed for %s",
             ctx->model_id);
    return FALSE;
  }
  ctx->ai_ctx = cpu;
  return TRUE;
}


/* The backend rewrites the output shape with the dimensions the model
 * REALLY produced and bounds its copy by MIN(what it has, what the caller
 * offered), reporting success either way. Another count than side*side
 * therefore means o_alpha holds a partial or foreign-geometry result whose
 * TAIL still carries the previous tile -- and this plane is hashed into a
 * provenance recipe, so that silent mixture would travel under a valid
 * fingerprint. The one way to it is a manifest whose input_sizes does not
 * describe its model, i.e. a config.json anybody may edit: untrusted
 * input, refuse rather than accumulate garbage. The check detect.c:399-413
 * documents, for the same reason, on the same kind of file. */
static gboolean _output_is_plane(const dt_ai_tensor_t *const output,
                                 const int s)
{
  int64_t got = 1;
  for(int d = 0; d < output->ndim; d++) got *= output->shape[d];
  if(got == (int64_t)((size_t)s * s)) return TRUE;
  dt_print(DT_DEBUG_AI,
           "[matte] model output carries %lld values where %dx%d were"
           " expected -- manifest input_sizes does not match the model",
           (long long)got, s, s);
  return FALSE;
}


/* One inference on the filled t_input. The CPU retry lives here so every
 * tile is covered by it, not just the first.
 *
 * Returns 0 when the configured provider answered, 1 when it refused and
 * the CPU took over (the plane is filled either way), and -1 on failure.
 * The caller has to tell those two successes apart -- see the restart at
 * the tile loop. */
static int _infer(dt_matte_context_t *ctx)
{
  const int s = ctx->side;
  int64_t shape_in[4] = { 1, 4, s, s };
  int64_t shape_out[4] = { 1, 1, s, s };
  dt_ai_tensor_t input = {
    .data = (void *)ctx->t_input,
    .shape = shape_in, .ndim = 4, .type = DT_AI_FLOAT
  };
  dt_ai_tensor_t output = {
    .data = ctx->o_alpha,
    .shape = shape_out, .ndim = 4, .type = DT_AI_FLOAT
  };

  if(dt_ai_run(ctx->ai_ctx, &input, 1, &output, 1) == 0)
    return _output_is_plane(&output, s) ? 0 : -1;

  // the observed failure is the GPU EP running out of VRAM mid-graph
  // (DirectML 8007000E) while darktable's OpenCL pipe holds the card: a
  // lost cause on that provider, a couple of minutes on the CPU. the flag
  // is raised BEFORE the attempt so a failed swap is not retried either
  const gboolean already = ctx->cpu_fallback_done;
  ctx->cpu_fallback_done = TRUE;
  if(already || !_reload_session_cpu(ctx))
  {
    dt_print(DT_DEBUG_AI, "[matte] inference failed");
    return -1;
  }
  // "the configured provider", not "the GPU": the retry fires on ANY
  // refusal of the session built above, and that session is whatever
  // plugins/ai/provider named -- the message must not assert a cause the
  // code never established. (The CPU case never reaches here:
  // _reload_session_cpu declines when the configured provider already is
  // the CPU, so a swap really is a change of provider.)
  dt_print(DT_DEBUG_AI,
           "[matte] inference failed on the configured provider;"
           " retrying on CPU");
  dt_control_log(_("AI matting: inference failed on the configured"
                   " provider, retrying on CPU"));

  if(dt_ai_run(ctx->ai_ctx, &input, 1, &output, 1) != 0)
  {
    dt_print(DT_DEBUG_AI, "[matte] inference failed on CPU too");
    return -1;
  }
  return _output_is_plane(&output, s) ? 1 : -1;
}


/* Distinct tile starts along one axis, the reference's own rule
 * (trimap.py:62-74): walk in `step`, clamp the start to the far edge, and
 * drop a start that repeats the previous one -- the clamp collapses the
 * trailing tiles onto the same start and each duplicate would cost a full
 * redundant inference. Writes at most `max_n` starts, returns how many. */
static int _tile_starts(const int extent,
                        const int side,
                        const int step,
                        int *const starts,
                        const int max_n)
{
  int n = 0;
  for(int t = 0; t * step < extent && n < max_n; t++)
  {
    const int st = MIN(t * step, MAX(0, extent - side));
    if(n == 0 || starts[n - 1] != st) starts[n++] = st;
  }
  return n;
}


static gboolean _vitmatte_run(const dt_matte_op_t *const op,
                              const dt_matte_stage_t *const stage,
                              float *const out)
{
  if(!op || !stage || !out) return FALSE;
  if(!stage->guide || !stage->hint_bin || !stage->trimap) return FALSE;
  const int W = stage->width, H = stage->height;
  if(W < 8 || H < 8) return FALSE;

  const size_t npix = (size_t)W * H;
  const float *const trimap = stage->trimap;

  dt_ai_environment_t *env = dt_ai_env_init(NULL);
  dt_matte_context_t *ctx = env ? dt_matte_load(env, op->model, op->task)
                                : NULL;
  if(!ctx)
  {
    // the caller decides what an absent operator means; this one only
    // reports that no plane was produced. a missing model is diagnosed
    // ahead of the render by the recipe's model-gap mirror, which is
    // where a user can still do something about it
    dt_print(DT_DEBUG_AI,
             "[matte] %s: model %s unavailable, no matting plane",
             op->id, op->model ? op->model : "?");
    if(env) dt_ai_env_destroy(env);
    return FALSE;
  }

  const int s = ctx->side;
  const size_t plane = (size_t)s * s;
  const int step = s - MATTE_OVERLAP;
  gboolean ok = FALSE;

  // the fused accumulator IS `out`: the operator owns that plane for the
  // duration of the call, so accumulating in place saves a full-region
  // buffer on a stage that already holds the guide, two hints, the trimap
  // and a summed-area table. the weight is the sum of the ramps of the
  // tiles that covered a pixel -- a float plane, one crossfade per seam
  float *weight = g_try_malloc0(npix * sizeof(float));
  int *tile_xs = g_try_malloc(sizeof(int) * (size_t)(W / MAX(step, 1) + 2));
  int *tile_ys = g_try_malloc(sizeof(int) * (size_t)(H / MAX(step, 1) + 2));
  if(!weight || !tile_xs || !tile_ys)
  {
    dt_print(DT_DEBUG_AI, "[matte] out of memory for the tile fusion");
    goto cleanup;
  }
  memset(out, 0, npix * sizeof(float));

  const int n_tx = _tile_starts(W, s, step, tile_xs, W / MAX(step, 1) + 2);
  const int n_ty = _tile_starts(H, s, step, tile_ys, H / MAX(step, 1) + 2);

  // local aligned copies: file-scope constants cannot be shared under
  // default(none)
  const dt_aligned_pixel_t mean
    = { MATTE_MEAN[0], MATTE_MEAN[1], MATTE_MEAN[2], 0.0f };
  const dt_aligned_pixel_t stdv
    = { MATTE_STD[0], MATTE_STD[1], MATTE_STD[2], 1.0f };

  int done = 0, skipped = 0;
  const double t_start = dt_get_wtime();
  // set once, when the region has already been restarted on the CPU. it is
  // not the same flag as ctx->cpu_fallback_done: that one says the SWAP has
  // been spent, this one says the ACCUMULATOR has been thrown away, and the
  // second can only ever happen once because the first gates it
  gboolean restarted = FALSE;

restart:
  for(int iy = 0; iy < n_ty; iy++)
  {
    const int y0 = tile_ys[iy];
    const int y1 = MIN(y0 + s, H);
    for(int ix = 0; ix < n_tx; ix++)
    {
      const int x0 = tile_xs[ix];
      const int x1 = MIN(x0 + s, W);
      const int tw = x1 - x0, th = y1 - y0;

      // skip a tile with no unknown at all: the network's whole job is the
      // unknown region, and on a known-only tile it would spend its seconds
      // reproducing the trimap it was handed. this is the reference's own
      // skip (trimap.py:82-84) and it is what makes the stage affordable --
      // on the bench's fur case 8 of 20 tiles never run.
      // "unknown" is tested as a band rather than == 0.5f: the values come
      // from dt_matte_trimap_from_band and are exactly 0, 0.5 or 1, so the
      // two tests agree, and the band one keeps agreeing if a future
      // producer ever emits a soft trimap
      gboolean has_unknown = FALSE;
      for(int y = y0; y < y1 && !has_unknown; y++)
      {
        const float *const row = trimap + (size_t)y * W;
        for(int x = x0; x < x1; x++)
          if(row[x] > 0.25f && row[x] < 0.75f) { has_unknown = TRUE; break; }
      }
      if(!has_unknown) { skipped++; continue; }

      if(stage->keep_going && !stage->keep_going(stage->user)) goto cleanup;

      // ---- fill the 4-channel square.
      // the PAD IS NOT ZERO on the image channels: the reference builds a
      // zeroed square, writes the tile into it and normalises the WHOLE
      // square, so the margin carries (0 - mean)/std and not 0. feeding
      // zeros there instead would hand the network a bright border it was
      // never shown. the trimap channel is padded with a plain 0 --
      // background -- because the reference pads it AFTER normalisation,
      // which never touches that channel
      for(int c = 0; c < 3; c++)
      {
        const float pad = (0.0f - mean[c]) / stdv[c];
        float *const ch = ctx->t_input + (size_t)c * plane;
        for(size_t k = 0; k < plane; k++) ch[k] = pad;
      }
      memset(ctx->t_input + 3 * plane, 0, plane * sizeof(float));

      DT_OMP_FOR(shared(mean, stdv))
      for(int y = 0; y < th; y++)
      {
        const float *const g = stage->guide + ((size_t)(y0 + y) * W + x0) * 4;
        const float *const tr = trimap + (size_t)(y0 + y) * W + x0;
        for(int x = 0; x < tw; x++)
        {
          for(int c = 0; c < 3; c++)
          {
            // through 8 bits on purpose. the reference reads an 8-bit
            // sRGB file, and so does every image this network was trained
            // on; the extra precision of the render buffer is precision
            // the model has no use for, and quantising here is what makes
            // a C render and a bench run comparable at all. the same
            // conversion the refine consumer's caller already performs
            const uint8_t q
              = (uint8_t)lrintf(CLAMPF(g[x * 4 + c], 0.0f, 1.0f) * 255.0f);
            ctx->t_input[(size_t)c * plane + (size_t)y * s + x]
              = ((float)q / 255.0f - mean[c]) / stdv[c];
          }
          ctx->t_input[3 * plane + (size_t)y * s + x] = tr[x];
        }
      }

      const int rc = _infer(ctx);
      if(rc < 0) goto cleanup;
      if(rc > 0 && !restarted)
      {
        // THE REGION STARTS OVER. The tiles already in the accumulator all
        // came off the provider that just refused this one, and a device
        // that fails a graph has earned no trust in the graphs it answered
        // before it: measured on the bench's failing adapter, the last tile
        // it accepted was wrong by up to 0.9 alpha over 28k pixels, and the
        // operator returned TRUE over it. Silent, and worse than a refusal
        // -- this plane is what the provenance recipe fingerprints, so a
        // mask that is half one provider and half another is exactly the
        // "two renders, one name" the whole table exists to prevent.
        //
        // Costs the tiles already done, once, and only on a machine where
        // the GPU actually failed; the swap itself stays single-shot, so a
        // second failure now falls through to -1 and the stage fails
        // outright rather than looping.
        restarted = TRUE;
        memset(out, 0, npix * sizeof(float));
        memset(weight, 0, npix * sizeof(float));
        done = skipped = 0;
        dt_print(DT_DEBUG_AI,
                 "[matte] %s: restarting the region on the CPU, the tiles"
                 " the GPU had already answered are discarded",
                 op->id);
        goto restart;
      }
      done++;

      // ---- accumulate, cropping the inner borders (trimap.py:99-113).
      // an edge that touches the region's own border is NOT cropped: there
      // is no neighbouring tile to take it over, and dropping it would
      // leave the region's rim with no tile at all
      int ax0 = x0, ay0 = y0, ax1 = x1, ay1 = y1;
      int px0 = 0, py0 = 0;
      if(x0 != 0) { ax0 += MATTE_CROP; px0 += MATTE_CROP; }
      if(y0 != 0) { ay0 += MATTE_CROP; py0 += MATTE_CROP; }
      if(x1 != W) ax1 -= MATTE_CROP;
      if(y1 != H) ay1 -= MATTE_CROP;

      // the ramp: 0 at the kept edge, 1 MATTE_RAMP pixels in, on the
      // edges a neighbouring tile shares; an edge on the region's own
      // border keeps full weight, nothing takes it over
      const gboolean rl = x0 != 0, rt = y0 != 0, rr = x1 != W, rb = y1 != H;
      for(int y = ay0; y < ay1; y++)
      {
        const float *const src
          = ctx->o_alpha + (size_t)(py0 + y - ay0) * s + px0;
        float *const dst = out + (size_t)y * W;
        float *const wgt = weight + (size_t)y * W;
        float wy = 1.0f;
        if(rt) wy = MIN(wy, (float)(y - ay0 + 1) / (float)MATTE_RAMP);
        if(rb) wy = MIN(wy, (float)(ay1 - y) / (float)MATTE_RAMP);
        for(int x = ax0; x < ax1; x++)
        {
          float w = wy;
          if(rl) w = MIN(w, (float)(x - ax0 + 1) / (float)MATTE_RAMP);
          if(rr) w = MIN(w, (float)(ax1 - x) / (float)MATTE_RAMP);
          w = CLAMPF(w, 0.0f, 1.0f);
          dst[x] += w * src[x - ax0];
          wgt[x] += w;
        }
      }
    }
  }

  dt_print(DT_DEBUG_AI,
           "[matte] %s: %d tiles run, %d skipped, %.1fs (%dx%d region,"
           " %d px tile)",
           op->id, done, skipped, dt_get_wtime() - t_start, W, H, s);

  // ---- fuse, then hand the known region back to the trimap.
  // two separate statements in the reference and two here, because they
  // say different things: the first averages the tiles that covered a
  // pixel and falls back to the trimap where none did (a skipped tile, or
  // a gap the clamped last start could leave); the second REVOKES the
  // network's answer everywhere the trimap was already sure. ViTMatte has
  // authority over the unknown region and nowhere else -- outside it the
  // band weight is zero anyway, and the composition ignores this plane
  DT_OMP_FOR()
  for(size_t k = 0; k < npix; k++)
  {
    const float fused = (weight[k] > 1e-6f)
                        ? out[k] / weight[k]
                        : trimap[k];
    out[k] = (trimap[k] > 0.25f && trimap[k] < 0.75f)
             ? CLAMPF(fused, 0.0f, 1.0f)
             : trimap[k];
  }
  ok = TRUE;

cleanup:
  g_free(weight);
  g_free(tile_xs);
  g_free(tile_ys);
  dt_matte_free(ctx);
  dt_ai_env_destroy(env);
  return ok;
}


const dt_matte_op_t dt_matte_op_vitmatte =
{
  .id = "vitmatte-b-912",
  // ALGORITHM revision, not a model version. every number in this file is
  // covered by it -- tile side, overlap, crop, normalisation, the skip
  // rule, the fusion -- because the recipe carrying it is hashed verbatim
  // to name a content-addressed file, and one name may never cover two
  // renders. the model's own version is the registry's business and is
  // checked at load
  .version = "2",
  .task = TASK_MATTE,
  .model = MODEL_MATTE,
  .caps = DT_MATTE_NEEDS_TRIMAP | DT_MATTE_TILED | DT_MATTE_NEEDS_MODEL
          | DT_MATTE_CPU_OK,
  // FULL authority inside the band: this operator does not refine the
  // guided filter's answer, it replaces it. the reference composition
  // (trimap.py tail_vitmatte) is exactly wband*alpha_matting +
  // (1-wband)*hint_soft, which is the three-term form at wmatte = 1
  .wmatte = 1.0f,
  .run = _vitmatte_run,
};

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
