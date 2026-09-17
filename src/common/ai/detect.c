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

/* One-shot detection.
 *
 * Third instance of the model-consumer pattern: segmentation.c drives the
 * interactive SAM stack, refine.c the CascadePSP chain, this file the
 * detectors that take the photo alone -- no click -- and answer with a
 * soft mask: the salient subject, or the class a text prompt names. One
 * fixed-size network, one inference, the result resampled back onto the
 * caller's grid.
 *
 * Unlike its siblings this consumer is entirely manifest-driven: input
 * side, letterboxing, normalisation statistics and output activation are
 * read from the model's config.json, because the task families it serves
 * share the calling convention and differ only in those constants -- the
 * bench-selected model replaces the development one by packaging alone.
 * A text-conditioned model (CLIPSeg exported with the text embedding as
 * a second input) is the same consumer with one more tensor: the
 * embeddings of its prompts are frozen at export into prompts.bin next
 * to the model, listed by name in the manifest, and the caller picks one
 * by index -- no text tower runs here.
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
#include "control/control.h"

#include <json-glib/json-glib.h>
#include <math.h>
#include <string.h>

// ImageNet statistics on the [0,1] scale, the default of the BiRefNet
// family; a manifest overrides them with "mean" / "std" attributes
static const float DETECT_MEAN[3] = { 0.485f, 0.456f, 0.406f };
static const float DETECT_STD[3] = { 0.229f, 0.224f, 0.225f };

struct dt_detect_context_t
{
  dt_ai_context_t *ai_ctx;
  dt_ai_environment_t *env;  // borrowed: the caller's env, which every
                             // call site destroys after the context --
                             // held for the one CPU reload below
  char *model_id;
  gboolean cpu_fallback_done; // the CPU retry is single-shot: once the
                              // session was swapped (or the swap
                              // failed), a failure is final
  int side;             // attributes.input_sizes[0]
  gboolean letterbox;   // attributes.letterbox: keep aspect, pad with 0
  gboolean sigmoid;     // output_activation != "none": logits -> sigmoid
  int crop_border;      // attributes.crop_border: rows and columns of
                        // the network output discarded on each side
                        // before the resample back (CLIPSeg's edge
                        // roll-off); 0 = the whole map is trusted
  int out_ndim;         // rank of the model's declared output: the
                        // output tensor is bound with that rank, the
                        // count is side x side either way
  float mean[3];
  float stdv[3];

  float *t_image;       // 3 * side * side, CHW
  float *t_prompt;      // prompt_dim floats, the embedding of the prompt
                        // this context answers about; NULL for a model
                        // that takes the image alone
  int prompt_dim;       // attributes.prompt_dim
  int text_input;       // slot of the text tensor among the two inputs
                        // (dt_ai_run binds by position)
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


/* The prompt list of a text-conditioned model: attributes.prompts is a
 * JSON array of strings, which the numeric accessors of the backend do
 * not read, so the attribute block is parsed here. Returns the length
 * of the list (0 when absent) and, through `name`, the entry at `index`
 * when there is one (caller frees). */
static int _prompt_list(const dt_ai_model_info_t *info,
                        const int index,
                        char **name)
{
  if(name) *name = NULL;
  if(!info->attributes) return 0;

  int n = 0;
  JsonParser *parser = json_parser_new();
  if(json_parser_load_from_data(parser, info->attributes, -1, NULL))
  {
    JsonNode *root = json_parser_get_root(parser);
    JsonObject *obj = root && JSON_NODE_HOLDS_OBJECT(root)
      ? json_node_get_object(root) : NULL;
    JsonNode *node = obj && json_object_has_member(obj, "prompts")
      ? json_object_get_member(obj, "prompts") : NULL;
    if(node && JSON_NODE_HOLDS_ARRAY(node))
    {
      JsonArray *arr = json_node_get_array(node);
      n = (int)json_array_get_length(arr);
      if(name && index >= 0 && index < n)
      {
        JsonNode *e = json_array_get_element(arr, index);
        if(e && JSON_NODE_HOLDS_VALUE(e)
           && json_node_get_value_type(e) == G_TYPE_STRING)
          *name = g_strdup(json_node_get_string(e));
      }
    }
  }
  g_object_unref(parser);
  return n;
}


/* The embedding of prompt `index`, read from prompts.bin in the model's
 * folder and checked against the manifest: the file holds exactly the
 * embeddings the list names, prompt_dim float32 each, little-endian, in
 * the order of the list. Newly allocated, prompt_dim floats; NULL with
 * the reason journaled. The folder is the registry's, the same one the
 * model card is read from. */
static float *_load_prompt(const dt_ai_model_info_t *info,
                           const char *model_id,
                           const int index,
                           int *dim_out)
{
  char *name = NULL;
  const int n = _prompt_list(info, index, &name);
  const int dim = dt_ai_model_attribute_int(info, "prompt_dim", 0);
  float *out = NULL;
  char *dir = NULL, *path = NULL, *data = NULL;
  gsize len = 0, expected = 0;

  if(n <= 0 || dim <= 0)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] model %s takes a text prompt but its manifest names"
             " no prompts / prompt_dim", model_id);
    goto out;
  }
  if(index < 0 || index >= n)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] model %s: prompt %d is not among its %d prompts",
             model_id, index, n);
    goto out;
  }

  dir = dt_ai_models_get_path(model_id);
  path = dir ? g_build_filename(dir, "prompts.bin", NULL) : NULL;
  if(!path || !g_file_get_contents(path, &data, &len, NULL))
  {
    dt_print(DT_DEBUG_AI, "[detect] model %s: cannot read %s", model_id,
             path ? path : "prompts.bin");
    goto out;
  }
  expected = (gsize)n * (gsize)dim * sizeof(float);
  if(len != expected)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] model %s: prompts.bin holds %" G_GSIZE_FORMAT
             " bytes where %d x %d floats (%" G_GSIZE_FORMAT ") were"
             " expected", model_id, len, n, dim, expected);
    goto out;
  }

  out = g_try_malloc((size_t)dim * sizeof(float));
  if(!out)
  {
    dt_print(DT_DEBUG_AI, "[detect] out of memory for the prompt embedding");
    goto out;
  }
  for(int k = 0; k < dim; k++)
  {
    uint32_t u;
    memcpy(&u, data + ((gsize)index * dim + k) * sizeof(float), sizeof(u));
    u = GUINT32_FROM_LE(u);
    memcpy(&out[k], &u, sizeof(u));
  }
  *dim_out = dim;
  dt_print(DT_DEBUG_AI, "[detect] model %s: prompt %d '%s' (%d floats)",
           model_id, index, name ? name : "?", dim);

out:
  g_free(data);
  g_free(path);
  g_free(dir);
  g_free(name);
  return out;
}


dt_detect_context_t *dt_detect_load(dt_ai_environment_t *env,
                                    const char *model_id,
                                    const char *task,
                                    const int prompt_index)
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

  // the session is the arbiter of the input count, whatever the
  // manifest says: one input is the image, two are the image and the
  // text embedding, anything else is not a detector
  const int n_inputs = dt_ai_get_input_count(ai);
  if((n_inputs != 1 && n_inputs != 2) || dt_ai_get_output_count(ai) != 1)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] expected 1-2 inputs / 1 output, got %d / %d",
             n_inputs, dt_ai_get_output_count(ai));
    dt_ai_unload_model(ai);
    return NULL;
  }

  dt_detect_context_t *ctx = g_malloc0(sizeof(dt_detect_context_t));
  ctx->ai_ctx = ai;
  ctx->env = env;
  ctx->model_id = g_strdup(model_id);
  ctx->side = side;
  ctx->letterbox = dt_ai_model_attribute_bool(info, "letterbox");

  if(n_inputs == 2)
  {
    ctx->t_prompt = _load_prompt(info, model_id, prompt_index,
                                 &ctx->prompt_dim);
    if(!ctx->t_prompt)
    {
      dt_detect_free(ctx);
      return NULL;
    }
    // dt_ai_run binds the inputs by position, so the text tensor's slot
    // is read from the session: the input the model names
    // text_embedding, the second one when it names neither
    ctx->text_input = 1;
    for(int i = 0; i < 2; i++)
      if(g_strcmp0(dt_ai_get_input_name(ai, i), "text_embedding") == 0)
        ctx->text_input = i;
  }

  // the models of this family emit logits; an absent attribute means
  // "sigmoid", only an explicit "none" opts out
  char *act = dt_ai_model_attribute_string(info, "output_activation");
  ctx->sigmoid = !act || strcmp(act, "none") != 0;
  g_free(act);

  // the untrusted manifest again: a border that would eat the whole map
  // is clamped to leave at least one usable row and column
  ctx->crop_border
    = CLAMP(dt_ai_model_attribute_int(info, "crop_border", 0), 0,
            (side - 1) / 2);

  // the family shares the element count (side x side) but not the rank
  // of its output: BiRefNet emits [1,1,S,S], CLIPSeg [1,S,S]. the runtime
  // verifies a pre-bound output against the model's declared shape, so
  // the tensor must be bound with the rank the model declares; an
  // undeclarable rank falls back to the 4-D convention, and the count
  // check after the run stays the arbiter either way
  {
    int64_t shape[16];
    const int ndim = dt_ai_get_output_shape(ai, 0, shape, 16);
    ctx->out_ndim = (ndim == 2 || ndim == 3) ? ndim : 4;
  }

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
  g_free(ctx->t_prompt);
  g_free(ctx->o_mask);
  g_free(ctx->model_id);
  if(ctx->ai_ctx) dt_ai_unload_model(ctx->ai_ctx);
  g_free(ctx);
}


int dt_detect_get_side(const dt_detect_context_t *ctx)
{
  return ctx ? ctx->side : 0;
}


/* Swap the failing session for one on the CPU provider, the
 * dt_restore_reload_session_cpu pattern: the swapped session lives on
 * the context, so later runs reuse it instead of rebuilding one per
 * call, and dt_detect_free releases it like any other. */
static gboolean _reload_session_cpu(dt_detect_context_t *ctx)
{
  if(!ctx || !ctx->env || !ctx->model_id) return FALSE;

  // nothing to fall back FROM when the configured provider already is
  // the CPU: a swap would rebuild the same session, run the doomed
  // inference twice and toast a GPU failure that never happened. the
  // refine consumer draws the same line (dt_refine_load)
  if(dt_ai_env_get_provider(ctx->env) == DT_AI_PROVIDER_CPU)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] configured provider is CPU, no fallback");
    return FALSE;
  }

  // unload the old session BEFORE creating the new one: on GPU EPs the
  // failing session may still hold VRAM, and the CPU session creation
  // happens to be cheaper if no other ORT state is in flight
  dt_ai_unload_model(ctx->ai_ctx);
  ctx->ai_ctx = NULL;

  dt_ai_context_t *cpu = dt_ai_load_model_ext(ctx->env, ctx->model_id,
                                              NULL, DT_AI_PROVIDER_CPU,
                                              DT_AI_OPT_DEFAULT, NULL, 0);
  if(!cpu)
  {
    dt_print(DT_DEBUG_AI,
             "[detect] CPU fallback session load failed for %s",
             ctx->model_id);
    return FALSE;
  }
  ctx->ai_ctx = cpu;
  return TRUE;
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
  // zeros of the memset. the plain stretch is anisotropic on purpose:
  // a 2:3 frame lands on the 1:1 tensor with its subject distorted,
  // which is exactly how the family is trained, and the inverse
  // mapping below restores the frame's geometry on the way back --
  // "fixing" it with a letterbox the manifest does not declare would
  // feed the model a margin it never saw in training
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

  // the journal must show what the session is REALLY served: the
  // caller logs its render as "rendering WxH for encoding", and a
  // failure right after that line reads as if the frame size had
  // reached the model. it never does -- the tensor is pinned to the
  // manifest's side by construction -- and this line is the proof in
  // the very journal such a hunt starts from
  dt_print(DT_DEBUG_AI,
           "[detect] frame %dx%d -> %dx%d tensor (used %dx%d)",
           rgb_w, rgb_h, s, s, uw, uh);

  int64_t shape_img[4] = { 1, 3, s, s };
  // the output bound at the model's own rank (see dt_detect_load): the
  // trailing dimensions are the map, the leading ones are 1
  int64_t shape_1c[4] = { 1, 1, s, s };
  const int out_ndim = ctx->out_ndim;
  if(out_ndim == 3)
  {
    shape_1c[0] = 1; shape_1c[1] = s; shape_1c[2] = s;
  }
  else if(out_ndim == 2)
  {
    shape_1c[0] = s; shape_1c[1] = s;
  }
  // the image, and for a text-conditioned model the prompt embedding
  // in the slot the session declared for it (see dt_detect_load)
  int64_t shape_txt[2] = { 1, ctx->prompt_dim };
  dt_ai_tensor_t inputs[2];
  const int n_inputs = ctx->t_prompt ? 2 : 1;
  const int img_input = ctx->t_prompt ? 1 - ctx->text_input : 0;
  inputs[img_input] = (dt_ai_tensor_t){
    .data = (void *)ctx->t_image,
    .shape = shape_img, .ndim = 4, .type = DT_AI_FLOAT
  };
  if(ctx->t_prompt)
    inputs[ctx->text_input] = (dt_ai_tensor_t){
      .data = (void *)ctx->t_prompt,
      .shape = shape_txt, .ndim = 2, .type = DT_AI_FLOAT
    };
  dt_ai_tensor_t output = {
    .data = ctx->o_mask,
    .shape = shape_1c, .ndim = out_ndim, .type = DT_AI_FLOAT
  };
  if(dt_ai_run(ctx->ai_ctx, inputs, n_inputs, &output, 1) != 0)
  {
    // one retry on the CPU provider, the restore_* convention: the
    // observed failure mode is the GPU EP running out of VRAM
    // mid-graph (DirectML 8007000E) while darktable's own OpenCL pipe
    // holds the card -- a lost cause on that provider, a few seconds
    // on the CPU. the CPU failing too is the final answer, and the
    // caller's message stays the one the user reads. the flag is set
    // BEFORE the attempt: a failed swap must not be retried either
    const gboolean already = ctx->cpu_fallback_done;
    ctx->cpu_fallback_done = TRUE;
    if(already || !_reload_session_cpu(ctx))
    {
      dt_print(DT_DEBUG_AI, "[detect] inference failed");
      return FALSE;
    }
    dt_print(DT_DEBUG_AI,
             "[detect] GPU inference failed; retrying on CPU");
    dt_control_log(_("AI detection: GPU inference failed, "
                     "falling back to CPU"));
    if(dt_ai_run(ctx->ai_ctx, inputs, n_inputs, &output, 1) != 0)
    {
      dt_print(DT_DEBUG_AI, "[detect] inference failed");
      return FALSE;
    }
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
  // inverse of the mapping above (pixel-centre both ways). a manifest
  // crop_border narrows the used area further: the outermost rows and
  // columns of the map are dropped and the rest is stretched over the
  // whole frame, so the frame edge reads the first TRUSTED row of the
  // map, never the unreliable one -- zeroing the border instead would
  // print a rim of "no detection" along a sky that reaches the top of
  // the frame. the margin is a whole percent of the frame (4 of 352),
  // the map's own resolution (22x22 patches) is far coarser than that
  const int cb = MIN(ctx->crop_border, (MIN(uw, uh) - 1) / 2);
  const int cw = uw - 2 * cb, ch = uh - 2 * cb;
  const float *const inner = ctx->o_mask + (size_t)cb * s + cb;
  const float fx_back = (float)cw / (float)rgb_w;
  const float fy_back = (float)ch / (float)rgb_h;

  DT_OMP_FOR()
  for(int y = 0; y < rgb_h; y++)
  {
    const float sy = ((float)y + 0.5f) * fy_back - 0.5f;
    float *const out = mask + (size_t)y * rgb_w;
    for(int x = 0; x < rgb_w; x++)
    {
      const float sx = ((float)x + 0.5f) * fx_back - 0.5f;
      out[x] = CLAMPF(_sample_1c(inner, s, cw, ch, sx, sy), 0.0f, 1.0f);
    }
  }

  return TRUE;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
