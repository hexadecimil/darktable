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

#pragma once

#include <glib.h>
#include <stdint.h>

// the provenance recipe of an AI-generated raster mask file.
//
// the mask PNG written by the AI object mask is a local cache: it can always
// be regenerated from the image and the parameters that produced it. this
// struct records those parameters. it is embedded verbatim in the params of
// the external raster mask module (iop/rasterfile.c), so it travels with the
// history, the XMP sidecar and styles, while the heavy PNG stays on the
// machine. a library opened on another machine can therefore recompute the
// missing file instead of showing a broken mask.
//
// the struct is hashed verbatim to derive the cache file name
// (content-addressed), so its layout obeys three rules:
//  - every byte is deterministic: no implicit padding, memset before filling;
//  - only types the introspection scanner accepts (int64_t per the overlay
//    module precedent -- no uint64_t, no size_t, no enums from elsewhere);
//  - the layout is frozen: any change bumps the EMBEDDING module's params
//    version, never this struct in place. an unknown `version` is treated
//    like no recipe at all.

#define DT_RF_RECIPE_MAGIC 0x64745243  // "dtRC"; 0 = no recipe
// version 1: clicked prompts, extension block all zero. version 2 (EXT)
// is REQUIRED as soon as any extension field below is nonzero: a build
// that only knows version 1 must refuse such a recipe (and fall back to
// plain file display) rather than replay it without the recorded extra
// stages and write different bytes under its fingerprint. clicked
// sessions with a zeroed extension keep writing version 1, so their
// recipes stay byte-identical to what earlier builds produced
#define DT_RF_RECIPE_VERSION 1
#define DT_RF_RECIPE_VERSION_EXT 2
#define DT_RF_RECIPE_MAX_POINTS 32
#define DT_RF_RECIPE_MODEL_ID_LEN 64
#define DT_RF_RECIPE_MODEL_VERSION_LEN 16
#define DT_RF_RECIPE_MATTING_ID_LEN 28
#define DT_RF_RECIPE_MATTING_VERSION_LEN 12

// how the mask was prompted -- the discriminant of the extension block.
// plain defines, not an enum: the struct field must stay int32_t for the
// introspection scanner, and a version-1 blob's zero must read as POINTS
#define DT_RF_PROMPT_POINTS 0    // clicked prompts; points[] replay
#define DT_RF_PROMPT_SUBJECT 1   // one-shot salient subject; no points
#define DT_RF_PROMPT_SEMANTIC 2  // semantic classes (class_bits); no points

typedef struct dt_rf_recipe_point_t
{
  // normalized input-space coordinates, the space every mask form is stored
  // in: backtransformed through the pipe, divided by the input dimensions
  float x, y;
  int32_t label;         // 1 = foreground, 0 = background
  // 1 when a decode was actually launched right after this point landed.
  // clicks can outpace the compute, so N points do not mean N decodes: a
  // replay must decode exactly at these recorded boundaries to reproduce
  // the iterative refinement context (prev_mask) of the original session
  int32_t decode_after;
  float threshold;       // threshold used by that decode, 0 otherwise
  int32_t _pad;          // explicit, keep zeroed
} dt_rf_recipe_point_t;

typedef struct dt_rf_recipe_t
{
  int32_t magic;         // DT_RF_RECIPE_MAGIC, 0 = no recipe
  int32_t version;       // DT_RF_RECIPE_VERSION
  // distortion-modules hash of the develop history the encoding was made on
  // (crop/rotate/lens change it, exposure/color do not)
  int64_t distort_hash;
  char seg_model[DT_RF_RECIPE_MODEL_ID_LEN];
  char seg_model_version[DT_RF_RECIPE_MODEL_VERSION_LEN];
  char refine_model[DT_RF_RECIPE_MODEL_ID_LEN];       // empty = no refinement
  char refine_model_version[DT_RF_RECIPE_MODEL_VERSION_LEN];
  // the EFFECTIVE encoding render dimensions of the session, not the
  // configured target: a reused .seg cache may have been encoded at other
  // dimensions than the current preference, and the replay must render at
  // exactly these to reproduce the prompt-point geometry
  int32_t encode_w, encode_h;
  int32_t render_size;   // render cap for the native finalisation pass
  int32_t refine_passes;
  float threshold;       // final threshold at finalisation time
  int32_t crf_enabled;
  int32_t crf_iterations;
  float crf_sigma_color;
  float crf_w_bilateral;
  int32_t ai_refine;     // CascadePSP contour refinement enabled
  float ai_refine_margin;
  int32_t cleanup;
  float smoothing;
  float feather;
  // prompt count: 1..DT_RF_RECIPE_MAX_POINTS when prompt_kind is POINTS,
  // exactly 0 for the promptless kinds
  int32_t n_points;
  int32_t _pad0;         // explicit, keep zeroed
  dt_rf_recipe_point_t points[DT_RF_RECIPE_MAX_POINTS];
  // ---- extension block: carved out of the 64 bytes version 1 reserved
  // as zero-filled headroom, so the struct size, the layout of every
  // field above and the fingerprint of any version-1 blob are all
  // unchanged -- a v1 recipe reads back with prompt_kind ==
  // DT_RF_PROMPT_POINTS and every extension stage disabled, which IS its
  // original meaning. the matting fields are reserved from day one so
  // enabling that stage later is a value change (plus the version bump
  // to EXT), never a second layout migration
  int32_t prompt_kind;     // DT_RF_PROMPT_*; selects the replay family
  int32_t detect_input;    // detector model input side; 0 for POINTS
  int64_t class_bits;      // semantic class set; 0 unless SEMANTIC
  // the matting stage. all four are written together or not at all, and
  // only from version EXT: a session that ran the stage records enabled,
  // the operator's table id, its ALGORITHM revision (bumped by any numeric
  // change, because this blob is hashed verbatim to name a
  // content-addressed file and one name may never cover two renders) and
  // the single user scalar. a session that did not leaves all four zero,
  // which is what keeps its recipe byte-identical to a version-1 one
  int32_t matting_enabled; // 1 when the stage ran; 0 = no stage, all zero
  float matting_band;      // band width scale, [0.5, 2.0]; 0 when disabled
  char matting_id[DT_RF_RECIPE_MATTING_ID_LEN];          // NUL-terminated
  char matting_version[DT_RF_RECIPE_MATTING_VERSION_LEN]; // NUL-terminated
} dt_rf_recipe_t;

// the layout is load-bearing twice over: the struct is embedded in module
// params (a size drift silently invalidates every stored history) and it
// is hashed verbatim (an implicit-padding hole would make the fingerprint
// build-dependent). freeze both
G_STATIC_ASSERT(sizeof(dt_rf_recipe_point_t) == 24);
G_STATIC_ASSERT(sizeof(dt_rf_recipe_t)
                == 240 + DT_RF_RECIPE_MAX_POINTS * sizeof(dt_rf_recipe_point_t)
                   + 64);
// the extension block must occupy EXACTLY the 64 bytes version 1 kept as
// reserved[16]: pin its first field to where that array began and its
// last to the end of the struct -- combined with the size assert above
// (the field sizes sum to 64) this leaves no room for implicit padding
// anywhere in the block
G_STATIC_ASSERT(G_STRUCT_OFFSET(dt_rf_recipe_t, prompt_kind)
                == 240
                   + DT_RF_RECIPE_MAX_POINTS * sizeof(dt_rf_recipe_point_t));
G_STATIC_ASSERT(G_STRUCT_OFFSET(dt_rf_recipe_t, matting_version)
                   + DT_RF_RECIPE_MATTING_VERSION_LEN
                == sizeof(dt_rf_recipe_t));

// a recipe is only acted upon when fully understood: an unknown version is
// deliberately NOT valid, the embedding module then falls back to plain
// path/file resolution without touching the recipe bytes. clicked recipes
// carry their prompt points; promptless ones carry none -- a blob mixing
// the two families describes no session anybody could have recorded. a
// promptless kind IS an extension field in use, so the version contract
// above applies: version EXT is mandatory for it, and a version-1 blob
// claiming one marks a writer that forgot the bump -- refused here, so
// the mistake surfaces on the new builds instead of silently degrading
// on the version-1-only ones
static inline gboolean dt_rf_recipe_valid(const dt_rf_recipe_t *r)
{
  if(!r || r->magic != DT_RF_RECIPE_MAGIC)
    return FALSE;
  if(r->version != DT_RF_RECIPE_VERSION
     && r->version != DT_RF_RECIPE_VERSION_EXT)
    return FALSE;
  if(r->prompt_kind == DT_RF_PROMPT_POINTS)
    return r->n_points > 0 && r->n_points <= DT_RF_RECIPE_MAX_POINTS;
  return r->version == DT_RF_RECIPE_VERSION_EXT
         && (r->prompt_kind == DT_RF_PROMPT_SUBJECT
             || r->prompt_kind == DT_RF_PROMPT_SEMANTIC)
         && r->n_points == 0;
}

// the local root folder for raster mask files: the user preference when set,
// a per-machine data dir otherwise. caller frees
gchar *dt_rasterfile_mask_root(void);

// content-addressed fingerprint of a recipe applied to an image: the recipe
// blob hashed verbatim with the image basename, its sensor dimensions and
// its capture datetime. model versions are part of the blob, so a model
// update yields a new name and thus a recompute. the capture datetime is
// what separates two same-named images of the same camera when a history
// (and thus a verbatim recipe) is copied between them -- without it the
// wrong image's mask would resolve silently. known limitation, documented:
// a style or a pasted history carries the ORIGINAL image's clicked
// coordinates; the recompute then produces a mask that is geometrically
// meaningless on the target image, under the target's own name
uint64_t dt_rasterfile_recipe_fingerprint(const dt_rf_recipe_t *recipe,
                                          const char *image_basename,
                                          const int32_t sensor_width,
                                          const int32_t sensor_height,
                                          const int64_t datetime_taken);

// derived cache file name "<image basename>_<fingerprint>.png". caller frees
gchar *dt_rasterfile_recipe_filename(const dt_rf_recipe_t *recipe,
                                     const char *image_basename,
                                     const int32_t sensor_width,
                                     const int32_t sensor_height,
                                     const int64_t datetime_taken);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
