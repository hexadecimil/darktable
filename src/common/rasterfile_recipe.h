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
#define DT_RF_RECIPE_VERSION 1
#define DT_RF_RECIPE_MAX_POINTS 32
#define DT_RF_RECIPE_MODEL_ID_LEN 64
#define DT_RF_RECIPE_MODEL_VERSION_LEN 16

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
  int32_t n_points;      // 1..DT_RF_RECIPE_MAX_POINTS when valid
  int32_t _pad0;         // explicit, keep zeroed
  dt_rf_recipe_point_t points[DT_RF_RECIPE_MAX_POINTS];
  int32_t reserved[16];  // zero-filled headroom for compatible extensions
} dt_rf_recipe_t;

// the layout is load-bearing twice over: the struct is embedded in module
// params (a size drift silently invalidates every stored history) and it
// is hashed verbatim (an implicit-padding hole would make the fingerprint
// build-dependent). freeze both
G_STATIC_ASSERT(sizeof(dt_rf_recipe_point_t) == 24);
G_STATIC_ASSERT(sizeof(dt_rf_recipe_t)
                == 240 + DT_RF_RECIPE_MAX_POINTS * sizeof(dt_rf_recipe_point_t)
                   + 64);

// a recipe is only acted upon when fully understood: an unknown version is
// deliberately NOT valid, the embedding module then falls back to plain
// path/file resolution without touching the recipe bytes
static inline gboolean dt_rf_recipe_valid(const dt_rf_recipe_t *r)
{
  return r
         && r->magic == DT_RF_RECIPE_MAGIC
         && r->version == DT_RF_RECIPE_VERSION
         && r->n_points > 0
         && r->n_points <= DT_RF_RECIPE_MAX_POINTS;
}

// the local root folder for raster mask files: the user preference when set,
// a per-machine data dir otherwise. caller frees
gchar *dt_rasterfile_mask_root(void);

// content-addressed fingerprint of a recipe applied to an image: the recipe
// blob hashed verbatim with the image basename and its sensor dimensions.
// model versions are part of the blob, so a model update yields a new name
// and thus a recompute. two same-named images of the same size collide only
// if every clicked coordinate matches too -- accepted and documented
uint64_t dt_rasterfile_recipe_fingerprint(const dt_rf_recipe_t *recipe,
                                          const char *image_basename,
                                          const int32_t sensor_width,
                                          const int32_t sensor_height);

// derived cache file name "<image basename>_<fingerprint>.png". caller frees
gchar *dt_rasterfile_recipe_filename(const dt_rf_recipe_t *recipe,
                                     const char *image_basename,
                                     const int32_t sensor_width,
                                     const int32_t sensor_height);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
