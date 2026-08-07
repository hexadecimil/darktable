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

#include "ai/backend.h"
#include "common/darktable.h"

#include <glib.h>
#include <stdint.h>

G_BEGIN_DECLS

/** Boundary refinement context. Not thread-safe: a context belongs to the
 *  thread that uses it. */
typedef struct dt_refine_context_t dt_refine_context_t;

/** TRUE when a model for the "refine" task is installed and active. */
gboolean dt_refine_available(void);

/** Load and warm up the active "refine" model.
 *
 * Returns NULL when the model is missing, malformed, or when the configured
 * execution provider is CPU -- a CPU inference measures around 20 s, which
 * would freeze the UI for a minute per click.
 *
 * Makes no GLib/GTK call visible to the main thread, so it may be called
 * from the segmentation encode worker.
 */
dt_refine_context_t *dt_refine_load(dt_ai_environment_t *env);

void dt_refine_free(dt_refine_context_t *ctx);

/** Duration of a single inference, in milliseconds, measured during warmup.
 *  A full dt_refine_run() chains three of them. */
double dt_refine_warmup_ms(const dt_refine_context_t *ctx);

/** Side of the static graph (912). */
int dt_refine_get_side(const dt_refine_context_t *ctx);

/**
 * @brief Refine the mask contour inside a region.
 *
 * The region is resampled to the network's fixed square input, refined, and
 * written back. Only pixels inside the region are touched.
 *
 * @param ctx       refinement context.
 * @param rgb       uint8 HWC image, 3 channels, rgb_w x rgb_h. Typically
 *                  dt_seg_get_encoded_rgb(), which shares the mask grid.
 * @param rgb_w     image width, in mask pixels.
 * @param rgb_h     image height, in mask pixels.
 * @param mask      float alpha in [0,1], same dimensions as rgb. Modified
 *                  in place inside the region only.
 * @param threshold binarisation threshold used to build the input seg; pass
 *                  the same value the preview uses.
 * @param roi_x     region origin x, in mask pixels.
 * @param roi_y     region origin y, in mask pixels.
 * @param roi_w     region width, in mask pixels.
 * @param roi_h     region height, in mask pixels.
 * @return TRUE when the mask was modified, FALSE when it was left intact.
 */
gboolean dt_refine_run(dt_refine_context_t *ctx,
                       const uint8_t *const rgb,
                       const int rgb_w,
                       const int rgb_h,
                       float *const mask,
                       const float threshold,
                       const int roi_x,
                       const int roi_y,
                       const int roi_w,
                       const int roi_h);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
