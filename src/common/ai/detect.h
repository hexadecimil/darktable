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

/** One-shot detection context: the photo alone, or the photo and a
 *  text prompt frozen at export. Not thread-safe: a context belongs to
 *  the thread that uses it. */
typedef struct dt_detect_context_t dt_detect_context_t;

/** TRUE when a model for the given detection task (e.g. "mask-subject")
 *  is installed and active. */
gboolean dt_detect_available(const char *task);

/** Load a detection model BY ID -- never "the active model of the task":
 *  the recipe replay must load the recorded model even when the user has
 *  since activated another one (the dt_refine_load lesson).
 *
 * `task` (nullable) is the registry task the caller expects; a model of
 * another task family is refused -- the id may come from an untrusted
 * recipe blob.
 *
 * The calling contract is read from the model's manifest attributes:
 * input_sizes (square input side), letterbox (aspect-preserving pad vs
 * plain stretch), output_activation ("sigmoid" over logits, the family
 * default, or "none"), mean/std (normalisation statistics on the [0,1]
 * scale, ImageNet by default), crop_border (rows and columns dropped
 * on each side of the map before the resample back, 0 by default).
 * The model output may be declared [1,1,S,S] or [1,S,S]: the rank is
 * read from the session, the element count is what is checked.
 *
 * A model with TWO inputs is text-conditioned: besides the image it
 * takes the embedding of the prompt it answers about, one of a fixed
 * set frozen at export. The set is named by attributes.prompts, its
 * embeddings lie in prompts.bin next to the model (n_prompts x
 * prompt_dim float32, little-endian, in the order of the list), and
 * `prompt_index` selects the one this context runs with. A model with
 * one input ignores `prompt_index`; a two-input model refuses a
 * negative index, an index past the list, or a prompts.bin whose size
 * disagrees with the manifest.
 *
 * No provider restriction and no warmup inference: a detection is a
 * single call on a background job, so a CPU fallback is slow but blocks
 * nothing, and a warmup would exactly double the cost.
 *
 * `env` is borrowed for the lifetime of the context (the CPU retry of
 * dt_detect_run reloads through it): destroy the context first, the
 * environment after, as every call site of this consumer already does.
 *
 * Makes no GLib/GTK call visible to the main thread, so it may be
 * called from a worker job.
 */
dt_detect_context_t *dt_detect_load(dt_ai_environment_t *env,
                                    const char *model_id,
                                    const char *task,
                                    const int prompt_index);

void dt_detect_free(dt_detect_context_t *ctx);

/** Side of the square model input (attributes.input_sizes[0]). */
int dt_detect_get_side(const dt_detect_context_t *ctx);

/**
 * @brief Detect on an RGB frame, one inference, no click.
 *
 * The frame is resampled to the model's square input (stretched, or
 * letterboxed with zero padding when the manifest says so), normalised,
 * run through the network, activated, and the result is resampled back
 * onto the frame's own grid -- the encoding grid of the mask pipeline,
 * so the caller feeds the finalisation exactly what the interactive
 * decoder feeds it.
 *
 * The model tensor is ALWAYS side x side, the size the manifest
 * declares: whatever the frame dimensions, the resample happens here
 * (bilinear, pixel-centre both ways) and no caller has to pre-fit its
 * render -- the frame size never reaches the session.
 *
 * A failed inference is retried ONCE on the CPU provider (the
 * restore_* convention, with its toast): the known failure mode is
 * the GPU provider running out of VRAM while darktable's own OpenCL
 * pipe holds the card. The swapped session stays on the context, so
 * a later call neither rebuilds it nor retries the doomed provider;
 * only when the CPU fails too does this return FALSE. A session the
 * user already configured on the CPU has nothing to fall back to,
 * and its failure is final at once, without the toast.
 *
 * @param ctx   detection context.
 * @param rgb   uint8 HWC image, 3 channels, rgb_w x rgb_h.
 * @param rgb_w frame width, in pixels.
 * @param rgb_h frame height, in pixels.
 * @param mask  caller-allocated rgb_w * rgb_h floats; on success holds
 *              the soft detection in [0,1]. Untouched on failure.
 * @return TRUE when the mask was written.
 */
gboolean dt_detect_run(dt_detect_context_t *ctx,
                       const uint8_t *const rgb,
                       const int rgb_w,
                       const int rgb_h,
                       float *const mask);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
