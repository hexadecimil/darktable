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

#include "common/darktable.h"
#include "common/rasterfile_recipe.h"

#include <glib.h>
#include <stdint.h>

G_BEGIN_DECLS

// the task x detector matrix, designed once and consulted everywhere: a
// promptless detection is identified in a recipe by (prompt_kind,
// class_bits), and this table is the single authority mapping that pair
// to the registry task supplying its models, the base name of the form
// it creates, and its post-processing. the replay gates, the rebind and
// the catalogue all resolve through it, so a diagnostic can never
// promise a repair the replay would refuse (they cannot diverge on WHICH
// task a recipe belongs to), and adding a detector is one row here plus
// its registry task -- never a change to the gate code. models
// themselves stay out of the table: a recipe pins its model by the
// recorded (id, version), the active model of a task comes from the
// registry, and the class set travels as a recipe parameter so eight
// semantic entries can share one model
typedef struct dt_detector_t
{
  const char *task;    // registry task whose models implement it
  const char *label;   // base name of the created form, lowercase msgid
  const char *glyph;   // monochrome text glyph heading the catalogue
                       // entry: menus carry no image icons
                       // (GtkImageMenuItem is deprecated), so the
                       // glyph is part of the label text -- it greys
                       // with the entry and follows the theme like
                       // any other character. UTF-8, never translated
  int32_t prompt_kind; // DT_RF_PROMPT_* recorded in the recipe
  int64_t class_bits;  // semantic class set; 0 = salient subject
  gboolean keep_seed;  // TRUE: single-object detection, keep only one
                       // connected component (the seed's, or the largest
                       // when no seed exists); FALSE: multi-component
                       // results are legitimate (a sky between branches)
  gboolean invert;     // TRUE: the shape renders the COMPLEMENT of the
                       // detection (DT_MASKS_RASTER_FLAG_INVERT on its
                       // point) -- same recipe, same file, the other side
                       // selected (Lightroom's "background")
} dt_detector_t;

static const dt_detector_t dt_detectors[] = {
  { "mask-subject", N_("subject"), "✦", DT_RF_PROMPT_SUBJECT, 0, TRUE, FALSE },
  { "mask-subject", N_("background"), "✦", DT_RF_PROMPT_SUBJECT, 0, TRUE, TRUE },
};

// resolve a recipe's (prompt_kind, class_bits) to its table row; NULL
// when no detector matches -- the caller treats that as an unusable
// recipe, never as "pick a default"
// the row a recipe replays through: the task and the post-processing.
// "subject" and "background" share one -- same detection, same file --
// so this answers the first of the two, which is all a replay needs
static inline const dt_detector_t *
dt_detector_find(const int32_t prompt_kind, const int64_t class_bits)
{
  for(size_t i = 0; i < G_N_ELEMENTS(dt_detectors); i++)
    if(dt_detectors[i].prompt_kind == prompt_kind
       && dt_detectors[i].class_bits == class_bits)
      return &dt_detectors[i];
  return NULL;
}

// the row a shape is NAMED after: the same pair, but the side the shape
// renders tells "subject" from "background" (DT_MASKS_RASTER_FLAG_INVERT
// on the raster point). falls back to the replay row when no row of the
// table renders that side, so a name is always found for a known pair
static inline const dt_detector_t *
dt_detector_find_side(const int32_t prompt_kind,
                      const int64_t class_bits,
                      const gboolean inverted)
{
  for(size_t i = 0; i < G_N_ELEMENTS(dt_detectors); i++)
    if(dt_detectors[i].prompt_kind == prompt_kind
       && dt_detectors[i].class_bits == class_bits
       && dt_detectors[i].invert == inverted)
      return &dt_detectors[i];
  return dt_detector_find(prompt_kind, class_bits);
}

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
