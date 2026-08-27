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

// the guided-filter operators of the matting table, and the table itself.
//
// compiled unconditionally, NOT behind HAVE_AI: these operators are plain
// C over float planes with no network anywhere near them, and the lookup
// dt_matte_find() has to answer for a build without AI support too -- that
// is exactly the build that must recognise a recipe it cannot reproduce
// instead of not knowing what the id means.

#include "common/matte.h"
#include "common/guided_filter.h"
#include "common/rasterfile_recipe.h"

#include <string.h>

// the recipe records these two strings verbatim in fixed-size fields; a
// line that did not fit would be silently truncated into another line's
// id. checked here, where a new line is written
G_STATIC_ASSERT(sizeof("gf-band") <= DT_RF_RECIPE_MATTING_ID_LEN);
G_STATIC_ASSERT(sizeof("1") <= DT_RF_RECIPE_MATTING_VERSION_LEN);

// gf-band -- THE WITNESS LINE.
//
// the band-limited guided filter of the finalisation, called at exactly
// the arguments the render core has always called it with: 4 channels,
// the plain band radius w_gf, sqrt_eps 1.0f, guide_weight 100.0f, output
// clamped to [0,1]. it exists so the table indirection itself can be
// falsified: routing the current stage through the table must produce the
// same plane, bit for bit, as the direct call it replaced. nothing here
// may ever be "improved" -- a different number makes this a different
// algorithm and the whole point of the line is lost.
static gboolean _gf_band_run(const dt_matte_op_t *const op,
                             const dt_matte_stage_t *const stage,
                             float *const out)
{
  if(!op || !stage || !out) return FALSE;
  if(!stage->guide || !stage->hint_bin) return FALSE;
  if(stage->width < 8 || stage->height < 8) return FALSE;

  guided_filter(stage->guide, stage->hint_bin, out,
                stage->width, stage->height, 4, stage->w_gf,
                1.0f, 100.0f, 0.0f, 1.0f);
  return TRUE;
}

const dt_matte_op_t dt_matte_op_gf_band =
{
  .id = "gf-band",
  .version = "1",
  .task = NULL,          // pure C, no model, no registry task
  .model = NULL,
  .caps = DT_MATTE_CPU_OK,
  // ZERO, and the one number in this file that may never change: it is
  // what makes the composition fold back to the two-term form the
  // finalisation has always computed. a nonzero weight here would make
  // the witness line witness nothing
  .wmatte = 0.0f,
  .run = _gf_band_run,
};

// the trimap view of the band. defined here, next to the table, and not
// with an operator: the CALLER builds it (matte.h, DT_MATTE_NEEDS_TRIMAP)
// and the caller is the render core, which is compiled whether or not
// this build carries a line that asks for one.
//
// no DT_OMP_FOR and no reduction: an element-wise map whose cost is a few
// tens of milliseconds on a 20 Mpix region, written serially so the plane
// cannot depend on a thread count -- the same rule the composition loop
// follows, for a plane that feeds a network whose output is recorded in a
// content-addressed file
void dt_matte_trimap_from_band(const float *const hint_bin,
                               const float *const wband,
                               float *const trimap,
                               const size_t npix)
{
  if(!hint_bin || !wband || !trimap) return;
  for(size_t k = 0; k < npix; k++)
    trimap[k] = (wband[k] > 0.0f)
                ? 0.5f
                : ((hint_bin[k] > 0.5f) ? 1.0f : 0.0f);
}

// the table. `gf-adaptive` (the same filter at the wide radius, driven by
// the adaptive band field) joins it with the stage that computes that
// field; until then the witness is the only line a shipped build carries,
// and asking for another by id correctly answers NULL.
//
// `vitmatte-b-912` is NOT a shipped line: it exists only in a build
// configured with USE_MATTE_VITMATTE, because its model weights are
// non-commercial (see the CMake option). Registering it here rather than
// anywhere else is the whole point of a table -- the private operator is
// one entry in the same array, reached through the same lookup, recorded
// in the recipe by the same writer, and refused by every other build
// through the same door
static const dt_matte_op_t *const _matte_table[] =
{
  &dt_matte_op_gf_band,
#ifdef HAVE_MATTE_VITMATTE
  &dt_matte_op_vitmatte,
#endif
};

const dt_matte_op_t *dt_matte_find(const char *id)
{
  if(!id || !*id) return NULL;
  for(size_t i = 0; i < G_N_ELEMENTS(_matte_table); i++)
    if(!strcmp(_matte_table[i]->id, id))
      return _matte_table[i];
  return NULL;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
