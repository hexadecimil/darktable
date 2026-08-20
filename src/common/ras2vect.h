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

/* Returns a list of path forms after having vectorized the raster mask.
   The coordinates are either on mask space: (0 x 0) -> (width x height)
   or if image is set (not NULL) on image spaces making the masks directly
   usable on the corresponding image.

   turdsize  – potrace turdsize: area of largest speckle to suppress.
               negative = historical default (50), suited to full-resolution
               masks; values >= 0 are honoured but floored at 2 px^2 to drop
               single-pixel thresholding noise.
   alphamax  – potrace alphamax: corner threshold (0 = all sharp, 1.0 = balanced,
               1.3 = maximum smoothing). Higher = fewer control points.
   opttolerance – potrace curve simplification tolerance, in pixels of the
               traced grid: tight (~0.3) on a coarse working grid, looser
               (~0.8) on a native-resolution mask.

   Returns NULL (and *out_signs set to NULL) if an allocation fails.

   If out_signs is not NULL, a parallel GList of GINT_TO_POINTER is
   returned: '+' for outer boundaries, '-' for holes.
   The caller must free this list with g_list_free().
*/
// pixels with mask < threshold are inside the form
GList *ras2forms(const float *mask,
                 const int width,
                 const int height,
                 const dt_image_t *const image,
                 const float threshold,
                 const int turdsize,
                 const double alphamax,
                 const double opttolerance,
                 GList **out_signs);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
