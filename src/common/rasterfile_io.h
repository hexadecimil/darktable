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

G_BEGIN_DECLS

// channel-selection bits, numerically identical to the
// DT_RASTERFILE_MODE_* bits of iop/rasterfile.c (RED=1, GREEN=2, BLUE=4,
// ALL=7) so the module passes its mode through unchanged
#define DT_RASTERFILE_IO_RED   1
#define DT_RASTERFILE_IO_GREEN 2
#define DT_RASTERFILE_IO_BLUE  4
#define DT_RASTERFILE_IO_ALL   7

// read a raster mask file (PNG 8/16 bit or PFM) into a single-channel
// float buffer in [0,1], taking the per-pixel MAX of the selected
// channels. returns a dt_iop_image_alloc'd buffer (free with
// dt_free_align) and its dimensions, or NULL on any failure. `quiet`
// suppresses the user-facing toast (kept for log). EXACT move of the
// former _read_rasterfile of iop/rasterfile.c -- behaviour frozen
float *dt_rasterfile_io_read(const char *filename,
                             const uint32_t channels,
                             const gboolean quiet,
                             int *swidth,
                             int *sheight);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
