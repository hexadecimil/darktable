/*
    This file is part of darktable,
    Copyright (C) 2025-2026 darktable developers.

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

#include "common/rasterfile_io.h"

#include "common/darktable.h"
#include "common/imagebuf.h"
#include "common/math.h"
#include "common/pfm.h"
#include "control/control.h"
#include "imageio/imageio_png.h"

float *dt_rasterfile_io_read(const char *filename,
                             const uint32_t channels,
                             const gboolean quiet,
                             int *swidth,
                             int *sheight)
{
  *swidth = 0;
  *sheight = 0;
  if(!filename || filename[0] == 0) return NULL;

  const char *extension = g_strrstr(filename, ".");
  const gboolean is_png = extension && !g_ascii_strcasecmp(extension, ".png");

  if(is_png)
  {
    dt_imageio_png_t png;
    if(!dt_imageio_png_read_header(filename, &png))
    {
      dt_print(DT_DEBUG_ALWAYS, "failed to read PNG header from '%s'", filename ? filename : "???");
      if(!quiet)
        dt_control_log(_("can't read raster mask file '%s'"), filename ? filename : "???");
      return NULL;
    }

    const size_t rowbytes = png_get_rowbytes(png.png_ptr, png.info_ptr);
    uint8_t *buf = dt_alloc_aligned((size_t)png.height * rowbytes);
    if(!buf)
    {
      fclose(png.f);
      png_destroy_read_struct(&png.png_ptr, &png.info_ptr, NULL);
      dt_print(DT_DEBUG_ALWAYS, "can't read raster mask file '%s'", filename ? filename : "???");
      if(!quiet)
        dt_control_log(_("can't read raster mask file '%s'"), filename ? filename : "???");
      return NULL;
    }

    if(!dt_imageio_png_read_image(&png, buf))
    {
      dt_free_align(buf);
      dt_print(DT_DEBUG_ALWAYS, "can't read raster mask file '%s'", filename ? filename : "???");
      if(!quiet)
        dt_control_log(_("can't read raster mask file '%s'"), filename ? filename : "???");
      return NULL;
    }

    const int width = png.width;
    const int height = png.height;
    float *mask = dt_iop_image_alloc(width, height, 1);
    if(!mask)
    {
      dt_free_align(buf);
      dt_print(DT_DEBUG_ALWAYS, "can't read raster mask file '%s'", filename ? filename : "???");
      if(!quiet)
        dt_control_log(_("can't read raster mask file '%s'"), filename ? filename : "???");
      return NULL;
    }

    if(png.bit_depth < 16)
    {
      const float normalizer = 1.0f / 255.0f;
      DT_OMP_FOR()
      for(size_t k = 0; k < (size_t)width * height; k++)
      {
        const size_t base = 3 * k;
        float val = 0.0f;
        if(channels & DT_RASTERFILE_IO_RED)   val = MAX(val, buf[base] * normalizer);
        if(channels & DT_RASTERFILE_IO_GREEN) val = MAX(val, buf[base + 1] * normalizer);
        if(channels & DT_RASTERFILE_IO_BLUE)  val = MAX(val, buf[base + 2] * normalizer);
        mask[k] = CLIP(val);
      }
    }
    else
    {
      const float normalizer = 1.0f / 65535.0f;
      DT_OMP_FOR()
      for(size_t k = 0; k < (size_t)width * height; k++)
      {
        const size_t base = 6 * k;
        const float red = (buf[base] * 256.0f + buf[base + 1]) * normalizer;
        const float green = (buf[base + 2] * 256.0f + buf[base + 3]) * normalizer;
        const float blue = (buf[base + 4] * 256.0f + buf[base + 5]) * normalizer;
        float val = 0.0f;
        if(channels & DT_RASTERFILE_IO_RED)   val = MAX(val, red);
        if(channels & DT_RASTERFILE_IO_GREEN) val = MAX(val, green);
        if(channels & DT_RASTERFILE_IO_BLUE)  val = MAX(val, blue);
        mask[k] = CLIP(val);
      }
    }

    dt_free_align(buf);
    *swidth = width;
    *sheight = height;
    return mask;
  }

  int width, height, ch, error = 0;
  float *image = dt_read_pfm(filename, &error, &width, &height, &ch, 3);
  float *mask = dt_iop_image_alloc(width, height, 1);
  if(!image || !mask)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "can't read raster mask file '%s'", filename ? filename : "???");
    if(!quiet)
      dt_control_log(_("can't read raster mask file '%s'"), filename ? filename : "???");

    dt_free_align(image);
    dt_free_align(mask);
    return NULL;
  }

  DT_OMP_FOR()
  for(size_t k = 0; k < (size_t)width * height; k++)
  {
    float val = 0.0f;
    if(channels & DT_RASTERFILE_IO_RED)   val = MAX(val, image[k*3]);
    if(channels & DT_RASTERFILE_IO_GREEN) val = MAX(val, image[k*3+1]);
    if(channels & DT_RASTERFILE_IO_BLUE)  val = MAX(val, image[k*3+2]);
    mask[k] = CLIP(val);
  }

  *swidth = width;
  *sheight = height;
  dt_free_align(image);
  return mask;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
