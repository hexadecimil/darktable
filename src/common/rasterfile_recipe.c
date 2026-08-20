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

#include "common/rasterfile_recipe.h"

#include "common/darktable.h"
#include "common/file_location.h"
#include "control/conf.h"

#include <inttypes.h>

gchar *dt_rasterfile_mask_root(void)
{
  gchar *root = dt_conf_get_string("plugins/darkroom/segments/def_path");
  if(!root || !*root)
  {
    g_free(root);
    // default under the config dir of THIS instance: on some platforms the
    // generic user data dir IS another instance's profile, which its
    // documented reset procedure deletes. the files here are a recomputable
    // cache, but they should never be at the mercy of a different install
    char configdir[PATH_MAX] = { 0 };
    dt_loc_get_user_config_dir(configdir, sizeof(configdir));
    root = g_build_filename(configdir, "masks", NULL);
  }
  return root;
}

uint64_t dt_rasterfile_recipe_fingerprint(const dt_rf_recipe_t *recipe,
                                          const char *image_basename,
                                          const int32_t sensor_width,
                                          const int32_t sensor_height,
                                          const int64_t datetime_taken)
{
  // the blob is hashed verbatim -- deterministic by the layout rules of
  // rasterfile_recipe.h (explicit padding, memset at capture)
  dt_hash_t hash = dt_hash(DT_INITHASH, recipe, sizeof(*recipe));
  if(image_basename)
    hash = dt_hash(hash, image_basename, strlen(image_basename));
  hash = dt_hash(hash, &sensor_width, sizeof(sensor_width));
  hash = dt_hash(hash, &sensor_height, sizeof(sensor_height));
  hash = dt_hash(hash, &datetime_taken, sizeof(datetime_taken));
  return hash;
}

gchar *dt_rasterfile_recipe_filename(const dt_rf_recipe_t *recipe,
                                     const char *image_basename,
                                     const int32_t sensor_width,
                                     const int32_t sensor_height,
                                     const int64_t datetime_taken)
{
  const uint64_t fp = dt_rasterfile_recipe_fingerprint
    (recipe, image_basename, sensor_width, sensor_height, datetime_taken);
  return g_strdup_printf("%s_%016" PRIx64 ".png",
                         image_basename ? image_basename : "mask", fp);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
