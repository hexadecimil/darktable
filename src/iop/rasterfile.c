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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "develop/tiling.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/imageop_math.h"
#include "common/interpolation.h"
#include "common/fast_guided_filter.h"
#include "common/pfm.h"
#include "common/ras2vect.h"
#include "common/rasterfile_recipe.h"
#include "common/utility.h"
#include "develop/masks.h"
#include "develop/masks/object_recipe.h"
#include "imageio/imageio_png.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"

#include <dirent.h>

#if defined (_WIN32)
#include "win/getdelim.h"
#include "win/scandir.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define SET_THRESHOLD 0.6f

DT_MODULE_INTROSPECTION(2, dt_iop_rasterfile_params_t)

typedef enum dt_iop_rasterfile_mode_t
{
  DT_RASTERFILE_MODE_ALL = 7,     // $DESCRIPTION: "all RGB channels"
  DT_RASTERFILE_MODE_RED = 1,     // $DESCRIPTION: "only red"
  DT_RASTERFILE_MODE_GREEN = 2,   // $DESCRIPTION: "only green"
  DT_RASTERFILE_MODE_BLUE = 4,   // $DESCRIPTION: "only blue"
  DT_RASTERFILE_MODE_REDGREEN = DT_RASTERFILE_MODE_RED | DT_RASTERFILE_MODE_GREEN, // $DESCRIPTION: "red and green"
  DT_RASTERFILE_MODE_REDBLUE = DT_RASTERFILE_MODE_RED | DT_RASTERFILE_MODE_BLUE, // $DESCRIPTION: "red and blue"
  DT_RASTERFILE_MODE_GREENBLUE = DT_RASTERFILE_MODE_GREEN | DT_RASTERFILE_MODE_BLUE, // $DESCRIPTION: "green and blue"
} dt_iop_rasterfile_mode_t;

#define RASTERFILE_MAXFILE 2048

typedef struct dt_iop_rasterfile_params_t
{
  dt_iop_rasterfile_mode_t mode;  // $DEFAULT: DT_RASTERFILE_MODE_ALL $DESCRIPTION: "mode"
  char path[RASTERFILE_MAXFILE];
  char file[RASTERFILE_MAXFILE];
  int32_t _pad;                   // explicit: aligns the recipe to 8 bytes
  // provenance of an AI-generated mask file, or all-zero for a plain file
  // picked by hand. when valid, the file is resolved content-addressed
  // against the LOCAL mask root (see commit_params); path/file above are
  // then informative only, so a library moved between machines keeps
  // resolving. inert bytes on builds without the AI subsystem
  dt_rf_recipe_t recipe;
} dt_iop_rasterfile_params_t;

typedef struct dt_iop_rasterfile_data_t
{
  dt_iop_rasterfile_mode_t mode;
  char filepath[PATH_MAX];
  // TRUE when the params carry a valid recipe: a missing or unreadable
  // file is then an expected transient state (the mask is recomputable and
  // a recompute is the answer), so the read path logs without toasting
  gboolean quiet;
  // committed copy of the provenance recipe, so the pixelpipe safety net
  // can schedule a recompute without touching the live GUI params
  dt_rf_recipe_t recipe;
} dt_iop_rasterfile_data_t;

typedef struct dt_rasterfile_cache_t
{
  dt_pthread_mutex_t lock;  // all access to cache data MUST be done in mutex locked state !
  volatile dt_hash_t hash;  // as the hash is shared between threads make sure it's actually read from mem
  int width, height;
  float *volatile mask;
} dt_rasterfile_cache_t;

const char *name()
{
  return _("external raster masks");
}

const char *aliases()
{
  return _("raster|mask");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self,
      _("read PFM/PNG files recorded for use as raster masks"),
      _("corrective or creative"),
      _("linear, raw, scene-referred"),
      _("linear, raw"),
      _("linear, raw, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL;
}

int flags()
{
  return IOP_FLAGS_WRITE_RASTER;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return (pipe && !dt_image_is_raw(&pipe->image)) ? IOP_CS_RGB : IOP_CS_RAW;
}

typedef struct dt_iop_rasterfile_gui_data_t
{
  GtkWidget *mode;
  GtkWidget *fbutton;
  GtkWidget *file;
  GtkWidget *vectorize;
  GtkWidget *recompute;
} dt_iop_rasterfile_gui_data_t;

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  typedef struct dt_iop_rasterfile_params_v1_t
  {
    dt_iop_rasterfile_mode_t mode;
    char path[RASTERFILE_MAXFILE];
    char file[RASTERFILE_MAXFILE];
  } dt_iop_rasterfile_params_v1_t;

  if(old_version == 1)
  {
    const dt_iop_rasterfile_params_v1_t *o = old_params;
    dt_iop_rasterfile_params_t *n = calloc(1, sizeof(dt_iop_rasterfile_params_t));

    n->mode = o->mode;
    memcpy(n->path, o->path, sizeof(n->path));
    memcpy(n->file, o->file, sizeof(n->file));
    // recipe stays zeroed: no provenance, plain path/file resolution

    *new_params = n;
    *new_params_size = sizeof(dt_iop_rasterfile_params_t);
    *new_version = 2;
    return 0;
  }
  return 1;
}

void modify_roi_in(dt_iop_module_t *self,
                   dt_dev_pixelpipe_iop_t *piece,
                   const dt_iop_roi_t *const roi_out,
                   dt_iop_roi_t *roi_in)
{
  *roi_in = *roi_out;
  roi_in->scale = 1.0f;
  roi_in->x = 0;
  roi_in->y = 0;
  roi_in->width = piece->buf_in.width;
  roi_in->height = piece->buf_in.height;
}

static void _vectorize_button_clicked(GtkWidget *widget,
                                      dt_iop_module_t *self)
{
  dt_develop_t *dev = darktable.develop;

  dt_rasterfile_cache_t *cd = self->data;

  dt_pthread_mutex_lock(&cd->lock);

  const dt_image_t *const image = &(self->dev->image_storage);
  // full-resolution file: a negative turdsize selects the historical speckle
  // cleanup (50) and 0.8 the matching curve tolerance, otherwise every tiny
  // speckle becomes a form and floods the mask manager with anchors
  GList *forms = ras2forms(cd->mask, cd->width, cd->height, image,
                           SET_THRESHOLD, -1, 0.0, 0.8, NULL);

  dt_pthread_mutex_unlock(&cd->lock);

  const int nbform = g_list_length(forms);
  if(nbform == 0)
  {
    dt_control_log(_("no mask extracted from the raster file\n"
                     "make sure the masks have proper contrast"));
  }
  else
  {
    dt_control_log(ngettext("%d mask extracted from the raster file",
                            "%d masks extracted from the raster file", nbform), nbform);

    // add all forms into the mask manager

    dt_masks_register_forms(dev, forms);
  }
}

static float *_read_rasterfile(char *filename,
                               const dt_iop_rasterfile_mode_t mode,
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
        if(mode & DT_RASTERFILE_MODE_RED)   val = MAX(val, buf[base] * normalizer);
        if(mode & DT_RASTERFILE_MODE_GREEN) val = MAX(val, buf[base + 1] * normalizer);
        if(mode & DT_RASTERFILE_MODE_BLUE)  val = MAX(val, buf[base + 2] * normalizer);
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
        if(mode & DT_RASTERFILE_MODE_RED)   val = MAX(val, red);
        if(mode & DT_RASTERFILE_MODE_GREEN) val = MAX(val, green);
        if(mode & DT_RASTERFILE_MODE_BLUE)  val = MAX(val, blue);
        mask[k] = CLIP(val);
      }
    }

    dt_free_align(buf);
    *swidth = width;
    *sheight = height;
    return mask;
  }

  int width, height, channels, error = 0;
  float *image = dt_read_pfm(filename, &error, &width, &height, &channels, 3);
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
    if(mode & DT_RASTERFILE_MODE_RED)   val = MAX(val, image[k*3]);
    if(mode & DT_RASTERFILE_MODE_GREEN) val = MAX(val, image[k*3+1]);
    if(mode & DT_RASTERFILE_MODE_BLUE)  val = MAX(val, image[k*3+2]);
    mask[k] = CLIP(val);
  }

  *swidth = width;
  *sheight = height;
  dt_free_align(image);
  return mask;
}

static int _check_extension(const struct dirent *namestruct)
{
  const char *filename = namestruct->d_name;
  if(!filename || !filename[0]) return 0;
  const char *p = g_strrstr(filename, ".");
  return p
         && (!g_ascii_strcasecmp(p, ".pfm")
             || !g_ascii_strcasecmp(p, ".png"));
}

static void _update_filepath(dt_iop_module_t *self)
{
  dt_iop_rasterfile_gui_data_t *g = self->gui_data;
  dt_iop_rasterfile_params_t *p = self->params;
  if(!p->path[0] || !p->file[0])
  {
    dt_bauhaus_combobox_clear(g->file);
    // Making the empty widget insensitive is very important, because
    // attempts to interact with it trigger a bug in GTK (as of 3.24.49)
    // that disables the display of tooltips
    gtk_widget_set_sensitive(g->file, FALSE);
    return;
  }
  gtk_widget_set_sensitive(g->file, TRUE);

  if(!dt_bauhaus_combobox_set_from_text(g->file, p->file))
  {
    struct dirent **entries;
    const int numentries = scandir(p->path, &entries, _check_extension, alphasort);
    dt_bauhaus_combobox_clear(g->file);

    for(int i = 0; i < numentries; i++)
    {
      const char *file = entries[i]->d_name;
      char *normalized_filename = g_locale_to_utf8(file, -1, NULL, NULL, NULL);
      dt_bauhaus_combobox_add_aligned(g->file, normalized_filename,
                                      DT_BAUHAUS_COMBOBOX_ALIGN_LEFT);
      free(entries[i]);
      g_free(normalized_filename);
    }
    if(numentries != -1)
      free(entries);

    if(!dt_bauhaus_combobox_set_from_text(g->file, p->file))
    { // file may have disappeared - show it
      char *invalidfilepath = g_strconcat(" ??? ", p->file, NULL);
      dt_bauhaus_combobox_add_aligned(g->file, invalidfilepath,
                                      DT_BAUHAUS_COMBOBOX_ALIGN_LEFT);
      dt_bauhaus_combobox_set_from_text(g->file, invalidfilepath);
      g_free(invalidfilepath);
    }
  }
}

static void _fbutton_clicked(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_rasterfile_gui_data_t *g = self->gui_data;
  dt_iop_rasterfile_params_t *p = self->params;

  gchar *mfolder = dt_conf_get_string("plugins/darkroom/segments/def_path");
  if(strlen(mfolder) == 0)
  {
    dt_print(DT_DEBUG_ALWAYS, "raster mask files root folder not defined");
    dt_control_log(_("raster mask files root folder not defined"));
    g_free(mfolder);
    return;
  }

  GtkWidget *win = dt_ui_main_window(darktable.gui->ui);
  GtkFileChooserNative *filechooser = gtk_file_chooser_native_new(
        _("select raster mask file"), GTK_WINDOW(win), GTK_FILE_CHOOSER_ACTION_OPEN,
        _("_select"), _("_cancel"));
  gtk_file_chooser_set_select_multiple(GTK_FILE_CHOOSER(filechooser), FALSE);
  gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(filechooser), mfolder);
  GtkFileFilter *filter = GTK_FILE_FILTER(gtk_file_filter_new());
  // only pfm/png files yet supported
  gtk_file_filter_add_pattern(filter, "*.pfm");
  gtk_file_filter_add_pattern(filter, "*.PFM");
  gtk_file_filter_add_pattern(filter, "*.png");
  gtk_file_filter_add_pattern(filter, "*.PNG");
  gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(filechooser), filter);
  gtk_file_chooser_set_filter(GTK_FILE_CHOOSER(filechooser), filter);

  if(gtk_native_dialog_run(GTK_NATIVE_DIALOG(filechooser)) == GTK_RESPONSE_ACCEPT)
  {
    gchar *filepath = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(filechooser));
    const gboolean within = (strlen(filepath) > strlen(mfolder))
                         && (memcmp(filepath, mfolder, strlen(mfolder)) == 0);
    if(within)
    {
      char *relativepath = g_path_get_dirname(filepath);
      dt_strlcpy_to_fixed(p->path, relativepath, sizeof(p->path));
      g_free(relativepath);

      char *bname = g_path_get_basename(filepath);
      dt_strlcpy_to_fixed(p->file, bname, sizeof(p->file));
      g_free(bname);

      // picking a file by hand detaches any AI provenance: from here on the
      // chosen file IS the mask, no fingerprint may override it
      memset(&p->recipe, 0, sizeof(p->recipe));

      _update_filepath(self);
      dt_dev_add_history_item(darktable.develop, self, TRUE);
    }
    else
    {
      dt_print(DT_DEBUG_ALWAYS, "selected file not within raster masks root folder");
      dt_control_log(_("selected file not within raster masks root folder"));
    }
    g_free(filepath);
    gtk_widget_set_sensitive(g->file, p->path[0] && p->file[0]);
    gtk_widget_set_sensitive(g->vectorize, p->path[0] && p->file[0]);
  }
  g_free(mfolder);
  g_object_unref(filechooser);
}

// provisional dev trigger for the AI mask edit session, replaced by the
// validated UX (the planned provenance section with its own edit button):
// behind a conf key defaulting to FALSE, the 'recompute mask' button
// becomes 'edit mask' when a valid recipe exists. only the entry point it
// calls, dt_object_mask_edit_begin, is meant to stay
#define CONF_RASTERFILE_DEV_EDIT_KEY "plugins/darkroom/masks/object/dev_edit_button"

static gboolean _dev_edit_enabled(void)
{
  return dt_conf_key_exists(CONF_RASTERFILE_DEV_EDIT_KEY)
         && dt_conf_get_bool(CONF_RASTERFILE_DEV_EDIT_KEY);
}

// redo the mask from its recipe with the models installed NOW. this is the
// user-consented escape when the recorded model is gone and the pinned
// replay rightly refuses to run: the recipe is rebound to the active
// models and persisted FIRST -- its fingerprint changes, so the redone
// mask lives under a new name and resolution follows automatically
static void _recompute_clicked(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_rasterfile_params_t *p = self->params;
  if(!dt_rf_recipe_valid(&p->recipe) || !self->dev)
    return;

  // provisional dev trigger (see _dev_edit_enabled): reopen the recorded
  // session interactively instead of recomputing headless
  if(_dev_edit_enabled())
  {
    dt_object_edit_target_t target = { 0 };
    target.kind = DT_OBJECT_EDIT_RASTER;
    target.raster_multi_priority = self->multi_priority;
    target.has_recipe = TRUE;
    target.recipe = p->recipe;
    dt_object_mask_edit_begin(self, &target);
    return;
  }

  dt_rf_recipe_t rebound = p->recipe;
  if(!dt_object_recipe_rebind_models(&rebound))
  {
    dt_control_log(_("no AI model available to recompute this mask"));
    return;
  }

  if(memcmp(&rebound, &p->recipe, sizeof(rebound)) != 0)
  {
    p->recipe = rebound;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
  }
  if(dt_object_recipe_schedule_recompute(&p->recipe,
                                         self->dev->image_storage.id))
    dt_control_log(_("recomputing the mask with the current model..."));
}

static void _file_callback(GtkWidget *widget, dt_iop_module_t *self)
{
  dt_iop_rasterfile_params_t *p = self->params;
  const gchar *select = dt_bauhaus_combobox_get_text(widget);
  dt_strlcpy_to_fixed(p->file, select, sizeof(p->file));
  // same detachment as the file chooser: a hand-picked file wins
  memset(&p->recipe, 0, sizeof(p->recipe));
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void _clear_cache(dt_rasterfile_cache_t *cache)
{
  dt_free_align(cache->mask);
  cache->mask = NULL;
  cache->width = cache->height = 0;
  cache->hash = DT_INVALID_HASH;
}

static inline dt_hash_t _get_cache_hash(dt_iop_module_t *self)
{
  dt_hash_t hash = dt_hash(DT_INITHASH, self->params, self->params_size);
  // not technically required but possibly reduces mem footprint
  hash = dt_hash(hash,
                 &self->dev->image_storage.id,
                 sizeof(self->dev->image_storage.id));
  return hash;
}

static float *_get_rasterfile_mask(dt_dev_pixelpipe_iop_t *piece,
                                   const dt_iop_roi_t *const roi,
                                   const dt_iop_roi_t *const roo)
{
  dt_iop_module_t *self = piece->module;
  dt_iop_rasterfile_data_t *d = piece->data;

  dt_rasterfile_cache_t *cd = self->data;
  float *res = NULL;

  dt_pthread_mutex_lock(&cd->lock);
  const dt_hash_t hash = _get_cache_hash(self);
  if(hash != cd->hash)
  {
    _clear_cache(cd);
    dt_print(DT_DEBUG_PIPE,
             "read image raster file `%s'", d->filepath);
    cd->mask = _read_rasterfile(d->filepath, d->mode, d->quiet,
                                &cd->width, &cd->height);
    cd->hash = cd->mask ? hash : DT_INVALID_HASH;
    dt_print(DT_DEBUG_PIPE,
             "got raster mask data %p %dx%d", cd->mask, cd->width, cd->height);

    // safety net: a valid recipe can regenerate a missing or unreadable
    // file. never from a thumbnail pipe (cheap pipes must not flood the
    // job queue; the full/export pipes will come). two regimes:
    //  - a pipe whose output leaves the machine (export) or a context
    //    with no job system at all (darktable-cli, whose export is the
    //    only reason its pipes run) blocks and recomputes NOW: these
    //    pipes run once, a wrong output is final;
    //  - the darkroom schedules asynchronously and gets reprocessed when
    //    the recompute lands. leaving cd->hash invalid meanwhile is
    //    deliberate: the next pipe run rereads the file
    if(!cd->mask
       && !(piece->pipe->type & DT_DEV_PIXELPIPE_THUMBNAIL)
       && dt_rf_recipe_valid(&d->recipe))
    {
      const gboolean sync_ctx
        = (piece->pipe->type & DT_DEV_PIXELPIPE_EXPORT)
          || !dt_control_running();
      const gboolean retry_now = sync_ctx
        ? dt_object_recipe_recompute_now(&d->recipe, piece->pipe->image.id)
        : dt_object_recipe_schedule_recompute(&d->recipe,
                                              piece->pipe->image.id);
      if(retry_now)
      {
        cd->mask = _read_rasterfile(d->filepath, d->mode, d->quiet,
                                    &cd->width, &cd->height);
        cd->hash = cd->mask ? hash : DT_INVALID_HASH;
      }
    }
  }
  if(cd->mask)
  {
    const gboolean scale = cd->width != roi->width || cd->height != roi->height;
    float *tmp = scale ? dt_iop_image_alloc(roi->width, roi->height, 1) : cd->mask;
    if(tmp)
    {
      if(scale)
        interpolate_bilinear(cd->mask, cd->width, cd->height, tmp,
                             roi->width, roi->height, 1);
      res = dt_iop_image_alloc(roo->width, roo->height, 1);
      if(res)
        self->distort_mask(self, piece, tmp, res, roi, roo);
      if(scale)
        dt_free_align(tmp);
    }
  }

  dt_pthread_mutex_unlock(&cd->lock);
  return res;
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  dt_dev_pixelpipe_t *pipe = piece->pipe;
  const ssize_t ch = pipe->dsc.filters ? 1 : 4;
  const gboolean fullpipe = dt_pipe_is_full(pipe);
  const gboolean visual = fullpipe && dt_iop_has_focus(self);

  const int devid = pipe->devid;
  cl_int err = DT_OPENCL_PROCESS_CL;

  if(visual) return err;

  if(roi_out->scale != roi_in->scale && ch == 4)
    err = dt_iop_clip_and_zoom_cl(devid, dev_out, dev_in, roi_out, roi_in);
  else
  {
    const size_t iorigin[2] = { roi_out->x, roi_out->y };
    const size_t region[2] = { roi_out->width, roi_out->height };
    err = dt_opencl_enqueue_copy_image(devid, dev_in, dev_out, iorigin, CLIMG_ORIGIN, region);
  }

  if(dt_iop_is_raster_mask_used(piece->module, BLEND_RASTER_ID)
     && (err == CL_SUCCESS))
  {
    float *mask = _get_rasterfile_mask(piece, roi_in, roi_out);
    if(mask)
      dt_iop_piece_set_raster(piece, mask, roi_in, roi_out);
    else
      dt_iop_piece_clear_raster(piece, NULL);
  }
  else
    dt_iop_piece_clear_raster(piece, NULL);

  return err;
}
#endif  // OpenCL

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const float *const in = (float *)ivoid;
  float *const out = (float *)ovoid;
  dt_dev_pixelpipe_t *pipe = piece->pipe;
  const uint32_t filters = pipe->dsc.filters;
  const ssize_t ch = filters ? 1 : 4;

  if(roi_out->scale != roi_in->scale && ch == 4)
  {
    const dt_interpolation_t *itor = dt_interpolation_new(DT_INTERPOLATION_USERPREF_WARP);
    dt_interpolation_resample(itor, out, roi_out, in, roi_in);
  }
  else
    dt_iop_copy_image_roi(out, in, ch, roi_in, roi_out);

  const gboolean fullpipe = dt_pipe_is_full(pipe);
  const gboolean request = dt_iop_is_raster_mask_used(piece->module, BLEND_RASTER_ID);
  const gboolean visual = fullpipe && dt_iop_has_focus(self);
  float *mask = visual || request ? _get_rasterfile_mask(piece, roi_in, roi_out) : NULL;

  if(visual)
  {
    pipe->mask_display = DT_DEV_PIXELPIPE_DISPLAY_PASSTHRU;
    if(ch == 1)
    {
      // simple blur to remove CFA colors:
      dt_box_mean(out, roi_out->height, roi_out->width, 1, 3, 2);
      DT_OMP_FOR()
      for(size_t k = 0; k < (size_t)roi_out->width * roi_out->height; k++)
        out[k] =
          0.2f
          * CLAMPF(sqrtf(out[k]), 0.0f, 0.5f)
          + (mask ? mask[k] : 0.0f);
    }
    else
    {
      DT_OMP_FOR()
      for(size_t k = 0; k < (size_t)roi_out->width * roi_out->height; k++)
      {
        const float val =
          0.2f
          * CLAMPF(sqrtf(0.33f * (out[4*k] + out[4*k+1]+ out[4*k+2])), 0.0f, 0.5f)
          + (mask ? mask[k] : 0.0f);
        for_three_channels(m) out[4*k+m] = val;
      }
    }
  }

  if(request && mask)
    dt_iop_piece_set_raster(piece, mask, roi_in, roi_out);
  else
  {
    dt_iop_piece_clear_raster(piece, NULL);
    dt_free_align(mask);
  }
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_rasterfile_params_t *p = (dt_iop_rasterfile_params_t *)p1;
  dt_iop_rasterfile_data_t *d = piece->data;

  d->mode = p->mode;
  // a recipe makes the file recomputable: its absence is then a transient
  // state handled by a recompute, not a toast (see _read_rasterfile)
  d->quiet = dt_rf_recipe_valid(&p->recipe);
  d->recipe = p->recipe;
  gchar *fullpath = NULL;
  if(dt_rf_recipe_valid(&p->recipe))
  {
    // content-addressed resolution: the recipe fingerprint names the file
    // under the LOCAL mask root, whatever path was recorded on the machine
    // that produced it
    const dt_image_t *img = &pipe->image;
    gchar *base = g_path_get_basename(img->filename);
    char *dot = g_strrstr(base, ".");
    if(dot) *dot = '\0';
    gchar *fname = dt_rasterfile_recipe_filename(&p->recipe, base,
                                                 img->width, img->height,
                                                 img->exif_datetime_taken);
    gchar *root = dt_rasterfile_mask_root();
    fullpath = g_build_filename(root, fname, NULL);
    g_free(root);
    g_free(fname);
    g_free(base);
  }
  else
    fullpath = g_build_filename(p->path, p->file, NULL);

  dt_strlcpy_to_fixed(d->filepath, fullpath, sizeof(d->filepath));
  g_free(fullpath);
}

void tiling_callback(dt_iop_module_t *self,
                     dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  tiling->maxbuf = 1.0f;
  tiling->align = 1;
  tiling->overhead = 0;
  tiling->factor = 2.0f;
}

void reload_defaults(dt_iop_module_t *self)
{
  // we might be called from presets update infrastructure => there is
  // no image
  if(!self->dev || !dt_is_valid_imgid(self->dev->image_storage.id))
    return;

  self->default_enabled = FALSE;
  dt_iop_rasterfile_params_t *dp = self->default_params;
  memset(dp->path, 0, sizeof(dp->path));
  memset(dp->file, 0, sizeof(dp->file));
  memset(&dp->recipe, 0, sizeof(dp->recipe));
}

void distort_mask(dt_iop_module_t *self,
                  dt_dev_pixelpipe_iop_t *piece,
                  const float *const in,
                  float *const out,
                  const dt_iop_roi_t *const roi_in,
                  const dt_iop_roi_t *const roi_out)
{
  if(roi_out->scale != roi_in->scale)
  {
    const dt_interpolation_t *itor = dt_interpolation_new(DT_INTERPOLATION_USERPREF_WARP);
    dt_interpolation_resample_mask(itor, out, roi_out, in, roi_in);
  }
  else
    dt_iop_copy_image_roi(out, in, 1, roi_in, roi_out);
}

void gui_changed(dt_iop_module_t *self,
                 GtkWidget *w,
                 void *previous)
{
  dt_iop_rasterfile_gui_data_t *g = self->gui_data;
  dt_iop_rasterfile_params_t *p = self->params;

  if(!w || w == g->mode)
    _update_filepath(self);

  if(!w)
  {
    dt_rasterfile_cache_t *cd = self->data;
    dt_pthread_mutex_lock(&cd->lock);
    const dt_hash_t hash = _get_cache_hash(self);
    const gboolean other = hash != cd->hash;
    if(other) _clear_cache(cd);
    dt_pthread_mutex_unlock(&cd->lock);

    if(other)
      dt_dev_reprocess_center(self->dev, self->iop_order);

    // proactive: a valid recipe whose cache file is missing (library moved
    // to this machine, cache purged) is regenerated without waiting for a
    // pipe to fail reading it. the anti-respawn table absorbs repeats
    if(dt_rf_recipe_valid(&p->recipe) && self->dev)
    {
      const dt_image_t *img = &self->dev->image_storage;
      gchar *base = g_path_get_basename(img->filename);
      char *dot = g_strrstr(base, ".");
      if(dot) *dot = '\0';
      gchar *fname = dt_rasterfile_recipe_filename(&p->recipe, base,
                                                   img->width, img->height,
                                                   img->exif_datetime_taken);
      gchar *root = dt_rasterfile_mask_root();
      gchar *fullpath = g_build_filename(root, fname, NULL);
      // an open AI edit session on this instance keeps the file
      // legitimately absent while it prepares a replacement: without the
      // gate every gui_update would schedule a headless recompute
      // concurrent with the session's own inference stack -- and pin a
      // fresh failure when the recorded model is still missing. the
      // session gate falls when the closing gesture frees the session
      // data, which is exactly when the finalisation job STARTS: the
      // second gate covers the seconds it then spends rendering and
      // inferring, when the replacement file still does not exist
      if(!g_file_test(fullpath, G_FILE_TEST_EXISTS)
         && !dt_object_mask_edit_active(self->op, self->multi_priority)
         && !dt_object_mask_finalize_running())
        dt_object_recipe_schedule_recompute(&p->recipe, img->id);
      g_free(fullpath);
      g_free(root);
      g_free(fname);
      g_free(base);
    }
  }

  gtk_widget_set_sensitive(g->vectorize, p->path[0] && p->file[0]);
  gtk_widget_set_sensitive(g->recompute, dt_rf_recipe_valid(&p->recipe));
  // provisional dev trigger (see _dev_edit_enabled): relabel the button
  // when it opens an edit session instead of a headless recompute -- only
  // when that session can actually run. without AI support, or with the
  // model not installed, dt_object_mask_edit_begin declines (the non-AI
  // stub does so silently) and the relabelled button would do nothing
#ifdef HAVE_AI
  const gboolean edits = _dev_edit_enabled()
                         && dt_rf_recipe_valid(&p->recipe)
                         && dt_masks_object_available();
#else
  const gboolean edits = FALSE;
#endif
  gtk_button_set_label(GTK_BUTTON(g->recompute),
                       edits ? _("edit mask") : _("recompute mask"));
}

void gui_update(dt_iop_module_t *self)
{
  gui_changed(self, NULL, NULL);
}

void init(dt_iop_module_t *self)
{
  dt_iop_default_init(self);

  dt_iop_rasterfile_params_t *d = self->default_params;
  memset(d->path, 0, sizeof(d->path));
  memset(d->file, 0, sizeof(d->file));
  memset(&d->recipe, 0, sizeof(d->recipe));

  /*
    Implementation note and reminder:

    Here we allocate per-module-instance memory shared by all pipes.
    To be sure data are valid and access is safe we

    a) ensure validity via a hash. Here it's just based on the
       parameters of the module's instance in other situations we
       might have to use the piece hash

    b) **always** access any module->data within a mutex-locked state.

    In this module the data do **not** depend on the using pipe.  In
    other cases, the pipe changing data according to a diffenrent hash
    must make sure the other pipes get restarted afterwards.
  */

  dt_rasterfile_cache_t *cd = calloc(1, sizeof(dt_rasterfile_cache_t));
  cd->hash = DT_INVALID_HASH;
  dt_pthread_mutex_init(&cd->lock, NULL);
  cd->mask = NULL;
  cd->width = cd->height = 0;
  self->data = cd;
}

void cleanup(dt_iop_module_t *self)
{
  dt_iop_default_cleanup(self);

  dt_rasterfile_cache_t *cd = self->data;
  _clear_cache(cd);
  dt_pthread_mutex_destroy(&cd->lock);
  free(cd);
  self->data = NULL;
}

void gui_focus(dt_iop_module_t *self, gboolean in)
{
  dt_dev_reprocess_center(self->dev, self->iop_order);
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_rasterfile_gui_data_t *g = IOP_GUI_ALLOC(rasterfile);

  g->mode = dt_bauhaus_combobox_from_params(self, "mode");
  gtk_widget_set_tooltip_text
    (g->mode,
     _("select the RGB channels taken into account to generate the raster mask"));

  g->fbutton = dtgtk_button_new(dtgtk_cairo_paint_directory, CPF_NONE, NULL);
  gtk_widget_set_name(g->fbutton, "non-flat");
  gtk_widget_set_tooltip_text
    (g->fbutton,
     _("select the PFM/PNG file recorded as a raster mask,\n"
       "CAUTION: path must be set in preferences/processing before choosing"));
  g_signal_connect(G_OBJECT(g->fbutton), "clicked",
                   G_CALLBACK(_fbutton_clicked), self);

  g->file = dt_bauhaus_combobox_new(self);
  dt_bauhaus_combobox_set_entries_ellipsis(g->file, PANGO_ELLIPSIZE_MIDDLE);
  gtk_widget_set_tooltip_text
    (g->file,
     _("the mask file path is saved with the image history"));
  g_signal_connect(G_OBJECT(g->file), "value-changed",
                   G_CALLBACK(_file_callback), self);

  // Vectorize button

  g->vectorize = gtk_button_new_with_label(_("vectorize"));
  gtk_widget_set_tooltip_text
    (g->vectorize,
     _("vectorize the current bitmap and create corresponding"
       " path masks in the mask manager"));
  g_signal_connect(g->vectorize, "clicked",
                   G_CALLBACK(_vectorize_button_clicked), self);

  // only meaningful for AI masks carrying a provenance recipe
  g->recompute = gtk_button_new_with_label(_("recompute mask"));
  gtk_widget_set_tooltip_text
    (g->recompute,
     _("regenerate this AI mask from its recorded recipe using the"
       " currently installed models.\nuse this when the mask file is"
       " missing and the model it was made with has been updated or"
       " removed: the mask is redone with the current model instead"));
  g_signal_connect(g->recompute, "clicked",
                   G_CALLBACK(_recompute_clicked), self);

  dt_gui_box_add(self->widget,
                 dt_gui_hbox(g->fbutton, dt_gui_expand(g->file)),
                 g->vectorize,
                 g->recompute);
}

#undef RASTERFILE_MAXFILE

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
