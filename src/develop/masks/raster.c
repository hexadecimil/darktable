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

#include "common/math.h"
#include "common/rasterfile_io.h"
#include "common/rasterfile_recipe.h"
#include "control/control.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/masks/object_recipe.h"

// the raster shape: a persistent mask form whose content is not a geometry
// but a reference to a raster mask file (see dt_masks_point_raster_t in
// masks.h). the form renders by sampling the decoded file, so it has no
// editable anchors and no creation gesture: instances are built by code
// (the precise-mask finalisation) and behave as regular group members
// everywhere else -- combination, opacity, duplication, removal

// ---- point access and path resolution ----

// the single serialized point of the form, or NULL when its layout is not
// the one this build knows: the form is then inert, never a wrong render
static dt_masks_point_raster_t *_raster_point(const dt_masks_form_t *form)
{
  if(!form || !form->points) return NULL;
  dt_masks_point_raster_t *pt = form->points->data;
  if(pt->magic != DT_MASKS_RASTER_POINT_MAGIC
     || pt->version != DT_MASKS_RASTER_POINT_VERSION)
    return NULL;
  return pt;
}

// resolve the mask file path for `img`, mirroring commit_params of
// iop/rasterfile.c: a valid recipe names the file content-addressed from
// its fingerprint under the LOCAL mask root, whatever machine produced
// it; without one, `file` is a leaf under that same root -- never an
// absolute path taken from the blob. NULL when the point references
// nothing at all. caller frees
static gchar *_raster_resolve_path(const dt_masks_point_raster_t *pt,
                                   const dt_image_t *img)
{
  gchar *root = dt_rasterfile_mask_root();
  gchar *path = NULL;
  if(dt_rf_recipe_valid(&pt->recipe))
  {
    gchar *base = g_path_get_basename(img->filename);
    char *dot = g_strrstr(base, ".");
    if(dot) *dot = '\0';
    gchar *fname = dt_rasterfile_recipe_filename(&pt->recipe, base,
                                                 img->width, img->height,
                                                 img->exif_datetime_taken);
    path = g_build_filename(root, fname, NULL);
    g_free(fname);
    g_free(base);
  }
  // the leaf arrives from a DB/XMP blob: refuse a corrupted one that
  // lost its NUL terminator rather than reading past the array
  else if(pt->file[0] && memchr(pt->file, '\0', sizeof(pt->file)))
    path = g_build_filename(root, pt->file, NULL);
  g_free(root);
  return path;
}

// ---- the decoded-file cache ----
//
// get_mask_roi runs on pixelpipe threads, several pipes in parallel, and
// a full-resolution PNG16 decode costs a noticeable fraction of a second:
// decoded planes are shared through a small global registry keyed by the
// resolved path. entries are immutable once published and refcounted, so
// sampling never takes any lock -- the global mutex guards the TABLE only
// (lookup, publication, eviction, refcounts); decoding and the stat run
// outside it. this deliberately does NOT reuse the locking of the
// rasterfile module's per-instance cache, whose mutex is held across the
// whole resample

#define RASTER_CACHE_MAX_DECODED 3   // LRU cap on decoded entries
#define RASTER_CACHE_MAX_NEGATIVE 8  // and on the tiny negative ones
#define RASTER_MIP_LEVELS 4          // level k is downscaled by 1 << k

typedef struct _raster_cache_entry_t
{
  gchar *path;   // the resolved file path: the key. owned
  // validity stamp: the file's (mtime, size) at decode time -- or at the
  // failed attempt for a negative entry, (0, -1) when it was absent.
  // recipe files are content-addressed and never rewritten in place;
  // the stamp covers hand-picked leaves and external overwrites
  gint64 mtime;
  gint64 fsize;
  // a negative entry records a failed decode (absent or unreadable
  // file). it carries no plane and is never handed out: acquire re-stats
  // the file and either returns NULL for the price of that stat, or
  // drops the entry and decodes anew when the stamp changed (a recompute
  // landed, the user copied the file back)
  gboolean negative;

  // ---- decoded data, immutable after publication ----
  // TRUE: PFM source kept as float (arbitrary penumbra precision);
  // FALSE: PNG source stored as uint16 -- lossless for the 8/16-bit
  // integer sources and half the memory of the reader's float plane
  gboolean f32;
  void *levels[RASTER_MIP_LEVELS];  // level 0 native, then 2x/4x/8x
  int lw[RASTER_MIP_LEVELS];
  int lh[RASTER_MIP_LEVELS];
  // bounding box of the pixels > 0 at level 0, inclusive coordinates;
  // an empty mask has bbox[2] < bbox[0]
  int bbox[4];
  // the canvas outline, memoized here lazily by the display code so a
  // re-expose never re-runs its extraction (arrives with the canvas
  // work; reserved now so the entry owns its whole decode product)
  float *outline;
  int outline_count;

  int refs;           // guarded by _raster_cache_lock
  gboolean in_table;  // FALSE once detached: dies at its last release
} _raster_cache_entry_t;

// guards the table only: the list, the refcounts, the in_table flags
static GMutex _raster_cache_lock;
static GList *_raster_cache = NULL;  // most recently used first

static void _raster_entry_free(_raster_cache_entry_t *entry)
{
  for(int k = 0; k < RASTER_MIP_LEVELS; k++)
    dt_free_align(entry->levels[k]);
  dt_free_align(entry->outline);
  g_free(entry->path);
  free(entry);
}

// under the table lock. walks from the LRU tail and never touches a
// referenced entry: it keeps its table slot (the next acquire of that
// path must keep hitting it) and merely makes the cache exceed its cap
// until its holders release
static void _raster_cache_trim(void)
{
  int decoded = 0, negative = 0;
  for(GList *l = _raster_cache; l; l = g_list_next(l))
  {
    if(((_raster_cache_entry_t *)l->data)->negative)
      negative++;
    else
      decoded++;
  }

  GList *l = g_list_last(_raster_cache);
  while(l
        && (decoded > RASTER_CACHE_MAX_DECODED
            || negative > RASTER_CACHE_MAX_NEGATIVE))
  {
    GList *prev = g_list_previous(l);
    _raster_cache_entry_t *entry = l->data;
    const gboolean over = entry->negative
      ? negative > RASTER_CACHE_MAX_NEGATIVE
      : decoded > RASTER_CACHE_MAX_DECODED;
    if(over && entry->refs == 0)
    {
      dt_print(DT_DEBUG_MASKS, "[masks raster] cache evict%s '%s'",
               entry->negative ? " negative" : "", entry->path);
      _raster_cache = g_list_delete_link(_raster_cache, l);
      if(entry->negative)
        negative--;
      else
        decoded--;
      _raster_entry_free(entry);
    }
    l = prev;
  }
}

// decode `path` into a fresh, private (unpublished) entry: single plane
// through the shared reader, uint16 quantization unless the source is
// PFM, 2x2-average mip pyramid, bounding box of the lit pixels. runs
// with NO lock held. NULL on any failure
static _raster_cache_entry_t *_raster_decode(const char *path,
                                             const gboolean quiet,
                                             const gint64 mtime,
                                             const gint64 fsize)
{
  double start = dt_get_debug_wtime();
  int width = 0, height = 0;
  float *plane = dt_rasterfile_io_read(path, DT_RASTERFILE_IO_ALL, quiet,
                                       &width, &height);
  if(!plane) return NULL;

  _raster_cache_entry_t *entry = calloc(1, sizeof(_raster_cache_entry_t));
  if(!entry)
  {
    dt_free_align(plane);
    return NULL;
  }
  entry->path = g_strdup(path);
  entry->mtime = mtime;
  entry->fsize = fsize;
  // the reader dispatches on the extension: everything non-PNG went
  // through its PFM path and keeps the float precision
  const char *ext = g_strrstr(path, ".");
  entry->f32 = !(ext && !g_ascii_strcasecmp(ext, ".png"));
  entry->lw[0] = width;
  entry->lh[0] = height;
  const size_t npix = (size_t)width * height;

  if(entry->f32)
    entry->levels[0] = plane;
  else
  {
    uint16_t *u16 = dt_alloc_aligned(npix * sizeof(uint16_t));
    if(!u16)
    {
      dt_free_align(plane);
      _raster_entry_free(entry);
      return NULL;
    }
    // the reader clips to [0,1]: the quantization is exact for what an
    // 8 or 16 bit integer file can hold
    DT_OMP_FOR()
    for(size_t k = 0; k < npix; k++)
      u16[k] = (uint16_t)(plane[k] * 65535.0f + 0.5f);
    dt_free_align(plane);
    entry->levels[0] = u16;
  }

  // 2x2 box-average pyramid for anti-aliased minification; odd
  // dimensions clamp the second tap so edge pixels average what exists
  for(int lev = 1; lev < RASTER_MIP_LEVELS; lev++)
  {
    const int sw = entry->lw[lev - 1], sh = entry->lh[lev - 1];
    const int dw = MAX(1, (sw + 1) / 2), dh = MAX(1, (sh + 1) / 2);
    entry->lw[lev] = dw;
    entry->lh[lev] = dh;
    entry->levels[lev]
      = dt_alloc_aligned((size_t)dw * dh
                         * (entry->f32 ? sizeof(float) : sizeof(uint16_t)));
    if(!entry->levels[lev])
    {
      _raster_entry_free(entry);
      return NULL;
    }
    if(entry->f32)
    {
      const float *const src = entry->levels[lev - 1];
      float *const dst = entry->levels[lev];
      DT_OMP_FOR()
      for(int j = 0; j < dh; j++)
        for(int i = 0; i < dw; i++)
        {
          const int j0 = 2 * j, j1 = MIN(2 * j + 1, sh - 1);
          const int i0 = 2 * i, i1 = MIN(2 * i + 1, sw - 1);
          dst[(size_t)j * dw + i]
            = 0.25f * (src[(size_t)j0 * sw + i0] + src[(size_t)j0 * sw + i1]
                       + src[(size_t)j1 * sw + i0]
                       + src[(size_t)j1 * sw + i1]);
        }
    }
    else
    {
      const uint16_t *const src = entry->levels[lev - 1];
      uint16_t *const dst = entry->levels[lev];
      DT_OMP_FOR()
      for(int j = 0; j < dh; j++)
        for(int i = 0; i < dw; i++)
        {
          const int j0 = 2 * j, j1 = MIN(2 * j + 1, sh - 1);
          const int i0 = 2 * i, i1 = MIN(2 * i + 1, sw - 1);
          const uint32_t sum = (uint32_t)src[(size_t)j0 * sw + i0]
                               + src[(size_t)j0 * sw + i1]
                               + src[(size_t)j1 * sw + i0]
                               + src[(size_t)j1 * sw + i1];
          dst[(size_t)j * dw + i] = (uint16_t)((sum + 2) >> 2);
        }
    }
  }

  // bounding box of the lit pixels: bounds the render's forward
  // transform to the part of the file that can contribute
  int x0 = width, y0 = height, x1 = -1, y1 = -1;
  if(entry->f32)
  {
    const float *const p = entry->levels[0];
    DT_OMP_FOR(reduction(min : x0, y0) reduction(max : x1, y1))
    for(int j = 0; j < height; j++)
      for(int i = 0; i < width; i++)
        if(p[(size_t)j * width + i] > 0.0f)
        {
          x0 = MIN(x0, i);
          x1 = MAX(x1, i);
          y0 = MIN(y0, j);
          y1 = MAX(y1, j);
        }
  }
  else
  {
    const uint16_t *const p = entry->levels[0];
    DT_OMP_FOR(reduction(min : x0, y0) reduction(max : x1, y1))
    for(int j = 0; j < height; j++)
      for(int i = 0; i < width; i++)
        if(p[(size_t)j * width + i])
        {
          x0 = MIN(x0, i);
          x1 = MAX(x1, i);
          y0 = MIN(y0, j);
          y1 = MAX(y1, j);
        }
  }
  entry->bbox[0] = x0;
  entry->bbox[1] = y0;
  entry->bbox[2] = x1;
  entry->bbox[3] = y1;

  size_t bytes = 0;
  for(int lev = 0; lev < RASTER_MIP_LEVELS; lev++)
    bytes += (size_t)entry->lw[lev] * entry->lh[lev]
             * (entry->f32 ? sizeof(float) : sizeof(uint16_t));
  dt_print(DT_DEBUG_MASKS,
           "[masks raster] decoded '%s' %dx%d %s, %.1f MB incl. mips, "
           "bbox %d..%d x %d..%d, took %0.04f sec",
           path, width, height, entry->f32 ? "float" : "uint16",
           bytes / (1024.0 * 1024.0), x0, x1, y0, y1,
           dt_get_lap_time(&start));
  return entry;
}

// publish a private entry (positive or negative) under the table lock,
// double-checked: a concurrent acquire may have published this path
// while we decoded -- the earlier decoded entry wins, ours is dropped.
// returns the acquired entry the caller must release, NULL for negative
static _raster_cache_entry_t *
_raster_cache_publish(_raster_cache_entry_t *fresh)
{
  g_mutex_lock(&_raster_cache_lock);
  for(GList *l = _raster_cache; l; l = g_list_next(l))
  {
    _raster_cache_entry_t *other = l->data;
    if(strcmp(other->path, fresh->path)) continue;
    if(!other->negative)
    {
      // a decoded entry raced in: adopt it, drop ours
      other->refs++;
      _raster_cache = g_list_remove(_raster_cache, other);
      _raster_cache = g_list_prepend(_raster_cache, other);
      g_mutex_unlock(&_raster_cache_lock);
      _raster_entry_free(fresh);
      return other;
    }
    if(fresh->negative)
    {
      // both negative: keep the incumbent
      g_mutex_unlock(&_raster_cache_lock);
      _raster_entry_free(fresh);
      return NULL;
    }
    // our decoded entry supersedes a raced-in negative one (never
    // referenced, refs is always 0)
    _raster_cache = g_list_delete_link(_raster_cache, l);
    _raster_entry_free(other);
    break;
  }
  fresh->in_table = TRUE;
  // captured before the trim: a negative entry keeps refs 0 and is fair
  // game for the eviction right below, so `fresh` must not be touched
  // after it
  const gboolean negative = fresh->negative;
  fresh->refs = negative ? 0 : 1;
  _raster_cache = g_list_prepend(_raster_cache, fresh);
  _raster_cache_trim();
  g_mutex_unlock(&_raster_cache_lock);
  return negative ? NULL : fresh;
}

// acquire the decoded entry for `path`, or NULL when the file cannot be
// read. costs one stat; a decode only runs when the table has no valid
// entry, and always outside the lock. the caller owns one reference and
// must _raster_cache_release it -- and must never touch the table's
// mutex while sampling, the reference alone keeps the entry alive
static _raster_cache_entry_t *_raster_cache_acquire(const char *path,
                                                    const gboolean quiet)
{
  // one stat per acquire, taken OUTSIDE the lock: it both validates a
  // hit against a rewritten file and re-arms a negative entry when the
  // failure state changed. (0, -1) stamps an absent file
  GStatBuf st;
  const gboolean present = g_stat(path, &st) == 0;
  const gint64 mtime = present ? (gint64)st.st_mtime : 0;
  const gint64 fsize = present ? (gint64)st.st_size : -1;

  g_mutex_lock(&_raster_cache_lock);
  _raster_cache_entry_t *hit = NULL;
  for(GList *l = _raster_cache; l; l = g_list_next(l))
  {
    _raster_cache_entry_t *entry = l->data;
    if(!strcmp(entry->path, path))
    {
      hit = entry;
      break;
    }
  }
  if(hit && hit->mtime == mtime && hit->fsize == fsize)
  {
    if(hit->negative)
    {
      // same failure state as the failed attempt: NULL for the price
      // of the stat, no decode I/O
      g_mutex_unlock(&_raster_cache_lock);
      return NULL;
    }
    hit->refs++;
    _raster_cache = g_list_remove(_raster_cache, hit);
    _raster_cache = g_list_prepend(_raster_cache, hit);
    g_mutex_unlock(&_raster_cache_lock);
    return hit;
  }
  if(hit)
  {
    // stale: the file changed (or came, or went) under the entry.
    // detach it -- a still-referenced one dies at its last release,
    // its holders keep sampling the bytes they acquired
    _raster_cache = g_list_remove(_raster_cache, hit);
    hit->in_table = FALSE;
    if(hit->refs == 0) _raster_entry_free(hit);
  }
  g_mutex_unlock(&_raster_cache_lock);

  _raster_cache_entry_t *fresh = NULL;
  if(present)
    fresh = _raster_decode(path, quiet, mtime, fsize);
  if(!fresh)
  {
    // absent, or present but unreadable: pin the failure under this
    // exact stamp so retries (one per expose is the worst case) cost a
    // stat each until the state changes
    fresh = calloc(1, sizeof(_raster_cache_entry_t));
    if(!fresh) return NULL;
    fresh->path = g_strdup(path);
    fresh->mtime = mtime;
    fresh->fsize = fsize;
    fresh->negative = TRUE;
  }
  return _raster_cache_publish(fresh);
}

static void _raster_cache_release(_raster_cache_entry_t *entry)
{
  if(!entry) return;
  g_mutex_lock(&_raster_cache_lock);
  entry->refs--;
  const gboolean dead = !entry->in_table && entry->refs == 0;
  g_mutex_unlock(&_raster_cache_lock);
  if(dead) _raster_entry_free(entry);
}

// ---- the render: sampling the file through the pipe's distortions ----

static inline float _raster_tap(const _raster_cache_entry_t *const entry,
                                const int level,
                                const int i,
                                const int j)
{
  // outside the file there is no mask
  if(i < 0 || j < 0 || i >= entry->lw[level] || j >= entry->lh[level])
    return 0.0f;
  const size_t k = (size_t)j * entry->lw[level] + i;
  return entry->f32
    ? ((const float *)entry->levels[level])[k]
    : ((const uint16_t *)entry->levels[level])[k] * (1.0f / 65535.0f);
}

// bilinear sample of one mip level at a LEVEL-0 file coordinate in the
// sampler's corner indexing (integer coordinate = level-0 pixel center).
// zero-padded outside the file, so the mask fades over its border pixel
// instead of cutting. NaN-safe: a folded distortion can hand us
// anything, and every comparison with NaN is false, so such coordinates
// take the early out-of-bounds zero before any int conversion
static inline float _raster_sample(const _raster_cache_entry_t *const entry,
                                   const int level,
                                   const float x0,
                                   const float y0)
{
  // level pixel (i) covers level-0 pixels [i * 2^lev, (i+1) * 2^lev):
  // aligning the centers maps level-0 index x0 to (x0 + 0.5) / 2^lev - 0.5
  const float inv = 1.0f / (float)(1 << level);
  const float x = (x0 + 0.5f) * inv - 0.5f;
  const float y = (y0 + 0.5f) * inv - 0.5f;
  if(!(x > -1.0f && y > -1.0f
       && x < (float)entry->lw[level] && y < (float)entry->lh[level]))
    return 0.0f;
  const float xf = floorf(x), yf = floorf(y);
  const int i = (int)xf, j = (int)yf;
  const float a = x - xf, b = y - yf;
  return (1.0f - a) * ((1.0f - b) * _raster_tap(entry, level, i, j)
                       + b * _raster_tap(entry, level, i, j + 1))
         + a * ((1.0f - b) * _raster_tap(entry, level, i + 1, j)
                + b * _raster_tap(entry, level, i + 1, j + 1));
}

// flow pattern: _circle_get_mask_roi (circle.c). two deliberate
// differences: the per-pixel step interpolates the file COORDINATES
// between the grid nodes and then samples the file there -- never mask
// values sampled at the grid, which would blur the penumbra a second
// time -- and the grid points sit at pixel CENTERS (+ 0.5), harmless on
// a continuous function but decisive for a sampler.
//
// no thresholding anywhere: the graded penumbra of the file is the
// whole point of this shape, every value passes through as read
static int _raster_get_mask_roi(const dt_iop_module_t *const restrict module,
                                const dt_dev_pixelpipe_iop_t *const restrict piece,
                                dt_masks_form_t *const form,
                                const dt_iop_roi_t *const roi,
                                float *const restrict buffer)
{
  double start1 = dt_get_debug_wtime();
  double start2 = start1;

  const int w = roi->width;
  const int h = roi->height;
  const int px = roi->x;
  const int py = roi->y;
  const float iscale = 1.0f / roi->scale;

  // the buffer contract: fully defined on every return 1 (zeros plus
  // the sampled bounding box), never a partial buffer
  memset(buffer, 0, sizeof(float) * w * h);

  dt_masks_point_raster_t *pt = _raster_point(form);
  if(!pt) return 0;  // unknown layout: the group skips this member

  const dt_image_t *img = &piece->pipe->image;
  gchar *path = _raster_resolve_path(pt, img);
  if(!path) return 0;

  _raster_cache_entry_t *entry
    = _raster_cache_acquire(path, dt_rf_recipe_valid(&pt->recipe));
  if(!entry)
  {
#ifdef HAVE_AI
    // ---- missing-file safety net, the rasterfile module's regime ----
    // (mirror of _get_rasterfile_mask, iop/rasterfile.c): a valid recipe
    // makes the absence a transient state that a recompute repairs.
    // never from a thumbnail pipe -- cheap pipes must not flood the job
    // queue, the full/export pipes will come. two regimes:
    //  - a pipe whose output leaves the machine (export) or a context
    //    with no job system at all (darktable-cli) recomputes NOW,
    //    blocking: these pipes run once, a wrong output is final;
    //  - the darkroom schedules asynchronously; the land reprocesses
    //    every pipe and the fresh runs re-acquire the file
    if(dt_rf_recipe_valid(&pt->recipe)
       && !(piece->pipe->type & DT_DEV_PIXELPIPE_THUMBNAIL))
    {
      const gboolean sync_ctx
        = (piece->pipe->type & DT_DEV_PIXELPIPE_EXPORT)
          || !dt_control_running();
      if(sync_ctx)
      {
        if(dt_object_recipe_recompute_now(&pt->recipe,
                                          piece->pipe->image.id))
          entry = _raster_cache_acquire(path, TRUE);
      }
      else
      {
        // the point blob does not change when the recompute lands, so
        // neither does dt_masks_group_hash: the module output cached
        // during THIS run (rendered without the mask) would be served
        // again after the land's reprocess. mark the pipe so its next
        // run drops the cachelines from this module on -- the pre-run
        // purge of pixelpipe_hb.c consumes and resets the mark. set
        // before the finalize gate below: whoever lands the file (our
        // schedule, another pipe's, or the finalisation job) must
        // become visible here
        piece->pipe->cache_obsolete_order
          = MIN(piece->pipe->cache_obsolete_order,
                (uint32_t)module->iop_order);
        // scheduling is safe from a pixelpipe thread: it marks the
        // anti-respawn table and enqueues a job, nothing more, and the
        // table absorbs repeated calls. during the precise-mask
        // finalisation the file is legitimately absent while the job
        // prepares it -- recomputing the OLD recipe then would race
        // the very job that replaces the file (the same gate the
        // rasterfile module's proactive repair takes)
        if(!dt_object_mask_finalize_running())
          dt_object_recipe_schedule_recompute(&pt->recipe,
                                              piece->pipe->image.id);
      }
    }
#else
    // regenerating a mask file from its provenance recipe is AI
    // machinery: without it a missing file has no repair path on this
    // machine. the member is skipped -- silent by design, logged below
#endif
    if(!entry)
    {
      dt_print(DT_DEBUG_MASKS,
               "[masks %s] raster mask file '%s' not readable, member skipped",
               form->name, path);
      g_free(path);
      return 0;
    }
  }
  g_free(path);

  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks %s] raster acquire took %0.04f sec",
           form->name, dt_get_lap_time(&start2));

  // nothing lit anywhere in the file: the member contributes plain zeros
  if(entry->bbox[2] < entry->bbox[0])
  {
    _raster_cache_release(entry);
    return 1;
  }

  const int wi = piece->pipe->iwidth, hi = piece->pipe->iheight;
  const float cropx = (float)img->crop_x, cropy = (float)img->crop_y;

  // ---- bounding box: the file's lit area transformed into the roi ----
  // perimeter of the nonzero bbox dilated by one pixel, converted from
  // the file frame to module coordinates by the exact inverse of the
  // per-pixel mapping below (the half-pixel terms drown in the one-pixel
  // dilation), then transformed forward. sampled densely along each
  // edge: the distortions can bend them -- the circle takes up to 360
  // points for the same reason
  const float fx0 = entry->bbox[0] - 1.0f, fy0 = entry->bbox[1] - 1.0f;
  const float fx1 = entry->bbox[2] + 1.0f, fy1 = entry->bbox[3] + 1.0f;
  const float to_mod_x = (float)wi / (float)img->width;
  const float to_mod_y = (float)hi / (float)img->height;

  const int edge_pts = 64;  // per edge; each edge carries its start corner
  const int npts = 4 * edge_pts;
  float *const restrict bpts = dt_alloc_align_float((size_t)npts * 2);
  if(bpts == NULL)
  {
    _raster_cache_release(entry);
    return 0;
  }
  for(int n = 0; n < edge_pts; n++)
  {
    const float t = (float)n / (float)edge_pts;
    const float xs[4] = { fx0 + t * (fx1 - fx0), fx1,
                          fx1 + t * (fx0 - fx1), fx0 };
    const float ys[4] = { fy0, fy0 + t * (fy1 - fy0),
                          fy1, fy1 + t * (fy0 - fy1) };
    for(int edge = 0; edge < 4; edge++)
    {
      const int idx = 2 * (edge * edge_pts + n);
      bpts[idx] = (xs[edge] + cropx) * to_mod_x;
      bpts[idx + 1] = (ys[edge] + cropy) * to_mod_y;
    }
  }

  // from module coordinates to the current point in the pixelpipe
  if(!dt_dev_distort_transform_plus(module->dev, piece->pipe,
                                    module->iop_order,
                                    DT_DEV_TRANSFORM_DIR_BACK_INCL,
                                    bpts, npts))
  {
    dt_free_align(bpts);
    _raster_cache_release(entry);
    return 0;
  }

  float xmin = FLT_MAX, ymin = FLT_MAX, xmax = FLT_MIN, ymax = FLT_MIN;
  for(int n = 0; n < npts; n++)
  {
    // just in case that transform throws surprising values
    if(!(dt_isnormal(bpts[2 * n]) && dt_isnormal(bpts[2 * n + 1])))
      continue;
    xmin = MIN(xmin, bpts[2 * n]);
    xmax = MAX(xmax, bpts[2 * n]);
    ymin = MIN(ymin, bpts[2 * n + 1]);
    ymax = MAX(ymax, bpts[2 * n + 1]);
  }
  dt_free_align(bpts);

  // every point rejected: nothing usable landed in this pipe
  if(xmin > xmax || ymin > ymax)
  {
    _raster_cache_release(entry);
    return 1;
  }

  // scale dependent grid resolution, exactly the circle's
  const int grid = CLAMP((10.0f * roi->scale + 2.0f) / 3.0f, 1, 4);
  const int gw = (w + grid - 1) / grid + 1;
  const int gh = (h + grid - 1) / grid + 1;

  // the roi-space bounding box with the circle's reserve
  const int bbxm = CLAMP((int)floorf(xmin / iscale - px) / grid - 1, 0, gw - 1);
  const int bbXM = CLAMP((int)ceilf(xmax / iscale - px) / grid + 2, 0, gw - 1);
  const int bbym = CLAMP((int)floorf(ymin / iscale - py) / grid - 1, 0, gh - 1);
  const int bbYM = CLAMP((int)ceilf(ymax / iscale - py) / grid + 2, 0, gh - 1);
  const int bbw = bbXM - bbxm + 1;
  const int bbh = bbYM - bbym + 1;

  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks %s] raster bounding box took %0.04f sec",
           form->name, dt_get_lap_time(&start2));

  // the shape lies outside this roi
  if(bbw <= 1 || bbh <= 1)
  {
    _raster_cache_release(entry);
    return 1;
  }

  float *const restrict points = dt_alloc_align_float((size_t)bbw * bbh * 2);
  if(points == NULL)
  {
    _raster_cache_release(entry);
    return 0;
  }

  // grid points in module coordinates, at the CENTER of the roi pixels
  DT_OMP_FOR(collapse(2) if(bbw * bbh > 50000))
  for(int j = bbym; j <= bbYM; j++)
    for(int i = bbxm; i <= bbXM; i++)
    {
      const size_t index = (size_t)(j - bbym) * bbw + i - bbxm;
      points[index * 2] = (grid * i + px + 0.5f) * iscale;
      points[index * 2 + 1] = (grid * j + py + 0.5f) * iscale;
    }

  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks %s] raster grid took %0.04f sec",
           form->name, dt_get_lap_time(&start2));

  // back to the pipe's input space
  if(!dt_dev_distort_backtransform_plus(module->dev, piece->pipe,
                                        module->iop_order,
                                        DT_DEV_TRANSFORM_DIR_BACK_INCL,
                                        points, (size_t)bbw * bbh))
  {
    dt_free_align(points);
    _raster_cache_release(entry);
    return 0;
  }

  // ---- THE mapping, established against circle.c (shapes normalize by
  // the pipe input dimensions) and object.c (normalized = (native +
  // rawprepare crop) / sensor width) and locked by the counter-review:
  // the backtransformed points live in the CURRENT pipe's input space,
  // downsampled in preview, and the file lives in the post-rawprepare
  // frame = sensor frame minus the metadata crop. so: normalize by the
  // pipe input size, denormalize to the sensor frame at scale 1,
  // subtract the crop -- and shift by the half pixel that converts
  // pixel-center coordinates to the sampler's corner indexing (file
  // pixel (0,0) has its center at (0.5, 0.5) of the continuous frame)
  const float to_file_x = (float)img->width / (float)wi;
  const float to_file_y = (float)img->height / (float)hi;
  DT_OMP_FOR(if(bbw * bbh > 50000))
  for(size_t k = 0; k < (size_t)bbw * bbh; k++)
  {
    points[k * 2] = points[k * 2] * to_file_x - cropx - 0.5f;
    points[k * 2 + 1] = points[k * 2 + 1] * to_file_y - cropy - 0.5f;
  }

  // ---- mip level for the minification at hand ----
  // adjacent grid nodes are grid * iscale pipe-input pixels apart, which
  // is grid * iscale * to_file_x FILE pixels (the preview pipe's input
  // is itself downsampled -- exactly where minification is strongest).
  // pick the level that keeps the sampling step at or below ~1.5 of its
  // own pixels
  const float step = grid * iscale * to_file_x;
  int level = 0;
  while(level < RASTER_MIP_LEVELS - 1
        && step > 1.5f * (float)(1 << level))
    level++;

  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks %s] raster mapping took %0.04f sec (mip level %d)",
           form->name, dt_get_lap_time(&start2), level);

  // per-pixel fill: interpolate the file COORDINATES between the four
  // surrounding grid nodes (at grid == 1 every pixel carries its exact
  // backtransformed coordinate), then sample the file bilinearly there
  const int endx = MIN(w, bbXM * grid);
  const int endy = MIN(h, bbYM * grid);
  DT_OMP_FOR()
  for(int j = bbym * grid; j < endy; j++)
  {
    const int jj = j % grid;
    const int mj = j / grid - bbym;
    for(int i = bbxm * grid; i < endx; i++)
    {
      const int ii = i % grid;
      const int mi = i / grid - bbxm;
      const size_t mindex = (size_t)mj * bbw + mi;
      const float fx
        = (points[mindex * 2] * (grid - ii) * (grid - jj)
           + points[(mindex + 1) * 2] * ii * (grid - jj)
           + points[(mindex + bbw) * 2] * (grid - ii) * jj
           + points[(mindex + bbw + 1) * 2] * ii * jj)
          / (grid * grid);
      const float fy
        = (points[mindex * 2 + 1] * (grid - ii) * (grid - jj)
           + points[(mindex + 1) * 2 + 1] * ii * (grid - jj)
           + points[(mindex + bbw) * 2 + 1] * (grid - ii) * jj
           + points[(mindex + bbw + 1) * 2 + 1] * ii * jj)
          / (grid * grid);
      buffer[(size_t)j * w + i] = _raster_sample(entry, level, fx, fy);
    }
  }

  dt_free_align(points);
  _raster_cache_release(entry);

  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks %s] raster fill took %0.04f sec",
           form->name, dt_get_lap_time(&start2));
  dt_print(DT_DEBUG_MASKS | DT_DEBUG_PERF,
           "[masks %s] raster total render took %0.04f sec",
           form->name, dt_get_lap_time(&start1));

  return 1;
}

static void _raster_set_form_name(dt_masks_form_t *const form,
                                  const size_t nb)
{
  snprintf(form->name, sizeof(form->name), _("precise mask #%d"), (int)nb);
}

static void _raster_duplicate_points(dt_develop_t *const dev,
                                     dt_masks_form_t *const base,
                                     dt_masks_form_t *const dest)
{
  (void)dev; // unused arg, keep compiler from complaining
  // the duplicate shares the same content-addressed file: correct by
  // construction, both render the same bytes. without this callback a
  // "duplicate shape" would silently produce an EMPTY form
  for(GList *pts = base->points; pts; pts = g_list_next(pts))
  {
    dt_masks_point_raster_t *pt = malloc(sizeof(dt_masks_point_raster_t));
    memcpy(pt, pts->data, sizeof(dt_masks_point_raster_t));
    dest->points = g_list_append(dest->points, pt);
  }
}

// the four mouse handlers and post_expose are called UNGUARDED on their
// member: stubs, never NULL. a raster shape has nothing to edit on canvas
static int _raster_events_mouse_moved(dt_iop_module_t *module,
                                      float pzx,
                                      float pzy,
                                      const double pressure,
                                      const int which,
                                      const float zoom_scale,
                                      dt_masks_form_t *form,
                                      const dt_imgid_t parentid,
                                      dt_masks_form_gui_t *gui,
                                      const int index)
{
  return 0;
}

static int _raster_events_mouse_scrolled(dt_iop_module_t *module,
                                         float pzx,
                                         float pzy,
                                         const gboolean up,
                                         uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_imgid_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  return 0;
}

static int _raster_events_button_pressed(dt_iop_module_t *module,
                                         float pzx,
                                         float pzy,
                                         const double pressure,
                                         const int which,
                                         const int type,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_imgid_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  return 0;
}

static int _raster_events_button_released(dt_iop_module_t *module,
                                          float pzx,
                                          float pzy,
                                          const int which,
                                          const uint32_t state,
                                          dt_masks_form_t *form,
                                          const dt_imgid_t parentid,
                                          dt_masks_form_gui_t *gui,
                                          const int index)
{
  return 0;
}

static void _raster_events_post_expose(cairo_t *cr,
                                       const float zoom_scale,
                                       dt_masks_form_gui_t *gui,
                                       const int index,
                                       const int num_points)
{
  // no outline yet: the canvas display comes with the render work
}

// the function table for raster shapes
const dt_masks_functions_t dt_masks_functions_raster = {
  .point_struct_size = sizeof(struct dt_masks_point_raster_t),
  .sanitize_config = NULL,
  .setup_mouse_actions = NULL,
  .set_form_name = _raster_set_form_name,
  .set_hint_message = NULL,
  .modify_property = NULL,
  .duplicate_points = _raster_duplicate_points,
  .initial_source_pos = NULL,
  .get_distance = NULL,
  .get_points = NULL,
  .get_points_border = NULL,
  .get_mask = NULL,
  .get_mask_roi = _raster_get_mask_roi,
  .get_area = NULL,
  .get_source_area = NULL,
  .mouse_moved = _raster_events_mouse_moved,
  .mouse_scrolled = _raster_events_mouse_scrolled,
  .button_pressed = _raster_events_button_pressed,
  .button_released = _raster_events_button_released,
  .post_expose = _raster_events_post_expose
};

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
