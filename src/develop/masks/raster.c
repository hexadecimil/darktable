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

#include "common/ai/detectors.h"
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

// ---- the panel's window into the shape (libs/masks.c) ----
// the missing-file badge and the recompute menu entry need the same two
// answers the render derives for itself: is the point one this build
// can read, and which file under the mask root does it name

dt_masks_point_raster_t *dt_masks_raster_point(const dt_masks_form_t *form)
{
  if(!form || !(form->type & DT_MASKS_RASTER)) return NULL;
  return _raster_point(form);
}

gchar *dt_masks_raster_resolve_path(const dt_masks_point_raster_t *pt,
                                    const dt_image_t *img)
{
  return _raster_resolve_path(pt, img);
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
  // the canvas outline: closed polylines in the level-0 file frame,
  // loops separated by DT_INVALID_COORDINATE sentinel pairs. extracted
  // lazily by the display code and memoized here so a re-expose never
  // re-runs marching squares. published under the table mutex, read
  // under it once, immutable afterwards -- never a bare write, the
  // entry is shared between the GUI and the pixelpipe threads
  float *outline;
  int outline_count;
  gboolean outline_done;  // computed -- an empty outline memoizes too

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

// ---- the canvas outline: marching squares on a display-sized mip ----
//
// what the photographer sees when editing the mask: the 0.5 level line
// of the file, as closed polylines. display only -- the render above
// never reads it, the file's graded values are the mask

// the level plane padded by one virtual ring of zeros, which closes
// every contour loop: _raster_tap already answers 0 out of bounds
static inline float _ms_sample(const _raster_cache_entry_t *const entry,
                               const int level,
                               const int i,
                               const int j)
{
  return _raster_tap(entry, level, i - 1, j - 1);
}

// the side the contour leaves a cell by, entered via side `entry`, or -1
// on a non-crossing entry (a logic failure the caller drops the loop on).
// corners: 0 (x,y), 1 (x+1,y), 2 (x+1,y+1), 3 (x,y+1); side s joins
// corners s and (s+1)&3 -- 0 bottom, 1 right, 2 top, 3 left
static int _ms_exit(const float v[4], const int entry)
{
  gboolean b[4];
  gboolean cross[4];
  int ncross = 0;
  for(int k = 0; k < 4; k++)
    b[k] = v[k] >= 0.5f;
  for(int s = 0; s < 4; s++)
  {
    cross[s] = b[s] != b[(s + 1) & 3];
    if(cross[s]) ncross++;
  }
  if(!cross[entry]) return -1;
  if(ncross == 4)
  {
    // the saddle: two segments in one cell. the center average decides
    // which diagonal the lit region connects across, hence which pair of
    // corners the contour isolates -- the unlit ones when the center is
    // lit -- and with it which crossing pairs with which. the rule only
    // depends on the cell, never on the walk direction, so the two
    // passes through the cell take distinct pairs
    const gboolean center_lit
      = 0.25f * (v[0] + v[1] + v[2] + v[3]) >= 0.5f;
    const gboolean isolate_02 = b[0] != center_lit;
    return entry ^ (isolate_02 ? 3 : 1);
  }
  for(int s = 0; s < 4; s++)
    if(cross[s] && s != entry) return s;
  return -1;
}

// extract the 0.5 contour of the entry as loops of level-0 file
// coordinates separated by DT_INVALID_COORDINATE pairs -- a convention
// internal to this type, our display callbacks are the only readers
// (the shared point-in-form helpers read the path shapes' (INVALID,
// index) skip convention instead). runs on the mip whose longest side
// fits a screen-sized budget: sub-pixel fidelity at canvas scale for an
// 8th of the work of the native plane. runs with NO lock held; NULL
// with *count == 0 when nothing reaches the threshold (or on alloc
// failure -- both render as "no outline", which is the right degraded
// display). caller owns the buffer (dt_free_align)
static float *_raster_extract_outline(const _raster_cache_entry_t *const entry,
                                      int *count)
{
  double start = dt_get_debug_wtime();
  *count = 0;
  int level = 0;
  while(level < RASTER_MIP_LEVELS - 1
        && MAX(entry->lw[level], entry->lh[level]) > 1024)
    level++;
  const float scale = (float)(1 << level);
  const int nx = entry->lw[level] + 2;  // the virtual zero ring
  const int ny = entry->lh[level] + 2;

  // every loop crosses a horizontal edge (the top row of its region), so
  // starts and the visited set only need those
  guint8 *hvisited = g_try_malloc0((size_t)(nx - 1) * ny);
  if(!hvisited) return NULL;
  dt_masks_dynbuf_t *dyn = dt_masks_dynbuf_init(1 << 14, "raster outline");
  if(!dyn)
  {
    g_free(hvisited);
    return NULL;
  }

  int nloops = 0;
  // padded rows 0 and ny-1 are all zeros: no crossing can involve them
  for(int j = 1; j < ny - 1; j++)
    for(int i = 0; i < nx - 1; i++)
    {
      if(hvisited[(size_t)j * (nx - 1) + i]) continue;
      const float e0 = _ms_sample(entry, level, i, j);
      const float e1 = _ms_sample(entry, level, i + 1, j);
      if((e0 >= 0.5f) == (e1 >= 0.5f)) continue;

      // walk the loop: append the crossing point of the current edge,
      // cross the adjacent cell to its exit edge, repeat until back at
      // the start edge. every crossing edge has both neighbour cells in
      // the padded range, so the walk needs no boundary cases
      const size_t lstart = dt_masks_dynbuf_position(dyn);
      gboolean horiz = TRUE;
      int ei = i, ej = j;                  // the current crossing edge
      int ci = i, cj = j, entry_side = 0;  // its cell above, by its bottom
      const size_t max_steps = (size_t)4 * nx * ny;  // defensive bound
      gboolean closed = FALSE;
      for(size_t step = 0; step < max_steps; step++)
      {
        float px, py;
        if(horiz)
        {
          const float v0 = _ms_sample(entry, level, ei, ej);
          const float v1 = _ms_sample(entry, level, ei + 1, ej);
          px = ei + (0.5f - v0) / (v1 - v0);
          py = ej;
          hvisited[(size_t)ej * (nx - 1) + ei] = TRUE;
        }
        else
        {
          const float v0 = _ms_sample(entry, level, ei, ej);
          const float v1 = _ms_sample(entry, level, ei, ej + 1);
          px = ei;
          py = ej + (0.5f - v0) / (v1 - v0);
        }
        // padded sample s sits on level pixel s-1, whose center is at
        // (s-1+0.5) * 2^level of the level-0 corner frame -- the same
        // frame the render maps to the sensor by adding the crop
        dt_masks_dynbuf_add_2(dyn, (px - 0.5f) * scale,
                              (py - 0.5f) * scale);

        const float v[4] = { _ms_sample(entry, level, ci, cj),
                             _ms_sample(entry, level, ci + 1, cj),
                             _ms_sample(entry, level, ci + 1, cj + 1),
                             _ms_sample(entry, level, ci, cj + 1) };
        const int exit_side = _ms_exit(v, entry_side);
        if(exit_side < 0) break;
        switch(exit_side)
        {
          case 0:  // out through the bottom, into the cell below
            horiz = TRUE;
            ei = ci;
            ej = cj;
            cj = cj - 1;
            entry_side = 2;
            break;
          case 2:  // out through the top, into the cell above
            horiz = TRUE;
            ei = ci;
            ej = cj + 1;
            cj = cj + 1;
            entry_side = 0;
            break;
          case 1:  // out through the right side
            horiz = FALSE;
            ei = ci + 1;
            ej = cj;
            ci = ci + 1;
            entry_side = 3;
            break;
          default:  // 3: out through the left side
            horiz = FALSE;
            ei = ci;
            ej = cj;
            ci = ci - 1;
            entry_side = 1;
            break;
        }
        if(horiz && ei == i && ej == j)
        {
          closed = TRUE;
          break;
        }
      }
      if(!closed)
      {
        // the guard cut a spin a logic failure would cause: forget the
        // partial loop rather than hand a broken polygon to the canvas
        dt_masks_dynbuf_reset_position(dyn, lstart);
        continue;
      }

      // light in-place simplification: marching squares emits exactly
      // collinear runs along the axis-aligned stretches of the contour.
      // the cross-product epsilon scales with the emitted coordinates
      float *lp = dt_masks_dynbuf_buffer(dyn) + lstart;
      const int n = (int)((dt_masks_dynbuf_position(dyn) - lstart) / 2);
      const float eps = 1e-3f * scale * scale;
      int m = 1;  // the first point always stays
      for(int k = 1; k < n; k++)
      {
        const float *prev = lp + 2 * (m - 1);
        const float *cur = lp + 2 * k;
        const float *next = lp + 2 * ((k + 1) % n);
        const float area2 = (cur[0] - prev[0]) * (next[1] - prev[1])
                            - (cur[1] - prev[1]) * (next[0] - prev[0]);
        if(fabsf(area2) <= eps) continue;
        lp[2 * m] = cur[0];
        lp[2 * m + 1] = cur[1];
        m++;
      }
      if(m < 3)
      {
        dt_masks_dynbuf_reset_position(dyn, lstart);
        continue;
      }
      dt_masks_dynbuf_reset_position(dyn, lstart + 2 * (size_t)m);
      dt_masks_dynbuf_add_2(dyn, DT_INVALID_COORDINATE,
                            DT_INVALID_COORDINATE);
      nloops++;
    }
  g_free(hvisited);

  const size_t total = dt_masks_dynbuf_position(dyn) / 2;
  float *out = total > 0 ? dt_masks_dynbuf_harvest(dyn) : NULL;
  dt_masks_dynbuf_free(dyn);
  dt_print(DT_DEBUG_MASKS,
           "[masks raster] outline '%s' mip %d: %d loop(s), %d points, "
           "took %0.04f sec",
           entry->path, level, nloops, (int)total, dt_get_lap_time(&start));
  *count = (int)total;
  return out;
}

// the memoized outline of an entry, extracted on first request. the
// extraction runs outside any lock on the entry's immutable planes; the
// result is published UNDER THE TABLE MUTEX (and read back under it),
// never by a bare write -- the entry is shared between the GUI thread
// asking here and the pipe threads holding references. once published
// the fields never change again, so the returned pointer stays valid
// lock-free for as long as the caller holds its entry reference
static const float *_raster_entry_outline(_raster_cache_entry_t *entry,
                                          int *count)
{
  g_mutex_lock(&_raster_cache_lock);
  gboolean done = entry->outline_done;
  const float *out = entry->outline;
  int n = entry->outline_count;
  g_mutex_unlock(&_raster_cache_lock);
  if(done)
  {
    *count = n;
    return out;
  }

  int fresh_count = 0;
  float *fresh = _raster_extract_outline(entry, &fresh_count);

  g_mutex_lock(&_raster_cache_lock);
  if(entry->outline_done)
  {
    // a concurrent expose extracted it first: the published one wins
    out = entry->outline;
    n = entry->outline_count;
    g_mutex_unlock(&_raster_cache_lock);
    dt_free_align(fresh);
    *count = n;
    return out;
  }
  entry->outline = fresh;
  entry->outline_count = fresh_count;
  entry->outline_done = TRUE;
  g_mutex_unlock(&_raster_cache_lock);
  *count = fresh_count;
  return fresh;
}

// the gpt polylines of the form: the memoized file-frame outline mapped
// to the preview pipe -- +metadata crop, normalized by the sensor size,
// scaled by the pipe input size, then through the pipe distortions, the
// same chain every drawn shape takes for its display points
static int _raster_get_points_border(dt_develop_t *dev,
                                     dt_masks_form_t *form,
                                     float **points,
                                     int *points_count,
                                     float **border,
                                     int *border_count,
                                     const int source,
                                     const dt_iop_module_t *const module)
{
  (void)module; // unused arg, keep compiler from complaining
  *points = NULL;
  *points_count = 0;
  if(border) *border = NULL;
  if(border_count) *border_count = 0;
  // no geometric border, and never a clone source
  if(source) return 0;

  dt_masks_point_raster_t *pt = _raster_point(form);
  if(!pt) return 0;

  // GUI side: the displayed image and the preview pipe set the frames
  const dt_image_t *img = &dev->image_storage;
  if(img->width <= 0 || img->height <= 0) return 0;

  gchar *path = _raster_resolve_path(pt, img);
  if(!path) return 0;
  // quiet: the expose retries while a recompute prepares the file, and
  // the negative cache prices every retry at one stat
  _raster_cache_entry_t *entry = _raster_cache_acquire(path, TRUE);
  g_free(path);
  // missing file: an empty gpt is tolerated everywhere. returning 0
  // leaves gui->pipe_hash unset, so the next expose retries -- cheap
  if(!entry) return 0;

  int ocount = 0;
  const float *outline = _raster_entry_outline(entry, &ocount);
  if(!outline || ocount < 4)
  {
    // nothing at or above the display threshold anywhere in the file:
    // succeed with an empty polygon set so the expose does not retry
    _raster_cache_release(entry);
    return 1;
  }

  float wd = 0.0f, ht = 0.0f;
  dt_masks_get_image_size(NULL, NULL, &wd, &ht);
  float *pts = (wd > 0.0f && ht > 0.0f)
    ? dt_alloc_align_float((size_t)ocount * 2)
    : NULL;
  if(!pts)
  {
    _raster_cache_release(entry);
    return 0;
  }

  // the sentinel pairs separating the loops must not travel through the
  // pipe transform (path.c writes its skip sentinels after the batch
  // transform for the same reason): their slot carries a copy of the
  // loop's first point for the ride and is overwritten below
  const float sx = wd / (float)img->width;
  const float sy = ht / (float)img->height;
  int lstart = 0;
  for(int k = 0; k < ocount; k++)
  {
    if(outline[2 * k] == DT_INVALID_COORDINATE)
    {
      pts[2 * k] = pts[2 * lstart];
      pts[2 * k + 1] = pts[2 * lstart + 1];
      lstart = k + 1;
    }
    else
    {
      pts[2 * k] = (outline[2 * k] + (float)img->crop_x) * sx;
      pts[2 * k + 1] = (outline[2 * k + 1] + (float)img->crop_y) * sy;
    }
  }

  if(!dt_dev_distort_transform(dev, pts, ocount))
  {
    dt_free_align(pts);
    _raster_cache_release(entry);
    return 0;
  }

  for(int k = 0; k < ocount; k++)
    if(outline[2 * k] == DT_INVALID_COORDINATE)
    {
      pts[2 * k] = DT_INVALID_COORDINATE;
      pts[2 * k + 1] = DT_INVALID_COORDINATE;
    }
  _raster_cache_release(entry);

  *points = pts;
  *points_count = ocount;
  return 1;
}

// squared distance from (x, y) to the segment (ax, ay)-(bx, by)
static float _raster_seg_dist2(const float x,
                               const float y,
                               const float ax,
                               const float ay,
                               const float bx,
                               const float by)
{
  const float dx = bx - ax, dy = by - ay;
  const float l2 = dx * dx + dy * dy;
  float t = l2 > 0.0f ? ((x - ax) * dx + (y - ay) * dy) / l2 : 0.0f;
  t = CLAMP(t, 0.0f, 1.0f);
  const float px = ax + t * dx - x;
  const float py = ay + t * dy - y;
  return px * px + py * py;
}

// canvas selection: even-odd over the outline loops of the gpt, squared
// distance to the nearest segment like the other shapes report. no
// border, no anchors, no source -- selecting the shape is all a click
// can do to it (the panel row selects it too, group.c guards this
// callback there)
static void _raster_get_distance(const float x,
                                 const float y,
                                 const float as,
                                 dt_masks_form_gui_t *gui,
                                 const int index,
                                 const int num_points,
                                 gboolean *inside,
                                 gboolean *inside_border,
                                 int *near,
                                 gboolean *inside_source,
                                 float *dist)
{
  (void)as;         // unused args, keep compiler from complaining
  (void)num_points;
  *inside = FALSE;
  *inside_border = FALSE;
  *near = -1;
  *inside_source = FALSE;
  *dist = FLT_MAX;

  if(!gui) return;
  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  if(!gpt || !gpt->points || gpt->points_count < 4) return;

  int crossings = 0;
  int start = 0;
  for(int k = 0; k <= gpt->points_count; k++)
  {
    const gboolean cut = (k == gpt->points_count)
      || gpt->points[2 * k] == DT_INVALID_COORDINATE;
    if(!cut) continue;
    if(k - start >= 3)
    {
      // each segment of the loop [start, k), closing edge included
      for(int p = start; p < k; p++)
      {
        const int q = (p + 1 == k) ? start : p + 1;
        const float ax = gpt->points[2 * p], ay = gpt->points[2 * p + 1];
        const float bx = gpt->points[2 * q], by = gpt->points[2 * q + 1];
        *dist = fminf(*dist, _raster_seg_dist2(x, y, ax, ay, bx, by));
        if((ay > y) != (by > y)
           && x < ax + (y - ay) / (by - ay) * (bx - ax))
          crossings++;
      }
    }
    start = k + 1;
  }
  *inside = (crossings & 1) != 0;
}

static void _raster_set_form_name(dt_masks_form_t *const form,
                                  const size_t nb)
{
  // a detection-made shape takes its detector's label ("subject", later
  // "sky"), resolved from the shape's own recipe through the detectors
  // table -- the single authority on that mapping; the clicked gesture
  // keeps its historical name. the caller's nb counts EVERY raster shape
  // (and retries it upward until the name is unique), so subtract the
  // shapes under other names to keep the label's own numbering dense:
  // "subject", then "subject 2" -- still strictly increasing in nb, so
  // the caller's uniqueness loop terminates as before
  const dt_masks_point_raster_t *pt = dt_masks_raster_point(form);
  const dt_detector_t *detector =
    (pt && dt_rf_recipe_valid(&pt->recipe)
     && pt->recipe.prompt_kind != DT_RF_PROMPT_POINTS)
    ? dt_detector_find(pt->recipe.prompt_kind, pt->recipe.class_bits)
    : NULL;
  if(detector)
  {
    const char *label = _(detector->label);
    size_t other = 0;
    if(darktable.develop)
      for(GList *l = darktable.develop->forms; l; l = g_list_next(l))
      {
        const dt_masks_form_t *f = l->data;
        if(f != form && f->type == form->type
           && strncmp(f->name, label, strlen(label)) != 0)
          other++;
      }
    const size_t label_nb = nb > other ? nb - other : 1;
    if(label_nb <= 1)
      g_strlcpy(form->name, label, sizeof(form->name));
    else
      snprintf(form->name, sizeof(form->name), "%s %d", label,
               (int)label_nb);
  }
  else
    snprintf(form->name, sizeof(form->name), _("precise mask #%d"),
             (int)nb);
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
  (void)num_points; // unused arg, keep compiler from complaining

  // never in creation mode: a raster shape is built by code, not drawn.
  // an empty or missing gpt (absent file, empty mask) draws nothing
  dt_masks_form_gui_points_t *gpt = g_list_nth_data(gui->points, index);
  if(!gpt || !gpt->points || gpt->points_count < 4) return;

  const gboolean selected = (gui->group_selected == index)
    && (gui->form_selected || gui->form_dragging);

  // every loop as one cairo subpath, stroked once through the shared
  // helper in its dashed style: the dashes read as "a precise mask",
  // not an editable outline -- there are no anchors to draw
  gboolean any = FALSE;
  int start = 0;
  for(int k = 0; k <= gpt->points_count; k++)
  {
    const gboolean cut = (k == gpt->points_count)
      || gpt->points[2 * k] == DT_INVALID_COORDINATE;
    if(!cut) continue;
    if(k - start >= 3)
    {
      cairo_move_to(cr, gpt->points[2 * start], gpt->points[2 * start + 1]);
      for(int p = start + 1; p < k; p++)
        cairo_line_to(cr, gpt->points[2 * p], gpt->points[2 * p + 1]);
      cairo_close_path(cr);
      any = TRUE;
    }
    start = k + 1;
  }
  if(any)
    dt_masks_line_stroke(cr, TRUE, FALSE, selected, zoom_scale);
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
  .get_distance = _raster_get_distance,
  .get_points = NULL,
  .get_points_border = _raster_get_points_border,
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
