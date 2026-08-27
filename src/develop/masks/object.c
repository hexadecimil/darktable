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

#include "common/ai/segmentation.h"
#include "common/ai/refine.h"
#include "common/ai/detect.h"
#include "common/ai/detectors.h"
#include "common/ai_models.h"
#include "common/colorspaces.h"
#include "common/debug.h"
#include "common/densecrf.h"
#include "common/matte.h"

#include "common/rasterfile_recipe.h"
// mirror of iop/rasterfile.c's parameter layout (v2), which lives in the
// module file only. keep in sync; introspection guards the size at history
// level, and the recipe struct itself is shared through rasterfile_recipe.h
#define RASTERFILE_MAXFILE 2048
typedef enum dt_iop_rasterfile_mode_t
{
  DT_RASTERFILE_MODE_ALL = 7,
} dt_iop_rasterfile_mode_t;
typedef struct dt_iop_rasterfile_params_t
{
  dt_iop_rasterfile_mode_t mode;
  char path[RASTERFILE_MAXFILE];
  char file[RASTERFILE_MAXFILE];
  int32_t _pad;
  dt_rf_recipe_t recipe;
} dt_iop_rasterfile_params_t;
#include "common/distance_transform.h"
#include "common/mipmap_cache.h"
#include "common/ras2vect.h"
#include "control/conf.h"
#include "control/control.h"
#include "control/jobs.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/masks.h"
#include "develop/masks/object_recipe.h"
#include "develop/openmp_maths.h"
#include "develop/pixelpipe_hb.h"
#include "gui/gtk.h"
#include "imageio/imageio_common.h"

#include <png.h>
#include <setjmp.h>
#include "views/view.h"

#include <limits.h>
#include <math.h>
#include <string.h>

#define CONF_OBJECT_THRESHOLD_KEY "plugins/darkroom/masks/object/threshold"
#define CONF_OBJECT_REFINE_PASSES_KEY "plugins/darkroom/masks/object/refine_passes"
#define CONF_OBJECT_CLEANUP_KEY "plugins/darkroom/masks/object/cleanup"
#define CONF_OBJECT_SMOOTHING_KEY "plugins/darkroom/masks/object/smoothing"
#define CONF_OBJECT_FEATHER_KEY "plugins/darkroom/masks/object/feather"
#define CONF_OBJECT_PERSIST_KEY "plugins/darkroom/masks/object/persist_model"
#define CONF_OBJECT_PATH_PREVIEW_KEY "plugins/darkroom/masks/object/path_preview"
#define CONF_OBJECT_REFINE_BOUNDARY_KEY "plugins/darkroom/masks/object/refine_boundary"
#define CONF_OBJECT_REFINE_BOUNDARY_ITER_KEY "plugins/darkroom/masks/object/refine_boundary_iterations"
#define CONF_OBJECT_REFINE_BOUNDARY_SIGMA_COLOR_KEY "plugins/darkroom/masks/object/refine_boundary_sigma_color"
#define CONF_OBJECT_REFINE_BOUNDARY_W_BILATERAL_KEY "plugins/darkroom/masks/object/refine_boundary_weight_bilateral"
#define CONF_OBJECT_AI_REFINE_KEY "plugins/darkroom/masks/object/ai_refine"
#define CONF_OBJECT_AI_REFINE_MARGIN_KEY "plugins/darkroom/masks/object/ai_refine_margin"

// default render target (longest side in pixels).
// the SAM encoder internally downscales to 1024 so encoding quality
// is the same, but higher render resolution gives the guided filter
// and vectorizer more detail for edge refinement.
// configurable via plugins/darkroom/masks/object/render_size
#define SEG_RENDER_DEFAULT 1536
#define CONF_OBJECT_RENDER_SIZE_KEY "plugins/darkroom/masks/object/render_size"

// --- per-session segmentation state (stored in gui->scratchpad) ---

typedef enum _encode_state_t
{
  ENCODE_ERROR = -1,
  ENCODE_IDLE = 0,
  ENCODE_MSG_SHOWN = 1, // busy message queued, waiting for next expose
  ENCODE_READY = 2,     // encoding complete, results available
  ENCODE_RUNNING = 3,   // background thread in progress
} _encode_state_t;

// minimum drag distance (preview pipe pixels) to distinguish click from drag
#define DRAG_THRESHOLD 5.0f

typedef enum _decode_state_t
{
  DECODE_ERROR = -1,    // compute failed; drained by the publication machine
  DECODE_IDLE = 0,      // no decode in flight
  DECODE_RUNNING = 1,   // compute thread working
  DECODE_READY = 2,     // result waiting for publication on the GUI thread
} _decode_state_t;

typedef struct _decode_job_t _decode_job_t;
typedef struct _edit_replay_job_t _edit_replay_job_t;
static void _decode_job_free(_decode_job_t *job);
static gboolean _modifier_poll(gpointer data);
static void _clear_selection(dt_masks_form_gui_t *gui);

typedef struct _object_data_t
{
  dt_ai_environment_t *env; // AI environment for model registry
  dt_seg_context_t *seg;    // SAM context (encoder+decoder)
  float *mask;              // current mask buffer (preview pipe size)
  int mask_w, mask_h;       // mask dimensions
  gboolean model_loaded;    // whether the model was loaded
  int encode_state;         // uses _encode_state_t values (atomic access)
  dt_imgid_t encoded_imgid; // image ID that was encoded
  dt_hash_t encoded_distort_hash; // distort hash at encode time (detects crop/rotate)
  int encode_w, encode_h;   // encoding resolution (for coordinate mapping)
  int encoded_render_size;  // render cap the CURRENT encoding was produced
                            // under, as the RAW preference value: every use
                            // point applies MAX(..., 1024), so the recipe
                            // records the preference and never the effective
                            // cap -- ONE convention, shared by the capture,
                            // the pinned encode and the headless replay.
                            // written by the launch (pinned or plain) and
                            // read unconditionally by _capture_recipe, so
                            // encode_w/h and render_size always describe the
                            // SAME render whatever became of the session
  guint modifier_poll_id;   // timer to detect shift key changes
  GThread *encode_thread;   // background encoding thread
  gboolean dragging;        // TRUE between press and release during click drag
  float drag_start_x;       // press position (preview pipe pixel space)
  float drag_start_y;
  gboolean has_selection;   // TRUE after first click, enables refinement mode
  // vectorization preview (auto-updated after each decode)
  GList *preview_forms;             // GList of dt_masks_form_t* (mask-space pixel coords)
  GList *preview_signs;             // parallel GList of sign values ('+' or '-')
  int preview_cleanup;              // current cleanup (potrace turdsize, 0-100)
  float preview_smoothing;          // current smoothing (potrace alphamax, 0.0-1.3)
  float preview_feather;            // path border/feather (0.0-0.5, normalized)
  gboolean preview_refine;          // run DenseCRF edge refinement on each decode
  dt_refine_context_t *refine;      // CascadePSP contour refinement, loaded lazily
  gboolean refine_failed;           // TRUE once loading failed, do not retry
  // -- asynchronous decode --
  int decode_state;                 // _decode_state_t values (atomic access)
  GThread *decode_thread;           // in-flight compute, joined at publication
  _decode_job_t *decode_job;        // job of the in-flight compute
  gboolean decode_pending;          // GUI thread only: clicks landed mid-decode
  int decode_launched_count;        // GUI thread only: point count at launch
  gboolean decode_busy_shown;       // GUI thread only: busy_enter needs a leave
  GArray *decode_marks;             // GUI thread only: one _decode_mark_t per
                                    // accumulated click, provenance recording
  // provenance: scalars of the last launched decode. the mask the
  // finalisation captures is the one that decode computed, so the recipe
  // must record these values, not the conf/preview state at capture time
  // (a toggle after the last decode would otherwise poison the recipe).
  // threshold stays per-boundary in decode_marks
  gboolean last_do_crf;
  int last_crf_iter;
  float last_crf_sigma_color;
  float last_crf_w_bilateral;
  gboolean last_do_refine;
  float last_refine_margin;
  int last_n_passes;
  // -- ai edit session (reopening a saved recipe interactively) --
  // owned by the edit machine below; every transition happens on the GUI
  // thread (post_expose, kept ticking by the 100 ms poll timer),
  // edit_pending is read atomically
  int edit_pending;            // _edit_pending_t values (atomic access)
  gboolean edit_valid;         // TRUE while an edit session drives this data
  gboolean edit_dirty;         // a clear happened: never report a no-op
  gboolean edit_model_pinned;  // seg was loaded BY ID for the session:
                               // never persist it in the active-model slot
  gboolean edit_no_sink;       // the session opened with no focused module
                               // and must close into none. SURVIVES
                               // _edit_session_end: the closing gesture is
                               // the last thing to run, and by then
                               // dev->gui_module may hold whatever module
                               // the user unfolded in the meantime
  dt_rf_recipe_t edit_recipe;  // owned copy of the recipe being replayed
  // -- the session's own parameters --
  // during an edit session these hold the RECORDED values, so a decode
  // added after the replay continues the recorded chain instead of
  // whatever the preferences drifted to since the mask was created. read
  // through the _session_* accessors below, which fall back to the conf
  // outside a session. the rule they enforce is greppable: no decode,
  // replay or finalisation path reads these conf keys directly, and none
  // of them is ever WRITTEN from a session (they are the user's global
  // persistent preferences, clobbering them on every edit would destroy
  // the user's values)
  float edit_threshold;        // NAN outside an edit session
  int edit_n_passes;
  int edit_crf_iter;
  float edit_crf_sigma_color;
  float edit_crf_w_bilateral;
  gboolean edit_do_refine;
  float edit_refine_margin;
  int edit_render_size;        // recorded render cap, raw (see
                               // encoded_render_size for the convention)
  // the matting stage of the session, seeded from the replayed recipe like
  // every parameter above. it exists because the stage is otherwise a
  // PREFERENCE, and a session that read the preference would re-finalise a
  // reopened mask with today's settings instead of the ones the mask was
  // made with -- the very drift the recorded threshold and render cap are
  // seeded to prevent. read through _session_matte(), never directly.
  // `edit_matte_op` points into the static table, so it is borrowed and
  // outlives any session by construction
  gboolean edit_matte_seeded;  // FALSE until edit_begin has seeded them
  gboolean edit_matte_enabled;
  const dt_matte_op_t *edit_matte_op;
  float edit_matte_band;
  // no-op baseline: the recipe the session WOULD re-capture at the end of
  // the replay. closing with a byte-identical re-capture means the session
  // produced nothing new. snapshotting the recipe instead of a point count
  // makes the criterion exhaustive by construction -- every parameter that
  // reaches the mask reaches the recipe, so no mutator has to remember to
  // raise a flag and no future field can escape the check
  dt_rf_recipe_t edit_baseline_recipe;
  gboolean edit_baseline_valid;
  GThread *edit_replay_thread; // in-flight replay; joined by the machine
                               // (or by _destroy_data on teardown), NEVER
                               // by _decode_finish
  _edit_replay_job_t *edit_replay_job;
} _object_data_t;

// clicks can outpace the compute (coalescing), so N points do not mean N
// decodes. the provenance recipe must replay decodes exactly at the
// boundaries where they really happened to reproduce the iterative
// refinement context, so each click records whether a decode was launched
// right after it, and with which threshold
typedef struct _decode_mark_t
{
  float threshold;
  gboolean launched;
} _decode_mark_t;

static void _marks_resize(_object_data_t *d, const int count)
{
  if(!d->decode_marks)
    d->decode_marks = g_array_new(FALSE, TRUE, sizeof(_decode_mark_t));
  g_array_set_size(d->decode_marks, count);
}

// ---------------------- ai edit session: state machine ----------------------
//
// an edit session reopened from a saved provenance recipe is a NORMAL
// creation session of the object tool, plus this machine driving it:
//
//   EDIT_NONE -> EDIT_WAIT_ENCODE       dt_object_mask_edit_begin
//   EDIT_WAIT_ENCODE -> EDIT_REPLAYING  post_expose at ENCODE_READY: join
//        of the encode thread (its decoder warmup outlives ENCODE_READY),
//        injection of the recorded clicks AND decode marks, replay thread
//        started on a stand-in _object_data_t
//   EDIT_REPLAYING -> EDIT_ACTIVE       post_expose when the replay thread
//        is done: joined by this machine EXCLUSIVELY, final mask published,
//        provenance scalars restored, no-op baseline snapshotted
//   EDIT_ACTIVE -> EDIT_NONE            every session exit (finalise,
//        no-op close, cancel, image change, teardown)
//   any -> EDIT_FAILED                  encode or replay failure; only the
//        edit bookkeeping ends, the tool session survives as a plain one
//
// while WAIT_ENCODE or REPLAYING the session data is FROZEN: clicks are
// refused, the publication machine consumes nothing, the invalidation
// branch defers its teardown -- the replay thread has exclusive rights on
// the shared segmentation context
typedef enum _edit_pending_t
{
  EDIT_FAILED = -1,
  EDIT_NONE = 0,
  EDIT_WAIT_ENCODE = 1,
  EDIT_REPLAYING = 2,
  EDIT_ACTIVE = 3,
} _edit_pending_t;

// the replay job: the recorded decode chain, executed on a worker thread
// against a STAND-IN _object_data_t whose seg/env/refine fields ALIAS the
// session's. passing the session's own data instead would publish
// DECODE_READY at every boundary (last store of _decode_thread_func) and
// the publication machine would join the replay thread in mid-flight from
// the GUI side, then let a click start a second thread on the same ONNX
// context. od.decode_state is stored per boundary and polled by nobody --
// the exact pattern of the headless replay's local stand-in; the machine
// polls `done`, this thread's very last store
struct _edit_replay_job_t
{
  _object_data_t od;    // stand-in; seg/env/refine ALIAS the session's
  dt_rf_recipe_t recipe;
  float *enc_pts;       // prompts in session encode space, snapshotted on
                        // the GUI thread at launch
  int n_points;
  int last_decode;      // index of the last recorded decode boundary
  float *out_mask;      // final boundary's mask, owned until published
  int out_w, out_h;
  gboolean ok;
  int done;             // atomic; set as the thread's last instruction
};

static gboolean _edit_replay_running(const _object_data_t *d)
{
  return d->edit_replay_thread && d->edit_replay_job
         && !g_atomic_int_get(&d->edit_replay_job->done);
}

// TRUE while the machine holds the session frozen (encode pending or
// replay in flight): clicks, publication and invalidation all test this
static gboolean _edit_frozen(const _object_data_t *d)
{
  const int ep = d ? g_atomic_int_get(&d->edit_pending) : EDIT_NONE;
  return ep == EDIT_WAIT_ENCODE || ep == EDIT_REPLAYING;
}

// ---------------------- the session parameter accessors ---------------------
//
// the ONE boundary between an edit session and the global preferences.
// during a session every scalar a decode consumes comes from the RECIPE:
// the format assumes these scalars are constant over a session, and the
// preferences have usually drifted since the mask was created -- a new
// click reading the conf would refine the replayed prefix with different
// parameters than the ones the re-captured recipe records, and the
// regenerated file would then diverge from the mask the user validated,
// under the very same fingerprint.
//
// the invariant is local and greppable: outside these accessors NO decode,
// replay or finalisation path reads CONF_OBJECT_THRESHOLD_KEY,
// CONF_OBJECT_REFINE_PASSES_KEY, CONF_OBJECT_REFINE_BOUNDARY_*_KEY,
// CONF_OBJECT_AI_REFINE_KEY or CONF_OBJECT_AI_REFINE_MARGIN_KEY. an
// override, never a conf write: those keys stay the user's persistent
// preferences

static float _session_threshold(const _object_data_t *d)
{
  if(d && d->edit_valid && !isnan(d->edit_threshold))
    return CLAMP(d->edit_threshold, 0.3f, 0.9f);
  return CLAMP(dt_conf_get_float(CONF_OBJECT_THRESHOLD_KEY), 0.3f, 0.9f);
}

static int _session_n_passes(const _object_data_t *d)
{
  if(d && d->edit_valid)
    return CLAMP(d->edit_n_passes, 1, 3);
  return CLAMP(dt_conf_get_int(CONF_OBJECT_REFINE_PASSES_KEY), 1, 3);
}

static int _session_crf_iter(const _object_data_t *d)
{
  if(d && d->edit_valid)
    return CLAMP(d->edit_crf_iter, 1, 10);
  return CLAMP(dt_conf_get_int(CONF_OBJECT_REFINE_BOUNDARY_ITER_KEY), 1, 10);
}

static float _session_crf_sigma_color(const _object_data_t *d)
{
  if(d && d->edit_valid)
    return CLAMP(d->edit_crf_sigma_color, 1.0f, 50.0f);
  return CLAMP(dt_conf_get_float(CONF_OBJECT_REFINE_BOUNDARY_SIGMA_COLOR_KEY),
               1.0f, 50.0f);
}

static float _session_crf_w_bilateral(const _object_data_t *d)
{
  if(d && d->edit_valid)
    return CLAMP(d->edit_crf_w_bilateral, 0.5f, 30.0f);
  return CLAMP(dt_conf_get_float(CONF_OBJECT_REFINE_BOUNDARY_W_BILATERAL_KEY),
               0.5f, 30.0f);
}

// refine_failed is deliberately NOT folded in here: it is a runtime
// outcome, not a parameter, and the callers that must honour it (the
// decode launch) differ from the ones that must not (the capture, which
// records what the decode really did)
static gboolean _session_do_refine(const _object_data_t *d)
{
  if(d && d->edit_valid)
    return d->edit_do_refine;
  return dt_conf_get_bool(CONF_OBJECT_AI_REFINE_KEY);
}

static float _session_refine_margin(const _object_data_t *d)
{
  if(d && d->edit_valid)
    return CLAMPF(d->edit_refine_margin, 0.0f, 0.5f);
  return CLAMPF(dt_conf_get_float(CONF_OBJECT_AI_REFINE_MARGIN_KEY),
                0.0f, 0.5f);
}

// the render cap a PLAIN session encodes under, raw: the single reader of
// the render-size preference. every use point applies MAX(..., 1024)
static int _conf_render_size(void)
{
  return dt_conf_key_exists(CONF_OBJECT_RENDER_SIZE_KEY)
    ? dt_conf_get_int(CONF_OBJECT_RENDER_SIZE_KEY)
    : SEG_RENDER_DEFAULT;
}

// end-of-session bookkeeping, shared by EVERY exit path (finalise, no-op
// close, cancel, failure, image or geometry change, teardown): the
// proactive-recompute gate falls, the threshold override falls with it.
// GUI thread; idempotent
static void _edit_session_end(_object_data_t *d, const int final_state)
{
  if(!d)
    return;
  if(d->edit_valid)
    dt_object_mask_edit_clear_active();
  d->edit_valid = FALSE;
  d->edit_threshold = NAN;
  // the matting override falls with the threshold one, and the borrowed
  // table pointer goes with it: edit_valid alone already sends
  // _session_matte back to the preference, this just leaves no stale
  // reading behind for a later session to inherit
  d->edit_matte_seeded = FALSE;
  d->edit_matte_op = NULL;
  d->edit_baseline_valid = FALSE;
  g_atomic_int_set(&d->edit_pending, final_state);
}

// THE single place that knows how to undo a pinned encode: the recorded
// model loaded BY ID and the render made at the recorded dimensions under
// the recorded cap. every non-nominal exit of the machine calls it, so a
// session that survives a failure is really the "plain, empty session" its
// toast promises -- otherwise it would keep decoding on the recorded model
// and capture recipes whose encode_w/h come from a cap the session no
// longer claims. dropping the encoding sends post_expose back through the
// plain launch (ACTIVE model, current preference).
//
// returns TRUE when something was dropped. GUI thread, and only where
// nothing computes on d->seg -- every caller is past the replay join. an
// encode that FAILED is left alone: nothing was pinned successfully and
// resetting its state would only spin a relaunch loop
static gboolean _edit_drop_pinned_encode(_object_data_t *d)
{
  if(!d || !d->edit_model_pinned
     || g_atomic_int_get(&d->encode_state) != ENCODE_READY)
    return FALSE;

  if(d->seg)
  {
    dt_seg_free(d->seg);
    d->seg = NULL;
  }
  d->model_loaded = FALSE;
  d->edit_model_pinned = FALSE;
  d->encode_w = d->encode_h = 0;
  d->encoded_render_size = 0;
  g_atomic_int_set(&d->encode_state, ENCODE_IDLE);
  return TRUE;
}

static _object_data_t *_get_data(dt_masks_form_gui_t *gui)
{
  return (gui && gui->scratchpad) ? (_object_data_t *)gui->scratchpad : NULL;
}

// the queryable form of "a computation owns this session", read STRAIGHT
// from the session machines -- the encode, the decode and the edit replay.
// no mirror flag to keep in sync, so a state a later version adds cannot
// escape the answer by omission; the price is that the three enums below
// are the contract, and any new one belongs here.
//
// the session data IS the scratchpad of the live mask gui: once _free_data
// cleared it the answer is rightly FALSE whatever threads still finish in
// the void, because there is no longer a session a new shape could
// destroy. deliberately NOT covering the finalisation and recompute jobs:
// they own no session either (the finalisation frees it before enqueuing),
// they compete for VRAM, which the compute paths already arbitrate on
// their own. GUI thread
gboolean dt_masks_object_session_busy(void)
{
  const dt_develop_t *dev = darktable.develop;
  const _object_data_t *d = dev ? _get_data(dev->form_gui) : NULL;
  if(!d)
    return FALSE;
  // MSG_SHOWN counts: the encode is committed, the next expose starts the
  // thread, and the frame in between must not become a hole in the lock
  const int es = g_atomic_int_get(&d->encode_state);
  return es == ENCODE_MSG_SHOWN
         || es == ENCODE_RUNNING
         || g_atomic_int_get(&d->decode_state) == DECODE_RUNNING
         || _edit_frozen(d)
         || _edit_replay_running(d);
}

// compute a hash of all distortion module parameters
// from a develop history — changes on crop/rotate/perspective/lens
// but NOT on exposure/color/masks
static dt_hash_t _compute_distort_hash(dt_develop_t *dev)
{
  dt_hash_t hash = DT_INITHASH;
  for(GList *l = dev->history; l; l = g_list_next(l))
  {
    const dt_dev_history_item_t *item = l->data;
    if(item->module
       && item->module->enabled
       && (item->module->operation_tags() & IOP_TAG_DISTORT))
    {
      hash = dt_hash(hash, item->params, item->module->params_size);
    }
  }
  return hash;
}

static void _on_view_changed(gpointer instance,
                             dt_view_t *old_view,
                             dt_view_t *new_view,
                             gpointer user_data)
{
  (void)instance;
  (void)new_view;
  (void)user_data;

  // free persistent model when leaving darkroom
  if(old_view && old_view->view(old_view) == DT_VIEW_DARKROOM)
  {
    dt_ai_seg_t *seg = &darktable.ai_seg;
    if(seg->ctx)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] freeing persistent model");
      dt_seg_free(seg->ctx);
      seg->ctx = NULL;
    }
    if(seg->env)
    {
      dt_ai_env_destroy(seg->env);
      seg->env = NULL;
    }
    seg->model_loaded = FALSE;

    DT_CONTROL_SIGNAL_DISCONNECT(_on_view_changed, NULL);
    seg->signal_connected = FALSE;
  }
}

// free vectorized preview forms (never registered in dev->forms)
static void _free_preview_forms(_object_data_t *d)
{
  if(!d) return;
  for(GList *l = d->preview_forms; l; l = g_list_next(l))
    dt_masks_free_form(l->data);
  g_list_free(d->preview_forms);
  d->preview_forms = NULL;
  g_list_free(d->preview_signs);
  d->preview_signs = NULL;
}

// free all resources in _object_data_t. joins any remaining worker thread
// itself -- never call while a compute is still RUNNING, the join would block
// the GUI thread. preserves seg+env in persistent statics so the model stays
// loaded
static void _destroy_data(_object_data_t *d)
{
  if(!d)
    return;
  if(d->modifier_poll_id)
    g_source_remove(d->modifier_poll_id);
  if(d->encode_thread)
    g_thread_join(d->encode_thread);
  if(d->decode_thread)
    g_thread_join(d->decode_thread);
  if(d->decode_job)
    _decode_job_free(d->decode_job);
  // teardown path of an edit replay the machine never got to finish (the
  // machine is the joiner on the live path; here the machine is gone).
  // _free_data defers to _deferred_cleanup while the thread still runs
  if(d->edit_replay_thread)
    g_thread_join(d->edit_replay_thread);
  if(d->edit_replay_job)
  {
    _edit_replay_job_t *rj = d->edit_replay_job;
    // a refinement context lazily loaded DURING the replay lives in the
    // stand-in only; free it unless it aliases the session's own
    if(rj->od.refine && rj->od.refine != d->refine)
      dt_refine_free(rj->od.refine);
    g_free(rj->enc_pts);
    g_free(rj->out_mask);
    g_free(rj);
  }
  if(d->decode_busy_shown)
    dt_control_busy_leave();

  // save model to persistent storage - keeps it loaded across
  // mask sessions, disk cache handles embedding persistence.
  // only persist if nobody already claimed the slot (guards
  // against deferred cleanup racing with a new session)
  dt_ai_seg_t *ps = &darktable.ai_seg;
  // the persistent slot is an optimization for the ACTIVE model; a model
  // an edit session loaded BY ID does not belong there (a later plain
  // session would discard it anyway, after paying the swap)
  const gboolean persist = dt_conf_get_bool(CONF_OBJECT_PERSIST_KEY)
                           && !d->edit_model_pinned;
  if(persist && !ps->ctx && d->seg)
  {
    dt_seg_reset_encoding(d->seg);
    ps->env = d->env;
    ps->ctx = d->seg;
    ps->model_loaded = d->model_loaded;
    d->env = NULL;
    d->seg = NULL;
  }
  else
  {
    if(d->seg) dt_seg_free(d->seg);
    if(d->env) dt_ai_env_destroy(d->env);
    d->seg = NULL;
    d->env = NULL;
  }

  if(d->refine) dt_refine_free(d->refine);
  if(d->decode_marks) g_array_free(d->decode_marks, TRUE);
  g_free(d->mask);
  _free_preview_forms(d);
  g_free(d);
}

// idle callback for deferred cleanup when background thread was still running.
// inherited limitation shared with the encode: if the main loop stops before
// the next tick, the worker thread leaks at exit
static gboolean _deferred_cleanup(gpointer data)
{
  _object_data_t *d = data;
  if(g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING
     || g_atomic_int_get(&d->decode_state) == DECODE_RUNNING
     || _edit_replay_running(d))
    return G_SOURCE_CONTINUE;
  _destroy_data(d);
  return G_SOURCE_REMOVE;
}

static void _free_data(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d)
    return;
  gui->scratchpad = NULL;

  // whatever ends the session data ends the edit session: do the
  // bookkeeping NOW, on the GUI thread, even when the destruction below
  // defers -- a later dt_object_mask_edit_begin (GUI thread too) must
  // never observe a stale gate from a session being torn down
  _edit_session_end(d, EDIT_NONE);

  // the session is gone and so is the reason to hold the shape buttons.
  // this is the ONE exit every teardown funnels through -- finalise, no-op
  // close, cancel, image or geometry change, module refocus, view change,
  // module destruction -- and the poll timer that keeps the lock fresh
  // dies with the data below, so the last word has to be said here
  dt_masks_update_shapes_sensitivity();

  if(g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING
     || g_atomic_int_get(&d->decode_state) == DECODE_RUNNING
     || _edit_replay_running(d))
  {
    // a thread still runs on this data: defer, don't block the UI. safe
    // because the compute works from its own snapshot -- the dynbufs and
    // preview structures freed by our caller are never touched by it
    g_timeout_add(200, _deferred_cleanup, d);
    return;
  }
  _destroy_data(d);
}

// data passed to the background encoding thread
typedef struct _encode_thread_data_t
{
  _object_data_t *d;
  dt_imgid_t imgid;        // image to encode (thread renders via export pipe)
  int32_t history_end;     // darkroom history_end (may be ahead of database)
  dt_hash_t distort_hash;  // hash from live darkroom state (for disk cache key)
  // -- ai edit session: reproduce the RECORDED encode --
  // when pinned, the thread loads the recorded model BY ID and renders at
  // the recorded dimensions instead of the active model at the current
  // render preference. every deviation from the plain session mirrors the
  // headless replay (dt_object_recipe_compute), the fidelity reference
  gboolean pinned;
  char pin_model[DT_RF_RECIPE_MODEL_ID_LEN];
  int pin_w, pin_h;        // recorded encode dimensions
  // the render cap this encoding runs under, raw: the recorded one when
  // pinned (it recovers the exact scale), the current preference
  // otherwise. always set, so the render never re-reads the conf and the
  // session can record the cap its encoding really came from
  int render_size;
  dt_rf_recipe_t pin_recipe; // to disable the edited rasterfile instance
} _encode_thread_data_t;

// convert the float RGBA backbuf of a processed export pipe to the uint8
// RGB layout the segmentation encoder expects. NULL when the render failed
// or on allocation failure. shared by the interactive encode thread and
// the headless recipe replay
static uint8_t *_backbuf_to_rgb8(const dt_dev_pixelpipe_t *pipe,
                                 const int w,
                                 const int h)
{
  if(!pipe->backbuf)
    return NULL;
  const float *outbuf = (const float *)pipe->backbuf;
  uint8_t *rgb = g_try_malloc((size_t)w * h * 3);
  if(!rgb)
    return NULL;
  for(size_t i = 0; i < (size_t)w * h; i++)
  {
    rgb[i * 3 + 0] = (uint8_t)CLAMP(outbuf[i * 4 + 0] * 255.0f + 0.5f, 0, 255);
    rgb[i * 3 + 1] = (uint8_t)CLAMP(outbuf[i * 4 + 1] * 255.0f + 0.5f, 0, 255);
    rgb[i * 3 + 2] = (uint8_t)CLAMP(outbuf[i * 4 + 2] * 255.0f + 0.5f, 0, 255);
  }
  return rgb;
}

// encode `rgb` with `*seg`, falling back to a CPU-provider reload of the
// model on failure. `model_id` names the model to reload; NULL reloads the
// active "mask" model (fetched lazily, only when the fallback triggers).
// on a failed reload *seg is NULL on return. shared by the interactive
// encode thread and the headless recipe replay -- keep both callers'
// behaviour identical when touching this
static gboolean _seg_encode_cpu_fallback(dt_seg_context_t **seg,
                                         dt_ai_environment_t *env,
                                         const char *model_id,
                                         const uint8_t *rgb,
                                         const int w,
                                         const int h)
{
  gboolean ok = dt_seg_encode_image(*seg, rgb, w, h);
  if(!ok)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] encoding failed, retrying with CPU provider");
    dt_seg_free(*seg);
    dt_ai_env_set_provider(env, DT_AI_PROVIDER_CPU);
    char *active = model_id ? NULL : dt_ai_models_get_active_for_task("mask");
    *seg = dt_seg_load(env, model_id ? model_id : active);
    g_free(active);
    if(*seg)
      ok = dt_seg_encode_image(*seg, rgb, w, h);
  }
  return ok;
}

// background thread: loads model, renders image via export pipe, and encodes,
// does ZERO GLib/GTK calls - only computation + atomic state set,
// the poll timer on the main thread detects completion
static gpointer _encode_thread_func(gpointer data)
{
  _encode_thread_data_t *td = data;
  _object_data_t *d = td->d;
  const dt_imgid_t imgid = td->imgid;
  const int32_t td_history_end = td->history_end;
  const dt_hash_t distort_hash = td->distort_hash;
  const gboolean pinned = td->pinned;
  char pin_model[DT_RF_RECIPE_MODEL_ID_LEN];
  g_strlcpy(pin_model, td->pin_model, sizeof(pin_model));
  const int pin_w = td->pin_w, pin_h = td->pin_h;
  const int render_size = td->render_size;
  const dt_rf_recipe_t pin_recipe = td->pin_recipe;
  g_free(td);

  // load model if needed
  if(!d->model_loaded)
  {
    if(!d->env)
      d->env = dt_ai_env_init(NULL);

    // an edit session loads the RECORDED segmentation model by id, exactly
    // like the headless replay: the session reproduces the recorded
    // decodes, and the active model may be another one even when the
    // model-gap verdict passed at edit_begin (the verdict checks installed
    // state, not active-ness, for the segmentation model). plain sessions
    // keep loading the active model
    char *model_id = pinned ? g_strdup(pin_model)
                            : dt_ai_models_get_active_for_task("mask");
    d->seg = dt_seg_load(d->env, model_id);
    g_free(model_id);

    if(!d->seg)
    {
      g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
      return NULL;
    }
    d->model_loaded = TRUE;
  }

  // render image at high resolution via temporary export pipeline
  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dt_dev_load_image(&dev, imgid);

  // the database's history_end may lag behind the darkroom's
  // in-memory state (crop/rotate not flushed yet), override
  // so synch_all applies all current edits
  if(td_history_end > 0 && td_history_end > dev.history_end)
    dev.history_end = td_history_end;

  // edit session: reproduce the FIRST capture's state, mechanism of the
  // headless replay -- the loaded history contains the very rasterfile
  // instance being edited; left enabled, its consumers would render with
  // the edited mask's own effect (or a zeroed one when the file is
  // missing) baked into this encode, which the original session -- made
  // before the mask existed -- never saw. disable the instances that
  // carry OUR recipe (verbatim compare); other raster masks keep their
  // effect. this only protects the RE-RENDERED path: a shared .seg cache
  // hit below returns embeddings made by whoever wrote the cache, possibly
  // with the carrier instance enabled. that divergence is DETERMINISTIC
  // (the carrier's effect is baked into the cached render) and merely
  // revealed at random by the state of the cache -- not ONNX variance. TC2
  // decides whether it is visible; if it is, the pinned path must refuse a
  // hit whose render could contain the carrier
  if(pinned)
    for(GList *l = dev.iop; l; l = g_list_next(l))
    {
      dt_iop_module_t *m = l->data;
      if(!strcmp(m->op, "rasterfile") && m->enabled && m->params
         && memcmp(&((dt_iop_rasterfile_params_t *)m->params)->recipe,
                   &pin_recipe, sizeof(pin_recipe)) == 0)
        m->enabled = FALSE;
    }

  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(&buf, imgid, DT_MIPMAP_FULL, DT_MIPMAP_BLOCKING, 'r');

  if(!buf.buf || !buf.width || !buf.height)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] failed to get image buffer for encoding");
    dt_mipmap_cache_release(&buf);
    dt_dev_cleanup(&dev);
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  const int wd = dev.image_storage.width;
  const int ht = dev.image_storage.height;

  dt_dev_pixelpipe_t pipe;
  if(!dt_dev_pixelpipe_init_export(&pipe, wd, ht, IMAGEIO_RGB | IMAGEIO_INT8,
                                   FALSE))
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] failed to init export pipe for encoding");
    dt_mipmap_cache_release(&buf);
    dt_dev_cleanup(&dev);
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  dt_dev_pixelpipe_set_icc(&pipe, DT_COLORSPACE_SRGB, NULL,
                           DT_INTENT_PERCEPTUAL);
  dt_dev_pixelpipe_set_input(&pipe, &dev, (float *)buf.buf,
                             buf.width, buf.height, buf.iscale);
  dt_dev_pixelpipe_create_nodes(&pipe, &dev);
  dt_dev_pixelpipe_synch_all(&pipe, &dev);

  dt_dev_pixelpipe_get_dimensions(&pipe, &dev, pipe.iwidth, pipe.iheight,
                                  &pipe.processed_width,
                                  &pipe.processed_height);

  // edit session: the render cap is the RECORDED one -- the recipe's
  // prompt geometry is bound to the recorded encode dimensions, and the
  // current preference may have moved since (headless replay rule: never
  // re-read the render-size conf for a replayed encode)
  // render_size is the raw cap the launch snapshotted (pinned or not); the
  // MAX is the one convention every use point applies
  const int render_target = MAX(render_size, 1024);
  const double scale = fmin((double)render_target / (double)pipe.processed_width,
                            (double)render_target / (double)pipe.processed_height);
  double final_scale = fmin(scale, 1.0); // don't upscale
  int out_w = (int)(final_scale * pipe.processed_width);
  int out_h = (int)(final_scale * pipe.processed_height);
  if(pinned)
  {
    // the recipe travels in the XMP and is untrusted input: the encode
    // render never upscales, so recorded dims above the processed frame
    // are necessarily corrupt (headless replay rule)
    if(pin_w <= 0 || pin_h <= 0
       || pin_w > pipe.processed_width || pin_h > pipe.processed_height)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] edit: recorded encode dims %dx%d unusable for"
               " the processed frame %dx%d",
               pin_w, pin_h, pipe.processed_width, pipe.processed_height);
      dt_dev_pixelpipe_cleanup(&pipe);
      dt_mipmap_cache_release(&buf);
      dt_dev_cleanup(&dev);
      g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
      return NULL;
    }
    if(out_w != pin_w || out_h != pin_h)
    {
      // recorded dims from a .seg cache made under another cap: fall back
      // to the dim ratio, mirror of the headless replay (sub-pixel
      // sampling difference, accepted for that rare case); reject dims
      // the render cannot reach even so
      final_scale = fmin((double)pin_w / (double)pipe.processed_width,
                         (double)pin_h / (double)pipe.processed_height);
      if((int)(final_scale * pipe.processed_width) + 1 < pin_w
         || (int)(final_scale * pipe.processed_height) + 1 < pin_h)
      {
        dt_print(DT_DEBUG_AI,
                 "[object mask] edit: recorded dims %dx%d not reachable"
                 " from the processed frame", pin_w, pin_h);
        dt_dev_pixelpipe_cleanup(&pipe);
        dt_mipmap_cache_release(&buf);
        dt_dev_cleanup(&dev);
        g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
        return NULL;
      }
    }
    out_w = pin_w;
    out_h = pin_h;
  }

  // use distort hash from darkroom's live state (passed by caller)
  // instead of computing from the thread's dev, which may have
  // stale history (not yet flushed to database)
  if(dt_seg_disk_cache_load(d->seg, imgid, distort_hash))
  {
    int cw = 0, ch = 0;
    dt_seg_get_encoded_rgb(d->seg, &cw, &ch);
    // edit session: the cache is validated without dimensions, and a hit
    // at other dims than the recipe records would put the replayed
    // prompts on the wrong grid -- re-encode then (headless replay rule)
    if(!pinned || (cw == pin_w && ch == pin_h))
    {
      dt_dev_pixelpipe_cleanup(&pipe);
      dt_mipmap_cache_release(&buf);
      dt_dev_cleanup(&dev);
      d->encode_w = cw;
      d->encode_h = ch;
      g_atomic_int_set(&d->encode_state, ENCODE_READY);
      dt_seg_warmup_decoder(d->seg);
      return NULL;
    }
    dt_print(DT_DEBUG_AI,
             "[object mask] edit: cached encoding is %dx%d, recipe records"
             " %dx%d, re-encoding", cw, ch, pin_w, pin_h);
    dt_seg_reset_encoding(d->seg);
  }

  dt_print(DT_DEBUG_AI,
           "[object mask] rendering %dx%d (scale=%.3f) for encoding...",
           out_w, out_h, final_scale);

  dt_dev_pixelpipe_process_no_gamma(&pipe, &dev, 0, 0, out_w, out_h, final_scale);

  // backbuf is float RGBA after process_no_gamma, convert to uint8 RGB for SAM.
  // edit session: a render that landed on other dims than the recorded ones
  // would put the replayed prompts on the wrong grid -- fail instead
  // (headless replay's own guard)
  uint8_t *rgb = NULL;
  if(!pinned
     || (pipe.backbuf
         && pipe.backbuf_width == out_w && pipe.backbuf_height == out_h))
    rgb = _backbuf_to_rgb8(&pipe, out_w, out_h);

  dt_dev_pixelpipe_cleanup(&pipe);
  dt_mipmap_cache_release(&buf);
  dt_dev_cleanup(&dev);

  if(!rgb)
  {
    dt_print(DT_DEBUG_AI, "[object mask] failed to render image for encoding");
    g_atomic_int_set(&d->encode_state, ENCODE_ERROR);
    return NULL;
  }

  // store encoding dimensions for coordinate mapping
  d->encode_w = out_w;
  d->encode_h = out_h;

  // encode the image, falling back to CPU when acceleration fails; an edit
  // session pins the fallback reload to the recorded model, as the replay does
  const gboolean ok
    = _seg_encode_cpu_fallback(&d->seg, d->env, pinned ? pin_model : NULL,
                               rgb, out_w, out_h);
  if(!d->seg)
    d->model_loaded = FALSE;

  // dt_seg_encode_image keeps its own copy of rgb for edge refinement.
  // edit session: never clobber an existing per-image cache slot -- it may
  // hold a plain session's working resolution (headless replay rule)
  if(ok && (!pinned || !dt_seg_disk_cache_exists(imgid)))
    dt_seg_disk_cache_save(d->seg, imgid, distort_hash,
                           rgb, out_w, out_h);
  g_free(rgb);

  // signal ready so the user can start placing points; warmup continues
  // on this thread; _launch_decode joins the thread on the first click to
  // avoid a race with warmup on the shared segmentation context
  g_atomic_int_set(&d->encode_state, ok ? ENCODE_READY : ENCODE_ERROR);

  // warm up decoder with real encoder embeddings so the first user click
  // doesn't pay ORT's lazy-init + arena-sizing cost on the main thread
  if(ok)
    dt_seg_warmup_decoder(d->seg);

  return NULL;
}

// keep only the connected component containing the seed pixel
// (seed_x, seed_y), if the seed is outside any foreground region,
// keep the largest component instead, operates in-place: non-selected
// foreground pixels are zeroed
static void _keep_seed_component(float *mask,
                                 const int w,
                                 const int h,
                                 const float threshold,
                                 const int seed_x,
                                 const int seed_y)
{
  const int npix = w * h;
  int16_t *labels = g_try_malloc0((size_t)npix * sizeof(int16_t));
  if(!labels)
    return;
  int *stack = g_try_malloc((size_t)npix * sizeof(int));
  if(!stack)
  {
    g_free(labels);
    return;
  }

  int16_t n_labels = 0;
  int16_t best_label = 0;
  int best_area = 0;
  int16_t seed_label = 0;

  for(int i = 0; i < npix; i++)
  {
    if(mask[i] <= threshold || labels[i] != 0)
      continue;
    if(n_labels >= INT16_MAX)
      break;

    n_labels++;
    const int16_t label = n_labels;
    int area = 0;
    int sp = 0;
    stack[sp++] = i;
    labels[i] = label;

    while(sp > 0)
    {
      const int p = stack[--sp];
      area++;
      const int px = p % w;
      const int py = p / w;

      if(px == seed_x && py == seed_y)
        seed_label = label;

      // 4-connected neighbors
      if(py > 0 && labels[p - w] == 0 && mask[p - w] > threshold)
      {
        labels[p - w] = label;
        stack[sp++] = p - w;
      }
      if(py < h - 1 && labels[p + w] == 0 && mask[p + w] > threshold)
      {
        labels[p + w] = label;
        stack[sp++] = p + w;
      }
      if(px > 0 && labels[p - 1] == 0 && mask[p - 1] > threshold)
      {
        labels[p - 1] = label;
        stack[sp++] = p - 1;
      }
      if(px < w - 1 && labels[p + 1] == 0 && mask[p + 1] > threshold)
      {
        labels[p + 1] = label;
        stack[sp++] = p + 1;
      }
    }

    if(area > best_area)
    {
      best_area = area;
      best_label = label;
    }
  }

  // prefer component containing the seed point; fall back to largest
  const int16_t keep = (seed_label > 0) ? seed_label : best_label;

  if(keep > 0)
  {
    for(int i = 0; i < npix; i++)
    {
      if(mask[i] > threshold && labels[i] != keep)
        mask[i] = 0.0f;
    }
  }

  g_free(stack);
  g_free(labels);
}

static float _mask_iou(const float *const restrict a,
                       const float *const restrict b,
                       const size_t n,
                       const float threshold)
{
  size_t inter = 0, uni = 0;
  DT_OMP_FOR(reduction(+:inter, uni))
  for(size_t i = 0; i < n; i++)
  {
    const int A = a[i] > threshold;
    const int B = b[i] > threshold;
    inter += A & B;
    uni   += A | B;
  }
  return uni > 0 ? (float)inter / (float)uni : 0.0f;
}

// peak of the (exact-Euclidean) distance transform of mask>threshold,
// excluding pixels within min_separation of any positive prompt
static gboolean _find_peak_point(const float *const restrict mask,
                                 const size_t w,
                                 const size_t h,
                                 const float threshold,
                                 const dt_seg_point_t *const exclude,
                                 const int n_exclude,
                                 const float min_separation,
                                 dt_seg_point_t *const out)
{
  float *const restrict dist = dt_alloc_align_float(w * h);
  if(!dist) return FALSE;

  // exact-euclidean DT: dist[i] = distance to nearest pixel where mask<thr
  // require ~4 px interior depth — shallower peaks aren't informative
  const float min_depth = 4.0f;
  const float max_dist
    = dt_image_distance_transform(mask, dist, w, h,
                                  threshold, DT_DISTANCE_TRANSFORM_MASK);
  if(max_dist <= min_depth) { dt_free_align(dist); return FALSE; }

  // zero out pixels too close to existing positive prompts so the
  // subsequent argmax never picks them
  const float min_sep_sq = min_separation * min_separation;
  for(int k = 0; k < n_exclude; k++)
  {
    if(exclude[k].label != 1) continue;
    const float px = exclude[k].x;
    const float py = exclude[k].y;
    const int x0 = MAX(0, (int)(px - min_separation));
    const int x1 = MIN((int)w - 1, (int)(px + min_separation));
    const int y0 = MAX(0, (int)(py - min_separation));
    const int y1 = MIN((int)h - 1, (int)(py + min_separation));
    DT_OMP_FOR(collapse(2))
    for(int y = y0; y <= y1; y++)
      for(int x = x0; x <= x1; x++)
      {
        const float dx = (float)x - px;
        const float dy = (float)y - py;
        if(dx * dx + dy * dy < min_sep_sq) dist[(size_t)y * w + x] = 0.0f;
      }
  }

  // single-threaded combined max+argmax (exclusion may have lowered
  // the peak below max_dist, so we can't reuse that value here)
  size_t best_idx = (size_t)-1;
  float best = min_depth;
  for(size_t i = 0; i < w * h; i++)
    if(dist[i] > best) { best = dist[i]; best_idx = i; }
  dt_free_align(dist);
  if(best_idx == (size_t)-1) return FALSE;

  const size_t py = best_idx / w;
  const size_t px = best_idx % w;
  out->x = (float)px;
  out->y = (float)py;
  out->label = 1;
  return TRUE;
}

// tight bbox around mask>threshold, padded by `padding` (fraction of
// bbox extent); FALSE if mask is empty
static gboolean _compute_bbox(const float *const restrict mask,
                              const int w,
                              const int h,
                              const float threshold,
                              const float padding,
                              dt_seg_point_t *const tl,
                              dt_seg_point_t *const br)
{
  // single-threaded: cheap, and avoids OMP-reduction identity surprises
  int min_x = INT_MAX, min_y = INT_MAX, max_x = INT_MIN, max_y = INT_MIN;
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      if(mask[(size_t)y * w + x] > threshold)
      {
        if(x < min_x) min_x = x;
        if(y < min_y) min_y = y;
        if(x > max_x) max_x = x;
        if(y > max_y) max_y = y;
      }
    }
  }
  if(max_x == INT_MIN) return FALSE;

  const int pad_x = (int)((max_x - min_x) * padding) + 1;
  const int pad_y = (int)((max_y - min_y) * padding) + 1;
  tl->x = (float)CLAMP(min_x - pad_x, 0, w - 1);
  tl->y = (float)CLAMP(min_y - pad_y, 0, h - 1);
  tl->label = 2;
  br->x = (float)CLAMP(max_x + pad_x, 0, w - 1);
  br->y = (float)CLAMP(max_y + pad_y, 0, h - 1);
  br->label = 3;
  return TRUE;
}

// ---------------------------- interactive decode ----------------------------
//
// three parts: _launch_decode snapshots every input on the GUI thread into a
// self-contained job and starts the compute thread; _decode_thread_func
// computes from that job alone -- its shared state is d->seg / d->refine /
// d->refine_failed / d->env, on which it has exclusive rights while a decode
// runs, plus the atomic decode_state it sets as its last instruction;
// _decode_finish joins the thread on the GUI side and publishes the result
// into d->mask (or drains it)

struct _decode_job_t
{
  _object_data_t *d;         // owner; stays valid while a decode runs
                             // (_free_data defers to _deferred_cleanup)
  // -- inputs, snapshotted on the GUI thread; the compute never reads the
  // live dynbufs, the preview pipe geometry or any gui->* field --
  dt_seg_point_t *points;    // mapped to encode space, with pass headroom
  int n_prompt_points;
  int n_passes;
  int seed_x, seed_y;        // unclamped; clamped against the mask dims
  gboolean reset_prev_mask;  // snapshotted; see _launch_decode
  float threshold;
  gboolean do_crf;
  int crf_iter;
  float crf_sigma_color;
  float crf_w_bilateral;
  gboolean do_refine;
  float refine_margin;
  // -- private output of the compute --
  float *out_mask;
  int out_w, out_h;
};

static void _decode_job_free(_decode_job_t *job)
{
  if(!job) return;
  g_free(job->points);
  g_free(job->out_mask);
  g_free(job);
}

// the compute. runs from the job snapshot only; writes its result into the
// job, never into d->mask
static gpointer _decode_thread_func(gpointer data)
{
  _decode_job_t *job = data;
  _object_data_t *d = job->d;

  if(job->reset_prev_mask)
    dt_seg_reset_prev_mask(d->seg);

  dt_seg_point_t *points = job->points;
  int n_points = job->n_prompt_points;
  const float threshold = job->threshold;
  const gboolean supports_box = dt_seg_supports_box(d->seg);
  int mw = 0, mh = 0;
  float *mask = NULL;
  gboolean box_added = FALSE;

  for(int pass = 0; pass < job->n_passes; pass++)
  {
    float *new_mask = dt_seg_compute_mask(d->seg, points, n_points, &mw, &mh);
    if(!new_mask) break;

    if(mask && _mask_iou(mask, new_mask, (size_t)mw * mh, threshold) > 0.99f)
    {
      g_free(mask);
      mask = new_mask;
      dt_print(DT_DEBUG_AI,
               "[object mask] converged at pass %d/%d", pass + 1, job->n_passes);
      break;
    }
    g_free(mask);
    mask = new_mask;

    if(pass + 1 >= job->n_passes) break;

    gboolean any_added = FALSE;
    dt_seg_point_t peak;
    if(_find_peak_point(mask, mw, mh, threshold,
                        points, n_points, 8.0f, &peak))
    {
      points[n_points++] = peak;
      any_added = TRUE;
    }
    if(supports_box && !box_added)
    {
      dt_seg_point_t tl, br;
      if(_compute_bbox(mask, mw, mh, threshold, 0.05f, &tl, &br))
      {
        points[n_points++] = tl;
        points[n_points++] = br;
        box_added = TRUE;
        any_added = TRUE;
      }
    }
    if(!any_added) break;
  }

  if(mask)
  {
    // remove disconnected blobs: keep only the component at the seed point
    const int seed_x = CLAMP(job->seed_x, 0, mw - 1);
    const int seed_y = CLAMP(job->seed_y, 0, mh - 1);
    _keep_seed_component(mask, mw, mh, threshold, seed_x, seed_y);

    // optional DenseCRF edge refinement using the encoded RGB as guide
    if(job->do_crf)
    {
      int rgb_w = 0, rgb_h = 0;
      const uint8_t *rgb = dt_seg_get_encoded_rgb(d->seg, &rgb_w, &rgb_h);
      if(rgb && rgb_w == mw && rgb_h == mh)
      {
        const double t0 = dt_get_wtime();
        dt_dense_crf_binary(mask, rgb, mw, mh,
                            5.0f, job->crf_sigma_color,
                            3.0f, job->crf_w_bilateral, job->crf_iter);
        dt_print(DT_DEBUG_AI,
                 "[object mask] CRF refinement: %dx%d (%.2fs)",
                 mw, mh, dt_get_wtime() - t0);
      }
    }

    // optional CascadePSP contour refinement. the segmentation decoder emits
    // a fixed 256x256 mask, so on a large image one mask pixel spans dozens
    // of image pixels and no resampling can recover the contour. a dedicated
    // network re-derives it from the image, given the coarse mask as a hint.
    if(job->do_refine)
    {
      int rgb_w = 0, rgb_h = 0;
      const uint8_t *rgb = dt_seg_get_encoded_rgb(d->seg, &rgb_w, &rgb_h);
      if(rgb && rgb_w == mw && rgb_h == mh)
      {
        // lazy-load here on purpose: while a decode runs this code has
        // exclusive rights on d->refine / d->refine_failed
        if(!d->refine && !d->refine_failed)
        {
          d->refine = dt_refine_load(d->env);
          if(!d->refine)
          {
            d->refine_failed = TRUE;
            dt_print(DT_DEBUG_AI,
                     "[object mask] contour refinement unavailable, disabled"
                     " for this mask");
          }
        }

        if(d->refine)
        {
          dt_seg_point_t tl, br;
          if(_compute_bbox(mask, mw, mh, threshold, job->refine_margin,
                           &tl, &br))
          {
            const int rx = CLAMP((int)tl.x, 0, mw - 1);
            const int ry = CLAMP((int)tl.y, 0, mh - 1);
            const int rw = CLAMP((int)br.x - rx + 1, 1, mw - rx);
            const int rh = CLAMP((int)br.y - ry + 1, 1, mh - ry);
            const double t1 = dt_get_wtime();
            if(dt_refine_run(d->refine, rgb, mw, mh, mask, threshold,
                             rx, ry, rw, rh))
              dt_print(DT_DEBUG_AI,
                       "[object mask] contour refinement: %dx%d region (%.2fs)",
                       rw, rh, dt_get_wtime() - t1);
          }
        }
      }
    }
  }

  job->out_mask = mask;
  job->out_w = mw;
  job->out_h = mh;
  // last instruction: hand the result to the GUI side. the publication
  // machine in post_expose (woken by the 100 ms poll timer redraw) joins
  // the thread and publishes
  g_atomic_int_set(&job->d->decode_state, mask ? DECODE_READY : DECODE_ERROR);
  return NULL;
}

// GUI thread: snapshot every input into a job and start the compute thread.
// coalesces on its own: while a decode is in flight it only flags
// decode_pending -- the dynbuf holds the clicks and IS the queue
static void _launch_decode(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d || !d->seg || !dt_seg_is_encoded(d->seg))
    return;
  if(gui->guipoints_count <= 0)
    return;

  // frozen while an edit replay owns the segmentation context: the event
  // handlers refuse clicks first, this is the structural backstop -- a
  // decode launched here would run a second thread on the same ONNX
  // context as the replay
  if(_edit_frozen(d))
    return;

  // a decode is already in flight: the point is accumulated in the dynbuf
  // (which IS the queue); publication will relaunch with ALL points
  if(g_atomic_int_get(&d->decode_state) != DECODE_IDLE)
  {
    d->decode_pending = TRUE;
    return;
  }

  // wait for encode thread: warmup may still be running after ENCODE_READY
  if(d->encode_thread)
  {
    g_thread_join(d->encode_thread);
    d->encode_thread = NULL;
  }

  // non-modal busy indicator: a modal grab would swallow the very clicks
  // the coalescing is meant to accumulate
  if(!d->decode_busy_shown)
  {
    dt_control_busy_enter();
    d->decode_busy_shown = TRUE;
  }

  const float *gp = dt_masks_dynbuf_buffer(gui->guipoints);
  const float *gpp = dt_masks_dynbuf_buffer(gui->guipoints_payload);

  // points are stored in preview pipe pixel space, scale to encoding space
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
  const float sx = (wd > 0) ? (float)d->encode_w / wd : 1.0f;
  const float sy = (ht > 0) ? (float)d->encode_h / ht : 1.0f;

  _decode_job_t *job = g_malloc0(sizeof(_decode_job_t));
  job->d = d;
  job->n_prompt_points = gui->guipoints_count;
  // always FALSE from the interactive path today: has_selection is set
  // before the launch. snapshotted to preserve the historical behaviour bit
  // for bit -- do not attach first-click semantics to it
  job->reset_prev_mask = (gui->guipoints_count <= 1 && !d->has_selection);
  job->n_passes = _session_n_passes(d);

  // headroom: one peak point per pass + 2 box corners (SAM only)
  job->points
    = g_new(dt_seg_point_t, job->n_prompt_points + job->n_passes + 2);
  for(int i = 0; i < job->n_prompt_points; i++)
  {
    job->points[i].x = gp[i * 2 + 0] * sx;
    job->points[i].y = gp[i * 2 + 1] * sy;
    job->points[i].label = (int)gpp[i];
  }

  // seed point for the connected component filter: last positive point
  job->seed_x = -1;
  job->seed_y = -1;
  for(int i = gui->guipoints_count - 1; i >= 0; i--)
  {
    if((int)gpp[i] == 1)
    {
      job->seed_x = (int)(gp[i * 2 + 0] * sx);
      job->seed_y = (int)(gp[i * 2 + 1] * sy);
      break;
    }
  }

  // through the session accessors: an edit session decodes with the
  // recipe's recorded scalars, not the preferences of the day. a click
  // added after the replay must refine the recorded chain with the
  // parameters that built it -- the recipe format assumes they are
  // constant over the session, and the re-capture records them
  job->threshold = _session_threshold(d);
  job->do_crf = d->preview_refine;
  job->crf_iter = _session_crf_iter(d);
  job->crf_sigma_color = _session_crf_sigma_color(d);
  job->crf_w_bilateral = _session_crf_w_bilateral(d);
  job->do_refine = _session_do_refine(d) && !d->refine_failed;
  job->refine_margin = _session_refine_margin(d);

  d->decode_pending = FALSE;
  d->decode_launched_count = gui->guipoints_count;

  // provenance: scalars of the last launched decode. the mask a later
  // finalisation captures is the one THIS job computes; the conf and the
  // preview toggles may move before the capture, so the recipe must read
  // these snapshots, never the live state (threshold is recorded
  // per-boundary through the mark below)
  d->last_do_crf = job->do_crf;
  d->last_crf_iter = job->crf_iter;
  d->last_crf_sigma_color = job->crf_sigma_color;
  d->last_crf_w_bilateral = job->crf_w_bilateral;
  d->last_do_refine = job->do_refine;
  d->last_refine_margin = job->refine_margin;
  d->last_n_passes = job->n_passes;

  // provenance: this decode's boundary is the last point it covers
  _marks_resize(d, gui->guipoints_count);
  _decode_mark_t *mark = &g_array_index(d->decode_marks, _decode_mark_t,
                                        job->n_prompt_points - 1);
  mark->threshold = job->threshold;
  mark->launched = TRUE;

  d->decode_job = job;
  g_atomic_int_set(&d->decode_state, DECODE_RUNNING);
  // the poll timer (started with the encode) redraws every 100 ms; the
  // publication machine in post_expose detects completion
  if(!d->modifier_poll_id)
    d->modifier_poll_id = g_timeout_add(100, _modifier_poll, NULL);
  d->decode_thread = g_thread_new("ai-mask-decode", _decode_thread_func, job);
}

// run vectorization with current preview parameters, store result in scratchpad,
// called automatically after each decode and on scroll parameter changes
static void _update_preview(_object_data_t *d)
{
  _free_preview_forms(d);
  if(!d->mask || d->mask_w <= 0 || d->mask_h <= 0)
    return;

  // skip vectorization when path preview is disabled
  if(dt_conf_key_exists(CONF_OBJECT_PATH_PREVIEW_KEY)
     && !dt_conf_get_bool(CONF_OBJECT_PATH_PREVIEW_KEY))
    return;

  // ras2forms inherits potrace's convention: pixels < threshold are
  // "inside the form" (black ink on white paper). our AI mask uses the
  // opposite — high values = inside the object — so we invert both the
  // mask and the threshold here. result: the path traces the same
  // contour as the red overlay (mask > user_threshold)
  const size_t n = (size_t)d->mask_w * d->mask_h;
  float *inv_mask = g_try_malloc(n * sizeof(float));
  if(!inv_mask) return;

  for(size_t i = 0; i < n; i++)
    inv_mask[i] = 1.0f - d->mask[i];

  const float thresh = 1.0f - _session_threshold(d);
  d->preview_forms = ras2forms(inv_mask, d->mask_w, d->mask_h, NULL,
                               thresh,
                               d->preview_cleanup, (double)d->preview_smoothing,
                               0.3, &d->preview_signs);
  g_free(inv_mask);

  // apply feather to all path points
  const float feather = d->preview_feather;
  for(GList *fl = d->preview_forms; fl; fl = g_list_next(fl))
  {
    dt_masks_form_t *f = fl->data;
    for(GList *p = f->points; p; p = g_list_next(p))
    {
      dt_masks_point_path_t *pt = p->data;
      pt->border[0] = feather;
      pt->border[1] = feather;
    }
  }
}

// GUI thread: join a finished decode and either publish its result or drop
// it. dropping (publish=FALSE) is the drain used by _clear_selection, the
// invalidation branch and the stale-geometry check in the publication
// machine: without it, a posthumous publication would resurrect a mask the
// user just tore down. handles the coalesced relaunch. called from the
// expose handler like the encode machine -- keep this path light
static void _decode_finish(dt_masks_form_gui_t *gui, const gboolean publish)
{
  _object_data_t *d = _get_data(gui);
  if(!d) return;
  const int st = g_atomic_int_get(&d->decode_state);
  if(st != DECODE_READY && st != DECODE_ERROR) return;

  if(d->decode_thread)
  {
    g_thread_join(d->decode_thread);
    d->decode_thread = NULL;
  }
  _decode_job_t *job = d->decode_job;
  d->decode_job = NULL;

  // a failed decode never produced a boundary: retract the provenance mark
  // posted optimistically at launch, or a later capture would record a
  // decode the session never completed and the replay would diverge
  if(st == DECODE_ERROR && d->decode_marks
     && d->decode_launched_count > 0
     && d->decode_launched_count <= (int)d->decode_marks->len)
  {
    _decode_mark_t *mark = &g_array_index(d->decode_marks, _decode_mark_t,
                                          d->decode_launched_count - 1);
    mark->launched = FALSE;
    mark->threshold = 0.0f;
  }

  gboolean published = FALSE;
  if(publish && st == DECODE_READY && job && job->out_mask)
  {
    g_free(d->mask);
    d->mask = job->out_mask;
    d->mask_w = job->out_w;
    d->mask_h = job->out_h;
    job->out_mask = NULL;   // ownership moved
    published = TRUE;
  }
  _decode_job_free(job);
  g_atomic_int_set(&d->decode_state, DECODE_IDLE);

  if(published)
  {
    _update_preview(d);
    if(darktable.develop->proxy.masks.module)
      darktable.develop->proxy.masks.list_change(
        darktable.develop->proxy.masks.module);
  }

  // coalescing: clicks landed while the compute ran -- relaunch once with
  // ALL accumulated points ("last click wins" by accumulation). after an
  // error, only an explicit pending click relaunches (no error loop)
  gboolean relaunch = FALSE;
  if(publish)
  {
    if(st == DECODE_READY)
      relaunch = d->decode_pending
                 || gui->guipoints_count != d->decode_launched_count;
    else
      relaunch = d->decode_pending;
  }
  d->decode_pending = FALSE;

  if(relaunch && gui->guipoints_count > 0)
    _launch_decode(gui);        // keeps the busy indicator shown

  // default branch, not an else: the enter/leave pairing must hold locally,
  // whatever early-return the relaunch attempt above may have taken
  if(g_atomic_int_get(&d->decode_state) != DECODE_RUNNING
     && d->decode_busy_shown)
  {
    dt_control_busy_leave();
    d->decode_busy_shown = FALSE;
  }

  // a failed compute otherwise ends with a silent busy-indicator removal,
  // and the user is left clicking into the void
  if(publish && st == DECODE_ERROR
     && g_atomic_int_get(&d->decode_state) != DECODE_RUNNING)
    dt_control_log(_("object mask: computation failed"));

  if(published)
  {
    // dismiss "computing mask..." unless a coalesced relaunch took off
    if(g_atomic_int_get(&d->decode_state) != DECODE_RUNNING)
      dt_control_log_ack_all();
    dt_control_queue_redraw_center();
  }
}

// self-documenting wrappers for the two lifecycle transitions above
static void _decode_publish(dt_masks_form_gui_t *gui)
{
  _decode_finish(gui, TRUE);
}

static void _decode_drain(dt_masks_form_gui_t *gui)
{
  _decode_finish(gui, FALSE);
}

// write a mask as a 16-bit RGB PNG to the raster mask folder. the external
// raster mask module reads that depth with a 1/65535 normaliser and no
// thresholding, so the alpha survives intact; the three channels carry the
// same value and deflate collapses them
static gboolean _write_mask_png16(const char *path,
                                  const float *const restrict mask,
                                  const int w,
                                  const int h)
{
  FILE *f = g_fopen(path, "wb");
  if(!f) return FALSE;

  uint16_t *row = g_try_malloc((size_t)w * 3 * sizeof(uint16_t));
  if(!row)
  {
    fclose(f);
    return FALSE;
  }

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png)
  {
    g_free(row);
    fclose(f);
    return FALSE;
  }
  png_infop info = png_create_info_struct(png);
  if(!info || setjmp(png_jmpbuf(png)))
  {
    // libpng error path: free everything and remove the truncated file
    png_destroy_write_struct(&png, info ? &info : NULL);
    g_free(row);
    fclose(f);
    g_unlink(path);
    return FALSE;
  }

  png_init_io(png, f);
  png_set_IHDR(png, info, w, h, 16, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_set_compression_level(png, 6);
  png_write_info(png, info);
#if G_BYTE_ORDER == G_LITTLE_ENDIAN
  png_set_swap(png);   // PNG wants big-endian samples
#endif

  for(int y = 0; y < h; y++)
  {
    const float *const src = mask + (size_t)y * w;
    for(int x = 0; x < w; x++)
    {
      const uint16_t v = (uint16_t)lrintf(CLAMPF(src[x], 0.0f, 1.0f) * 65535.0f);
      row[x * 3 + 0] = v;
      row[x * 3 + 1] = v;
      row[x * 3 + 2] = v;
    }
    png_write_row(png, (png_bytep)row);
  }

  png_write_end(png, NULL);
  png_destroy_write_struct(&png, &info);
  g_free(row);
  if(fclose(f) != 0)
  {
    // the final flush can fail on a full disk: a truncated file must never
    // be reported as success, the caller would wire it into the history
    g_unlink(path);
    return FALSE;
  }
  return TRUE;
}

// cheap validation of an existing mask file: non-empty and carrying the
// PNG signature. a zero-byte or truncated file under a content-addressed
// name would otherwise be trusted forever ("exists = success") and the
// broken mask could never self-repair
// this validation must be at least as strict as the raster module's reader:
// a file it accepts but the reader rejects would loop forever between the
// pipe's failed read and a recompute that trusts the existing file. the
// signature catches wrong files, the trailing IEND chunk catches truncation
// -- the two ways a PNG breaks without a decoder
static gboolean _mask_png_valid(const char *path)
{
  GStatBuf st;
  // 8 signature bytes + IHDR (25) + IEND (12) is the bare minimum
  if(g_stat(path, &st) != 0 || st.st_size < 45)
    return FALSE;
  FILE *f = g_fopen(path, "rb");
  if(!f)
    return FALSE;
  guchar sig[8] = { 0 };
  guchar tail[12] = { 0 };
  const gboolean ok =
    fread(sig, 1, sizeof(sig), f) == sizeof(sig)
    && png_sig_cmp(sig, 0, sizeof(sig)) == 0
    && fseek(f, -(long)sizeof(tail), SEEK_END) == 0
    && fread(tail, 1, sizeof(tail), f) == sizeof(tail)
    && memcmp(tail + 4, "IEND", 4) == 0;
  fclose(f);
  return ok;
}

// content-addressed write: to a temp name unique to this writer, then an
// atomic move. an existing valid file under the same fingerprint was made
// from the same recipe -- it already is this content, keep it; an invalid
// one (truncated by a crash, zero bytes) is replaced. the unique temp name
// matters: the interactive finalisation and a headless replay of the same
// fingerprint, or two instances sharing the mask root, must never
// interleave writes into the same temp file. shared by the finalisation
// job and the headless recipe replay
static gboolean _write_mask_png16_atomic(const char *outpath,
                                         const float *const restrict mask,
                                         const int w,
                                         const int h)
{
  gchar *tmp = g_strdup_printf("%s.%d-%p.tmp", outpath,
                               (int)getpid(), (void *)g_thread_self());
  gboolean written = _write_mask_png16(tmp, mask, w, h);
  if(written)
  {
    if(_mask_png_valid(outpath))
      g_unlink(tmp);
    else
    {
      // no-op when the target does not exist; drops an invalid leftover
      g_unlink(outpath);
      if(g_rename(tmp, outpath) != 0)
      {
        g_unlink(tmp);
        // a concurrent writer of the same fingerprint may have won the
        // rename race: the file now under the target name is the same
        // content by construction, that still is a success
        written = g_file_test(outpath, G_FILE_TEST_EXISTS);
      }
    }
  }
  else
    g_unlink(tmp);
  g_free(tmp);
  return written;
}


/* Build a unique output path <def_path>/<image>_mask[_N].png. NULL on error.
 * May be called from a worker job: dt_control_log is thread-safe. */
static gchar *_build_mask_path(const dt_imgid_t imgid)
{
  if(!dt_is_valid_imgid(imgid)) return NULL;

  // the mask file is a local cache, not part of the library: the recipe
  // lives in the history/XMP, the file can always be regenerated with a new
  // finalisation
  gchar *root = dt_rasterfile_mask_root();
  if(g_mkdir_with_parents(root, 0755) != 0)
  {
    dt_print(DT_DEBUG_AI, "[object mask] cannot create folder: %s", root);
    dt_control_log(_("cannot create raster mask folder"));
    g_free(root);
    return NULL;
  }

  char imgpath[PATH_MAX] = { 0 };
  dt_image_full_path(imgid, imgpath, sizeof(imgpath), NULL);
  gchar *basename = g_path_get_basename(imgpath);
  char *dot = g_strrstr(basename, ".");
  if(dot) *dot = '\0';

  gchar *mask_name = g_strdup_printf("%s_mask.png", basename);
  gchar *outpath = g_build_filename(root, mask_name, NULL);
  g_free(mask_name);

  for(int seq = 1;
      g_file_test(outpath, G_FILE_TEST_EXISTS) && seq < 1000;
      seq++)
  {
    g_free(outpath);
    mask_name = g_strdup_printf("%s_mask_%d.png", basename, seq);
    outpath = g_build_filename(root, mask_name, NULL);
    g_free(mask_name);
  }
  g_free(basename);
  g_free(root);
  return outpath;
}


/* ------------------- native-resolution mask finalisation -------------------
 *
 * The interactive loop works on a render capped at 1536 px, where one pixel
 * spans ~4 native pixels: whatever the segmentation and its refinement
 * achieve, the contour position stays quantised to that grid. Finalisation
 * re-renders only the subject's bounding box at scale 1.0, re-derives the
 * alpha there with a guided filter -- whose local affine model in RGB is
 * exactly the inversion of the compositing equation I = a*F + (1-a)*B --
 * inside a narrow band around the contour, maps the result back to
 * full-frame *input* space (the space every mask form is normalised to, so
 * a crop added or changed later keeps working), and writes a 16-bit PNG for
 * the external raster mask module.
 *
 * Runs once, on a worker job, a few seconds. The interactive loop is not
 * touched. */

// the preferences of the matting stage. ONE spelling of each key, shared
// by the reader that resolves a session and by the panel widgets that
// write them: two literals that drift apart make a control that silently
// stops controlling anything
#define CONF_MATTE_ENABLED_KEY "plugins/ai/matting_enabled"
#define CONF_MATTE_OP_KEY "plugins/ai/matting_op"
#define CONF_MATTE_BAND_KEY "plugins/ai/matting_band_scale"

// the single user degree of freedom of the matting stage, clamped on the
// way in: the preference file is hand-editable and the recipe blob is
// untrusted, so no path may hand the render core a band scale it did not
// vet. the law's own constants stay compiled in, covered by the operator
// version
#define DT_MATTE_BAND_SCALE_MIN 0.5f
#define DT_MATTE_BAND_SCALE_MAX 2.0f
#define DT_MATTE_BAND_SCALE_DEFAULT 1.0f

// the matting stage of ONE finalisation, resolved once by the caller and
// carried in the request. the render core never reads the configuration
// itself: the interactive job resolves the stage from the preferences at
// launch, the recipe replay resolves it from what the recipe RECORDED, and
// a core that consulted dt_conf would make the replay reproduce today's
// preferences instead of the render it is replaying.
// declared here, ahead of the job, because the launch SNAPSHOTS it into the
// job like it snapshots the render cap -- see _finalize_job_t::matte
typedef struct _matte_session_t
{
  gboolean enabled;         // the stage runs; FALSE is the current chain
  float band_scale;         // [DT_MATTE_BAND_SCALE_MIN, ..._MAX]
  const dt_matte_op_t *op;  // table line, NULL when the stage is off
} _matte_session_t;

typedef struct _finalize_job_t
{
  dt_imgid_t imgid;
  int32_t history_end;
  float *hint;          // copy of the working-grid mask, owned by the job
  int hint_w, hint_h;
  int bx, by, bw, bh;   // subject bounding box on the hint grid, margin included
  float threshold;
  // target module for auto-wiring the produced mask, identified by name --
  // never by pointer, the job outlives any UI guarantee
  char target_op[32];
  int target_multi_priority;
  gboolean has_target;
  gboolean vectorize;   // TRUE: produce path forms instead of a raster file
  int cleanup;          // potrace turdsize, working-grid px^2 (scaled inside)
  float smoothing;      // potrace alphamax
  float feather;        // border applied to the resulting path points
  int render_target;    // effective cap of the native render, snapshotted
                        // at launch from the cap the ENCODING ran under --
                        // the very value the captured recipe records, so
                        // this render and the headless regeneration of that
                        // recipe land on the same pixels. re-reading the
                        // preference here would silently detach the two in
                        // an edit session (whose cap is the recorded one)
  _matte_session_t matte;  // matting stage, snapshotted at launch for the
                           // same reason as render_target: the recipe is
                           // captured on the GUI thread at launch, so the
                           // stage the recipe will record and the stage
                           // this render runs must be ONE resolution. the
                           // job body executes after an unbounded queue
                           // wait -- resolving there would let a
                           // preference toggled in between detach the
                           // render from its own provenance
  dt_hash_t distort_hash;  // distortion state at launch; revalidated at
                           // APPLY time on the GUI thread (same context)
  // provenance of the mask, captured on the GUI thread at launch. when
  // valid, outpath holds the content-addressed target file name derived
  // from its fingerprint; otherwise the job falls back to a sequential name
  dt_rf_recipe_t recipe;
  gboolean has_recipe;
  gchar *outpath;       // owned by the job
  // TRUE once _finalize_job_run entered: the serialisation token is reset
  // by the run itself, so the destroy callback needs to know whether a run
  // ever happened to release it for a job the queue discarded
  gboolean ran;
} _finalize_job_t;

// the matting block of a recipe, from the session the CALLER resolved for
// the render that recipe describes. the single writer of these four
// fields, and the single place the version is moved to EXT for them.
//
// an off stage writes NOTHING -- not the block, not the version bump. that
// is the rule the recipe header states for the whole extension area, and
// it is what makes "matting off produces the bytes earlier builds
// produced" a property of one branch here instead of a claim about the
// whole capture.
//
// FALSE means the session cannot be described. an id or a revision that
// did not fit its fixed-size field would be silently truncated into a
// DIFFERENT operator's name, and a fingerprint naming the wrong algorithm
// is the exact poison the table exists to prevent; the caller then records
// no recipe at all, which costs a content-addressed name and nothing else.
// the G_STATIC_ASSERTs standing next to each table line make this
// unreachable -- this is the runtime half of that guard, for the line
// somebody adds one day without them
static gboolean _recipe_set_matting(dt_rf_recipe_t *recipe,
                                    const _matte_session_t *matte)
{
  if(!matte || !matte->enabled || !matte->op)
    return TRUE;

  if(g_strlcpy(recipe->matting_id, matte->op->id,
               sizeof(recipe->matting_id)) >= sizeof(recipe->matting_id)
     || g_strlcpy(recipe->matting_version, matte->op->version,
                  sizeof(recipe->matting_version))
        >= sizeof(recipe->matting_version))
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[object mask] matting: operator '%s' v'%s' does not fit the"
             " recipe fields -- recording no provenance for this mask",
             matte->op->id, matte->op->version);
    memset(recipe->matting_id, 0, sizeof(recipe->matting_id));
    memset(recipe->matting_version, 0, sizeof(recipe->matting_version));
    return FALSE;
  }
  recipe->version = DT_RF_RECIPE_VERSION_EXT;
  recipe->matting_enabled = 1;
  recipe->matting_band = CLAMPF(matte->band_scale,
                                DT_MATTE_BAND_SCALE_MIN,
                                DT_MATTE_BAND_SCALE_MAX);
  return TRUE;
}

// GUI thread: record everything needed to regenerate the finalised mask
// file from the raw -- the provenance recipe stored with the raster
// module's params, so a library opened on another machine (or after the
// cache was purged) can recompute the file instead of showing a broken
// mask. returns FALSE when the session cannot be described (more clicks
// than the recipe holds); the caller then falls back to a plain file.
// `matte` is the matting stage of the render this recipe will name, as the
// caller ALREADY resolved it -- a nullable pointer, NULL meaning "this
// route runs no matting stage". it is a parameter and not a dt_conf read
// here on purpose: the recipe and the render it describes must be one
// resolution of the preference, and the render's own resolution happens at
// launch (see _finalize_job_t::matte)
static gboolean _capture_recipe(_object_data_t *d,
                                dt_masks_form_gui_t *gui,
                                const _matte_session_t *matte,
                                dt_rf_recipe_t *recipe)
{
  memset(recipe, 0, sizeof(*recipe));
  const int n = gui->guipoints_count;
  if(n <= 0 || n > DT_RF_RECIPE_MAX_POINTS)
    return FALSE;

  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);
  if(iwidth <= 0 || iheight <= 0)
    return FALSE;

  recipe->magic = DT_RF_RECIPE_MAGIC;
  recipe->version = DT_RF_RECIPE_VERSION;
  // the hash of the distortion state the ENCODING was made on -- the render
  // the prompt points were clicked against, not whatever the history holds
  // at capture time
  recipe->distort_hash = (int64_t)d->encoded_distort_hash;

  const char *seg_id = dt_seg_get_model_id(d->seg);
  if(seg_id)
  {
    g_strlcpy(recipe->seg_model, seg_id, sizeof(recipe->seg_model));
    const char *v = dt_ai_model_get_version(seg_id);
    if(v)
      g_strlcpy(recipe->seg_model_version, v,
                sizeof(recipe->seg_model_version));
  }

  // the mask being finalised was computed by the LAST launched decode:
  // record that job's scalars, not the live conf/preview state -- a toggle
  // flipped after the last decode never reached the mask. refine_failed
  // still absorbs a load failure: the decode then ran without refinement,
  // so ai_refine=0 is exact
  recipe->ai_refine = (d->last_do_refine && !d->refine_failed) ? 1 : 0;
  if(recipe->ai_refine)
  {
    char *refine_id = dt_ai_models_get_active_for_task("refine");
    if(refine_id)
    {
      g_strlcpy(recipe->refine_model, refine_id,
                sizeof(recipe->refine_model));
      const char *v = dt_ai_model_get_version(refine_id);
      if(v)
        g_strlcpy(recipe->refine_model_version, v,
                  sizeof(recipe->refine_model_version));
      g_free(refine_id);
    }
  }

  recipe->encode_w = d->encode_w;
  recipe->encode_h = d->encode_h;
  // the cap the ENCODING was really made under, whatever the session's
  // state is now: the truth is carried by the encode, never re-derived
  // from the session validity or the current preference. an edit session
  // encoded under the RECORDED cap, and it still holds that encoding after
  // a failure ended the edit bookkeeping -- reading the conf here would
  // then name a cap encode_w/h never came from, and the recipe's own
  // headless replay would fall into the dim-ratio path for no reason.
  // stored raw (see encoded_render_size): the value the capture writes is
  // the one a gen-1 recipe wrote, so a re-capture stays byte-identical
  recipe->render_size = d->encoded_render_size;
  recipe->refine_passes = CLAMP(d->last_n_passes, 1, 3);
  // the final threshold is genuinely a finalisation-time input (the native
  // pass and its bbox read it at the right click), so the live read is the
  // correct capture here -- unlike the per-decode scalars above. through
  // the session accessor: an edit session records its recorded override
  recipe->threshold = _session_threshold(d);
  recipe->crf_enabled = d->last_do_crf ? 1 : 0;
  recipe->crf_iterations = d->last_crf_iter;
  recipe->crf_sigma_color = d->last_crf_sigma_color;
  recipe->crf_w_bilateral = d->last_crf_w_bilateral;
  recipe->ai_refine_margin = d->last_refine_margin;
  recipe->cleanup = d->preview_cleanup;
  recipe->smoothing = d->preview_smoothing;
  recipe->feather = d->preview_feather;

  // points: preview-pipe pixel space -> normalized input space, the same
  // convention every stored mask form uses
  const float *gp = dt_masks_dynbuf_buffer(gui->guipoints);
  const float *gpp = dt_masks_dynbuf_buffer(gui->guipoints_payload);
  float *pts = g_new(float, (size_t)n * 2);
  memcpy(pts, gp, (size_t)n * 2 * sizeof(float));
  dt_dev_distort_backtransform(darktable.develop, pts, n);

  recipe->n_points = n;
  for(int i = 0; i < n; i++)
  {
    dt_rf_recipe_point_t *rp = &recipe->points[i];
    rp->x = pts[i * 2 + 0] / iwidth;
    rp->y = pts[i * 2 + 1] / iheight;
    rp->label = (int32_t)gpp[i];
    if(d->decode_marks && i < (int)d->decode_marks->len)
    {
      const _decode_mark_t *mark
        = &g_array_index(d->decode_marks, _decode_mark_t, i);
      rp->decode_after = mark->launched ? 1 : 0;
      rp->threshold = mark->launched ? mark->threshold : 0.0f;
    }
  }
  g_free(pts);
  // the matting stage LAST, over an otherwise finished version-1 blob: it
  // is the only field group whose presence moves the recipe to version
  // EXT, and writing it here makes "an off stage leaves a byte-identical
  // v1 recipe" readable in one place instead of inferred from a branch
  // three hundred lines up
  return _recipe_set_matting(recipe, matte);
}

// content hash of a group of paths for the ai trailer: the group's own
// serialized blob (the dt_masks_point_group_t sequence the masks history
// writes, trailer excluded) followed by each child's point blob in list
// order. THE single definition shared by the capture at creation time and
// the re-edit arbitration -- child states (union/difference), opacities,
// list order and membership all change it. children are resolved in
// `forms`; a child that does not resolve still contributes its group
// entry, so the hash cannot accidentally match a later state where the
// child resolves again
static int64_t _ai_trailer_group_hash(GList *forms,
                                      const dt_masks_form_t *grp)
{
  dt_hash_t hash = DT_INITHASH;
  for(GList *l = grp->points; l; l = g_list_next(l))
    hash = dt_hash(hash, l->data, sizeof(dt_masks_point_group_t));
  for(GList *l = grp->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    const dt_masks_form_t *child = dt_masks_get_from_id_ext(forms, pt->formid);
    // nested groups are deliberately skipped, not recursed into: a
    // non-path child forces the context fallback at edit time anyway.
    // must mirror the recalculation in outil/verifier_trailer.py
    if(!child || !child->functions || (child->type & DT_MASKS_GROUP))
      continue;
    const size_t point_size = child->functions->point_struct_size;
    for(GList *p = child->points; p; p = g_list_next(p))
      hash = dt_hash(hash, p->data, point_size);
  }
  return (int64_t)hash;
}

// identity of the image the session ran on, folded to the trailer's 32-bit
// field: basename without extension, sensor dimensions, capture datetime --
// the rasterfile fingerprint's ingredients. detects a history pasted onto
// another image. mirrored by outil/verifier_trailer.py
static uint32_t _ai_trailer_image_hash(const dt_image_t *img)
{
  gchar *base = g_path_get_basename(img->filename);
  char *dot = g_strrstr(base, ".");
  if(dot) *dot = '\0';
  dt_hash_t h = dt_hash(DT_INITHASH, base, strlen(base));
  const int32_t w = img->width, ht = img->height;
  h = dt_hash(h, &w, sizeof(w));
  h = dt_hash(h, &ht, sizeof(ht));
  const int64_t taken = img->exif_datetime_taken;
  h = dt_hash(h, &taken, sizeof(taken));
  g_free(base);
  return (uint32_t)h;
}

// GUI thread: stamp the ai provenance trailer on a freshly created group
// of paths -- the session recipe plus the content hash of the group as
// just built. `forms` is the list the group's children live in
// (dev->forms); must run after the group is fully assembled and before
// the masks history item is committed
static void _ai_trailer_stamp(GList *forms,
                              dt_masks_form_t *grp,
                              const dt_rf_recipe_t *recipe)
{
  memset(&grp->ai_trailer, 0, sizeof(grp->ai_trailer));
  grp->ai_trailer.magic = DT_MASKS_AI_TRAILER_MAGIC;
  grp->ai_trailer.version = DT_MASKS_AI_TRAILER_VERSION;
  grp->ai_trailer.flags = 0;  // no synthetic context on the creation routes
  grp->ai_trailer.image_hash
    = _ai_trailer_image_hash(&darktable.develop->image_storage);
  grp->ai_trailer.recipe = *recipe;
  grp->ai_trailer.paths_hash = _ai_trailer_group_hash(forms, grp);
}

// GUI thread: the content-addressed target path of a recipe, under the
// local mask root. must derive exactly what rasterfile.c's commit_params
// derives on resolution. NULL when the root folder cannot be created
static gchar *_recipe_outpath(const dt_rf_recipe_t *recipe)
{
  const dt_image_t *img = &darktable.develop->image_storage;
  gchar *base = g_path_get_basename(img->filename);
  char *dot = g_strrstr(base, ".");
  if(dot) *dot = '\0';
  gchar *fname = dt_rasterfile_recipe_filename(recipe, base,
                                               img->width, img->height,
                                               img->exif_datetime_taken);
  gchar *root = dt_rasterfile_mask_root();
  gchar *outpath = NULL;
  if(g_mkdir_with_parents(root, 0755) == 0)
    outpath = g_build_filename(root, fname, NULL);
  else
  {
    dt_print(DT_DEBUG_AI, "[object mask] cannot create folder: %s", root);
    dt_control_log(_("cannot create raster mask folder"));
  }
  g_free(root);
  g_free(fname);
  g_free(base);
  return outpath;
}

/* Payload handed from the worker job to the GUI thread once the file is
 * written: everything needed to wire the mask into the pipeline. */
typedef struct _finalize_apply_t
{
  dt_imgid_t imgid;
  gchar *outpath;
  char target_op[32];
  int target_multi_priority;
  gboolean has_target;
  gboolean vectorize;
  GList *forms;         // dt_masks_form_t*, points already input-normalized
  GList *signs;
  dt_rf_recipe_t recipe;  // provenance stored with the raster module params
  gboolean has_recipe;
  dt_hash_t distort_hash; // distortion state at launch; applying a result
                          // computed on a since-changed geometry would wire
                          // a misaligned mask
  // the detector row of a one-shot detection, NULL for the clicked
  // routes: it words the apply toast ("'subject' applied to exposure").
  // a row of the static table, so the pointer outlives any job
  const dt_detector_t *detector;
} _finalize_apply_t;

// destroy-notify of the apply idle: owns everything the payload carries.
// runs on source destruction, so the payload is freed exactly once whether
// the callback ran or the main loop went down before its tick
static void _finalize_apply_free(gpointer data)
{
  _finalize_apply_t *a = data;
  if(a->forms) g_list_free_full(a->forms, (GDestroyNotify)dt_masks_free_form);
  g_list_free(a->signs);
  g_free(a->outpath);
  g_free(a);
}

// GUI thread: activate the external raster mask module on the produced file
// and connect the target module's blending to it. this is exactly what the
// user would do by hand in the raster-mask combo (blend_gui.c), automated.
// frees nothing: the payload belongs to _finalize_apply_free
static gboolean _finalize_apply_idle(gpointer data)
{
  _finalize_apply_t *a = data;
  dt_develop_t *dev = darktable.develop;

  // leaving the darkroom does not reset image_storage.id, and it tears
  // down dev->iop and the form lists this idle works on -- check the
  // actual view, not just the image
  if(!dev || dev->image_storage.id != a->imgid
     || dt_view_get_current() != DT_VIEW_DARKROOM
     || !dev->form_gui)
  {
    if(a->vectorize)
      dt_control_log(_("image changed, precise paths discarded"));
    else
      dt_control_log(_("precise raster mask saved (image changed, not applied)"));
    return G_SOURCE_REMOVE;
  }

  // same develop, same GUI thread as the launch-time capture: this compares
  // like with like, and covers the whole job lifetime (queue wait included)
  if(_compute_distort_hash(dev) != a->distort_hash)
  {
    dt_control_log(_("image geometry changed while the precise mask was"
                     " computed, result discarded"));
    return G_SOURCE_REMOVE;
  }

  if(a->vectorize)
  {
    // wrap the native-resolution paths in a group and attach it to the
    // target module -- mirrors _register_vectorized_forms and the classic
    // right-click path, with coordinates already input-normalized by the job
    if(!a->forms)
    {
      dt_control_log(_("no mask extracted from AI segmentation"));
      return G_SOURCE_REMOVE;
    }

    // forms were created on the worker thread; their ids are atomic but
    // unicity against this dev's forms is only checkable here, on the GUI
    // thread, right before insertion. siblings must be checked too: they
    // are not in dev->forms yet, and the trailer's content hash resolves
    // children BY id -- a duplicate id would make it cover the wrong form
    for(GList *l = a->forms; l; l = g_list_next(l))
    {
      dt_masks_form_t *f = l->data;
      gboolean clash = TRUE;
      while(clash)
      {
        clash = dt_masks_get_from_id(dev, f->formid) != NULL;
        for(GList *k = a->forms; !clash && k != l; k = g_list_next(k))
          clash = ((dt_masks_form_t *)k->data)->formid == f->formid;
        if(clash) f->formid++;
      }
    }

    const char *group_prefix = _("ai object group");
    const char *path_prefix = _("ai object");
    guint grp_nb = 0, path_nb = 0;
    for(GList *l = dev->forms; l; l = g_list_next(l))
    {
      const dt_masks_form_t *f = l->data;
      if(strncmp(f->name, group_prefix, strlen(group_prefix)) == 0) grp_nb++;
      if(strncmp(f->name, path_prefix, strlen(path_prefix)) == 0) path_nb++;
    }
    grp_nb++;
    path_nb++;
    for(GList *l = a->forms; l; l = g_list_next(l))
    {
      dt_masks_form_t *f = l->data;
      snprintf(f->name, sizeof(f->name), "%s #%d", path_prefix, (int)path_nb++);
    }

    dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
    snprintf(grp->name, sizeof(grp->name), "%s #%d", group_prefix, (int)grp_nb);

    for(GList *l = a->forms; l; l = g_list_next(l))
      dev->forms = g_list_append(dev->forms, l->data);

    GList *sg = a->signs;
    for(GList *l = a->forms; l; l = g_list_next(l), sg = sg ? g_list_next(sg) : NULL)
    {
      const int sign = sg ? GPOINTER_TO_INT(sg->data) : '+';
      dt_masks_point_group_t *grpt = dt_masks_group_add_form(grp, l->data);
      if(grpt && sign == '-')
        grpt->state = (grpt->state & ~DT_MASKS_STATE_UNION) | DT_MASKS_STATE_DIFFERENCE;
    }
    dev->forms = g_list_append(dev->forms, grp);

    // stamp the provenance trailer on the fully assembled group: the
    // recorded session clicks travel with it, a later ai edit can reopen
    // the group from its exact prompts (the hash detects manual
    // retouching in between). must precede the history item below
    if(a->has_recipe)
      _ai_trailer_stamp(dev->forms, grp, &a->recipe);

    // attach to the target module's blend mask group, as the classic path does
    dt_iop_module_t *target = NULL;
    if(a->has_target)
      for(GList *l = dev->iop; l; l = g_list_next(l))
      {
        dt_iop_module_t *m = l->data;
        if(!strcmp(m->op, a->target_op)
           && m->multi_priority == a->target_multi_priority)
        {
          target = m;
          break;
        }
      }

    if(target && target->blend_params)
    {
      dt_masks_form_t *mod_grp
        = dt_masks_get_from_id(dev, target->blend_params->mask_id);
      if(!mod_grp)
      {
        mod_grp = dt_masks_create(DT_MASKS_GROUP);
        gchar *label = dt_history_item_get_name(target);
        snprintf(mod_grp->name, sizeof(mod_grp->name), _("group '%s'"), label);
        g_free(label);
        dev->forms = g_list_append(dev->forms, mod_grp);
        target->blend_params->mask_id = mod_grp->formid;
      }
      dt_masks_point_group_t *grpt = dt_masks_group_add_form(mod_grp, grp);
      if(grpt)
        grpt->opacity = dt_conf_get_float("plugins/darkroom/masks/opacity");
      // additive: never tear down a conditional blend already configured
      target->blend_params->mask_mode
        |= DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK;
      dt_dev_add_masks_history_item(dev, target, TRUE);
      if(target->gui_data) dt_iop_gui_update(target);
      dt_control_log(_("precise paths applied to %s"), target->name());
      // entering edit mode clears any form_gui in creation -- if the user
      // started a new mask session while the job ran, leave their session
      // alone, the group is attached and committed either way
      if(!dev->form_gui->creation && !dev->form_visible)
      {
        dt_masks_set_edit_mode(target, DT_MASKS_EDIT_FULL);
        dt_masks_iop_update(target);
      }
    }
    else
    {
      dt_dev_add_masks_history_item(dev, NULL, TRUE);
      dt_control_log(_("precise paths created"));
    }

    g_list_free(a->forms);   // cells only: ownership moved to dev->forms
    a->forms = NULL;
    dt_dev_reprocess_all(dev);
    dt_control_queue_redraw_center();
    return G_SOURCE_REMOVE;
  }

  // ---- the precise mask becomes a real mask: a raster shape ----
  // a member of the target's group, combinable with drawn shapes, with
  // per-member opacity and operator. the rasterfile channel of the old
  // wiring stays untouched for existing edits and hand-picked files;
  // only the NEW gesture lands here
  dt_masks_form_t *rform = dt_masks_create(DT_MASKS_RASTER);
  dt_masks_point_raster_t *pt = rform
    ? calloc(1, sizeof(dt_masks_point_raster_t))
    : NULL;
  if(!pt)
  {
    dt_masks_free_form(rform);
    dt_control_log(_("precise raster mask saved (out of memory)"));
    return G_SOURCE_REMOVE;
  }
  pt->magic = DT_MASKS_RASTER_POINT_MAGIC;
  pt->version = DT_MASKS_RASTER_POINT_VERSION;
  if(a->has_recipe)
    pt->recipe = a->recipe;  // content-addressed reference
  else
  {
    // sequential output file: a leaf under the local mask root
    gchar *leaf = g_path_get_basename(a->outpath);
    g_strlcpy(pt->file, leaf, sizeof(pt->file));
    g_free(leaf);
  }
  rform->points = g_list_append(rform->points, pt);

  // ---- sink side: the module the mask was created from ----
  dt_iop_module_t *target = NULL;
  if(a->has_target)
    for(GList *l = dev->iop; l; l = g_list_next(l))
    {
      dt_iop_module_t *m = l->data;
      if(!strcmp(m->op, a->target_op)
         && m->multi_priority == a->target_multi_priority)
      {
        target = m;
        break;
      }
    }

  // a target that somehow lost its blend params cannot wear a mask
  // group (_group_from_module reads through them): file the shape
  // unattached then, exactly as with no target at all
  if(target && !target->blend_params) target = NULL;

  // registers the form (unique name "precise mask #n"), appends it to
  // dev->forms, creates or joins the module's blend group with the armed
  // or default operator and the conf opacity, and commits the masks
  // history -- the one type-agnostic door every created shape goes
  // through. gui NULL: the target's gui is updated explicitly below
  dt_masks_gui_form_save_creation(dev, target, rform, NULL);

  if(target && target->blend_params)
  {
    // additive, the whole point: a drawn mask already on the module is
    // KEPT and the raster shape joins it in the group. never
    // DEVELOP_MASK_RASTER -- that is the exclusive channel of the
    // rasterfile module, and exclusivity is what this gesture ends
    target->blend_params->mask_mode
      |= DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK;
    dt_dev_add_masks_history_item(dev, target, TRUE);
    if(target->gui_data) dt_iop_gui_update(target);
    if(a->detector)
      dt_control_log(_("'%s' applied to %s"),
                     _(a->detector->label), target->name());
    else
      dt_control_log(_("precise mask applied to %s"), target->name());
    // entering edit mode clears any form_gui in creation -- if the user
    // started a new mask session while the job ran, leave their session
    // alone, the shape is attached and committed either way
    if(!dev->form_gui->creation && !dev->form_visible)
    {
      dt_masks_set_edit_mode(target, DT_MASKS_EDIT_FULL);
      dt_masks_iop_update(target);
    }
  }
  else if(a->detector)
    dt_control_log(_("'%s' mask created"), _(a->detector->label));
  else
    dt_control_log(_("precise mask created"));

  dt_dev_masks_list_change(dev);
  dt_dev_reprocess_all(dev);
  dt_control_queue_redraw_center();
  return G_SOURCE_REMOVE;
}

// one finalisation at a time; the job resets this when it completes
static volatile gint _finalize_running = 0;

// TRUE while that job is in flight. the proactive missing-file recompute
// of iop/rasterfile.c reads it as a second gate: the edit-session gate
// falls the moment the session data is freed, which is the moment the
// finalisation job STARTS -- for the seconds it then spends rendering and
// inferring, the mask file is still legitimately absent and an ungated
// gui_update would schedule a headless recompute of the OLD recipe
// against it. covers plain finalisations too, which have no session gate
gboolean dt_object_mask_finalize_running(void)
{
  return g_atomic_int_get(&_finalize_running) != 0;
}

// runs on every outcome the job system takes charge of -- finished,
// cancelled, replaced, discarded. a job the queue discarded before running
// never reached the reset at the end of _finalize_job_run, and the token
// it claimed at launch would then pin dt_object_mask_finalize_running for
// the rest of the session: no further finalisation, no headless replay
// (both take the same token), and the proactive recompute gated forever.
// same shape as _recompute_job_destroy and its `ran` field
static void _finalize_job_destroy(void *p)
{
  _finalize_job_t *j = p;
  if(!j) return;
  if(!j->ran)
    g_atomic_int_set(&_finalize_running, 0);
  g_free(j->hint);
  g_free(j->outpath);
  g_free(j);
}

// bilinear sample of a single-channel plane, pixel-centre convention
static inline float _sample_plane(const float *const restrict src,
                                  const int sw,
                                  const int sh,
                                  const float fx,
                                  const float fy)
{
  const float x = CLAMP(fx, 0.0f, (float)(sw - 1));
  const float y = CLAMP(fy, 0.0f, (float)(sh - 1));
  const int x0 = (int)x, y0 = (int)y;
  const int x1 = MIN(x0 + 1, sw - 1), y1 = MIN(y0 + 1, sh - 1);
  const float ax = x - (float)x0, ay = y - (float)y0;
  return src[(size_t)y0 * sw + x0] * (1.0f - ax) * (1.0f - ay)
       + src[(size_t)y0 * sw + x1] * ax * (1.0f - ay)
       + src[(size_t)y1 * sw + x0] * (1.0f - ax) * ay
       + src[(size_t)y1 * sw + x1] * ax * ay;
}

// the step hook of the two long AI mask jobs. both compute paths poll a
// keep_going between their expensive steps and nowhere else, which makes
// this the one thing that runs regularly for the whole job without the
// compute having to report anything -- so it carries the feedback as well
// as the cancellation:
//  - the toast: a dt_control_log expires after DT_CTL_LOG_TIMEOUT (5 s)
//    while these jobs run for tens of seconds, and control.c dedups a
//    repeated message, so re-emitting it here is what keeps the darkroom
//    saying that something IS computing. it is thread-safe;
//  - the progress bar of the background jobs module: created by
//    dt_control_job_add_progress, it would otherwise sit at 0 for the
//    whole job, which is indistinguishable from a hung one.
// the bar carries no fraction on purpose. these jobs poll between steps of
// wildly unequal cost (a tiled network pass dwarfs a render), so any
// fraction derived from the poll count reads as stalled on the long step
// and as almost-done a third of the way in -- worse than no number. what
// the job does know is WHICH step it is on, and dt_control_job_set_progress
// with a negative value keeps the bar in its indeterminate state
static gboolean _job_step(dt_job_t *job, const char *msg)
{
  // the cancellation test comes FIRST and returns without emitting: a
  // cancelled job must not leave its "computing..." toast alive for
  // another five seconds, nor nudge a bar nobody will finish. also bails
  // out when the whole job system goes down -- a multi-minute compute
  // must not hold the application's exit hostage
  if(!dt_control_running()
     || dt_control_job_get_state(job) == DT_JOB_STATE_CANCELLED)
    return FALSE;

  if(msg)
  {
    // the step name goes to the progress entry (which lives as long as the
    // job) and the same string is kept alive as a toast: a dt_control_log
    // expires after five seconds, and control.c deduplicates a repeat
    dt_control_job_set_progress_message(job, "%s", msg);
    dt_control_log("%s", msg);
  }
  return TRUE;
}

// the interactive finalisation. the message is the very string its launch
// toast used, taken from the job's own params: the user reads one message
// that simply stays alive, not two alternating ones
static gboolean _finalize_keep_going(void *p)
{
  dt_job_t *job = p;
  const _finalize_job_t *j = job ? dt_control_job_get_params(job) : NULL;
  return _job_step(job, (j && j->vectorize)
                        ? _("computing precise paths...")
                        : _("computing precise raster mask..."));
}

// the stage is off unless an operator is actually there to run it. an
// unavailable operator DISABLES the stage, it never selects another line:
// two different algorithms under one recipe id is the one failure this
// whole table exists to prevent
static _matte_session_t _matte_session_off(void)
{
  _matte_session_t s = { .enabled = FALSE,
                         .band_scale = DT_MATTE_BAND_SCALE_DEFAULT,
                         .op = NULL };
  return s;
}

// what the user asked for, for a render that is being made now. WHICH
// operator is a preference naming a table id -- not a compiled-in choice
// and not a combo box over an enum: the table is the single authority on
// what exists, so an id no line carries (the default `gf-adaptive`, in a
// build that does not register it yet) leaves the stage inert with the
// preference on, which is precisely the state this step has to be able to
// prove harmless. looked up by id, like the replay does, so both paths
// meet the same "no such operator" answer through the same door -- and a
// preference naming a line that was removed disables the stage instead of
// falling back to another one
static _matte_session_t _matte_session_from_conf(void)
{
  _matte_session_t s = _matte_session_off();
  if(!dt_conf_get_bool(CONF_MATTE_ENABLED_KEY))
    return s;

  gchar *id = dt_conf_get_string(CONF_MATTE_OP_KEY);
  const dt_matte_op_t *const op = dt_matte_find(id);
  if(!op)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] matting: no operator '%s' in this build,"
             " running the plain band stage", id ? id : "");
    g_free(id);
    return s;
  }
  g_free(id);
  s.op = op;
  s.enabled = TRUE;
  s.band_scale = CLAMPF((float)dt_conf_get_float(CONF_MATTE_BAND_KEY),
                        DT_MATTE_BAND_SCALE_MIN, DT_MATTE_BAND_SCALE_MAX);
  return s;
}

// the matting stage a given session runs, the last of the _session_*
// accessors -- and the only one that could not sit with the others up at
// the top of the file, because it needs the table lookup and the
// preference keys, both of which the finalisation block introduces.
//
// same rule as _session_threshold and its siblings: inside an edit session
// the value is the one the REPLAYED RECIPE recorded, outside it the
// preference. that rule is what closes the gap the previous step could
// only warn about -- reopening a mask made with a matting stage used to
// re-finalise it with whatever the preferences said today, so a session
// that changed one click silently changed the algorithm too.
//
// the seeded stage is used verbatim, WITHOUT re-resolving it against the
// preference: edit_begin resolved it from the recipe (and refused the
// session outright when this build could not), so second-guessing it here
// would reintroduce exactly the drift the seeding removes
static _matte_session_t _session_matte(const _object_data_t *d)
{
  if(d && d->edit_valid && d->edit_matte_seeded)
  {
    _matte_session_t s = _matte_session_off();
    if(d->edit_matte_enabled && d->edit_matte_op)
    {
      s.enabled = TRUE;
      s.op = d->edit_matte_op;
      s.band_scale = CLAMPF(d->edit_matte_band,
                            DT_MATTE_BAND_SCALE_MIN, DT_MATTE_BAND_SCALE_MAX);
    }
    return s;
  }
  return _matte_session_from_conf();
}

// the operator the preference NAMES, whether or not the stage is switched
// on. the two matting widgets need exactly that and _matte_session_from_conf
// cannot answer it: that one returns an off session as soon as the enable
// preference is off, so asking it would hide the very toggle that switches
// the stage on. same door as everything else -- dt_matte_find by id -- so a
// preference naming a line this build does not carry answers NULL here too
static const dt_matte_op_t *_matte_conf_op(void)
{
  gchar *id = dt_conf_get_string(CONF_MATTE_OP_KEY);
  const dt_matte_op_t *const op = dt_matte_find(id);
  g_free(id);
  return op;
}

// the operator a recipe's matting block names, resolved against THIS
// build's table. NULL means "this build cannot reproduce that stage", and
// it deliberately covers three different blobs:
//  - an id no line carries: another build's operator, or a private one;
//  - a line whose algorithm revision differs from the recorded one -- the
//    same name over different numbers, the single thing the version field
//    exists to catch, since the recipe is hashed verbatim to name a
//    content-addressed file and two renders may never share one name;
//  - a blob whose fixed-size strings are not NUL-terminated. the recipe
//    travels in module params, an XMP sidecar and styles, so its bytes are
//    read as DATA: handing dt_matte_find() a char array that runs off the
//    end of the struct is how untrusted provenance becomes a read fault.
//  - a blob announcing the stage at version 1. the matting fields live in
//    the extension block, and the header's contract is that any nonzero
//    field there makes version EXT mandatory: a version-1 blob claiming a
//    stage was written by nobody, so it is a forgery or a corruption, and
//    honouring it would let a recipe every version-1-only build replays
//    WITHOUT the stage be replayed WITH it here -- two renders, one
//    fingerprint. the belt every other reader of the block already wears
//    (dt_rf_recipe_valid demands EXT for the promptless kinds), worn here
//    too, where the last reader of the block was missing it.
//
// ONE lookup, shared by the replay gate and by the session the replay then
// runs. a gate that admitted a recipe the reconstruction could not honour
// would replay WITHOUT the recorded stage under a fingerprint promising it
static const dt_matte_op_t *_matte_op_recorded(const dt_rf_recipe_t *recipe)
{
  if(!recipe || !recipe->matting_enabled)
    return NULL;
  if(recipe->version != DT_RF_RECIPE_VERSION_EXT)
    return NULL;
  if(recipe->matting_id[DT_RF_RECIPE_MATTING_ID_LEN - 1] != '\0'
     || recipe->matting_version[DT_RF_RECIPE_MATTING_VERSION_LEN - 1] != '\0')
    return NULL;
  const dt_matte_op_t *const op = dt_matte_find(recipe->matting_id);
  if(!op || g_strcmp0(op->version, recipe->matting_version))
    return NULL;
  return op;
}

// the queryable mirror of the matting half of the replay gate, for the UX
// surfaces outside this file: TRUE when this build could reproduce whatever
// matting stage the recipe records -- trivially TRUE for the recipes that
// record none, which is every recipe a build with the stage off ever wrote.
//
// exported because the refusal was, until now, honest ONLY headless. four
// surfaces answered a model-gap verdict for a cause that is not a model:
// the recompute button ("no AI model available"), the masks panel row
// ("file missing, recomputing" -- a promise the replay then refuses and
// pins as a deterministic failure), its context menu entry, and the edit
// session, which opened on a recipe it could never re-finalise. no install
// and no activation moves a build capability, so every one of them has to
// ask THIS question instead, and all of them ask it through this one door
gboolean dt_object_recipe_matting_reproducible(const dt_rf_recipe_t *recipe)
{
  if(!recipe || !recipe->matting_enabled)
    return TRUE;
  return _matte_op_recorded(recipe) != NULL;
}

// what a recipe RECORDED, for a render that is being reproduced. built
// from the RECIPE and from nothing else: the operator is the table line
// its id names, the band scale is the number it stored. a preference
// toggled since -- or a machine that never had the stage on at all --
// therefore cannot change what a replay computes, which is the whole
// contract of a content-addressed cache. the replay gate in
// dt_object_recipe_compute has already refused every recipe this would
// answer an off session for, so the off branch here is the recipe that
// records no matting, never a silent downgrade of one that does
static _matte_session_t _matte_session_recorded(const dt_rf_recipe_t *recipe)
{
  _matte_session_t s = _matte_session_off();
  const dt_matte_op_t *const op = _matte_op_recorded(recipe);
  if(!op)
    return s;
  s.op = op;
  s.enabled = TRUE;
  // the recorded scale is vetted exactly like the preference is, and for
  // the same reason: params can be hand-edited and a sidecar can be
  // truncated. CLAMPed rather than refused, following the recorded
  // render_size right below (MAX(recipe->render_size, 1024) at the replay
  // request): a scalar outside its domain is a corrupt comfort setting,
  // not a different algorithm, and refusing it would leave a regenerable
  // mask broken. the mapping stays a FUNCTION of the recipe bytes -- the
  // same blob always yields the same clamped scale -- so the fingerprint
  // still names exactly one render. what may NOT be repaired is the
  // identity of the algorithm, and that is what _matte_op_recorded above
  // refuses outright. a NaN fails the first comparison of CLAMPF and
  // lands on the minimum, the conservative end
  s.band_scale = CLAMPF(recipe->matting_band,
                        DT_MATTE_BAND_SCALE_MIN, DT_MATTE_BAND_SCALE_MAX);
  return s;
}

// request of the native render core below, grouped so the two call sites
// name every field instead of threading fifteen positional arguments (an
// inverted bbox would compile without a sound)
typedef struct _finalize_render_req_t
{
  const float *hint;    // working-grid mask the finalisation works from
  int hint_w, hint_h;
  int bx, by, bw, bh;   // subject bounding box on the hint grid
  float threshold;
  int render_target;    // encode-render size cap the hint was made under
  gboolean interactive; // FALSE suppresses every dt_control_log
  _matte_session_t matte; // matting stage of this render, already resolved
} _finalize_render_req_t;

// the shared render core of the native finalisation: from a working-grid
// hint (and its subject bbox) to the alpha plane of the full
// post-rawprepare frame. renders the bbox at scale 1.0 through an export
// pipe on the ALREADY LOADED dev, re-runs the refinement network tiled at
// native scale, re-derives sub-pixel coverage with a guided filter inside
// a band around the contour, and maps the result back through the
// geometry chain. shared by the interactive finalisation job and the
// headless recipe replay: req->interactive=FALSE suppresses every
// dt_control_log (the replay may run outside any GUI). req->render_target
// is the encode-render size cap the hint was made under, used to recover
// the exact hint->native scale factor; keep_going (nullable) is polled at
// the expensive steps. returns the (*out_pw x *out_ph) alpha plane, freed
// by the caller with g_free, or NULL on failure/cancellation
static float *_finalize_render_alpha(dt_develop_t *dev,
                                     const _finalize_render_req_t *req,
                                     gboolean (*keep_going)(void *),
                                     void *user,
                                     int *out_pw,
                                     int *out_ph)
{
  const float *const hint = req->hint;
  const int hint_w = req->hint_w, hint_h = req->hint_h;
  const int bx = req->bx, by = req->by, bw = req->bw, bh = req->bh;
  const float threshold = req->threshold;
  const int render_target = req->render_target;
  const gboolean interactive = req->interactive;
  const _matte_session_t matte = req->matte;
  gboolean ok = FALSE;
  float *hint_soft = NULL, *hint_bin = NULL, *alpha_gf = NULL;
  float *alpha_full = NULL, *grid = NULL;
  // the matting operator's plane and the trimap view it may ask for. both
  // stay NULL for every session whose operator carries no weight, which is
  // every session of a build that ships only the witness line
  float *alpha_op = NULL, *trimap = NULL;
  double *sat = NULL;
  gboolean pipe_ready = FALSE;
  int rx = 0, ry = 0, gw = 0, gh = 0;

  // pipe-input dimensions (sensor incl. borders) and the post-rawprepare
  // frame. the external raster mask module sits after rawprepare (iop order
  // 3.1 vs 1.0) and interprets its file in the post-rawprepare frame, so
  // that is the space the file must be written in; rawprepare's crop offset
  // is added back when entering the full forward transform, which includes
  // rawprepare's own distort_transform.
  const int iw = dev->image_storage.width;
  const int ih = dev->image_storage.height;
  const int pw = dev->image_storage.p_width > 0 ? dev->image_storage.p_width : iw;
  const int ph = dev->image_storage.p_height > 0 ? dev->image_storage.p_height : ih;
  *out_pw = pw;
  *out_ph = ph;
  // rawprepare's default sensor crop, from the image metadata. a hand-edited
  // rawprepare margin is not reflected here -- accepted limitation: the
  // raster file module reads its file against the same metadata geometry,
  // so both sides drift together for that (rare) case
  const int cropx = dev->image_storage.crop_x;
  const int cropy = dev->image_storage.crop_y;

  dt_dev_pixelpipe_t pipe;
  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(&buf, dev->image_storage.id, DT_MIPMAP_FULL,
                      DT_MIPMAP_BLOCKING, 'r');
  if(!buf.buf || !buf.width || !buf.height)
  {
    dt_print(DT_DEBUG_AI, "[object mask] finalise: cannot get the image buffer");
    if(interactive)
      dt_control_log(_("precise mask: cannot get the image buffer"));
    goto cleanup;
  }

  if(!dt_dev_pixelpipe_init_export(&pipe, iw, ih, IMAGEIO_RGB | IMAGEIO_INT8,
                                   FALSE))
  {
    dt_print(DT_DEBUG_AI, "[object mask] finalise: cannot init the render pipe");
    if(interactive)
      dt_control_log(_("precise mask: cannot init the render pipe"));
    goto cleanup;
  }
  pipe_ready = TRUE;

  dt_dev_pixelpipe_set_icc(&pipe, DT_COLORSPACE_SRGB, NULL,
                           DT_INTENT_PERCEPTUAL);
  dt_dev_pixelpipe_set_input(&pipe, dev, (float *)buf.buf,
                             buf.width, buf.height, buf.iscale);
  dt_dev_pixelpipe_create_nodes(&pipe, dev);
  dt_dev_pixelpipe_synch_all(&pipe, dev);
  dt_dev_pixelpipe_get_dimensions(&pipe, dev, pipe.iwidth, pipe.iheight,
                                  &pipe.processed_width,
                                  &pipe.processed_height);

  {
    // the encode render maps by pure scale (out = trunc(final_scale * proc)),
    // so the exact hint->native factor is 1/final_scale, isotropic. deriving
    // it from the truncated integer sizes would introduce an anisotropic
    // drift of several native pixels at the far corner. recompute final_scale
    // with the same formula as _encode_thread_func.
    const double e_scale
      = fmin((double)render_target / (double)pipe.processed_width,
             (double)render_target / (double)pipe.processed_height);
    const double final_scale = fmin(e_scale, 1.0);
    double fx = 1.0 / final_scale;
    double fy = fx;
    // the render-size setting may have changed since the mask was encoded
    // (or, on a replay, the recorded dims may come from a reused .seg
    // cache); fall back to the integer ratio when the grids do not match
    if((int)(final_scale * pipe.processed_width) != hint_w
       || (int)(final_scale * pipe.processed_height) != hint_h)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] finalise: hint grid %dx%d does not match "
               "current render scale, falling back to integer ratios",
               hint_w, hint_h);
      fx = (double)pipe.processed_width / (double)hint_w;
      fy = (double)pipe.processed_height / (double)hint_h;
    }

    // the hint carries several native pixels of position error (one working
    // pixel spans ~fx native ones), so both the correction band and the
    // filter radius must scale with the resolution ratio
    const int R = MAX(8, (int)ceil(2.5 * fx));
    const int w_gf = MAX(4, R / 2);

    rx = (int)floor(bx * fx) - R;
    ry = (int)floor(by * fy) - R;
    int rw = (int)ceil((bx + bw) * fx) + R - rx;
    int rh = (int)ceil((by + bh) * fy) + R - ry;
    rx = CLAMP(rx, 0, pipe.processed_width - 8);
    ry = CLAMP(ry, 0, pipe.processed_height - 8);
    rw = CLAMP(rw, 8, pipe.processed_width - rx);
    rh = CLAMP(rh, 8, pipe.processed_height - ry);

    dt_print(DT_DEBUG_AI,
             "[object mask] finalise: native region %dx%d at (%d,%d) of %dx%d"
             " (R=%d, w=%d)",
             rw, rh, rx, ry, pipe.processed_width, pipe.processed_height,
             R, w_gf);
    // the matting stage as the CALLER resolved it, printed where the band
    // geometry is: with the stage off (the only state this build can
    // reach) nothing below changes, and that silence is the point
    if(matte.enabled && matte.op)
      dt_print(DT_DEBUG_AI,
               "[object mask] finalise: matting stage %s v%s, band scale %.2f",
               matte.op->id, matte.op->version, (double)matte.band_scale);

    // only the region goes through the pipe: demosaic's modify_roi_in
    // restricts the sensor read to what the ROI needs
    dt_dev_pixelpipe_process_no_gamma(&pipe, dev, rx, ry, rw, rh, 1.0);

    if(keep_going && !keep_going(user)) goto cleanup;

    // for a non-display pipe these are the requested dimensions echoed back;
    // kept as the single source of truth for the buffer we read
    const float *const guide = (const float *)pipe.backbuf;
    gw = pipe.backbuf_width;
    gh = pipe.backbuf_height;
    if(!guide || gw < 8 || gh < 8)
    {
      dt_print(DT_DEBUG_AI, "[object mask] finalise: native render failed");
      if(interactive)
        dt_control_log(_("precise mask: native render failed"));
      goto cleanup;
    }

    const size_t npix = (size_t)gw * gh;
    hint_soft = dt_alloc_align_float(npix);
    hint_bin = dt_alloc_align_float(npix);
    alpha_gf = dt_alloc_align_float(npix);
    sat = g_try_malloc((size_t)(gw + 1) * (gh + 1) * sizeof(double));
    if(!hint_soft || !hint_bin || !alpha_gf || !sat)
    {
      dt_print(DT_DEBUG_AI, "[object mask] finalise: out of memory");
      if(interactive)
        dt_control_log(_("precise mask: out of memory"));
      goto cleanup;
    }

    // hint on the native grid, pixel-centre mapping. the soft values are kept:
    // away from the contour they carry genuine partial coverage (defocused
    // edges, veils) that a hard threshold would destroy. the binarised copy
    // drives the band logic and the filter, which want a clean step.
    // parallel over rows: every iteration writes its own two pixels and
    // reads nothing another one writes, so the plane does not depend on
    // the thread count -- no reduction, no ordering, same bytes
    DT_OMP_FOR()
    for(int y = 0; y < gh; y++)
    {
      const float hy = ((float)(ry + y) + 0.5f) / (float)fy - 0.5f;
      for(int x = 0; x < gw; x++)
      {
        const float hx = ((float)(rx + x) + 0.5f) / (float)fx - 0.5f;
        const float v = _sample_plane(hint, hint_w, hint_h, hx, hy);
        hint_soft[(size_t)y * gw + x] = CLAMPF(v, 0.0f, 1.0f);
        hint_bin[(size_t)y * gw + x] = (v > threshold) ? 1.0f : 0.0f;
      }
    }

    // ---- tiled network pass at native scale ----
    // the interactive loop's refinement worked at ~3 native px per network
    // sample; here the same network re-derives the contour at 1:1, one
    // inference per overlapping tile. this is what actually moves the
    // contour onto the true edge; the guided filter afterwards only shapes
    // sub-pixel coverage inside the band. degradation is graceful: without
    // a usable model (missing, CPU provider, OOM) the hint stays as it is.
    {
      dt_ai_environment_t *net_env = dt_ai_env_init(NULL);
      dt_refine_context_t *net = net_env ? dt_refine_load(net_env) : NULL;
      if(net)
      {
        uint8_t *rgb8 = g_try_malloc((size_t)gw * gh * 3);
        float *mask_net = dt_alloc_align_float(npix);
        if(rgb8 && mask_net)
        {
          DT_OMP_FOR()
          for(size_t k = 0; k < npix; k++)
          {
            for(int c = 0; c < 3; c++)
              rgb8[k * 3 + c]
                = (uint8_t)lrintf(CLAMPF(guide[k * 4 + c], 0.0f, 1.0f) * 255.0f);
            mask_net[k] = hint_soft[k];
          }

          const double t_net = dt_get_wtime();
          if(dt_refine_run_tiled(net, rgb8, gw, gh, mask_net, threshold,
                                 keep_going, user))
          {
            dt_print(DT_DEBUG_AI,
                     "[object mask] finalise: tiled network pass (%.1fs)",
                     dt_get_wtime() - t_net);
            // same shape as the loop that filled rgb8/mask_net above: an
            // element-wise copy-back, parallel and order-independent
            DT_OMP_FOR()
            for(size_t k = 0; k < npix; k++)
            {
              hint_soft[k] = mask_net[k];
              hint_bin[k] = (mask_net[k] > threshold) ? 1.0f : 0.0f;
            }
          }
        }
        g_free(rgb8);
        dt_free_align(mask_net);
      }
      else
        dt_print(DT_DEBUG_AI,
                 "[object mask] finalise: refine model unavailable, "
                 "keeping the interactive hint");
      if(net) dt_refine_free(net);
      if(net_env) dt_ai_env_destroy(net_env);
      if(keep_going && !keep_going(user)) goto cleanup;
    }

    // summed-area table of the binary hint, for the band weight below
    for(int x = 0; x <= gw; x++) sat[x] = 0.0;
    for(int y = 1; y <= gh; y++)
    {
      double rowsum = 0.0;
      sat[(size_t)y * (gw + 1)] = 0.0;
      for(int x = 1; x <= gw; x++)
      {
        rowsum += hint_bin[(size_t)(y - 1) * gw + (x - 1)];
        sat[(size_t)y * (gw + 1) + x]
          = sat[(size_t)(y - 1) * (gw + 1) + x] + rowsum;
      }
    }

    // guided filter: local affine model in RGB -- the inversion of
    // I = a*F + (1-a)*B that a segmentation confidence lacks. the effective
    // regularisation comes from guide_weight scaling the covariances; these
    // values measure as a genuine edge snap on real sRGB contrasts, not a
    // box blur. high-ISO noise in the guide does transfer into the alpha
    // inside the band; that is the accepted trade.
    //
    // routed through the matting table as its witness line `gf-band`
    // rather than called directly: the line IS this call, at these
    // arguments, so the table indirection is falsifiable -- if the plane
    // changes, the table is wrong, not the algorithm
    const dt_matte_stage_t stage = {
      .guide = guide,
      .hint_bin = hint_bin,
      .hint_soft = hint_soft,
      .radius = NULL,          // uniform band; the field comes with the law
      .width = gw, .height = gh,
      .r_base = R, .r_max = R,
      .w_gf = w_gf,
      .band_scale = matte.band_scale,
      .keep_going = keep_going,
      .user = user,
    };
    if(!dt_matte_run(&dt_matte_op_gf_band, &stage, alpha_gf))
    {
      dt_print(DT_DEBUG_AI, "[object mask] finalise: band filter failed");
      if(interactive)
        dt_control_log(_("precise mask: band filter failed"));
      goto cleanup;
    }

    if(keep_going && !keep_going(user)) goto cleanup;

    // ---- the matting operator of the session, when there is one.
    //
    // gated on the table's OWN datum, `wmatte`, and not on the operator's
    // identity: a line whose plane carries no weight contributes nothing
    // the composition would read, so running it would cost a second
    // full-region plane and a second filter pass to produce a value
    // discarded a few lines below. the witness line is exactly such a line
    // -- it is already what produced alpha_gf -- so a session naming
    // `gf-band` walks past this block and composes the two-term form, and
    // a build carrying no other line can reach no other outcome
    float wmatte = 0.0f;
    if(matte.enabled && matte.op && matte.op->wmatte > 0.0f)
    {
      const dt_matte_op_t *const op = matte.op;
      alpha_op = dt_alloc_align_float(npix);
      if(op->caps & DT_MATTE_NEEDS_TRIMAP) trimap = dt_alloc_align_float(npix);
      if(!alpha_op || ((op->caps & DT_MATTE_NEEDS_TRIMAP) && !trimap))
      {
        dt_print(DT_DEBUG_AI,
                 "[object mask] finalise: out of memory for the matting stage");
        if(interactive)
          dt_control_log(_("precise mask: out of memory for the matting stage"));
        goto cleanup;
      }

      if(trimap)
      {
        // the band weight, materialised once so the trimap can be a VIEW
        // of it rather than a second definition of the same band. read
        // from the very summed-area table the composition reads, at the
        // very radius it uses, so "unknown" below and "the composition
        // gives the operator some authority here" are the same predicate
        // by construction. the plane is temporary: the composition loop
        // re-reads the four SAT corners per pixel as it always has, which
        // is what keeps that loop byte-identical for the witness path
        float *wband_plane = dt_alloc_align_float(npix);
        if(!wband_plane)
        {
          dt_print(DT_DEBUG_AI,
                   "[object mask] finalise: out of memory for the band view");
          if(interactive)
            dt_control_log(_("precise mask: out of memory for the matting stage"));
          goto cleanup;
        }
        DT_OMP_FOR()
        for(int y = 0; y < gh; y++)
        {
          for(int x = 0; x < gw; x++)
          {
            const int x0 = MAX(x - R, 0), x1 = MIN(x + R + 1, gw);
            const int y0 = MAX(y - R, 0), y1 = MIN(y + R + 1, gh);
            const double area = (double)(x1 - x0) * (y1 - y0);
            const double inside
              = sat[(size_t)y1 * (gw + 1) + x1] - sat[(size_t)y0 * (gw + 1) + x1]
              - sat[(size_t)y1 * (gw + 1) + x0] + sat[(size_t)y0 * (gw + 1) + x0];
            const double f = inside / area;
            wband_plane[(size_t)y * gw + x] = (float)(2.0 * MIN(f, 1.0 - f));
          }
        }
        dt_matte_trimap_from_band(hint_bin, wband_plane, trimap, npix);
        dt_free_align(wband_plane);
      }

      const dt_matte_stage_t op_stage = {
        .guide = guide,
        .hint_bin = hint_bin,
        .hint_soft = hint_soft,
        .radius = NULL,          // uniform band; the field comes with the law
        .trimap = trimap,        // NULL unless the line asked for one
        .width = gw, .height = gh,
        .r_base = R, .r_max = R,
        .w_gf = w_gf,
        .band_scale = matte.band_scale,
        .keep_going = keep_going,
        .user = user,
      };
      const double t_op = dt_get_wtime();
      if(!dt_matte_run(op, &op_stage, alpha_op))
      {
        // NOT a graceful degradation, unlike the network pass above. that
        // one only sharpens a hint nobody recorded; this stage is written
        // into the provenance recipe, and the recipe is hashed verbatim to
        // name a content-addressed file. finishing the render without the
        // operator would put a DIFFERENT mask under a fingerprint that
        // promises this one -- the single failure the whole table exists
        // to prevent. so the finalisation fails and the mask stays as it
        // was, which a user can act on
        dt_print(DT_DEBUG_AI,
                 "[object mask] finalise: matting operator %s v%s failed",
                 op->id, op->version);
        // one FALSE covers two situations the stage contract deliberately
        // does not tell apart (matte.h): the operator failed, or the user
        // cancelled while it was polling keep_going. asking the predicate
        // ourselves recovers the distinction, and a cancellation the user
        // just requested is not a failure to toast at them. the same
        // recovery the headless replay already performs on the render
        // core's own FALSE (see the DT_OBJECT_RECIPE_RETRY branch of
        // dt_object_recipe_compute)
        if(interactive && !(keep_going && !keep_going(user)))
          dt_control_log(_("precise mask: the matting stage failed"));
        goto cleanup;
      }
      wmatte = op->wmatte;
      dt_print(DT_DEBUG_AI,
               "[object mask] finalise: matting stage %s ran in %.1fs"
               " (weight %.2f)",
               op->id, dt_get_wtime() - t_op, (double)wmatte);

      if(keep_going && !keep_going(user)) goto cleanup;
    }

    // composition with a continuous band weight: 1 on the binary edge,
    // fading to 0 with Chebyshev distance R. a hard band boundary would
    // truncate the filter's ramp and leave a visible alpha step on guides
    // with no local edge (plain sky).
    //
    // written in the generic three-term form of the stage,
    //   alpha = wband*(wmatte*alpha_op + (1-wmatte)*alpha_gf)
    //           + (1-wband)*hint_soft
    // with `wmatte` zero unless an operator above actually produced a
    // plane: there is then no second plane to read, alpha_op is NULL, the
    // inner term is never evaluated, and what runs is the two-term
    // composition this stage has always run -- the same two products and
    // the same sum, in the same order. that is the identity anchor:
    // matting off must render the very same bytes, not equivalent ones.
    // the two are moved in lockstep by the block above (a plane without
    // its weight, or a weight without its plane, is never reachable), so
    // the ternary's NULL test and compose_px's zero test agree
    DT_OMP_FOR()
    for(int y = 0; y < gh; y++)
    {
      for(int x = 0; x < gw; x++)
      {
        const int x0 = MAX(x - R, 0), x1 = MIN(x + R + 1, gw);
        const int y0 = MAX(y - R, 0), y1 = MIN(y + R + 1, gh);
        const double area = (double)(x1 - x0) * (y1 - y0);
        const double inside
          = sat[(size_t)y1 * (gw + 1) + x1] - sat[(size_t)y0 * (gw + 1) + x1]
          - sat[(size_t)y1 * (gw + 1) + x0] + sat[(size_t)y0 * (gw + 1) + x0];
        const double f = inside / area;
        const float wband = (float)(2.0 * MIN(f, 1.0 - f));
        const size_t k = (size_t)y * gw + x;
        const float refined = CLAMPF(alpha_gf[k], 0.0f, 1.0f);
        const float op_px = alpha_op ? CLAMPF(alpha_op[k], 0.0f, 1.0f) : refined;
        alpha_gf[k]
          = dt_matte_compose_px(hint_soft[k], refined, op_px, wband, wmatte);
      }
    }

    // map back to the full post-rawprepare frame through the geometry chain,
    // on a coarse grid of forward-transformed nodes. grid nodes are indexed
    // in the post-rawprepare frame (the file's space), shifted by
    // rawprepare's crop to enter the full DIR_ALL transform, which includes
    // rawprepare's own distort_transform. a crop module added or changed
    // later keeps working; pixels outside the *current* crop stay 0 in the
    // file, so relaxing an existing crop reveals a hard edge -- accepted
    const int G = 8;
    const int gnx = pw / G + 2, gny = ph / G + 2;
    grid = g_try_malloc((size_t)gnx * gny * 2 * sizeof(float));
    alpha_full = g_try_malloc0((size_t)pw * ph * sizeof(float));
    if(!grid || !alpha_full)
    {
      dt_print(DT_DEBUG_AI, "[object mask] finalise: out of memory");
      if(interactive)
        dt_control_log(_("precise mask: out of memory"));
      goto cleanup;
    }
    for(int y = 0; y < gny; y++)
      for(int x = 0; x < gnx; x++)
      {
        grid[((size_t)y * gnx + x) * 2 + 0] = (float)(x * G + cropx);
        grid[((size_t)y * gnx + x) * 2 + 1] = (float)(y * G + cropy);
      }
    dt_dev_distort_transform_plus(dev, &pipe, 0.0, DT_DEV_TRANSFORM_DIR_ALL,
                                  grid, (size_t)gnx * gny);

    for(int y = 0; y < ph; y++)
    {
      const int cy = y / G;
      const float wy = (float)(y - cy * G) / (float)G;
      for(int x = 0; x < pw; x++)
      {
        const int cx = x / G;
        const float wx = (float)(x - cx * G) / (float)G;
        const float *n00 = grid + ((size_t)cy * gnx + cx) * 2;
        const float *n01 = grid + ((size_t)cy * gnx + cx + 1) * 2;
        const float *n10 = grid + ((size_t)(cy + 1) * gnx + cx) * 2;
        const float *n11 = grid + ((size_t)(cy + 1) * gnx + cx + 1) * 2;
        const float px = n00[0] * (1.0f - wx) * (1.0f - wy)
                       + n01[0] * wx * (1.0f - wy)
                       + n10[0] * (1.0f - wx) * wy + n11[0] * wx * wy;
        const float py = n00[1] * (1.0f - wx) * (1.0f - wy)
                       + n01[1] * wx * (1.0f - wy)
                       + n10[1] * (1.0f - wx) * wy + n11[1] * wx * wy;
        const float lx = px - (float)rx;
        const float ly = py - (float)ry;
        if(lx > -1.0f && ly > -1.0f && lx < (float)gw && ly < (float)gh)
          alpha_full[(size_t)y * pw + x]
            = _sample_plane(alpha_gf, gw, gh, lx, ly);
      }
    }
  }

  ok = TRUE;

cleanup:
  g_free(grid);
  g_free(sat);
  dt_free_align(trimap);
  dt_free_align(alpha_op);
  dt_free_align(alpha_gf);
  dt_free_align(hint_bin);
  dt_free_align(hint_soft);
  if(pipe_ready) dt_dev_pixelpipe_cleanup(&pipe);
  dt_mipmap_cache_release(&buf);
  if(!ok)
  {
    g_free(alpha_full);
    alpha_full = NULL;
  }
  return alpha_full;
}

static int32_t _finalize_job_run(dt_job_t *job)
{
  _finalize_job_t *const j = dt_control_job_get_params(job);
  // from here the reset of the serialisation token is ours, on every exit
  j->ran = TRUE;
  gboolean ok = FALSE;
  float *alpha_full = NULL;
  gchar *outpath = NULL;
  int pw = 0, ph = 0;

  const double t_start = dt_get_wtime();

  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dt_dev_load_image(&dev, j->imgid);
  if(j->history_end > 0 && j->history_end > dev.history_end)
    dev.history_end = j->history_end;

  // geometry drift while the job runs is checked at APPLY time, on the GUI
  // thread, against the hash captured at launch -- both sides then compute
  // on the same develop. comparing the launch hash against this headless
  // dev's hash looked equivalent but is not: the two contexts disagree on
  // the hash even for an unchanged history (systematic false positive that
  // cancelled every finalisation), and the apply-time check covers the
  // whole job lifetime anyway, which this early check never did

  const _finalize_render_req_t req = {
    .hint = j->hint,
    .hint_w = j->hint_w,
    .hint_h = j->hint_h,
    .bx = j->bx, .by = j->by, .bw = j->bw, .bh = j->bh,
    .threshold = j->threshold,
    .render_target = j->render_target,
    .interactive = TRUE,
    // a render being MADE: the stage the LAUNCH resolved from the user's
    // configuration, snapshotted alongside the render cap and the recipe
    .matte = j->matte,
  };
  alpha_full = _finalize_render_alpha(&dev, &req,
                                      _finalize_keep_going, job, &pw, &ph);
  if(!alpha_full)
    goto cleanup;

  if(dt_control_job_get_state(job) == DT_JOB_STATE_CANCELLED) goto cleanup;

  if(j->vectorize)
  {
    // native-frame geometry, same derivation as the render core above
    const int iw = dev.image_storage.width;
    const int ih = dev.image_storage.height;
    const int cropx = dev.image_storage.crop_x;
    const int cropy = dev.image_storage.crop_y;

    // trace the native alpha instead of writing a raster file. tolerance and
    // cleanup are expressed in native pixels here: one traced pixel is one
    // sensor pixel, so a ~1 px tolerance keeps the drift invisible while
    // keeping the anchor count manageable
    float *inv = g_try_malloc((size_t)pw * ph * sizeof(float));
    if(!inv)
    {
      dt_control_log(_("precise mask: out of memory"));
      goto cleanup;
    }
    for(size_t k = 0; k < (size_t)pw * ph; k++)
      inv[k] = 1.0f - alpha_full[k];

    // cleanup is expressed in working-grid area; scale it by the real
    // surface ratio between the native frame and the working grid
    const double area_ratio
      = ((double)iw * ih) / MAX(1.0, (double)j->hint_w * j->hint_h);
    GList *signs = NULL;
    GList *forms = ras2forms(inv, pw, ph, NULL, 1.0f - j->threshold,
                             MAX(2, (int)(j->cleanup * area_ratio)),
                             (double)j->smoothing, 1.2, &signs);
    g_free(inv);

    // normalize points straight into input space: the native grid is the
    // post-rawprepare frame, so input coord = native coord + rawprepare crop
    float wd_, ht_, iwidth_, iheight_;
    (void)wd_;
    (void)ht_;
    iwidth_ = (float)iw;
    iheight_ = (float)ih;
    for(GList *l = forms; l; l = g_list_next(l))
    {
      dt_masks_form_t *f = l->data;
      for(GList *p = f->points; p; p = g_list_next(p))
      {
        dt_masks_point_path_t *pt = p->data;
        pt->corner[0] = (pt->corner[0] + cropx) / iwidth_;
        pt->corner[1] = (pt->corner[1] + cropy) / iheight_;
        pt->ctrl1[0] = (pt->ctrl1[0] + cropx) / iwidth_;
        pt->ctrl1[1] = (pt->ctrl1[1] + cropy) / iheight_;
        pt->ctrl2[0] = (pt->ctrl2[0] + cropx) / iwidth_;
        pt->ctrl2[1] = (pt->ctrl2[1] + cropy) / iheight_;
        // a precise contour needs no imposed falloff: below the legacy
        // floor, collapse the border to a sub-pixel sliver -- visually a
        // hard edge, structurally still a valid border for the path editor
        const float fb = (j->feather < 0.0005f) ? 0.00002f : j->feather;
        pt->border[0] = fb;
        pt->border[1] = fb;
      }
    }

    _finalize_apply_t *a = g_malloc0(sizeof(_finalize_apply_t));
    a->imgid = j->imgid;
    a->vectorize = TRUE;
    a->forms = forms;
    a->signs = signs;
    a->has_target = j->has_target;
    memcpy(a->target_op, j->target_op, sizeof(a->target_op));
    a->target_multi_priority = j->target_multi_priority;
    // the recipe becomes the produced group's provenance trailer, stamped
    // by the apply idle once the group exists
    a->has_recipe = j->has_recipe;
    if(j->has_recipe)
      a->recipe = j->recipe;
    a->distort_hash = j->distort_hash;
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, _finalize_apply_idle, a,
                    _finalize_apply_free);
    dt_print(DT_DEBUG_AI,
             "[object mask] precise paths: %d form(s) traced at %dx%d (%.1fs)",
             g_list_length(forms), pw, ph, dt_get_wtime() - t_start);
    ok = TRUE;
    goto cleanup;
  }

  gboolean written = FALSE;
  if(j->has_recipe && j->outpath)
  {
    // content-addressed target: atomic write, and an existing file under
    // the same fingerprint already is this content
    outpath = g_strdup(j->outpath);
    written = _write_mask_png16_atomic(outpath, alpha_full, pw, ph);
  }
  else
  {
    outpath = _build_mask_path(j->imgid);
    written = outpath && _write_mask_png16(outpath, alpha_full, pw, ph);
  }

  if(written)
  {
    size_t soft = 0;
    for(size_t k = 0; k < (size_t)pw * ph; k++)
      if(alpha_full[k] > 0.05f && alpha_full[k] < 0.95f) soft++;
    GStatBuf st;
    const double mo = (g_stat(outpath, &st) == 0) ? st.st_size / 1048576.0 : 0.0;
    dt_print(DT_DEBUG_AI,
             "[object mask] precise mask saved: %s (%dx%d, %.1f MB, "
             "%.3f%% soft, %.1fs)",
             outpath, pw, ph, mo, 100.0 * soft / ((double)pw * ph),
             dt_get_wtime() - t_start);
    dt_control_log(_("precise raster mask saved (%.1f MB, %.1fs)"),
                   mo, dt_get_wtime() - t_start);

    // hand over to the GUI thread to wire the mask into the pipeline
    _finalize_apply_t *a = g_malloc0(sizeof(_finalize_apply_t));
    a->imgid = j->imgid;
    a->outpath = g_strdup(outpath);
    a->has_target = j->has_target;
    memcpy(a->target_op, j->target_op, sizeof(a->target_op));
    a->target_multi_priority = j->target_multi_priority;
    a->has_recipe = j->has_recipe;
    if(j->has_recipe)
      a->recipe = j->recipe;
    a->distort_hash = j->distort_hash;
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, _finalize_apply_idle, a,
                    _finalize_apply_free);
    ok = TRUE;
  }
  else
    dt_control_log(_("failed to save the precise raster mask"));

cleanup:
  g_free(outpath);
  g_free(alpha_full);
  dt_dev_cleanup(&dev);
  g_atomic_int_set(&_finalize_running, 0);
  return ok ? 0 : 1;
}

// GUI thread: snapshot the working mask and hand it to a worker job
static gboolean _launch_native_finalize(_object_data_t *d,
                                        dt_masks_form_gui_t *gui,
                                        dt_iop_module_t *target,
                                        const gboolean vectorize)
{
  if(!g_atomic_int_compare_and_exchange(&_finalize_running, 0, 1))
  {
    dt_control_log(_("precise mask finalisation already running"));
    return FALSE;
  }

  // the job re-reads the history from the database; unflushed edits
  // (exposure, crop, ...) would silently be missing from the render
  dt_dev_write_history(darktable.develop);
  // distortion state the hint was made on; the job revalidates it after
  // loading, since the history can move while the job waits in the queue
  const dt_hash_t launch_distort_hash
    = _compute_distort_hash(darktable.develop);

  // session accessors: an edit session finalises at its recorded threshold
  // and its recorded margin, the values the recipe records
  const float thresh = _session_threshold(d);
  const float margin = _session_refine_margin(d);

  dt_seg_point_t tl, br;
  if(!_compute_bbox(d->mask, d->mask_w, d->mask_h, thresh, margin, &tl, &br))
  {
    dt_control_log(_("empty mask, nothing to finalise"));
    g_atomic_int_set(&_finalize_running, 0);
    return FALSE;
  }

  _finalize_job_t *j = g_malloc0(sizeof(_finalize_job_t));
  j->imgid = darktable.develop->image_storage.id;
  j->history_end = darktable.develop->history_end;
  j->hint_w = d->mask_w;
  j->hint_h = d->mask_h;
  j->hint = g_malloc((size_t)d->mask_w * d->mask_h * sizeof(float));
  memcpy(j->hint, d->mask, (size_t)d->mask_w * d->mask_h * sizeof(float));
  j->bx = CLAMP((int)tl.x, 0, d->mask_w - 1);
  j->by = CLAMP((int)tl.y, 0, d->mask_h - 1);
  j->bw = CLAMP((int)br.x - j->bx + 1, 1, d->mask_w - j->bx);
  j->bh = CLAMP((int)br.y - j->by + 1, 1, d->mask_h - j->by);
  j->threshold = thresh;
  j->vectorize = vectorize;
  j->cleanup = d->preview_cleanup;
  j->smoothing = d->preview_smoothing;
  j->feather = d->preview_feather;
  // the cap the encoding ran under, through the same MAX every use point
  // applies: identical to the recipe captured below, so a later headless
  // regeneration renders exactly what this job renders
  j->render_target = MAX(d->encoded_render_size, 1024);
  // the matting stage of THIS render, resolved on the GUI thread here and
  // not in the job body: the recipe below is captured at this same instant,
  // and the two must describe one stage.
  // through the session accessor, so a finalisation launched from an EDIT
  // session runs the stage that session was seeded with (the recorded one)
  // instead of the preference -- the same rule the threshold, the passes
  // and the render cap have always followed on this path
  j->matte = _session_matte(d);
  j->distort_hash = launch_distort_hash;

  // both finalisation routes capture the provenance recipe. the raster
  // route additionally derives the content-addressed file name from it
  // (an outpath failure then falls back to a plain sequential file, and
  // the recipe is dropped with it); the vector route stores the recipe
  // with the produced group of paths instead -- its capture is decoupled
  // from any output path on purpose, a recipe without a file is exactly
  // what the group trailer needs. sessions the recipe cannot hold (too
  // many clicks) proceed without provenance on either route
  // the stage resolved fourteen lines up, NOT a second reading of the
  // preference: the recipe names the render this job is about to make
  j->has_recipe = _capture_recipe(d, gui, &j->matte, &j->recipe);
  if(!vectorize && j->has_recipe)
  {
    j->outpath = _recipe_outpath(&j->recipe);
    if(!j->outpath)
      j->has_recipe = FALSE;
  }
  if(!j->has_recipe)
    memset(&j->recipe, 0, sizeof(j->recipe));

  if(target)
  {
    g_strlcpy(j->target_op, target->op, sizeof(j->target_op));
    j->target_multi_priority = target->multi_priority;
    j->has_target = TRUE;
  }

  dt_job_t *job = dt_control_job_create(_finalize_job_run,
                                        "precise mask finalisation");
  if(!job)
  {
    // the destroy releases the token itself (j->ran is FALSE): the one
    // path that frees these params, whether the job system calls it or we
    // do, is also the one path that gives the token back
    _finalize_job_destroy(j);
    return FALSE;
  }
  dt_control_job_set_params(job, j, _finalize_job_destroy);
  dt_control_job_add_progress(job, _("precise mask finalisation"), TRUE);
  // announce before queueing: a worker may pick the job up at once and
  // report a failure, and that report must not be overwritten by this
  dt_control_log(vectorize ? _("computing precise paths...")
                           : _("computing precise raster mask..."));
  dt_control_add_job(DT_JOB_QUEUE_USER_BG, job);
  return TRUE;
}

/* ------------------------ headless recipe replay ------------------------
 *
 * Regenerate the finalised mask PNG from its provenance recipe alone: the
 * same render at the recorded encode dimensions, the recorded segmentation
 * model pinned to its recorded version, the decodes replayed at the
 * recorded boundaries, the same native finalisation, the same
 * content-addressed file name. Runs on a worker job or inline in a CLI
 * context: this code itself calls no GTK and no dt_control_log -- dt_print
 * only -- but the replayed pipes run every module of the history (see the
 * self-reference note below). The replay never produces path forms; those
 * stay a GUI-session gesture.
 *
 * Documented limitations, all confined to the guide render and accepted as
 * ONNX-variance-class (T2) deviations:
 *  - self-reference: both replay pipes synch the full history, including
 *    the rasterfile instance whose file is the very one being regenerated.
 *    That instance resolves to a missing file, blend falls back to a
 *    zeroed mask, and the RGB guide through the target module may differ
 *    from the original run (where an earlier file could be present).
 *    rasterfile.c stays silent about the missing file when a recipe is
 *    present -- the recompute is the answer, not a toast.
 *  - divergent distortion history: when the current history no longer
 *    matches the recorded distort_hash the replay still proceeds (see the
 *    comment at the check below); the regenerated bytes then depend on the
 *    CURRENT history, which the fingerprint does not cover.
 *  - the recipe does not record whether the original run's native tiled
 *    refinement pass succeeded; the replay's own attempt may degrade
 *    differently (model pinning above only guarantees the same weights). */

// render the recorded encode frame through the already prepared export
// pipe and hand it back as the uint8 RGB the model consumers take. the
// sampling scale is recovered with the encode-thread formula from the
// recorded cap; when the recorded dims come from a .seg made under
// another cap the formula cannot land on them, fall back to the dim
// ratio (sub-pixel sampling difference, accepted for that rare case).
// shared by the clicked replay (which encodes the render) and the
// promptless replay (which runs the detector on it directly). NULL on
// failure, reported inside
static uint8_t *_replay_render_rgb8(dt_develop_t *dev,
                                    dt_dev_pixelpipe_t *pipe,
                                    const dt_rf_recipe_t *recipe,
                                    const int enc_w,
                                    const int enc_h)
{
  const int render_cap = MAX(recipe->render_size, 1024);
  const double e_scale
    = fmin((double)render_cap / (double)pipe->processed_width,
           (double)render_cap / (double)pipe->processed_height);
  double final_scale = fmin(e_scale, 1.0);
  if((int)(final_scale * pipe->processed_width) != enc_w
     || (int)(final_scale * pipe->processed_height) != enc_h)
  {
    final_scale = fmin((double)enc_w / (double)pipe->processed_width,
                       (double)enc_h / (double)pipe->processed_height);
    // bound the requested ROI to the scaled extent of the processed
    // frame: the ratio truncates, so allow the accepted one-pixel slack,
    // but reject a recipe whose dims the render cannot reach -- the
    // prompts would land on the wrong grid anyway
    if((int)(final_scale * pipe->processed_width) + 1 < enc_w
       || (int)(final_scale * pipe->processed_height) + 1 < enc_h)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: recorded dims %dx%d not reachable"
               " from the processed frame, rejecting", enc_w, enc_h);
      return NULL;
    }
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: recorded dims %dx%d do not match the "
             "recorded render cap, using the dim ratio", enc_w, enc_h);
  }

  dt_print(DT_DEBUG_AI,
           "[object mask] replay: rendering %dx%d for encoding...",
           enc_w, enc_h);
  // the return value signals "pipe altered mid-flight", not success --
  // the backbuf and its dimensions are the check that matters
  dt_dev_pixelpipe_process_no_gamma(pipe, dev, 0, 0, enc_w, enc_h,
                                    final_scale);
  if(!pipe->backbuf
     || pipe->backbuf_width != enc_w || pipe->backbuf_height != enc_h)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: render for encoding returned %dx%d,"
             " expected %dx%d",
             pipe->backbuf_width, pipe->backbuf_height, enc_w, enc_h);
    return NULL;
  }

  // backbuf is float RGBA, convert to uint8 RGB -- same as the encode
  // thread
  uint8_t *rgb = _backbuf_to_rgb8(pipe, enc_w, enc_h);
  if(!rgb)
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: render for encoding failed");
  return rgb;
}

dt_object_recipe_status_t
dt_object_recipe_compute(const dt_rf_recipe_t *recipe,
                         const dt_imgid_t imgid,
                         gboolean (*keep_going)(void *),
                         void *user)
{
  if(!dt_rf_recipe_valid(recipe) || !dt_is_valid_imgid(imgid))
    return DT_OBJECT_RECIPE_FAILED;

  // the two replay families part here: a clicked recipe replays its
  // recorded decode boundaries through the segmentation stack, a
  // promptless one replays a single detector inference. the detectors
  // table is the single authority mapping the recorded prompt kind to a
  // registry task and its post-processing; a pair it does not know is an
  // unusable recipe, not a "pick a default"
  const gboolean promptless = recipe->prompt_kind != DT_RF_PROMPT_POINTS;
  const dt_detector_t *detector = NULL;
  if(promptless)
  {
    detector = dt_detector_find(recipe->prompt_kind, recipe->class_bits);
    if(!detector)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: no detector for prompt kind %d /"
               " class bits %llx",
               recipe->prompt_kind,
               (unsigned long long)recipe->class_bits);
      return DT_OBJECT_RECIPE_FAILED;
    }
  }

  // the matting stage the recipe records must be reproducible BY THIS
  // BUILD: the operator is looked up by id and its algorithm revision must
  // match the recorded one too. replaying without the stage -- or with
  // another revision of it -- would write different bytes under a
  // fingerprint that promises these ones, so there is no fallback here on
  // purpose: an unreproducible stage FAILS, loudly, and the mask stays
  // missing until a build that carries the operator opens the library.
  // deterministic (like the boundary check below) and outside the
  // model-gap mirror: it is a capability of the build, not a model to
  // install, and no download can move it.
  // ONE lookup with the session the replay runs under below, so the
  // verdict here and the stage that then executes cannot disagree
  if(recipe->matting_enabled && !_matte_op_recorded(recipe))
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: recipe records a matting stage this"
             " build cannot reproduce (id '%.*s' v'%.*s') -- not replaying",
             DT_RF_RECIPE_MATTING_ID_LEN, recipe->matting_id,
             DT_RF_RECIPE_MATTING_VERSION_LEN, recipe->matting_version);
    return DT_OBJECT_RECIPE_FAILED;
  }

  // the recorded decode boundaries drive the clicked replay; the last one
  // produced the mask the finalisation worked from. a clicked recipe
  // without any boundary defines no refinement chain and cannot be
  // replayed. stays -1 for a promptless recipe: the decode loop below
  // then never iterates
  int last_decode = -1;
  if(!promptless)
  {
    for(int i = 0; i < recipe->n_points; i++)
      if(recipe->points[i].decode_after)
        last_decode = i;
    if(last_decode < 0)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: recipe records no decode boundary");
      return DT_OBJECT_RECIPE_FAILED;
    }
  }

  // pin the models to the recorded versions. the fingerprint names the
  // exact weights; replaying with anything else would write different
  // content under the same name and poison a shared store forever. hard
  // failure on any deviation: it is retryable once the right models are
  // installed, and a mask-to-0 in the meantime beats silent divergence.
  // the seg_model slot of a promptless recipe carries the recorded
  // DETECTOR -- the same pin applies unchanged, and the model gap
  // diagnostic below stays its exact mirror for both families
  const char *seg_ver = dt_ai_model_get_version(recipe->seg_model);
  if(g_strcmp0(seg_ver, recipe->seg_model_version) != 0)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: segmentation model '%s' is version '%s',"
             " the recipe records '%s' -- not replaying",
             recipe->seg_model, seg_ver ? seg_ver : "?",
             recipe->seg_model_version);
    return DT_OBJECT_RECIPE_FAILED;
  }
  if(recipe->ai_refine)
  {
    // dt_refine_load offers no per-id loading: the replay can only use the
    // ACTIVE refine model, so that one must be the recorded one
    char *refine_id = dt_ai_models_get_active_for_task("refine");
    const char *refine_ver
      = refine_id ? dt_ai_model_get_version(refine_id) : NULL;
    const gboolean pinned = refine_id
      && strcmp(refine_id, recipe->refine_model) == 0
      && g_strcmp0(refine_ver, recipe->refine_model_version) == 0;
    if(!pinned)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: active refine model '%s' (version"
               " '%s') does not match the recorded '%s' (version '%s') --"
               " not replaying",
               refine_id ? refine_id : "(none)",
               refine_ver ? refine_ver : "?",
               recipe->refine_model, recipe->refine_model_version);
      g_free(refine_id);
      return DT_OBJECT_RECIPE_FAILED;
    }
    g_free(refine_id);
  }

  dt_object_recipe_status_t status = DT_OBJECT_RECIPE_FAILED;
  gchar *outpath = NULL;
  float *enc_pts = NULL;
  uint8_t *rgb = NULL;
  float *hint = NULL;
  float *alpha_full = NULL;
  int hint_w = 0, hint_h = 0;
  dt_ai_environment_t *env = NULL;
  dt_seg_context_t *seg = NULL;
  dt_detect_context_t *det = NULL;
  dt_dev_pixelpipe_t pipe;
  dt_mipmap_buffer_t buf;
  gboolean pipe_ready = FALSE, buf_ready = FALSE;
  // minimal stand-in for the session data _decode_thread_func works on: the
  // compute only touches seg/refine/refine_failed/env, plus a final atomic
  // store to decode_state that nobody polls here
  _object_data_t od = { 0 };

  const double t_start = dt_get_wtime();

  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dt_dev_load_image(&dev, imgid);
  // no history_end override here: headless, the database is the source of
  // truth (the interactive threads only override it because the darkroom
  // can be ahead of the database)

  // the recipe describes the distortion state the prompts were clicked on.
  // if the history moved since, the replay still renders the CURRENT state
  // -- the closest available to what the recipe names -- and says so; a
  // stale mask is repaired by a manual re-finalisation. proceeding (where
  // the interactive job cancels) is deliberate: cancelling would leave a
  // permanent mask-to-0 on the common cross-machine case (crop added after
  // finalisation), and the input-space-normalised points make the replay
  // largely self-correcting. accepted deviation from the content-addressed
  // promise: the regenerated bytes then depend on the CURRENT history,
  // which the fingerprint does not cover -- divergence is confined to the
  // guide render (the mask lives in input space) and is treated as
  // ONNX-variance-class (T2), see the block comment above
  // NOTE: the recorded hash was computed on the GUI develop and this one on
  // a headless develop; the two contexts have been observed to disagree for
  // an unchanged history, so this trace can be a false positive. purely
  // informational (the replay proceeds either way) -- root cause of the
  // cross-context divergence still to be established
  const dt_hash_t cur_hash = _compute_distort_hash(&dev);
  if((dt_hash_t)recipe->distort_hash != cur_hash)
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: distortion state differs from capture"
             " (may be a cross-context artefact), proceeding");

  // reproduce the FIRST capture's state: the original encoding was made
  // before this very mask existed, but the loaded history contains the
  // rasterfile instance carrying our recipe -- during the replay's renders
  // it would resolve to the missing file and its consumers would render a
  // zeroed mask, making the regenerated bytes depend on which files happen
  // to exist. disable the instances that carry OUR recipe (verbatim
  // compare); other raster masks of the image keep their effect
  for(GList *l = dev.iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    if(!strcmp(m->op, "rasterfile") && m->enabled && m->params
       && memcmp(&((dt_iop_rasterfile_params_t *)m->params)->recipe,
                 recipe, sizeof(*recipe)) == 0)
      m->enabled = FALSE;
  }

  // content-addressed target under the LOCAL mask root, derived from the
  // loaded dev -- never darktable.develop. must mirror what _recipe_outpath
  // and rasterfile.c's commit_params resolve
  {
    const dt_image_t *img = &dev.image_storage;
    gchar *base = g_path_get_basename(img->filename);
    char *dot = g_strrstr(base, ".");
    if(dot) *dot = '\0';
    gchar *fname = dt_rasterfile_recipe_filename(recipe, base,
                                                 img->width, img->height,
                                                 img->exif_datetime_taken);
    gchar *root = dt_rasterfile_mask_root();
    if(g_mkdir_with_parents(root, 0755) == 0)
      outpath = g_build_filename(root, fname, NULL);
    else
      dt_print(DT_DEBUG_AI, "[object mask] replay: cannot create folder: %s",
               root);
    g_free(root);
    g_free(fname);
    g_free(base);
  }
  if(!outpath)
    goto cleanup;

  // an existing file under this fingerprint was produced from this very
  // recipe: it already is the requested content -- but only when it is an
  // actual PNG. a zero-byte or truncated leftover would otherwise be
  // trusted forever and the pipe would loop on an unreadable file
  if(g_file_test(outpath, G_FILE_TEST_EXISTS))
  {
    if(_mask_png_valid(outpath))
    {
      dt_print(DT_DEBUG_AI, "[object mask] replay: %s already exists",
               outpath);
      status = DT_OBJECT_RECIPE_OK;
      goto cleanup;
    }
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: %s exists but is not a valid PNG,"
             " regenerating", outpath);
    g_unlink(outpath);
  }

  // load the RECORDED model, not the currently active one: the replay
  // reproduces the original session. a missing model is a clean failure
  // -- the caller may retry once it is installed
  env = dt_ai_env_init(NULL);
  if(promptless)
  {
    det = env ? dt_detect_load(env, recipe->seg_model, detector->task)
              : NULL;
    if(!det)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: detector model '%s' unavailable",
               recipe->seg_model);
      goto cleanup;
    }
    // the input side shapes the whole resample geometry: a model
    // repackaged at another side would produce different bytes under the
    // same recorded (id, version), so the side is pinned like the
    // version is
    if(dt_detect_get_side(det) != recipe->detect_input)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: detector input side %d differs from"
               " the recorded %d -- not replaying",
               dt_detect_get_side(det), recipe->detect_input);
      goto cleanup;
    }
  }
  else
  {
    seg = env ? dt_seg_load(env, recipe->seg_model) : NULL;
    if(!seg)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: segmentation model '%s' unavailable",
               recipe->seg_model);
      goto cleanup;
    }
    od.env = env;
    od.seg = seg;
  }

  const int enc_w = recipe->encode_w;
  const int enc_h = recipe->encode_h;
  if(enc_w <= 0 || enc_h <= 0)
  {
    dt_print(DT_DEBUG_AI, "[object mask] replay: invalid encode dimensions");
    goto cleanup;
  }

  // export render pipe at full input dimensions, as _encode_thread_func
  // builds it. it serves both the encode render and the prompt mapping
  dt_mipmap_cache_get(&buf, imgid, DT_MIPMAP_FULL, DT_MIPMAP_BLOCKING, 'r');
  buf_ready = TRUE;
  if(!buf.buf || !buf.width || !buf.height)
  {
    dt_print(DT_DEBUG_AI, "[object mask] replay: cannot get the image buffer");
    goto cleanup;
  }
  const int iw = dev.image_storage.width;
  const int ih = dev.image_storage.height;
  if(!dt_dev_pixelpipe_init_export(&pipe, iw, ih, IMAGEIO_RGB | IMAGEIO_INT8,
                                   FALSE))
  {
    dt_print(DT_DEBUG_AI, "[object mask] replay: cannot init the render pipe");
    goto cleanup;
  }
  pipe_ready = TRUE;
  dt_dev_pixelpipe_set_icc(&pipe, DT_COLORSPACE_SRGB, NULL,
                           DT_INTENT_PERCEPTUAL);
  dt_dev_pixelpipe_set_input(&pipe, &dev, (float *)buf.buf,
                             buf.width, buf.height, buf.iscale);
  dt_dev_pixelpipe_create_nodes(&pipe, &dev);
  dt_dev_pixelpipe_synch_all(&pipe, &dev);
  dt_dev_pixelpipe_get_dimensions(&pipe, &dev, pipe.iwidth, pipe.iheight,
                                  &pipe.processed_width,
                                  &pipe.processed_height);
  if(pipe.processed_width <= 0 || pipe.processed_height <= 0)
  {
    dt_print(DT_DEBUG_AI, "[object mask] replay: empty processed dimensions");
    goto cleanup;
  }

  // the recipe travels in the XMP and is untrusted input: the encode
  // render never upscales, so recorded dims above the processed frame are
  // necessarily corrupt -- reject them before they size any allocation
  if(enc_w > pipe.processed_width || enc_h > pipe.processed_height)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: recorded encode dims %dx%d exceed the"
             " processed frame %dx%d, rejecting",
             enc_w, enc_h, pipe.processed_width, pipe.processed_height);
    goto cleanup;
  }

  // prompt points back to encode-render pixels, the exact inverse of the
  // capture convention (backtransform through the pipe, divided by its
  // input dims): scale by THIS pipe's input dims, forward through the full
  // distortion chain -- the same transform_plus/DIR_ALL the finalisation
  // grid uses -- then the plain processed->encode factor of _launch_decode
  // (which uses no pixel-centre offset either). the normalised coordinates
  // are scale-invariant, so the preview pipe of the capture and this export
  // pipe agree by the same convention every stored mask form relies on.
  // a promptless recipe has no points to map
  if(!promptless)
  {
    const int n = recipe->n_points;
    enc_pts = g_new(float, (size_t)n * 2);
    for(int k = 0; k < n; k++)
    {
      enc_pts[k * 2 + 0] = recipe->points[k].x * (float)pipe.iwidth;
      enc_pts[k * 2 + 1] = recipe->points[k].y * (float)pipe.iheight;
    }
    dt_dev_distort_transform_plus(&dev, &pipe, 0.0,
                                  DT_DEV_TRANSFORM_DIR_ALL, enc_pts, n);
    const float psx = (float)enc_w / (float)pipe.processed_width;
    const float psy = (float)enc_h / (float)pipe.processed_height;
    for(int k = 0; k < n; k++)
    {
      enc_pts[k * 2 + 0] *= psx;
      enc_pts[k * 2 + 1] *= psy;
    }
  }

  // both families render at EXACTLY the recorded dims -- never re-read
  // the render-size conf for this render, a changed preference would
  // shift the geometry the recipe describes
  if(promptless)
  {
    // no embedding cache to consult: the detector keeps no per-image
    // state, every replay renders and infers
    rgb = _replay_render_rgb8(&dev, &pipe, recipe, enc_w, enc_h);
    if(!rgb)
      goto cleanup;

    hint_w = enc_w;
    hint_h = enc_h;
    hint = g_try_malloc((size_t)enc_w * enc_h * sizeof(float));
    if(!hint)
      goto cleanup;
    if(!dt_detect_run(det, rgb, enc_w, enc_h, hint))
    {
      dt_print(DT_DEBUG_AI, "[object mask] replay: detection failed");
      goto cleanup;
    }
    g_free(rgb);
    rgb = NULL;
  }
  else
  {
    // encoder embeddings: the disk cache first, keyed exactly like the
    // interactive session (imgid + current distortion state + model). the
    // cache is validated without dimensions, so a hit at other dims than
    // the recipe records would put the prompts on the wrong grid --
    // re-encode then
    gboolean encoded = FALSE;
    if(dt_seg_disk_cache_load(seg, imgid, cur_hash))
    {
      int cw = 0, ch = 0;
      dt_seg_get_encoded_rgb(seg, &cw, &ch);
      if(cw == enc_w && ch == enc_h)
        encoded = TRUE;
      else
      {
        dt_print(DT_DEBUG_AI,
                 "[object mask] replay: cached encoding is %dx%d, recipe "
                 "records %dx%d, re-encoding", cw, ch, enc_w, enc_h);
        dt_seg_reset_encoding(seg);
      }
    }

    if(!encoded)
    {
      rgb = _replay_render_rgb8(&dev, &pipe, recipe, enc_w, enc_h);
      if(!rgb)
        goto cleanup;

      // same CPU fallback as the interactive encode thread, pinned to the
      // recorded model
      encoded = _seg_encode_cpu_fallback(&seg, env, recipe->seg_model,
                                         rgb, enc_w, enc_h);
      od.seg = seg;
      // populate the disk cache for repeated replays -- but never
      // clobber a file the interactive session wrote: a single slot
      // exists per image, and overwriting it with the recipe dims would
      // silently degrade the session's working resolution on its next
      // cache hit
      if(encoded && !dt_seg_disk_cache_exists(imgid))
        dt_seg_disk_cache_save(seg, imgid, cur_hash, rgb, enc_w, enc_h);
      g_free(rgb);
      rgb = NULL;
      if(!encoded)
      {
        dt_print(DT_DEBUG_AI, "[object mask] replay: encoding failed");
        goto cleanup;
      }
    }
  }

  // this pipe served the encode render and the prompt mapping; the
  // finalisation core below builds its own
  dt_dev_pixelpipe_cleanup(&pipe);
  pipe_ready = FALSE;
  dt_mipmap_cache_release(&buf);
  buf_ready = FALSE;

  if(keep_going && !keep_going(user))
  {
    status = DT_OBJECT_RECIPE_RETRY;   // cancelled, not broken
    goto cleanup;
  }

  // replay the decodes at the recorded boundaries: decode i covers points
  // 0..i, with the threshold recorded when that decode really ran. the
  // whole per-decode sequence (multi-pass refinement with peak points and
  // box prompt, IoU convergence, seed-component filter, CRF and CascadePSP
  // when they were active) IS _decode_thread_func, called synchronously on
  // a per-boundary job -- zero duplicated logic, zero divergence. prev_mask
  // is never reset between decodes, as in the interactive session; the
  // context is freshly loaded, so the first decode starts clean anyway
  for(int i = 0; i <= last_decode; i++)
  {
    if(!recipe->points[i].decode_after)
      continue;
    if(keep_going && !keep_going(user))
    {
      status = DT_OBJECT_RECIPE_RETRY;
      goto cleanup;
    }

    const int n_prompt = i + 1;
    _decode_job_t *djob = g_malloc0(sizeof(_decode_job_t));
    djob->d = &od;
    djob->n_prompt_points = n_prompt;
    // always FALSE, like every launch of the interactive path (see
    // _launch_decode); do not attach first-click semantics to it
    djob->reset_prev_mask = FALSE;
    // the clamps mirror the conf reads of _launch_decode that recorded
    // these values, and shield against a hand-edited recipe
    djob->n_passes = CLAMP(recipe->refine_passes, 1, 3);
    // headroom: one peak point per pass + 2 box corners (SAM only)
    djob->points = g_new(dt_seg_point_t, n_prompt + djob->n_passes + 2);
    for(int k = 0; k < n_prompt; k++)
    {
      djob->points[k].x = enc_pts[k * 2 + 0];
      djob->points[k].y = enc_pts[k * 2 + 1];
      djob->points[k].label = (int)recipe->points[k].label;
    }
    // seed for the connected-component filter: last positive point, as in
    // _launch_decode
    djob->seed_x = -1;
    djob->seed_y = -1;
    for(int k = n_prompt - 1; k >= 0; k--)
      if(recipe->points[k].label == 1)
      {
        djob->seed_x = (int)enc_pts[k * 2 + 0];
        djob->seed_y = (int)enc_pts[k * 2 + 1];
        break;
      }
    djob->threshold = CLAMP(recipe->points[i].threshold, 0.3f, 0.9f);
    djob->do_crf = recipe->crf_enabled != 0;
    djob->crf_iter = CLAMP(recipe->crf_iterations, 1, 10);
    djob->crf_sigma_color = CLAMP(recipe->crf_sigma_color, 1.0f, 50.0f);
    djob->crf_w_bilateral = CLAMP(recipe->crf_w_bilateral, 0.5f, 30.0f);
    djob->do_refine = recipe->ai_refine != 0 && !od.refine_failed;
    djob->refine_margin = CLAMPF(recipe->ai_refine_margin, 0.0f, 0.5f);

    _decode_thread_func(djob);

    // the interactive session absorbs a refine load failure gracefully
    // (refine_failed, decode proceeds unrefined) -- the replay must not:
    // it would write an unrefined mask under a fingerprint that promises
    // refinement. hard failure, retryable once the model loads again
    if(recipe->ai_refine && od.refine_failed)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: refine model failed to load,"
               " cannot reproduce the recorded refinement");
      _decode_job_free(djob);
      goto cleanup;
    }

    if(i == last_decode && djob->out_mask)
    {
      hint = djob->out_mask;
      hint_w = djob->out_w;
      hint_h = djob->out_h;
      djob->out_mask = NULL;   // ownership moved
    }
    const gboolean decode_ok = (i == last_decode) ? (hint != NULL)
                                                  : (djob->out_mask != NULL);
    _decode_job_free(djob);
    if(!decode_ok)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] replay: decode at point %d failed", i + 1);
      goto cleanup;
    }
  }

  // the model stack of either family is done; release it before the heavy
  // native pass (which loads its own refinement context, as the
  // interactive job does)
  if(det)
  {
    dt_detect_free(det);
    det = NULL;
  }
  if(od.refine)
  {
    dt_refine_free(od.refine);
    od.refine = NULL;
  }
  dt_seg_free(seg);
  seg = NULL;
  od.seg = NULL;
  dt_ai_env_destroy(env);
  env = NULL;
  od.env = NULL;

  // subject bbox on the working grid, exactly as _launch_native_finalize
  // derives it: final threshold, CascadePSP margin as padding
  const float thresh = CLAMP(recipe->threshold, 0.3f, 0.9f);
  const float margin = CLAMPF(recipe->ai_refine_margin, 0.0f, 0.5f);
  // the component filter of a single-object detection: no clicked seed
  // exists, so the filter keeps the largest component. multi-component
  // detectors (a sky between branches) skip it by their table row
  if(promptless && detector->keep_seed)
    _keep_seed_component(hint, hint_w, hint_h, thresh, -1, -1);
  dt_seg_point_t tl, br;
  if(!_compute_bbox(hint, hint_w, hint_h, thresh, margin, &tl, &br))
  {
    dt_print(DT_DEBUG_AI, "[object mask] replay: empty mask, nothing to write");
    goto cleanup;
  }
  const int bx = CLAMP((int)tl.x, 0, hint_w - 1);
  const int by = CLAMP((int)tl.y, 0, hint_h - 1);
  const int bw = CLAMP((int)br.x - bx + 1, 1, hint_w - bx);
  const int bh = CLAMP((int)br.y - by + 1, 1, hint_h - by);

  // the native pass is the peak of VRAM use and shares its serialisation
  // token with the interactive finalisation job: never run both at once.
  // busy means a finalisation (or another replay) is in flight -- fail
  // cleanly, the caller may retry later
  if(!g_atomic_int_compare_and_exchange(&_finalize_running, 0, 1))
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] replay: a finalisation is already running,"
             " try again later");
    status = DT_OBJECT_RECIPE_RETRY;
    goto cleanup;
  }
  int pw = 0, ph = 0;
  const _finalize_render_req_t req = {
    .hint = hint,
    .hint_w = hint_w,
    .hint_h = hint_h,
    .bx = bx, .by = by, .bw = bw, .bh = bh,
    .threshold = thresh,
    .render_target = MAX(recipe->render_size, 1024),
    .interactive = FALSE,
    // a render being REPRODUCED: the stage is what the recipe recorded,
    // never what this machine happens to prefer today
    .matte = _matte_session_recorded(recipe),
  };
  alpha_full = _finalize_render_alpha(&dev, &req, keep_going, user, &pw, &ph);
  g_atomic_int_set(&_finalize_running, 0);
  if(!alpha_full)
  {
    // the render core reports failure and cancellation alike; recover the
    // distinction here so a cancelled replay frees its anti-respawn slot
    if(keep_going && !keep_going(user))
      status = DT_OBJECT_RECIPE_RETRY;
    goto cleanup;
  }

  if(!_write_mask_png16_atomic(outpath, alpha_full, pw, ph))
  {
    dt_print(DT_DEBUG_AI, "[object mask] replay: cannot write %s", outpath);
    goto cleanup;
  }

  dt_print(DT_DEBUG_AI,
           "[object mask] replay: %s regenerated (%dx%d, %.1fs)",
           outpath, pw, ph, dt_get_wtime() - t_start);
  status = DT_OBJECT_RECIPE_OK;

cleanup:
  g_free(rgb);
  g_free(enc_pts);
  g_free(hint);
  g_free(alpha_full);
  g_free(outpath);
  if(pipe_ready) dt_dev_pixelpipe_cleanup(&pipe);
  if(buf_ready) dt_mipmap_cache_release(&buf);
  if(od.refine) dt_refine_free(od.refine);
  if(seg) dt_seg_free(seg);
  if(det) dt_detect_free(det);
  if(env) dt_ai_env_destroy(env);
  dt_dev_cleanup(&dev);
  // a deterministic failure deserves a visible trace even without -d ai:
  // the user otherwise faces a silent zeroed mask with no clue
  if(status == DT_OBJECT_RECIPE_FAILED)
    dt_print(DT_DEBUG_ALWAYS,
             "[object mask] could not regenerate the mask of image %d from"
             " its recipe (run with -d ai for details)", imgid);
  return status;
}

// --------------------------- model gap diagnostic ---------------------------
//
// the queryable mirror of the replay gates above: OK means exactly "a
// replay would pass its model checks" -- same version lookup, same string
// comparison, refinement examined only when the recipe enables it, and
// the active == recorded rule for the refinement model. a diagnostic
// that disagreed with the replay would promise repairs that change
// nothing, so any change to the gates in dt_object_recipe_compute must
// land here too. anything but OK is refined into the cause an UX surface
// can act on; the verdict is cheap and never cached (installs, rebinds
// and edits all change it). the promptless family shares the
// (seg_model, version) pin verbatim, and the extra replay gates that a
// registry or manifest change can move -- the detector table row, the
// task family of the recorded id, the model's input side -- are
// mirrored below. the one deviation, stated: the matting-capability
// gate stays outside the mirror, a build property no download can
// move, like the boundary check

// the priority when the recipe's two models disagree: the verdict
// needing the heaviest user action wins. NOT the enum order --
// INSTALLABLE outranks both drifts
static int _model_gap_rank(const dt_object_recipe_model_gap_t gap)
{
  switch(gap)
  {
    case DT_OBJECT_RECIPE_MODELS_AI_OFF:       return 5;
    case DT_OBJECT_RECIPE_MODELS_UNKNOWN:      return 4;
    case DT_OBJECT_RECIPE_MODELS_INSTALLABLE:  return 3;
    case DT_OBJECT_RECIPE_MODELS_DRIFT_BEHIND: return 2;
    case DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD:  return 1;
    default:                                   return 0;
  }
}

static const char *_model_gap_name(const dt_object_recipe_model_gap_t gap)
{
  switch(gap)
  {
    case DT_OBJECT_RECIPE_MODELS_OK:           return "ok";
    case DT_OBJECT_RECIPE_MODELS_INSTALLABLE:  return "installable";
    case DT_OBJECT_RECIPE_MODELS_DRIFT_BEHIND: return "drift-behind";
    case DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD:  return "drift-ahead";
    case DT_OBJECT_RECIPE_MODELS_UNKNOWN:      return "unknown";
    case DT_OBJECT_RECIPE_MODELS_AI_OFF:       return "ai-off";
    default:                                   return "?";
  }
}

// verdict for one recorded (id, version) pair against the registry.
// `missing` collects the ids a download could move toward the recorded
// state (the INSTALLABLE and DRIFT_BEHIND ones)
static dt_object_recipe_model_gap_t
_model_gap_one(const char *model_id,
               const char *recorded_version,
               GPtrArray *missing)
{
  dt_ai_model_t *model = dt_ai_models_get_by_id(model_id);
  if(!model)
    // not even the registry knows the id: no download can produce it,
    // only a local .dtmodel install can
    return DT_OBJECT_RECIPE_MODELS_UNKNOWN;

  dt_object_recipe_model_gap_t gap = DT_OBJECT_RECIPE_MODELS_OK;

  if(model->status == DT_AI_MODEL_NOT_DOWNLOADED
     || model->status == DT_AI_MODEL_DOWNLOADING
     || model->status == DT_AI_MODEL_ERROR)
    // known deviation from a literal gate mirror: a recipe recording the
    // "0.0" placeholder version would slip past the replay's version
    // comparison only to fail at model load time; "install it" is the
    // truthful verdict either way
    gap = DT_OBJECT_RECIPE_MODELS_INSTALLABLE;
  else
  {
    // the exact comparison the replay gate refuses on. when it would
    // refuse, split the drift by direction: only a lagging install is
    // repairable by a download -- an UPDATE_* status marks the remote as
    // ahead of the install, so downloading can still move the version
    // even when the plain "X.Y" comparison cannot order the two strings.
    // when the versions MATCH the replay passes whatever the update
    // status says: an available update must not grow a repair button for
    // a mask that is not broken
    const char *installed = dt_ai_model_get_version(model_id);
    if(g_strcmp0(installed, recorded_version) != 0)
      gap = (dt_ai_models_version_compare(installed, recorded_version) < 0
             || model->status == DT_AI_MODEL_UPDATE_AVAILABLE
             || model->status == DT_AI_MODEL_UPDATE_REQUIRED)
        ? DT_OBJECT_RECIPE_MODELS_DRIFT_BEHIND
        : DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD;
  }
  dt_ai_model_free(model);

  if(missing
     && (gap == DT_OBJECT_RECIPE_MODELS_INSTALLABLE
         || gap == DT_OBJECT_RECIPE_MODELS_DRIFT_BEHIND))
    g_ptr_array_add(missing, g_strdup(model_id));

  return gap;
}

#ifdef HAVE_MATTE_VITMATTE
// THE PERSONAL MIRROR (Q7). In a shipped build a recorded matting stage is
// answered by one question -- does this build carry the operator? -- and
// dt_object_recipe_matting_reproducible answers it before any model
// verdict is asked. That is right there, because a shipped build's
// operators need no model at all.
//
// A build configured with USE_MATTE_VITMATTE carries an operator that
// does, and the two questions come apart: the operator is present, its
// revision matches, the replay gate is satisfied -- and the render still
// cannot run, because the weights are not on this machine. Reported as a
// MODEL gap so the four UX surfaces say "a model is missing" instead of
// "this build cannot reproduce that stage", which would be false and would
// point the user away from the one action that repairs it.
//
// The verdict for an absent model is UNKNOWN and not INSTALLABLE, and that
// is not an approximation: these weights are non-commercial, so they are
// in no download catalogue and dt_ai_models_get_by_id cannot know the id
// until the user has installed the package by hand. UNKNOWN is precisely
// the registry's word for "no download produces this" -- the same verdict
// a locally-packaged model gets anywhere else in this file.
static dt_object_recipe_model_gap_t
_matte_model_gap(const dt_matte_op_t *op, GPtrArray *missing)
{
  if(!op || !op->model || !(op->caps & DT_MATTE_NEEDS_MODEL))
    return DT_OBJECT_RECIPE_MODELS_OK;

  dt_ai_model_t *model = dt_ai_models_get_by_id(op->model);
  if(!model)
    return DT_OBJECT_RECIPE_MODELS_UNKNOWN;

  // the same statuses _model_gap_one treats as "not usable yet". no
  // version comparison: the recipe records the OPERATOR's algorithm
  // revision, never a model version, so there is no recorded number to
  // compare against and inventing one would refuse installs that are fine
  const gboolean pending = model->status == DT_AI_MODEL_NOT_DOWNLOADED
                           || model->status == DT_AI_MODEL_DOWNLOADING
                           || model->status == DT_AI_MODEL_ERROR;
  dt_ai_model_free(model);

  if(!pending) return DT_OBJECT_RECIPE_MODELS_OK;
  if(missing) g_ptr_array_add(missing, g_strdup(op->model));
  return DT_OBJECT_RECIPE_MODELS_INSTALLABLE;
}
#endif

// full diagnostic: the combined verdict plus the per-model ones (the
// failure toast needs to name the model at fault). the public wrapper
// below discards the details
static dt_object_recipe_model_gap_t
_recipe_model_gap(const dt_rf_recipe_t *recipe,
                  gchar ***missing,
                  dt_object_recipe_model_gap_t *seg_gap_out,
                  dt_object_recipe_model_gap_t *refine_gap_out,
                  dt_object_recipe_model_gap_t *matte_gap_out)
{
  dt_object_recipe_model_gap_t seg_gap = DT_OBJECT_RECIPE_MODELS_OK;
  dt_object_recipe_model_gap_t refine_gap = DT_OBJECT_RECIPE_MODELS_OK;
  // OK in every shipped build and never assigned there: the matting
  // operators of an upstream table need no model, so this stays the
  // neutral element of the rank below and the verdict is the one the
  // build without the private line computes
  dt_object_recipe_model_gap_t matte_gap = DT_OBJECT_RECIPE_MODELS_OK;

  if(missing) *missing = NULL;

  // an invalid recipe names no models: nothing to diagnose. callers gate
  // their surfaces on dt_rf_recipe_valid, not on this verdict
  const gboolean valid = dt_rf_recipe_valid(recipe);
  if(valid)
  {
    if(!dt_ai_registry_is_enabled())
      // distinct from UNKNOWN: with AI off the registry holds no models,
      // and every lookup below would misdiagnose the recipe as recording
      // ids nobody has heard of
      seg_gap = refine_gap = DT_OBJECT_RECIPE_MODELS_AI_OFF;
    else
    {
      // a registry created while AI was disabled at startup answers NULL
      // to every lookup: complete the deferred init first, or every
      // verdict would be a false UNKNOWN. no-op when already initialized
      dt_ai_models_init_lazy();

      GPtrArray *ids = missing ? g_ptr_array_new() : NULL;

      seg_gap = _model_gap_one(recipe->seg_model,
                               recipe->seg_model_version, ids);

      // the promptless replay additionally resolves its detector row
      // and loads BY ID with a task check and an input-side pin (the
      // gates of dt_object_recipe_compute and dt_detect_load): registry
      // and manifest properties, so OK must consult them too or it
      // would promise a replay those gates refuse
      if(recipe->prompt_kind != DT_RF_PROMPT_POINTS)
      {
        const dt_detector_t *detector
          = dt_detector_find(recipe->prompt_kind, recipe->class_bits);
        if(!detector)
          // the recipe names a detector this build's table does not
          // know: like an unknown id, nothing a download can produce
          seg_gap = DT_OBJECT_RECIPE_MODELS_UNKNOWN;
        else
        {
          // a catalogue update can reassign an id to another task; the
          // replay then refuses the load and no download repairs it --
          // "redo with the current models" is the only repair, the
          // DRIFT_AHEAD offer (the refine active-check precedent)
          dt_ai_model_t *m = dt_ai_models_get_by_id(recipe->seg_model);
          if(m && g_strcmp0(m->task, detector->task) != 0)
            seg_gap = DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD;
          dt_ai_model_free(m);

          if(seg_gap == DT_OBJECT_RECIPE_MODELS_OK)
          {
            // the input-side pin: a model repackaged locally at another
            // side under the same (id, version). only checkable once
            // the install matches (OK), the manifest exists then
            dt_ai_environment_t *env = dt_ai_registry_get_env();
            const dt_ai_model_info_t *info = env
              ? dt_ai_get_model_info_by_id(env, recipe->seg_model)
              : NULL;
            int n = 0;
            int *sizes = info
              ? dt_ai_model_attribute_int_array(info, "input_sizes", &n)
              : NULL;
            const int side = (sizes && n > 0) ? sizes[0] : 0;
            g_free(sizes);
            // the manifest carries its OWN task field and that is the
            // one dt_detect_load actually refuses on -- a hand-edited
            // config.json can move it independently of the registry
            // task above, so consult it too (a NULL info short-circuits
            // on the zero side before the dereference)
            if(side <= 0 || side != recipe->detect_input
               || g_strcmp0(info->task_type, detector->task) != 0)
              seg_gap = DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD;
          }
        }
      }

      if(recipe->ai_refine)
      {
        refine_gap = _model_gap_one(recipe->refine_model,
                                    recipe->refine_model_version, ids);
        if(refine_gap == DT_OBJECT_RECIPE_MODELS_OK)
        {
          // dt_refine_load offers no per-id loading, so the replay gate
          // additionally demands that the recorded model be the ACTIVE
          // one (see dt_object_recipe_compute). installed right but not
          // active is not repairable by any download: report it as
          // DRIFT_AHEAD, whose UX offer -- redo with the current models
          // -- is the only repair that exists for it
          char *active = dt_ai_models_get_active_for_task("refine");
          if(g_strcmp0(active, recipe->refine_model) != 0)
            refine_gap = DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD;
          g_free(active);
        }
      }

#ifdef HAVE_MATTE_VITMATTE
      // the personal mirror: an operator this build DOES carry, whose
      // model this machine may not have. folded into the same verdict as
      // the other two, so no surface needs a third question
      if(recipe->matting_enabled)
        matte_gap = _matte_model_gap(_matte_op_recorded(recipe), ids);
#endif

      if(ids)
      {
        if(ids->len)
        {
          g_ptr_array_add(ids, NULL);
          *missing = (gchar **)g_ptr_array_free(ids, FALSE);
        }
        else
          g_ptr_array_free(ids, TRUE);
      }
    }
  }

  dt_object_recipe_model_gap_t verdict
    = _model_gap_rank(refine_gap) > _model_gap_rank(seg_gap)
      ? refine_gap : seg_gap;
  if(_model_gap_rank(matte_gap) > _model_gap_rank(verdict))
    verdict = matte_gap;

  if(valid)
  {
    // the headless test surface: every diagnostic states its verdict and
    // both causes under -d ai
    gchar *refine_part = !recipe->ai_refine ? NULL
      : g_strdup_printf(", refine '%s' recorded v%s installed v%s (%s)",
                        recipe->refine_model,
                        recipe->refine_model_version,
                        dt_ai_model_get_version(recipe->refine_model),
                        _model_gap_name(refine_gap));
    // declared OUTSIDE the guard and filled inside it: the trace is the
    // one diagnostic surface an upstream build also prints, and a build
    // without the private operator must print the upstream line to the
    // byte. Left unguarded, this states a matting cause for a recipe that
    // same build refuses to replay at all -- one differing byte of output
    // is contamination, whatever the option was set to
    gchar *matte_part = NULL;
#ifdef HAVE_MATTE_VITMATTE
    // the matting cause states the MODEL it needs, not the operator: the
    // operator is the other gate's subject and says so in its own words
    if(recipe->matting_enabled)
      matte_part = g_strdup_printf(", matting '%.*s' (%s)",
                                   DT_RF_RECIPE_MATTING_ID_LEN,
                                   recipe->matting_id,
                                   _model_gap_name(matte_gap));
#endif
    dt_print(DT_DEBUG_AI,
             "[object mask] model gap: %s -- seg '%s' recorded v%s"
             " installed v%s (%s)%s%s",
             _model_gap_name(verdict),
             recipe->seg_model,
             recipe->seg_model_version,
             dt_ai_model_get_version(recipe->seg_model),
             _model_gap_name(seg_gap),
             refine_part ? refine_part : "",
             matte_part ? matte_part : "");
    g_free(refine_part);
    g_free(matte_part);
  }

  if(seg_gap_out) *seg_gap_out = seg_gap;
  if(refine_gap_out) *refine_gap_out = refine_gap;
  // OK in every build without the private line, and the surfaces that ask
  // for it test it LAST: a caller that never mentions matting therefore
  // behaves exactly as it did before this stage existed
  if(matte_gap_out) *matte_gap_out = matte_gap;
  return verdict;
}

dt_object_recipe_model_gap_t
dt_object_recipe_model_gap(const dt_rf_recipe_t *recipe, gchar ***missing)
{
  return _recipe_model_gap(recipe, missing, NULL, NULL, NULL);
}

// ------------------------ one-shot detection job ----------------------------
//
// "select subject" and its future siblings: a promptless detection as a
// cancellable background job, no interactive session, no canvas freeze.
// the job is the promptless replay run FORWARD: it captures a fresh v2
// promptless recipe from the live state, then performs exactly the steps
// dt_object_recipe_compute performs on that recipe -- the same render
// helper at the same dimensions, the same detector call, the same
// seed-component filter and bbox, the same native finalisation core and
// the same content-addressed file name -- so the recipe it stores really
// regenerates the file it wrote. what the replay cannot do is added
// around that spine: the model may be DOWNLOADED first (the catalogue's
// download->detect chaining), the _job_step hook keeps one toast alive
// for the whole 15-40 s, and the produced file is applied as a raster
// shape through the finalisation apply idle, unchanged

typedef struct _detect_job_t
{
  dt_imgid_t imgid;
  int32_t history_end;
  const dt_detector_t *detector;  // row of the static table, never freed
  // resolved at launch so the job is self-contained; the recipe records
  // this very id, and the version once the model is installed
  char model_id[DT_RF_RECIPE_MODEL_ID_LEN];
  gboolean download;              // model absent: download it first
  int render_size;                // raw render-cap preference at launch
  float threshold;                // session threshold at launch (clamped)
  float margin;                   // bbox padding at launch (clamped)
  char target_op[32];
  int target_multi_priority;
  gboolean has_target;
  dt_hash_t distort_hash;         // launch state; recorded in the recipe
                                  // and revalidated at APPLY time
  gchar *msg;                     // the one toast the job keeps alive
  gboolean download_cancelled;    // set by the progress callback
  // TRUE once the run entered: the serialisation token is reset by the
  // run itself, the destroy releases it for a discarded job (the
  // _finalize_job_destroy pattern)
  gboolean ran;
} _detect_job_t;

static void _detect_job_destroy(void *p)
{
  _detect_job_t *j = p;
  if(!j) return;
  if(!j->ran)
    g_atomic_int_set(&_finalize_running, 0);
  g_free(j->msg);
  g_free(j);
}

// the step hook of the detection job: one message ("detecting the
// subject...") that simply stays alive, the _finalize_keep_going pattern
static gboolean _detect_keep_going(void *p)
{
  dt_job_t *job = p;
  const _detect_job_t *j = job ? dt_control_job_get_params(job) : NULL;
  return _job_step(job, j ? j->msg : NULL);
}

#ifdef HAVE_AI_DOWNLOAD
// worker thread, driven by dt_ai_models_download_sync: the download is
// the one phase with an honest fraction, so the bar carries it; the
// cancel button reaches the download through the flag the sync call polls
static void _detect_download_progress(const char *model_id,
                                      const double progress,
                                      gpointer user_data)
{
  dt_job_t *job = user_data;
  _detect_job_t *j = dt_control_job_get_params(job);
  if(!_job_step(job, _("downloading the detection model...")))
    j->download_cancelled = TRUE;
  else
    dt_control_job_set_progress(job, progress);
}
#endif

static int32_t _detect_job_run(dt_job_t *job)
{
  _detect_job_t *const j = dt_control_job_get_params(job);
  // from here the reset of the serialisation token is ours, on every exit
  j->ran = TRUE;
  gboolean ok = FALSE;
  uint8_t *rgb = NULL;
  float *hint = NULL;
  float *alpha_full = NULL;
  gchar *outpath = NULL;
  dt_ai_environment_t *env = NULL;
  dt_detect_context_t *det = NULL;
  dt_dev_pixelpipe_t pipe;
  dt_mipmap_buffer_t buf;
  gboolean pipe_ready = FALSE, buf_ready = FALSE;
  int out_w = 0, out_h = 0;
  int pw = 0, ph = 0;
  dt_rf_recipe_t recipe;
  memset(&recipe, 0, sizeof(recipe));

  const double t_start = dt_get_wtime();

  dt_develop_t dev;
  dt_dev_init(&dev, FALSE);
  dt_dev_load_image(&dev, j->imgid);
  // the darkroom can be ahead of the database (the interactive threads'
  // rule); the launch flushed the history, the override covers the rest
  if(j->history_end > 0 && j->history_end > dev.history_end)
    dev.history_end = j->history_end;

#ifdef HAVE_AI_DOWNLOAD
  // the catalogue's chaining: the entry said "downloads the model", so
  // the download IS the first step of the detection, on this very job --
  // same progress entry, same cancel button
  if(j->download)
  {
    char *error = dt_ai_models_download_sync(j->model_id,
                                             _detect_download_progress,
                                             job,
                                             &j->download_cancelled);
    if(error)
    {
      if(!j->download_cancelled)
        dt_control_log(_("model download failed: %s"), error);
      g_free(error);
      goto cleanup;
    }
    // the version the recipe must record lives in the freshly extracted
    // config.json; the registry scan is what reads it back
    dt_ai_models_refresh_status();
    // the fraction was the download's; the steps that follow carry none
    // (the _job_step contract), so a negative value hands the bar back
    // to its indeterminate state instead of parking it full
    dt_control_job_set_progress(job, -1.0);
  }
#endif

  if(!_job_step(job, j->msg)) goto cleanup;

  // a fresh environment scans the installed models, so it must be
  // created after the download; loading is BY the resolved id, with the
  // task check of the detect consumer
  env = dt_ai_env_init(NULL);
  det = env ? dt_detect_load(env, j->model_id, j->detector->task) : NULL;
  if(!det)
  {
    dt_control_log(_("cannot load the detection model"));
    goto cleanup;
  }

  // the matting stage of this render, resolved BEFORE the recipe is filled
  // and read by nothing else afterwards: same rule as every other parameter
  // below -- what the compute runs under is what the recipe records. reading
  // the configuration again down at the render request would put a second
  // dt_conf lookup between the fingerprint and the pixels it names
  const _matte_session_t matte = _matte_session_from_conf();

  // ---- the recipe: the promptless v2 blob a headless replay consumes.
  // filled before the compute, so every parameter the compute reads below
  // is a parameter the recipe records -- the two cannot diverge. the
  // decode scalars and the refinement pin stay zero: no decode chain ran.
  // this family is version EXT whatever the matting stage does, its
  // prompt kind being an extension field in use by itself
  recipe.magic = DT_RF_RECIPE_MAGIC;
  recipe.version = DT_RF_RECIPE_VERSION_EXT;
  recipe.distort_hash = (int64_t)j->distort_hash;
  g_strlcpy(recipe.seg_model, j->model_id, sizeof(recipe.seg_model));
  {
    const char *v = dt_ai_model_get_version(j->model_id);
    if(v)
      g_strlcpy(recipe.seg_model_version, v,
                sizeof(recipe.seg_model_version));
  }
  recipe.render_size = j->render_size;   // raw, the capture convention
  recipe.threshold = j->threshold;
  recipe.ai_refine_margin = j->margin;
  recipe.prompt_kind = j->detector->prompt_kind;
  recipe.detect_input = dt_detect_get_side(det);
  recipe.class_bits = j->detector->class_bits;
  // the matting stage from the session hoisted above, through the same
  // single writer the clicked capture uses. an operator that cannot be
  // named in the recipe is a mask with no reproducible provenance, and
  // this route has no plain-file fallback to offer: refuse the whole job
  // rather than write a file whose recipe describes another render
  if(!_recipe_set_matting(&recipe, &matte))
  {
    dt_control_log(_("precise mask: the matting stage cannot be recorded"));
    goto cleanup;
  }

  // export render pipe at full input dimensions, as the replay builds it
  dt_mipmap_cache_get(&buf, j->imgid, DT_MIPMAP_FULL, DT_MIPMAP_BLOCKING,
                      'r');
  buf_ready = TRUE;
  if(!buf.buf || !buf.width || !buf.height)
  {
    dt_control_log(_("precise mask: cannot get the image buffer"));
    goto cleanup;
  }
  const int iw = dev.image_storage.width;
  const int ih = dev.image_storage.height;
  if(!dt_dev_pixelpipe_init_export(&pipe, iw, ih,
                                   IMAGEIO_RGB | IMAGEIO_INT8, FALSE))
  {
    dt_control_log(_("precise mask: cannot init the render pipe"));
    goto cleanup;
  }
  pipe_ready = TRUE;
  dt_dev_pixelpipe_set_icc(&pipe, DT_COLORSPACE_SRGB, NULL,
                           DT_INTENT_PERCEPTUAL);
  dt_dev_pixelpipe_set_input(&pipe, &dev, (float *)buf.buf,
                             buf.width, buf.height, buf.iscale);
  dt_dev_pixelpipe_create_nodes(&pipe, &dev);
  dt_dev_pixelpipe_synch_all(&pipe, &dev);
  dt_dev_pixelpipe_get_dimensions(&pipe, &dev, pipe.iwidth, pipe.iheight,
                                  &pipe.processed_width,
                                  &pipe.processed_height);
  if(pipe.processed_width <= 0 || pipe.processed_height <= 0)
  {
    dt_control_log(_("precise mask: cannot init the render pipe"));
    goto cleanup;
  }

  // the effective encode dimensions under the launch's cap, with the
  // encode-thread formula -- the recipe records THESE, and the shared
  // render helper below recovers the same scale from them
  {
    const int render_target = MAX(j->render_size, 1024);
    const double e_scale
      = fmin((double)render_target / (double)pipe.processed_width,
             (double)render_target / (double)pipe.processed_height);
    const double final_scale = fmin(e_scale, 1.0);
    out_w = (int)(final_scale * pipe.processed_width);
    out_h = (int)(final_scale * pipe.processed_height);
  }
  if(out_w < 8 || out_h < 8)
  {
    dt_control_log(_("precise mask: cannot init the render pipe"));
    goto cleanup;
  }
  recipe.encode_w = out_w;
  recipe.encode_h = out_h;

  // the recipe is complete: derive its content-addressed target, exactly
  // as the replay does from ITS loaded dev
  {
    const dt_image_t *img = &dev.image_storage;
    gchar *base = g_path_get_basename(img->filename);
    char *dot = g_strrstr(base, ".");
    if(dot) *dot = '\0';
    gchar *fname = dt_rasterfile_recipe_filename(&recipe, base,
                                                 img->width, img->height,
                                                 img->exif_datetime_taken);
    gchar *root = dt_rasterfile_mask_root();
    if(g_mkdir_with_parents(root, 0755) == 0)
      outpath = g_build_filename(root, fname, NULL);
    else
      dt_control_log(_("cannot create raster mask folder"));
    g_free(root);
    g_free(fname);
    g_free(base);
  }
  if(!outpath)
    goto cleanup;

  // a valid file under this fingerprint IS this detection, already made
  // (same recipe, same image): skip the whole compute and just apply.
  // no instance disabling here, unlike the replay -- so a REdetection of
  // a subject whose file was purged renders under a history that may
  // already apply the first one through its module. the replay's
  // documented self-influence class (and raster forms sit outside its
  // instance disabling anyway): assumed, not solved
  if(g_file_test(outpath, G_FILE_TEST_EXISTS) && _mask_png_valid(outpath))
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] detect: %s already exists, reusing", outpath);
    goto apply;
  }

  rgb = _replay_render_rgb8(&dev, &pipe, &recipe, out_w, out_h);
  if(!rgb)
  {
    dt_control_log(_("precise mask: native render failed"));
    goto cleanup;
  }

  // this pipe served the encode render; the finalisation core below
  // builds its own
  dt_dev_pixelpipe_cleanup(&pipe);
  pipe_ready = FALSE;
  dt_mipmap_cache_release(&buf);
  buf_ready = FALSE;

  if(!_job_step(job, j->msg)) goto cleanup;

  hint = g_try_malloc((size_t)out_w * out_h * sizeof(float));
  if(!hint)
  {
    dt_control_log(_("precise mask: out of memory"));
    goto cleanup;
  }
  if(!dt_detect_run(det, rgb, out_w, out_h, hint))
  {
    dt_control_log(_("the detection failed on this image"));
    goto cleanup;
  }
  g_free(rgb);
  rgb = NULL;

  // the detector is done; release it before the heavy native pass, which
  // loads its own refinement context (the replay's VRAM rule)
  dt_detect_free(det);
  det = NULL;
  dt_ai_env_destroy(env);
  env = NULL;

  if(!_job_step(job, j->msg)) goto cleanup;

  // seed filter and subject bbox, exactly as the replay derives them
  // from this very recipe
  {
    const float thresh = CLAMP(recipe.threshold, 0.3f, 0.9f);
    const float margin = CLAMPF(recipe.ai_refine_margin, 0.0f, 0.5f);
    if(j->detector->keep_seed)
      _keep_seed_component(hint, out_w, out_h, thresh, -1, -1);
    dt_seg_point_t tl, br;
    if(!_compute_bbox(hint, out_w, out_h, thresh, margin, &tl, &br))
    {
      dt_control_log(_("nothing detected in this image"));
      goto cleanup;
    }
    const int bx = CLAMP((int)tl.x, 0, out_w - 1);
    const int by = CLAMP((int)tl.y, 0, out_h - 1);
    const int bw = CLAMP((int)br.x - bx + 1, 1, out_w - bx);
    const int bh = CLAMP((int)br.y - by + 1, 1, out_h - by);

    // the native finalisation, unchanged: the launch already holds the
    // serialisation token for the whole job, so unlike the replay there
    // is nothing to claim here
    const _finalize_render_req_t req = {
      .hint = hint,
      .hint_w = out_w,
      .hint_h = out_h,
      .bx = bx, .by = by, .bw = bw, .bh = bh,
      .threshold = thresh,
      .render_target = MAX(recipe.render_size, 1024),
      .interactive = TRUE,
      // also a render being made: same door as the interactive job, walked
      // above the recipe so the recorded stage and the run stage are one
      .matte = matte,
    };
    alpha_full = _finalize_render_alpha(&dev, &req, _detect_keep_going,
                                        job, &pw, &ph);
  }
  if(!alpha_full)
    goto cleanup;

  if(dt_control_job_get_state(job) == DT_JOB_STATE_CANCELLED)
    goto cleanup;

  if(!_write_mask_png16_atomic(outpath, alpha_full, pw, ph))
  {
    dt_control_log(_("failed to save the precise raster mask"));
    goto cleanup;
  }
  dt_print(DT_DEBUG_AI,
           "[object mask] detect: %s written (%dx%d, %.1fs)",
           outpath, pw, ph, dt_get_wtime() - t_start);

apply:
  // hand over to the GUI thread: the finalisation apply idle wires the
  // raster shape, named from its recipe by the detectors table
  {
    _finalize_apply_t *a = g_malloc0(sizeof(_finalize_apply_t));
    a->imgid = j->imgid;
    a->outpath = g_strdup(outpath);
    a->has_target = j->has_target;
    memcpy(a->target_op, j->target_op, sizeof(a->target_op));
    a->target_multi_priority = j->target_multi_priority;
    a->has_recipe = TRUE;
    a->recipe = recipe;
    a->detector = j->detector;
    a->distort_hash = j->distort_hash;
    g_idle_add_full(G_PRIORITY_DEFAULT_IDLE, _finalize_apply_idle, a,
                    _finalize_apply_free);
  }
  ok = TRUE;

cleanup:
  g_free(rgb);
  g_free(hint);
  g_free(alpha_full);
  g_free(outpath);
  if(pipe_ready) dt_dev_pixelpipe_cleanup(&pipe);
  if(buf_ready) dt_mipmap_cache_release(&buf);
  if(det) dt_detect_free(det);
  if(env) dt_ai_env_destroy(env);
  dt_dev_cleanup(&dev);
  g_atomic_int_set(&_finalize_running, 0);
  return ok ? 0 : 1;
}

// resolve the model a detection of `detector` would use NOW: the active
// model of its task when one is set, otherwise the registry's default
// for the task (the download candidate), otherwise any model serving it.
// registry copy, caller frees; NULL when the registry knows none
static dt_ai_model_t *_detector_resolve_model(const dt_detector_t *det)
{
  // a registry created while AI was disabled at startup answers NULL to
  // every lookup: complete the deferred init first (no-op otherwise)
  dt_ai_models_init_lazy();

  char *id = dt_ai_models_get_active_for_task(det->task);
  dt_ai_model_t *model
    = (id && id[0]) ? dt_ai_models_get_by_id(id) : NULL;
  g_free(id);
  if(model) return model;

  dt_ai_model_t *candidate = NULL;
  const int count = dt_ai_models_get_count();
  for(int i = 0; i < count; i++)
  {
    dt_ai_model_t *m = dt_ai_models_get_by_index(i);
    if(m && m->task && !strcmp(m->task, det->task))
    {
      if(m->is_default)
      {
        dt_ai_model_free(candidate);
        return m;
      }
      if(!candidate)
      {
        candidate = m;
        continue;
      }
    }
    dt_ai_model_free(m);
  }
  return candidate;
}

dt_masks_object_detect_state_t
dt_masks_object_detect_state(const struct dt_detector_t *detector)
{
  if(!detector)
    return DT_MASKS_OBJECT_DETECT_UNAVAILABLE;
  if(!dt_ai_registry_is_enabled())
    return DT_MASKS_OBJECT_DETECT_AI_OFF;

  dt_ai_model_t *model = _detector_resolve_model(detector);
  if(!model)
    return DT_MASKS_OBJECT_DETECT_UNAVAILABLE;

  dt_masks_object_detect_state_t state;
  switch(model->status)
  {
    case DT_AI_MODEL_DOWNLOADED:
    case DT_AI_MODEL_UPDATE_AVAILABLE:  // the installed model still works
      state = DT_MASKS_OBJECT_DETECT_READY;
      break;
    case DT_AI_MODEL_DOWNLOADING:
      state = DT_MASKS_OBJECT_DETECT_DOWNLOADING;
      break;
    default:  // NOT_DOWNLOADED, UPDATE_REQUIRED, ERROR
#ifdef HAVE_AI_DOWNLOAD
      state = DT_MASKS_OBJECT_DETECT_DOWNLOAD;
#else
      state = DT_MASKS_OBJECT_DETECT_UNAVAILABLE;
#endif
      break;
  }
  dt_ai_model_free(model);
  return state;
}

gboolean dt_masks_object_detect_launch(const struct dt_detector_t *detector,
                                       dt_iop_module_t *module)
{
  dt_develop_t *dev = darktable.develop;
  if(!detector || !dev || !dt_is_valid_imgid(dev->image_storage.id))
    return FALSE;

  // the state the catalogue labelled the entry with, re-derived: the
  // registry may have moved between the menu and the click. a refused
  // click must say so -- a silent no-op is a broken button (decision 13)
  const dt_masks_object_detect_state_t state
    = dt_masks_object_detect_state(detector);
  if(state != DT_MASKS_OBJECT_DETECT_READY
     && state != DT_MASKS_OBJECT_DETECT_DOWNLOAD)
  {
    dt_control_log(_("AI model is not available. Check preferences > AI"));
    return FALSE;
  }

  dt_ai_model_t *model = _detector_resolve_model(detector);
  if(!model)
  {
    dt_control_log(_("AI model is not available. Check preferences > AI"));
    return FALSE;
  }

  // one heavy AI mask job at a time: the very token the interactive
  // finalisation and the headless replay serialise on
  if(!g_atomic_int_compare_and_exchange(&_finalize_running, 0, 1))
  {
    dt_control_log(_("mask still computing, try again in a moment"));
    dt_ai_model_free(model);
    return FALSE;
  }

  // the job re-reads the history from the database; unflushed edits
  // (exposure, crop, ...) would silently be missing from the render
  dt_dev_write_history(dev);

  _detect_job_t *j = g_malloc0(sizeof(_detect_job_t));
  j->imgid = dev->image_storage.id;
  j->history_end = dev->history_end;
  j->detector = detector;
  g_strlcpy(j->model_id, model->id ? model->id : "", sizeof(j->model_id));
  j->download = state == DT_MASKS_OBJECT_DETECT_DOWNLOAD;
  j->render_size = _conf_render_size();
  // through the session accessors, NULL session: the conf branch, with
  // the clamps the replay re-applies -- the recorded value is the used one
  j->threshold = _session_threshold(NULL);
  j->margin = _session_refine_margin(NULL);
  j->distort_hash = _compute_distort_hash(dev);
  j->msg = g_strdup_printf(_("detecting the %s..."), _(detector->label));
  if(module)
  {
    g_strlcpy(j->target_op, module->op, sizeof(j->target_op));
    j->target_multi_priority = module->multi_priority;
    j->has_target = TRUE;
  }
  dt_ai_model_free(model);

  dt_job_t *job = dt_control_job_create(_detect_job_run,
                                        "one-shot mask detection");
  if(!job)
  {
    // the destroy releases the token itself (j->ran is FALSE): the one
    // path that frees these params is also the one that gives it back
    _detect_job_destroy(j);
    return FALSE;
  }
  dt_control_job_set_params(job, j, _detect_job_destroy);
  dt_control_job_add_progress(job, _("automatic selection"), TRUE);
  // announce before queueing, as the finalisation launch does
  dt_control_log("%s", j->msg);
  dt_control_add_job(DT_JOB_QUEUE_USER_BG, job);
  return TRUE;
}

// ----------------------------- edit session gate ----------------------------
//
// while an interactive AI edit session is open on a rasterfile instance,
// the proactive missing-file branch of iop/rasterfile.c gui_changed must
// not schedule a headless recompute of the very mask the session is
// preparing to replace: in the nominal repair flow the file IS missing
// for the whole session, so every gui_update (signal, focus change,
// w == NULL refresh) would otherwise race a second inference stack
// against the session -- and pin a fresh FAILED slot when the recorded
// model is still absent. a single (op, multi_priority) slot suffices:
// edit sessions are modal, at most one per process

static GMutex _edit_active_mutex;
static gboolean _edit_active = FALSE;
static char _edit_active_op[32] = { 0 };
static int32_t _edit_active_priority = -1;

gboolean dt_object_mask_edit_active(const char *op,
                                    const int32_t multi_priority)
{
  g_mutex_lock(&_edit_active_mutex);
  const gboolean active = _edit_active
    && op
    && !g_strcmp0(op, _edit_active_op)
    && multi_priority == _edit_active_priority;
  g_mutex_unlock(&_edit_active_mutex);
  return active;
}

void dt_object_mask_edit_set_active(const char *op,
                                    const int32_t multi_priority)
{
  g_mutex_lock(&_edit_active_mutex);
  _edit_active = TRUE;
  g_strlcpy(_edit_active_op, op ? op : "", sizeof(_edit_active_op));
  _edit_active_priority = multi_priority;
  g_mutex_unlock(&_edit_active_mutex);
}

void dt_object_mask_edit_clear_active(void)
{
  g_mutex_lock(&_edit_active_mutex);
  _edit_active = FALSE;
  _edit_active_op[0] = '\0';
  _edit_active_priority = -1;
  g_mutex_unlock(&_edit_active_mutex);
}

// edit sessions are modal: at most one per process, whatever instance.
// the flag is BOOKKEEPING, not thread ownership: _free_data drops it as
// soon as the session ends, while a replay thread torn down mid-flight
// keeps running until its deferred join. a new session may therefore start
// alongside an orphaned replay -- no shared data (separate stand-in, own
// ONNX contexts, per-thread temp files with atomic rename, the persistent
// slot protected by edit_model_pinned), only two resident models and two
// inferences for the time it takes. same class as the pre-existing
// deferred encode, and accepted for the same reason
static gboolean _edit_any_active(void)
{
  g_mutex_lock(&_edit_active_mutex);
  const gboolean active = _edit_active;
  g_mutex_unlock(&_edit_active_mutex);
  return active;
}

// ------------------------- ai edit session: opening -------------------------
//
// see the machine's contract at the _edit_pending_t definition. the code
// below implements the three GUI-thread legs: the opening
// (dt_object_mask_edit_begin), the WAIT_ENCODE -> REPLAYING transition
// (_edit_replay_start) and the REPLAYING -> ACTIVE transition
// (_edit_replay_finish), both driven from post_expose. the replay itself
// (_edit_replay_thread_func) is the recorded per-boundary decode chain of
// the headless replay, executed against the LIVE session's encoded
// context so the interactive flow continues natively afterwards

// worker thread: replay the recorded decodes at their recorded boundaries.
// this mirrors the boundary loop of dt_object_recipe_compute -- the
// fidelity reference -- with two justified deviations: (a) the prompts
// were mapped to encode space on the GUI thread with the SESSION's own
// preview-to-encode scale (the mapping a later re-capture inverts; it
// agrees with the replay's export-pipe mapping because the normalised
// coordinates are scale-invariant), and (b) the decodes run on the
// session's already-encoded context through the stand-in, not on a
// context of their own
static gpointer _edit_replay_thread_func(gpointer data)
{
  _edit_replay_job_t *rj = data;
  const dt_rf_recipe_t *recipe = &rj->recipe;
  gboolean ok = TRUE;

  // decode i covers points 0..i with the threshold recorded when that
  // decode really ran; prev_mask is never reset between decodes, exactly
  // like the interactive session (the encode reset it, so the first
  // decode starts clean) and the headless replay
  for(int i = 0; i <= rj->last_decode && ok; i++)
  {
    if(!recipe->points[i].decode_after)
      continue;

    const int n_prompt = i + 1;
    _decode_job_t *djob = g_malloc0(sizeof(_decode_job_t));
    djob->d = &rj->od;
    djob->n_prompt_points = n_prompt;
    // always FALSE, like every launch of the interactive path (see
    // _launch_decode); do not attach first-click semantics to it
    djob->reset_prev_mask = FALSE;
    // the clamps mirror the conf reads of _launch_decode that recorded
    // these values, and shield against a hand-edited recipe
    djob->n_passes = CLAMP(recipe->refine_passes, 1, 3);
    // headroom: one peak point per pass + 2 box corners (SAM only)
    djob->points = g_new(dt_seg_point_t, n_prompt + djob->n_passes + 2);
    for(int k = 0; k < n_prompt; k++)
    {
      djob->points[k].x = rj->enc_pts[k * 2 + 0];
      djob->points[k].y = rj->enc_pts[k * 2 + 1];
      djob->points[k].label = (int)recipe->points[k].label;
    }
    // seed for the connected-component filter: last positive point, as in
    // _launch_decode
    djob->seed_x = -1;
    djob->seed_y = -1;
    for(int k = n_prompt - 1; k >= 0; k--)
      if(recipe->points[k].label == 1)
      {
        djob->seed_x = (int)rj->enc_pts[k * 2 + 0];
        djob->seed_y = (int)rj->enc_pts[k * 2 + 1];
        break;
      }
    djob->threshold = CLAMP(recipe->points[i].threshold, 0.3f, 0.9f);
    djob->do_crf = recipe->crf_enabled != 0;
    djob->crf_iter = CLAMP(recipe->crf_iterations, 1, 10);
    djob->crf_sigma_color = CLAMP(recipe->crf_sigma_color, 1.0f, 50.0f);
    djob->crf_w_bilateral = CLAMP(recipe->crf_w_bilateral, 0.5f, 30.0f);
    djob->do_refine = recipe->ai_refine != 0 && !rj->od.refine_failed;
    djob->refine_margin = CLAMPF(recipe->ai_refine_margin, 0.0f, 0.5f);

    _decode_thread_func(djob);

    // the interactive session absorbs a refine load failure gracefully --
    // the replay must not: the reopened working state would silently lack
    // the refinement every recorded decode had (headless replay rule)
    if(recipe->ai_refine && rj->od.refine_failed)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] edit: refine model failed to load, cannot"
               " reproduce the recorded refinement");
      _decode_job_free(djob);
      ok = FALSE;
      break;
    }

    if(i == rj->last_decode && djob->out_mask)
    {
      rj->out_mask = djob->out_mask;
      rj->out_w = djob->out_w;
      rj->out_h = djob->out_h;
      djob->out_mask = NULL;   // ownership moved
    }
    const gboolean decode_ok = (i == rj->last_decode)
      ? (rj->out_mask != NULL)
      : (djob->out_mask != NULL);
    _decode_job_free(djob);
    if(!decode_ok)
    {
      dt_print(DT_DEBUG_AI,
               "[object mask] edit: replayed decode at point %d failed",
               i + 1);
      ok = FALSE;
    }
  }

  rj->ok = ok && rj->out_mask != NULL;
  // last instruction: hand the result to the GUI side. the 100 ms poll
  // timer keeps post_expose ticking; the edit machine is the ONLY joiner
  // of this thread on the live path (_destroy_data joins on teardown)
  g_atomic_int_set(&rj->done, 1);
  return NULL;
}

// GUI thread, from post_expose at ENCODE_READY: the WAIT_ENCODE ->
// REPLAYING transition. injects the recorded session into the live gui
// state and starts the replay thread; any failure ends the edit
// bookkeeping (EDIT_FAILED) and leaves the tool as a plain session
static void _edit_replay_start(dt_masks_form_gui_t *gui, _object_data_t *d)
{
  // ENCODE_READY is signalled BEFORE the decoder warmup running on the
  // same thread ends (see _encode_thread_func): starting the replay now
  // would race the warmup on the shared segmentation context. join on the
  // GUI side -- the warmup is bounded -- exactly like _launch_decode
  // protects the first interactive decode. in practice the encode machine
  // above already joined; this keeps the transition self-sufficient
  if(d->encode_thread)
  {
    g_thread_join(d->encode_thread);
    d->encode_thread = NULL;
  }

  // the replay validates its own precondition instead of trusting the
  // invalidation branch, which is DEFERRED while the session is frozen: a
  // geometry change during the encode (a ctrl+Z of a crop, seconds are
  // available) leaves the encoding on the old state while the injection
  // below maps the recorded points through the CURRENT distortion. every
  // prompt would land on the wrong grid -- a systematic offset, not the
  // sub-pixel variance the comment further down accepts -- and the whole
  // replay would burn on false prompts before publishing a wrong mask the
  // user can click for a frame
  if(_compute_distort_hash(darktable.develop) != d->encoded_distort_hash)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] edit: the geometry moved while the session was"
             " frozen, the encoding no longer reflects the live state");
    dt_control_log(_("could not reload the recorded mask:"
                     " the geometry changed"));
    _edit_session_end(d, EDIT_FAILED);
    _edit_drop_pinned_encode(d);
    return;
  }

  const dt_rf_recipe_t *recipe = &d->edit_recipe;
  const int n = recipe->n_points;

  // the recorded decode boundaries drive the replay; a recipe without any
  // defines no refinement chain (mirror of the headless replay's gate)
  int last_decode = -1;
  for(int i = 0; i < n; i++)
    if(recipe->points[i].decode_after)
      last_decode = i;

  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // the pinned encode guarantees the recorded dims; belt and braces, a
  // mismatch here would put every replayed prompt on the wrong grid
  if(last_decode < 0 || wd <= 0 || ht <= 0 || iwidth <= 0 || iheight <= 0
     || d->encode_w != recipe->encode_w || d->encode_h != recipe->encode_h)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] edit: cannot start the replay (boundaries %d,"
             " encode %dx%d vs recorded %dx%d)",
             last_decode, d->encode_w, d->encode_h,
             recipe->encode_w, recipe->encode_h);
    dt_control_log(_("could not reload the recorded mask"));
    _edit_session_end(d, EDIT_FAILED);
    _edit_drop_pinned_encode(d);
    return;
  }

  // the recipe describes the distortion state the prompts were clicked
  // on; when the history moved since, the session -- like the headless
  // replay -- proceeds on the CURRENT state: the input-space-normalised
  // points make the injection largely self-correcting. divergence is
  // confined to the render the decodes see, ONNX-variance class
  if((dt_hash_t)recipe->distort_hash != d->encoded_distort_hash)
    dt_print(DT_DEBUG_AI,
             "[object mask] edit: distortion state differs from capture,"
             " proceeding on the current state");

  if(!gui->guipoints)
    gui->guipoints = dt_masks_dynbuf_init(200000, "object guipoints");
  if(!gui->guipoints_payload)
    gui->guipoints_payload = dt_masks_dynbuf_init(100000,
                                                  "object guipoints_payload");
  if(!gui->guipoints || !gui->guipoints_payload)
  {
    dt_control_log(_("could not reload the recorded mask"));
    _edit_session_end(d, EDIT_FAILED);
    _edit_drop_pinned_encode(d);
    return;
  }

  // inject the recorded clicks into the live gui state, in preview-pipe
  // pixel space: the exact inverse of _capture_recipe's mapping
  // (backtransform through the pipe, divided by the input dimensions).
  // points a later crop pushed off-frame stay unclamped, like live clicks
  // (only the decode seed clamps) -- accepted ONNX-variance-class case
  float *pts = g_new(float, (size_t)n * 2);
  for(int i = 0; i < n; i++)
  {
    pts[i * 2 + 0] = recipe->points[i].x * iwidth;
    pts[i * 2 + 1] = recipe->points[i].y * iheight;
  }
  dt_dev_distort_transform(darktable.develop, pts, n);

  dt_masks_dynbuf_reset(gui->guipoints);
  dt_masks_dynbuf_reset(gui->guipoints_payload);
  for(int i = 0; i < n; i++)
  {
    dt_masks_dynbuf_add_2(gui->guipoints, pts[i * 2 + 0], pts[i * 2 + 1]);
    dt_masks_dynbuf_add(gui->guipoints_payload,
                        (float)recipe->points[i].label);
  }
  gui->guipoints_count = n;
  d->has_selection = TRUE;

  // ... AND the decode marks (launched + per-boundary threshold): without
  // them a later re-capture would record no boundary for the replayed
  // prefix, and the gen-2 recipe's own headless replay would regenerate
  // different bytes under the new fingerprint -- a silent store poisoning
  // invisible to any visual check
  _marks_resize(d, n);
  for(int i = 0; i < n; i++)
  {
    _decode_mark_t *mark = &g_array_index(d->decode_marks, _decode_mark_t, i);
    mark->launched = recipe->points[i].decode_after != 0;
    mark->threshold = mark->launched ? recipe->points[i].threshold : 0.0f;
  }

  // prompts to session encode space with the session's own mapping -- the
  // very scale _launch_decode applies to live clicks and _capture_recipe
  // inverts at re-capture, keeping the round-trip stable. snapshotted on
  // this thread: the replay never reads the live dynbufs
  _edit_replay_job_t *rj = g_malloc0(sizeof(_edit_replay_job_t));
  const float sx = (wd > 0) ? (float)d->encode_w / wd : 1.0f;
  const float sy = (ht > 0) ? (float)d->encode_h / ht : 1.0f;
  rj->enc_pts = g_new(float, (size_t)n * 2);
  for(int i = 0; i < n; i++)
  {
    rj->enc_pts[i * 2 + 0] = pts[i * 2 + 0] * sx;
    rj->enc_pts[i * 2 + 1] = pts[i * 2 + 1] * sy;
  }
  g_free(pts);
  rj->recipe = *recipe;
  rj->n_points = n;
  rj->last_decode = last_decode;
  // the stand-in: seg/env/refine ALIAS the session's -- the replay needs
  // the session's encoded context, and the session gets the lazily loaded
  // refinement context back at the join. the session data itself stays
  // FROZEN (publication and clicks gated) until then
  rj->od.env = d->env;
  rj->od.seg = d->seg;
  rj->od.refine = d->refine;
  rj->od.refine_failed = d->refine_failed;

  // non-modal busy indicator, the decode machine's own pattern; no decode
  // can run while frozen, so no enter/leave interleaving is possible
  if(!d->decode_busy_shown)
  {
    dt_control_busy_enter();
    d->decode_busy_shown = TRUE;
  }
  dt_control_log(_("reloading the recorded mask..."));

  d->edit_replay_job = rj;
  g_atomic_int_set(&d->edit_pending, EDIT_REPLAYING);
  d->edit_replay_thread
    = g_thread_new("ai-mask-edit-replay", _edit_replay_thread_func, rj);
}

// GUI thread, from post_expose while REPLAYING: poll the replay, and on
// completion perform the REPLAYING -> ACTIVE transition (or EDIT_FAILED)
static void _edit_replay_finish(dt_masks_form_gui_t *gui, _object_data_t *d)
{
  _edit_replay_job_t *rj = d->edit_replay_job;
  if(!rj)
  {
    // cannot happen on the live path; recover instead of freezing forever
    _edit_session_end(d, EDIT_FAILED);
    _edit_drop_pinned_encode(d);
    return;
  }
  if(!g_atomic_int_get(&rj->done))
  {
    // keep the toast alive; control.c dedups the repeated message
    dt_control_log(_("reloading the recorded mask..."));
    return;
  }

  // join by the edit machine EXCLUSIVELY: _decode_finish never sees this
  // thread, and the publication machine is gated while REPLAYING
  if(d->edit_replay_thread)
  {
    g_thread_join(d->edit_replay_thread);
    d->edit_replay_thread = NULL;
  }
  d->edit_replay_job = NULL;

  // report the stand-in's refinement context back into the session: the
  // CascadePSP lazy-load during the replay landed in the stand-in, on
  // which the compute had exclusive rights
  d->refine = rj->od.refine;
  d->refine_failed = rj->od.refine_failed;

  // the same precondition _edit_replay_start checked, re-checked before
  // PUBLISHING: the geometry can also move while the replay runs (the
  // invalidation branch is deferred for the whole freeze). publishing then
  // would put a mask computed on the old grid on screen, clickable and
  // finalisable for the frame that separates us from the next expose
  const gboolean geometry_moved
    = _compute_distort_hash(darktable.develop) != d->encoded_distort_hash;
  if(geometry_moved)
    dt_print(DT_DEBUG_AI,
             "[object mask] edit: the geometry moved during the replay,"
             " dropping its result");

  if(rj->ok && !geometry_moved)
  {
    const dt_rf_recipe_t *recipe = &d->edit_recipe;

    g_free(d->mask);
    d->mask = rj->out_mask;
    d->mask_w = rj->out_w;
    d->mask_h = rj->out_h;
    rj->out_mask = NULL;   // ownership moved

    // restore the provenance scalars of the last replayed decode: the
    // boundary loop above ran from the recipe, not through _launch_decode,
    // so nothing recorded them -- without this a re-capture with no new
    // click would read stale zeros and the gen-2 recipe would diverge.
    // same clamps as the replay applied
    d->last_do_crf = recipe->crf_enabled != 0;
    d->last_crf_iter = CLAMP(recipe->crf_iterations, 1, 10);
    d->last_crf_sigma_color = CLAMP(recipe->crf_sigma_color, 1.0f, 50.0f);
    d->last_crf_w_bilateral = CLAMP(recipe->crf_w_bilateral, 0.5f, 30.0f);
    d->last_do_refine = recipe->ai_refine != 0;
    d->last_refine_margin = CLAMPF(recipe->ai_refine_margin, 0.0f, 0.5f);
    d->last_n_passes = CLAMP(recipe->refine_passes, 1, 3);

    // the decode marks were injected before the replay started; verify
    // they are still coherent with the session state (nothing may touch
    // them while frozen -- a mismatch here is a machine bug)
    if(!d->decode_marks || (int)d->decode_marks->len != rj->n_points
       || gui->guipoints_count != rj->n_points)
      dt_print(DT_DEBUG_ALWAYS,
               "[object mask] edit: injected state drifted during the"
               " replay (marks %d, points %d, expected %d)",
               d->decode_marks ? (int)d->decode_marks->len : -1,
               gui->guipoints_count, rj->n_points);

    // no-op baseline: the recipe this session would capture RIGHT NOW,
    // before the user touched anything. closing on a byte-identical
    // re-capture means the session produced nothing -- commit nothing
    // then. snapshotting the recipe rather than a point count is what
    // makes the criterion exhaustive: the recipe is by construction
    // everything that reaches the mask, so a parameter the user changes
    // without clicking (smoothing, cleanup, feather, refinement) is
    // covered, and so is any field a later version adds
    // the baseline is "what a capture would write RIGHT NOW", so it takes
    // the stage this SESSION runs -- the same resolution a finalisation
    // launched from it would take. the closing comparison re-resolves it
    // the same way, so a matting control moved during the session shows up
    // as a changed recipe and the session commits instead of closing as a
    // no-op
    const _matte_session_t baseline_matte = _session_matte(d);
    d->edit_baseline_valid
      = _capture_recipe(d, gui, &baseline_matte, &d->edit_baseline_recipe);
    d->edit_dirty = FALSE;

    _update_preview(d);
    if(darktable.develop->proxy.masks.module)
      darktable.develop->proxy.masks.list_change(
        darktable.develop->proxy.masks.module);

    g_atomic_int_set(&d->edit_pending, EDIT_ACTIVE);
    dt_control_log_ack_all();
    dt_control_log(_("recorded mask reloaded, refine with clicks or apply"));
  }
  else
  {
    dt_control_log_ack_all();
    dt_control_log(geometry_moved
                   ? _("could not reload the recorded mask:"
                       " the geometry changed")
                   : _("could not reload the recorded mask"));
    _edit_session_end(d, EDIT_FAILED);
    // drop the injected half-state: decode marks left alive would flow
    // into the next capture as boundaries this session never produced
    _clear_selection(gui);
    // ... and the pinned encoding with it (recorded model, recorded dims,
    // recorded cap): only then is the surviving tool really the "plain,
    // empty session" the toast promises, and only then will its own
    // captures describe the render they came from
    _edit_drop_pinned_encode(d);
  }

  g_free(rj->enc_pts);
  g_free(rj->out_mask);
  g_free(rj);

  if(d->decode_busy_shown)
  {
    dt_control_busy_leave();
    d->decode_busy_shown = FALSE;
  }
  dt_control_queue_redraw_center();
}

gboolean dt_object_mask_edit_begin(struct dt_iop_module_t *target_module,
                                   const dt_object_edit_target_t *target)
{
  // reserved for the in-place re-finalisation payload (C3); the C2 session
  // closes through the existing gestures, which create a NEW mask
  (void)target_module;

  dt_develop_t *dev = darktable.develop;
  if(!target || !target->has_recipe || !dt_rf_recipe_valid(&target->recipe))
    return FALSE;
  // a promptless recipe records no clicked session to reopen: revising
  // one is a redetect, not a decode replay -- refuse until that surface
  // exists
  if(target->recipe.prompt_kind != DT_RF_PROMPT_POINTS)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] edit: promptless recipes have no click"
             " session to reopen");
    return FALSE;
  }
  // opening a session goes through dt_masks_change_form_gui, which tears
  // down whatever session is in flight -- same guard as every other
  // creation entry point
  if(dt_masks_shapes_locked())
  {
    dt_control_log(_("mask still computing, try again in a moment"));
    return FALSE;
  }
  // C2 delivers family 1 (raster masks, recipe in the rasterfile params);
  // the paths and context families arrive with C4/C5 on this same entry
  if(target->kind != DT_OBJECT_EDIT_RASTER)
  {
    dt_print(DT_DEBUG_AI,
             "[object mask] edit: family %d not implemented yet",
             target->kind);
    return FALSE;
  }
  if(!dev || dt_view_get_current() != DT_VIEW_DARKROOM || !dev->form_gui
     || !dt_is_valid_imgid(dev->image_storage.id))
    return FALSE;
  if(!dt_masks_object_available())
  {
    dt_control_log(_("AI model is not available. Check preferences > AI"));
    return FALSE;
  }
  if(_edit_any_active())
  {
    dt_control_log(_("an AI mask edit session is already open"));
    return FALSE;
  }

  // the matting stage, asked BEFORE the model gap for the reason the
  // replay gate and the headless toast ask it first: it is a build
  // capability, and the model verdict below would answer "models check
  // out" while the session opened on a mask this build cannot re-finalise
  // the way it was made. two answers, one question:
  //  - unreproducible: refuse. the session would replay the recorded
  //    decodes faithfully and then finalise WITHOUT the recorded stage,
  //    handing the user a starting point that is not the mask on screen;
  //  - reproducible: open, and SEED the session with the recorded stage
  //    (see the pre-seed below), exactly as every other recorded parameter
  //    is seeded. that is what this session used to be unable to do, and
  //    what the warning it printed instead was standing in for
  if(!_matte_op_recorded(&target->recipe) && target->recipe.matting_enabled)
  {
    // untrusted bytes, terminated before anything formats them
    char id[DT_RF_RECIPE_MATTING_ID_LEN];
    memcpy(id, target->recipe.matting_id, sizeof(id));
    id[sizeof(id) - 1] = '\0';
    dt_control_log(_("cannot edit this mask: it records the matting stage"
                     " '%s', which this darktable cannot reproduce"),
                   id);
    return FALSE;
  }

  // the model-gap verdict is taken AT OPENING and never stored: installs,
  // rebinds and edits all change it under our feet. provisional C2 policy:
  // only OK opens -- the recorded models must be installed identically for
  // the session replay to be faithful. the INSTALLABLE dialogue of the
  // plan ("install the model" / "edit with the current model, approximate
  // starting point") lands with C6 once the download modal (D2) exists
  {
    dt_object_recipe_model_gap_t seg_gap, refine_gap, matte_gap;
    const dt_object_recipe_model_gap_t gap
      = _recipe_model_gap(&target->recipe, NULL, &seg_gap, &refine_gap,
                          &matte_gap);
    if(gap != DT_OBJECT_RECIPE_MODELS_OK)
    {
      // WHICH model the combined verdict came from. three causes now feed
      // that verdict, so a two-way selector necessarily misnames one of
      // them: on a matting gap `seg_gap != gap` used to elect the refine
      // model, which is a sane model on that machine -- or the empty
      // string when the recipe records no refinement at all. priority is
      // the ranking's own and unchanged for the first two: seg wins ties,
      // being the model every recipe records, then refine
      const gboolean seg_at_fault = seg_gap == gap;
      if(gap == DT_OBJECT_RECIPE_MODELS_AI_OFF)
        dt_control_log(_("cannot edit this mask: AI processing is disabled"));
#ifdef HAVE_MATTE_VITMATTE
      else if(!seg_at_fault && refine_gap != gap
              && matte_gap == gap && _matte_op_recorded(&target->recipe))
        // its OWN words: no version is recorded for this one (the recipe
        // pins the operator's algorithm revision, never a model version),
        // and no download catalogue carries these weights, so neither the
        // "does not match the recorded state" wording nor the AI models
        // preferences would tell the truth here
        dt_control_log(_("cannot edit this mask: its matting stage needs"
                         " the model '%s', which is not installed on this"
                         " machine"),
                       _matte_op_recorded(&target->recipe)->model);
#endif
      else
        dt_control_log(_("cannot edit this mask: model '%s' does not match"
                         " the recorded state (see the AI models"
                         " preferences, or use 'recompute mask')"),
                       seg_at_fault ? target->recipe.seg_model
                                    : target->recipe.refine_model);
      return FALSE;
    }
  }

  // in the nominal repair flow the mask file is missing for the whole
  // session: gate the proactive recompute of rasterfile.c gui_changed
  // before anything can trigger it, or every gui_update would race a
  // second inference stack against this session. cleared on EVERY session
  // exit by _edit_session_end
  dt_object_mask_edit_set_active("rasterfile", target->raster_multi_priority);

  // drop the module focus BEFORE entering the session: request_focus tears
  // the mask view down when a module loses focus, which would destroy the
  // session created below. side effect wanted for C2: with no focused
  // module the closing gesture wires the produced mask to no sink instead
  // of blending it into the rasterfile instance that happened to hold
  // focus (C3 carries the real target through the payload instead)
  dt_iop_request_focus(NULL);

  // enter the session by the masks panel's own add-shape pattern
  dt_masks_form_t *spot = dt_masks_create(DT_MASKS_OBJECT);
  dt_masks_change_form_gui(spot);
  dt_masks_form_gui_t *gui = dev->form_gui;
  gui->creation_module = NULL;
  gui->group_selected = 0;
  gui->edit_mode = DT_MASKS_EDIT_FULL;

  // pre-seed the session data the first post_expose would otherwise
  // create: the encode must start PINNED (recorded model, recorded dims),
  // and the preview parameters must restore the recipe's recorded values
  // -- a re-capture with no new click must reproduce them. deliberately
  // no persistent-model restore: the slot holds the ACTIVE model, the
  // session needs the recorded one loaded by id
  _object_data_t *d = g_new0(_object_data_t, 1);
  d->preview_cleanup = CLAMP(target->recipe.cleanup, 0, 100);
  d->preview_smoothing = CLAMPF(target->recipe.smoothing, 0.0f, 1.3f);
  d->preview_feather = CLAMPF(target->recipe.feather, 0.0f, 1.0f);
  d->preview_refine = target->recipe.crf_enabled != 0;
  d->edit_recipe = target->recipe;
  // the session's parameters, all of them, from the recipe: read back
  // through the _session_* accessors, which are the only readers of the
  // corresponding preferences. the clamps are the ones _launch_decode
  // applied when these values were recorded, and shield a hand-edited
  // recipe; the accessors re-apply them, so a raw copy would do too
  d->edit_threshold = CLAMP(target->recipe.threshold, 0.3f, 0.9f);
  d->edit_n_passes = CLAMP(target->recipe.refine_passes, 1, 3);
  d->edit_crf_iter = CLAMP(target->recipe.crf_iterations, 1, 10);
  d->edit_crf_sigma_color = CLAMP(target->recipe.crf_sigma_color, 1.0f, 50.0f);
  d->edit_crf_w_bilateral
    = CLAMP(target->recipe.crf_w_bilateral, 0.5f, 30.0f);
  d->edit_do_refine = target->recipe.ai_refine != 0;
  d->edit_refine_margin = CLAMPF(target->recipe.ai_refine_margin, 0.0f, 0.5f);
  // raw, like the recipe stores it: the MAX belongs to the use points
  d->edit_render_size = target->recipe.render_size;
  // the matting stage, through the SAME resolver the headless replay uses,
  // so an edit session and a regeneration of the same mask start from one
  // reading of the recipe. it answers an off stage for a recipe that
  // records none -- which is itself the recorded state and must be seeded
  // too, or reopening a plain mask would silently acquire whatever stage
  // the preferences name today. the refusal above has already rejected
  // every recipe this could fail to resolve
  {
    const _matte_session_t rec = _matte_session_recorded(&target->recipe);
    d->edit_matte_seeded = TRUE;
    d->edit_matte_enabled = rec.enabled;
    d->edit_matte_op = rec.op;
    d->edit_matte_band = rec.band_scale;
  }
  d->edit_model_pinned = TRUE;
  // the focus was dropped just above, so this session has no sink. carried
  // by the session because dev->gui_module can change under it: unfolding
  // a module mid-session refocuses from NULL, which skips the focus-loss
  // teardown and leaves the session alive with a module in hand
  d->edit_no_sink = TRUE;
  d->edit_valid = TRUE;
  g_atomic_int_set(&d->edit_pending, EDIT_WAIT_ENCODE);
  gui->scratchpad = d;
  gui->scratchpad_cleanup = _free_data;

  dt_print(DT_DEBUG_AI,
           "[object mask] edit: session opened on rasterfile instance %d"
           " (%d recorded points)",
           target->raster_multi_priority, target->recipe.n_points);
  dt_control_queue_redraw_center();
  return TRUE;
}

// ---------------------------- recompute scheduling --------------------------
//
// the anti-respawn table: one entry per (recipe, image) currently being
// recomputed, or having failed deterministically this session. its key is a
// local hash of the recipe blob and the image id -- cheaper than the file
// fingerprint (no image-cache access) and just as unique for this purpose.
// entries are claimed BEFORE any compute starts, so the pipes the replay
// itself runs (which traverse the very rasterfile instance being
// regenerated) cannot re-enter

typedef enum _recompute_state_t
{
  _RECOMPUTE_RUNNING = 1,
  _RECOMPUTE_FAILED = 2,   // deterministic: no automatic retry this session
} _recompute_state_t;

static GMutex _recompute_mutex;
static GHashTable *_recompute_table = NULL;   // gint64* key -> state

static gint64 _recompute_key(const dt_rf_recipe_t *recipe,
                             const dt_imgid_t imgid)
{
  dt_hash_t key = dt_hash(DT_INITHASH, recipe, sizeof(*recipe));
  key = dt_hash(key, &imgid, sizeof(imgid));
  return (gint64)key;
}

// claim the slot; FALSE when a recompute is running or has failed for good
static gboolean _recompute_claim(const gint64 key)
{
  g_mutex_lock(&_recompute_mutex);
  if(!_recompute_table)
    _recompute_table
      = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, NULL);
  if(g_hash_table_lookup(_recompute_table, &key))
  {
    g_mutex_unlock(&_recompute_mutex);
    return FALSE;
  }
  gint64 *k = g_new(gint64, 1);
  *k = key;
  g_hash_table_insert(_recompute_table, k,
                      GINT_TO_POINTER(_RECOMPUTE_RUNNING));
  g_mutex_unlock(&_recompute_mutex);
  return TRUE;
}

// settle a claimed slot: only a deterministic failure pins it -- success
// (the file exists) and transient outcomes (busy, cancelled, never ran)
// free it for a later attempt
static void _recompute_settle(const gint64 key,
                              const dt_object_recipe_status_t status)
{
  g_mutex_lock(&_recompute_mutex);
  if(_recompute_table)
  {
    if(status == DT_OBJECT_RECIPE_FAILED)
    {
      gint64 *k = g_new(gint64, 1);
      *k = key;
      g_hash_table_replace(_recompute_table, k,
                           GINT_TO_POINTER(_RECOMPUTE_FAILED));
    }
    else
      g_hash_table_remove(_recompute_table, &key);
  }
  g_mutex_unlock(&_recompute_mutex);
}

static gboolean _recompute_entry_failed(gpointer key,
                                        gpointer value,
                                        gpointer user_data)
{
  (void)key;
  (void)user_data;
  return GPOINTER_TO_INT(value) == _RECOMPUTE_FAILED;
}

void dt_object_recipe_reset_failed(void)
{
  guint cleared = 0;
  g_mutex_lock(&_recompute_mutex);
  if(_recompute_table)
    cleared = g_hash_table_foreach_remove(_recompute_table,
                                          _recompute_entry_failed, NULL);
  g_mutex_unlock(&_recompute_mutex);
  if(cleared)
    dt_print(DT_DEBUG_AI,
             "[object mask] recompute table: %u pinned failure%s cleared",
             cleared, cleared > 1 ? "s" : "");
}

typedef struct _recompute_job_t
{
  dt_rf_recipe_t recipe;
  dt_imgid_t imgid;
  gint64 key;
  gboolean ran;
  dt_object_recipe_status_t status;
} _recompute_job_t;

// GUI thread: a recompute landed for the displayed image. the failed reads
// left the raster module's caches unhashed (auto-repair), so fresh pipe
// runs reread the file; the thumbnails rendered with a zeroed mask in the
// meantime are dropped from the mipmap cache
static gboolean _recompute_landed_idle(gpointer data)
{
  const dt_imgid_t imgid = GPOINTER_TO_INT(data);

  dt_mipmap_cache_remove(imgid);

  // test the view before touching dev: leaving the darkroom can free the
  // develop while this idle is already queued
  dt_develop_t *dev = darktable.develop;
  if(dt_view_get_current() != DT_VIEW_DARKROOM
     || !dev || dev->image_storage.id != imgid)
    return G_SOURCE_REMOVE;

  // the image is about to change under the user's eyes, and he did not ask
  // for it: say what happened. emitted HERE and not from the job thread on
  // purpose -- past the two tests above, this is exactly the moment the
  // visible image is replaced. symmetric with the failure toasts below,
  // which say why it did NOT change
  dt_control_log_ack_all();
  dt_control_log(_("AI mask recomputed"));

  // the masks panel may carry a missing-file badge on a raster shape
  // row: rebuild the list now that the file is back, or the badge would
  // outlive the repair until some unrelated masks interaction
  dt_dev_masks_list_change(dev);

  // all pipes, not only the center: the preview rendered a zeroed mask too
  dt_dev_reprocess_all(dev);
  return G_SOURCE_REMOVE;
}

// the headless recompute. see _job_step: the message is the very one the
// scheduling emitted, so the toast the user already reads stays alive for
// the whole job instead of expiring after five seconds
static gboolean _recompute_keep_going(void *p)
{
  return _job_step((dt_job_t *)p, _("recomputing the AI mask..."));
}

static int32_t _recompute_job_run(dt_job_t *job)
{
  _recompute_job_t *p = dt_control_job_get_params(job);
  p->ran = TRUE;
  p->status = dt_object_recipe_compute(&p->recipe, p->imgid,
                                       _recompute_keep_going, job);
  if(p->status == DT_OBJECT_RECIPE_OK && dt_control_running())
    g_idle_add(_recompute_landed_idle, GINT_TO_POINTER(p->imgid));
  else if(p->status == DT_OBJECT_RECIPE_FAILED && dt_control_running()
          && p->recipe.matting_enabled && !_matte_op_recorded(&p->recipe))
  {
    // diagnosed FIRST and on its own, ahead of the model gap: the matting
    // stage is a build capability, so the verdict below would answer
    // "models check out" and send the user to the trace for a cause that
    // reads plainly here.
    // the recorded strings are untrusted bytes: copied into a terminated
    // buffer before anything formats them -- or looks them up
    char id[DT_RF_RECIPE_MATTING_ID_LEN];
    char ver[DT_RF_RECIPE_MATTING_VERSION_LEN];
    memcpy(id, p->recipe.matting_id, sizeof(id));
    memcpy(ver, p->recipe.matting_version, sizeof(ver));
    id[sizeof(id) - 1] = '\0';
    ver[sizeof(ver) - 1] = '\0';
    // ONE gate, two very different causes, and the message may not merge
    // them: an id no line carries is FINAL -- rebinding cannot bring an
    // operator this build does not have -- while the same id at another
    // revision is exactly what the matting branch of
    // dt_object_recipe_rebind_models repins, so 'recompute mask' really
    // does repair it and the message must say so. the condition mirrors
    // that branch's own acceptance (extension version, a line the table
    // carries); anything it would refuse is stated as the fact it is
    const dt_matte_op_t *const carried
      = (p->recipe.version == DT_RF_RECIPE_VERSION_EXT) ? dt_matte_find(id)
                                                        : NULL;
    if(carried)
      dt_control_log(_("AI mask not regenerated: it records revision v%s of"
                       " the matting stage '%s', this darktable carries"
                       " v%s.\nuse 'recompute mask' in the raster masks"
                       " module to redo it with the revision this build"
                       " has"),
                     ver, id, carried->version);
    else
      dt_control_log(_("AI mask not regenerated: it records the matting"
                       " stage '%s' (v%s), which this darktable cannot"
                       " reproduce.\nthe mask stays as it is until a build"
                       " carrying that stage opens this library"),
                     id, ver);
  }
  else if(p->status == DT_OBJECT_RECIPE_FAILED && dt_control_running())
  {
    // the mask stays zeroed and the user needs to know WHY, and the way
    // out: diagnose the model gap the replay refused on and name both
    // the cause and the actionable place. 'recompute mask' in the raster
    // masks module rebinds the recipe to the models installed NOW
    dt_object_recipe_model_gap_t seg_gap, refine_gap, matte_gap;
    const dt_object_recipe_model_gap_t gap
      = _recipe_model_gap(&p->recipe, NULL, &seg_gap, &refine_gap,
                          &matte_gap);
    // the model the combined verdict came from; the segmentation model
    // wins ties, being the one every recipe records, then the refinement
    // one. THREE causes feed the verdict, so the two-way selector below
    // needs the third one taken out first: on a matting gap it would
    // elect the refine model -- a model that is perfectly installed, or
    // the empty string when the recipe records no refinement at all
    const gboolean seg_at_fault = seg_gap == gap;
#ifdef HAVE_MATTE_VITMATTE
    if(!seg_at_fault && refine_gap != gap && matte_gap == gap
       && _matte_op_recorded(&p->recipe))
    {
      // answered HERE and in its own words. the recorded/installed
      // vocabulary below has nothing to fill in for this cause: a recipe
      // pins the OPERATOR's algorithm revision, never a version of the
      // model it loads, so there is no recorded number to compare. and
      // 'recompute mask' is not the way out either -- a rebind repins the
      // revision and keeps the stage, so the very same model would be
      // missing on the next attempt
      const char *const model = _matte_op_recorded(&p->recipe)->model;
      if(gap == DT_OBJECT_RECIPE_MODELS_INSTALLABLE)
        dt_control_log(_("AI mask not regenerated: its matting stage needs"
                         " the model '%s', which is not installed.\ninstall"
                         " it in the AI models preferences"),
                       model);
      else
        // UNKNOWN, which for these weights is the normal state and not an
        // accident: no catalogue offers them, so pointing at a download
        // -- or at a .dtmodel file to install -- would be a dead end
        dt_control_log(_("AI mask not regenerated: its matting stage needs"
                         " the model '%s', which this machine does not"
                         " have.\nno download provides it: install it into"
                         " your darktable models directory yourself, or"
                         " redo the selection with the matting stage"
                         " switched off"),
                       model);
      // this branch only runs on DT_OBJECT_RECIPE_FAILED
      return 1;
    }
#endif
    const char *id = seg_at_fault ? p->recipe.seg_model
                                  : p->recipe.refine_model;
    const char *recorded = seg_at_fault
      ? p->recipe.seg_model_version
      : p->recipe.refine_model_version;
    const char *installed = dt_ai_model_get_version(id);
    switch(gap)
    {
      case DT_OBJECT_RECIPE_MODELS_INSTALLABLE:
        dt_control_log(_("AI mask not regenerated: model '%s' (v%s) is"
                         " not installed.\ninstall it in the AI models"
                         " preferences, or use 'recompute mask' in the"
                         " raster masks module to redo it with the"
                         " current model"),
                       id, recorded);
        break;
      case DT_OBJECT_RECIPE_MODELS_DRIFT_BEHIND:
        dt_control_log(_("AI mask not regenerated: it records model '%s'"
                         " v%s, v%s is installed.\nupdating the model may"
                         " restore the exact mask; 'recompute mask' in"
                         " the raster masks module redoes it with the"
                         " current model"),
                       id, recorded, installed);
        break;
      case DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD:
        if(!g_strcmp0(installed, recorded))
          // the refine edge case: the recorded model is installed at the
          // recorded version but another model is the active one
          dt_control_log(_("AI mask not regenerated: model '%s' is"
                           " installed but not active.\nactivate it in"
                           " the AI models preferences, or use 'recompute"
                           " mask' in the raster masks module to redo it"
                           " with the current model"),
                         id);
        else
          dt_control_log(_("AI mask not regenerated: it records model"
                           " '%s' v%s, the newer v%s is installed.\nuse"
                           " 'recompute mask' in the raster masks module"
                           " to redo it with the installed model"),
                         id, recorded, installed);
        break;
      case DT_OBJECT_RECIPE_MODELS_UNKNOWN:
        dt_control_log(_("AI mask not regenerated: model '%s' is not"
                         " known to this darktable.\ninstall its .dtmodel"
                         " file, or use 'recompute mask' in the raster"
                         " masks module to redo it with the current"
                         " model"),
                       id);
        break;
      case DT_OBJECT_RECIPE_MODELS_AI_OFF:
        dt_control_log(_("AI mask not regenerated: AI processing is"
                         " disabled.\nenable it in the processing"
                         " preferences"));
        break;
      default:
        // models check out: the failure has another, rarer cause
        // (unusable recipe, render failure) -- point at the trace
        dt_control_log(_("AI mask not regenerated (run with -d ai for"
                         " details).\nuse 'recompute mask' in the raster"
                         " masks module to redo it"));
        break;
    }
  }
  return p->status == DT_OBJECT_RECIPE_OK ? 0 : 1;
}

// runs on every outcome the job system takes charge of -- finished,
// cancelled, replaced, discarded. a job still sitting in the queue when
// the process exits never reaches it: its params and table entry leak
// once, harmlessly, with the process
static void _recompute_job_destroy(void *data)
{
  _recompute_job_t *p = data;
  // a job cancelled before running keeps status FAILED from init: treat
  // not-ran as transient so a later trigger may claim the slot again
  _recompute_settle(p->key, p->ran ? p->status : DT_OBJECT_RECIPE_RETRY);
  g_free(p);
}

gboolean dt_object_recipe_schedule_recompute(const dt_rf_recipe_t *recipe,
                                             const dt_imgid_t imgid)
{
  if(!dt_rf_recipe_valid(recipe) || !dt_is_valid_imgid(imgid))
    return FALSE;
  // asynchronous by contract: without a running job system (darktable-cli,
  // GUI teardown) there is nobody to run the job -- callers needing the
  // file synchronously use dt_object_recipe_recompute_now instead
  if(!dt_control_running())
    return FALSE;

  const gint64 key = _recompute_key(recipe, imgid);
  if(!_recompute_claim(key))
    return FALSE;

  _recompute_job_t *p = g_new0(_recompute_job_t, 1);
  p->recipe = *recipe;
  p->imgid = imgid;
  p->key = key;
  p->status = DT_OBJECT_RECIPE_FAILED;

  dt_job_t *job = dt_control_job_create(_recompute_job_run,
                                        "AI mask recompute");
  if(!job)
  {
    _recompute_settle(key, DT_OBJECT_RECIPE_RETRY);
    g_free(p);
    return FALSE;
  }
  dt_control_job_set_params(job, p, _recompute_job_destroy);
  dt_control_job_add_progress(job, _("recomputing AI mask"), TRUE);
  // announce BEFORE queueing: a worker can pick the job up immediately and
  // fail with a diagnosis, and that diagnosis must not be overwritten by
  // this announcement arriving late.
  // announce it HERE, at the single point the three triggers funnel
  // through (the pixelpipe safety net and the proactive gui_update of
  // iop/rasterfile.c, plus its explicit button): a recompute the user
  // never asked for takes tens of seconds and used to be completely
  // silent, its only trace an entry at the bottom of the LEFT panel --
  // out of the darkroom's field of work and often folded away.
  // dt_control_log is thread-safe, so the pixelpipe trigger may use it
  dt_control_log(_("recomputing the AI mask..."));
  return TRUE;
}

gboolean dt_object_recipe_rebind_models(dt_rf_recipe_t *recipe)
{
  if(!dt_rf_recipe_valid(recipe))
    return FALSE;

  // the seg_model slot carries the interactive segmentation model of a
  // clicked recipe and the recorded detector of a promptless one: rebind
  // from the active model of the family's OWN task, or a subject recipe
  // would silently be rebound to the click-session SAM model
  const char *seg_task = "mask";
  if(recipe->prompt_kind != DT_RF_PROMPT_POINTS)
  {
    const dt_detector_t *detector
      = dt_detector_find(recipe->prompt_kind, recipe->class_bits);
    if(!detector)
      return FALSE;
    seg_task = detector->task;
  }

  char *seg_id = dt_ai_models_get_active_for_task(seg_task);
  if(!seg_id || !*seg_id)
  {
    g_free(seg_id);
    return FALSE;
  }

  if(recipe->prompt_kind != DT_RF_PROMPT_POINTS)
  {
    // the input side travels with the model: repin it to the new model's
    // manifest, or the rebound recipe would fail the side gate of its
    // own replay
    dt_ai_environment_t *env = dt_ai_registry_get_env();
    const dt_ai_model_info_t *info
      = env ? dt_ai_get_model_info_by_id(env, seg_id) : NULL;
    int n = 0;
    int *sizes
      = info ? dt_ai_model_attribute_int_array(info, "input_sizes", &n)
             : NULL;
    const int side = (sizes && n > 0) ? sizes[0] : 0;
    g_free(sizes);
    if(side <= 0)
    {
      g_free(seg_id);
      return FALSE;
    }
    recipe->detect_input = side;
  }

  memset(recipe->seg_model, 0, sizeof(recipe->seg_model));
  memset(recipe->seg_model_version, 0, sizeof(recipe->seg_model_version));
  g_strlcpy(recipe->seg_model, seg_id, sizeof(recipe->seg_model));
  const char *seg_ver = dt_ai_model_get_version(seg_id);
  if(seg_ver)
    g_strlcpy(recipe->seg_model_version, seg_ver,
              sizeof(recipe->seg_model_version));
  g_free(seg_id);

  if(recipe->ai_refine)
  {
    char *refine_id = dt_ai_models_get_active_for_task("refine");
    if(!refine_id || !*refine_id)
    {
      // the recipe promises contour refinement and no refine model is
      // active: rebinding would drop a recorded processing step
      g_free(refine_id);
      return FALSE;
    }
    memset(recipe->refine_model, 0, sizeof(recipe->refine_model));
    memset(recipe->refine_model_version, 0,
           sizeof(recipe->refine_model_version));
    g_strlcpy(recipe->refine_model, refine_id, sizeof(recipe->refine_model));
    const char *refine_ver = dt_ai_model_get_version(refine_id);
    if(refine_ver)
      g_strlcpy(recipe->refine_model_version, refine_ver,
                sizeof(recipe->refine_model_version));
    g_free(refine_id);
  }

  // the matting stage is a build capability, not a model: no install and
  // no activation can bring an operator this build does not carry, so an
  // id the table does not know makes the whole rebind FAIL -- "redo it
  // with what is here now" has no answer, and dropping the recorded stage
  // to make the rebind succeed would hand the replay a recipe describing a
  // render nobody asked for. an id that IS carried gets its revision
  // repinned to the build's own, which is exactly what rebinding means and
  // is the only reason this branch writes anything at all
  if(recipe->matting_enabled)
  {
    // the extension block is only meaningful at the extension version: a
    // version-1 blob announcing the stage is a forgery every other reader
    // of the block already refuses (_matte_op_recorded, and
    // dt_rf_recipe_valid for the promptless kinds). rebinding it would
    // mint a recipe no version-1-only build replays the same way
    if(recipe->version != DT_RF_RECIPE_VERSION_EXT)
      return FALSE;
    if(recipe->matting_id[DT_RF_RECIPE_MATTING_ID_LEN - 1] != '\0')
      return FALSE;
    const dt_matte_op_t *const op = dt_matte_find(recipe->matting_id);
    if(!op)
      return FALSE;
    // the id itself is kept -- it names the very line just found -- but
    // whatever follows its NUL is zeroed. those bytes are invisible to
    // every strcmp and fully visible to the verbatim hash that names the
    // cache file, so a blob carrying junk there would leave the rebound
    // recipe under a fingerprint no writer of this build can ever produce
    // again. the seg and refine ids get the same guarantee from the plain
    // memsets above, which can afford to clear the whole field because
    // they overwrite it; this one clears the tail because it must not
    const size_t idlen = strlen(recipe->matting_id);
    memset(recipe->matting_id + idlen, 0,
           sizeof(recipe->matting_id) - idlen);
    memset(recipe->matting_version, 0, sizeof(recipe->matting_version));
    if(g_strlcpy(recipe->matting_version, op->version,
                 sizeof(recipe->matting_version))
       >= sizeof(recipe->matting_version))
    {
      // a revision that does not fit would be truncated into another
      // one: leave the field zeroed and refuse, never half-written
      memset(recipe->matting_version, 0, sizeof(recipe->matting_version));
      return FALSE;
    }
  }
  return TRUE;
}

gboolean dt_object_recipe_recompute_now(const dt_rf_recipe_t *recipe,
                                        const dt_imgid_t imgid)
{
  if(!dt_rf_recipe_valid(recipe) || !dt_is_valid_imgid(imgid))
    return FALSE;

  const gint64 key = _recompute_key(recipe, imgid);
  if(!_recompute_claim(key))
    return FALSE;

  const dt_object_recipe_status_t status
    = dt_object_recipe_compute(recipe, imgid, NULL, NULL);
  _recompute_settle(key, status);
  return status == DT_OBJECT_RECIPE_OK;
}

// transform mask-space forms to input-normalized coords and register them,
// takes ownership of `forms` and `signs` lists (forms are appended to dev->forms).
// `gui` carries the live session whose provenance recipe is captured into
// the produced group's trailer; NULL skips the capture
static dt_masks_form_t *
_register_vectorized_forms(dt_iop_module_t *module,
                           dt_masks_form_gui_t *gui,
                           GList *forms,
                           GList *signs,
                           const int mask_w,
                           const int mask_h)
{
  (void)module;

  // darktable mask coordinates are stored in input-image-normalized space:
  //   coord = backtransform(backbuf_pixel) / iwidth
  // this undoes all geometric pipeline transforms (crop, rotation, lens, etc.)
  // so that the mask can be applied at any point in the pipeline
  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // vectorized coordinates are in mask space (encoding resolution),
  // dt_dev_distort_backtransform expects preview pipe pixel space
  const float msx = (mask_w > 0) ? wd / (float)mask_w : 1.0f;
  const float msy = (mask_h > 0) ? ht / (float)mask_h : 1.0f;

  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    const int npts = g_list_length(f->points);
    if(npts == 0)
      continue;

    // collect all coordinates into a flat array for batch backtransform,
    // each path point has 3 coordinate pairs: corner, ctrl1, ctrl2
    float *pts = g_new(float, npts * 6);
    int i = 0;
    for(GList *p = f->points; p; p = g_list_next(p))
    {
      dt_masks_point_path_t *pt = p->data;
      pts[i++] = pt->corner[0];
      pts[i++] = pt->corner[1];
      pts[i++] = pt->ctrl1[0];
      pts[i++] = pt->ctrl1[1];
      pts[i++] = pt->ctrl2[0];
      pts[i++] = pt->ctrl2[1];
    }

    // scale from mask space (encoding resolution) to preview pipe space
    for(int j = 0; j < npts * 6; j += 2)
    {
      pts[j + 0] *= msx;
      pts[j + 1] *= msy;
    }

    dt_dev_distort_backtransform(darktable.develop, pts, npts * 3);

    // write back and normalize by input image dimensions
    i = 0;
    for(GList *p = f->points; p; p = g_list_next(p))
    {
      dt_masks_point_path_t *pt = p->data;
      pt->corner[0] = pts[i++] / iwidth;
      pt->corner[1] = pts[i++] / iheight;
      pt->ctrl1[0] = pts[i++] / iwidth;
      pt->ctrl1[1] = pts[i++] / iheight;
      pt->ctrl2[0] = pts[i++] / iwidth;
      pt->ctrl2[1] = pts[i++] / iheight;
    }
    g_free(pts);
  }

  const int nbform = g_list_length(forms);
  if(nbform == 0)
  {
    g_list_free_full(forms, (GDestroyNotify)dt_masks_free_form);
    g_list_free(signs);
    dt_control_log(_("no mask extracted from AI segmentation"));
    return NULL;
  }

  // always wrap paths in a group; holes use difference mode

  // count existing AI object groups/paths for numbering
  dt_develop_t *dev = darktable.develop;
  const char *group_prefix = _("ai object group");
  const char *path_prefix = _("ai object");

  guint grp_nb = 0;
  guint path_nb = 0;
  for(GList *l = dev->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    if(strncmp(f->name, group_prefix, strlen(group_prefix)) == 0)
      grp_nb++;
    if(strncmp(f->name, path_prefix, strlen(path_prefix)) == 0)
      path_nb++;
  }
  grp_nb++;
  path_nb++;
  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    snprintf(f->name, sizeof(f->name),
             "%s #%d", path_prefix, (int)path_nb++);
  }

  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  snprintf(grp->name, sizeof(grp->name), "%s #%d", group_prefix, (int)grp_nb);

  // form ids come from a session counter reset at startup: collisions with
  // loaded forms (or between siblings) are possible, and the provenance
  // trailer resolves children BY id -- ensure unicity before registration,
  // like the native route does
  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    gboolean clash = TRUE;
    while(clash)
    {
      clash = dt_masks_get_from_id(dev, f->formid) != NULL;
      for(GList *k = forms; !clash && k != l; k = g_list_next(k))
        clash = ((dt_masks_form_t *)k->data)->formid == f->formid;
      if(clash) f->formid++;
    }
  }

  // register all path forms so they exist in dev->forms
  for(GList *l = forms; l; l = g_list_next(l))
  {
    dt_masks_form_t *f = l->data;
    dev->forms = g_list_append(dev->forms, f);
  }

  // add each path to the group; holes get difference mode
  GList *s = signs;
  for(GList *l = forms; l; l = g_list_next(l), s = s ? g_list_next(s) : NULL)
  {
    dt_masks_form_t *f = l->data;
    const int sign = s ? GPOINTER_TO_INT(s->data) : '+';
    dt_masks_point_group_t *grpt = dt_masks_group_add_form(grp, f);
    if(grpt && sign == '-')
    {
      grpt->state = (grpt->state & ~DT_MASKS_STATE_UNION) | DT_MASKS_STATE_DIFFERENCE;
    }
  }

  // register the group (history item added by caller after blend mask
  // assignment)
  dev->forms = g_list_append(dev->forms, grp);

  // stamp the provenance trailer on the fully assembled group: this
  // classic route finalises a prompt session too, so its group is made
  // re-editable by ai exactly like the native route's groups. a session
  // the recipe cannot describe simply leaves the trailer zeroed
  _object_data_t *sd = gui ? _get_data(gui) : NULL;
  dt_rf_recipe_t recipe;
  // NULL stage, and it is not an omission: this classic route traces the
  // working-grid mask straight into paths -- it never enters
  // _finalize_render_alpha, so no band and no matting operator ran, and a
  // recipe claiming one would describe a render nobody made
  if(sd && _capture_recipe(sd, gui, NULL, &recipe))
    _ai_trailer_stamp(dev->forms, grp, &recipe);

  g_list_free(forms);
  g_list_free(signs);

  dt_print(DT_DEBUG_AI, "[object mask] created %d paths", nbform);
  return grp;
}

// finalize using cached preview forms (steals ownership from scratchpad)
static dt_masks_form_t *
_finalize_from_preview(dt_iop_module_t *module, dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d || !d->preview_forms)
    return NULL;

  GList *forms = d->preview_forms;
  GList *signs = d->preview_signs;
  const int mw = d->mask_w;
  const int mh = d->mask_h;
  d->preview_forms = NULL;
  d->preview_signs = NULL;

  return _register_vectorized_forms(module, gui, forms, signs, mw, mh);
}

// finalize: vectorize the mask and register as a group of path forms,
// fallback when no preview forms are available
static dt_masks_form_t *_finalize_mask(dt_iop_module_t *module,
                                       dt_masks_form_t *form,
                                       dt_masks_form_gui_t *gui)
{
  (void)form;
  _object_data_t *d = _get_data(gui);
  if(!d || !d->mask)
    return NULL;

  const size_t n = (size_t)d->mask_w * d->mask_h;
  float *inv_mask = g_try_malloc(n * sizeof(float));
  if(!inv_mask)
    return NULL;

  for(size_t i = 0; i < n; i++)
    inv_mask[i] = 1.0f - d->mask[i];

  // the session's own vectorisation parameters, not the conf: they are
  // what the preview showed and what _capture_recipe records, and in an
  // edit session they come from the recipe. outside a session the two
  // agree anyway (the mutators keep preview_* and conf in step)
  const int cleanup = d->preview_cleanup;
  const float smoothing = d->preview_smoothing;
  const float thresh = 1.0f - _session_threshold(d);
  GList *signs = NULL;
  GList *forms = ras2forms(inv_mask, d->mask_w, d->mask_h, NULL,
                           thresh, cleanup, (double)smoothing, 0.3, &signs);
  g_free(inv_mask);

  return _register_vectorized_forms(module, gui, forms, signs,
                                    d->mask_w, d->mask_h);
}

// --- mask event handlers ---

static int _object_events_mouse_scrolled(dt_iop_module_t *module,
                                         const float pzx,
                                         const float pzy,
                                         const gboolean up,
                                         const uint32_t state,
                                         dt_masks_form_t *form,
                                         const dt_imgid_t parentid,
                                         dt_masks_form_gui_t *gui,
                                         const int index)
{
  _object_data_t *d = _get_data(gui);

  // vectorization parameter adjustment (after first click). the explicit
  // freeze test is the contract, not the d->mask NULL that happens to
  // stand in for it while the machine replays
  if(gui->creation && d && d->has_selection && d->mask && !_edit_frozen(d))
  {
    if(dt_modifier_is(state, 0))
    {
      // plain scroll: adjust smoothing (potrace alphamax)
      d->preview_smoothing = CLAMP(d->preview_smoothing + (up ? 0.05f : -0.05f),
                                   0.0f, 1.3f);
      // an edit session works on the RECIPE's parameters: writing them
      // back would destroy the user's global preference, the rule
      // _session_threshold states for the threshold
      if(!d->edit_valid)
        dt_conf_set_float(CONF_OBJECT_SMOOTHING_KEY, d->preview_smoothing);
      _update_preview(d);
      dt_toast_log(_("smoothing: %3.2f"), d->preview_smoothing);
      dt_dev_masks_list_change(darktable.develop);
      dt_control_queue_redraw_center();
      return 1;
    }
    if(dt_modifier_is(state, GDK_SHIFT_MASK))
    {
      // shift+scroll: adjust cleanup (potrace turdsize)
      d->preview_cleanup = CLAMP(d->preview_cleanup + (up ? 5 : -5), 0, 100);
      if(!d->edit_valid)
        dt_conf_set_int(CONF_OBJECT_CLEANUP_KEY, d->preview_cleanup);
      _update_preview(d);
      dt_toast_log(_("cleanup: %d"), d->preview_cleanup);
      dt_dev_masks_list_change(darktable.develop);
      dt_control_queue_redraw_center();
      return 1;
    }
  }

  // opacity control (ctrl+scroll). deliberately outside the session
  // boundary: opacity is not part of the recipe, it belongs to the group
  // entry the commit creates -- so it cannot make a session dirty, and a
  // no-op close (which creates no group) has nothing to apply it to
  if(gui->creation && dt_modifier_is(state, GDK_CONTROL_MASK))
  {
    float opacity = dt_conf_get_float("plugins/darkroom/masks/opacity");
    opacity = CLAMP(opacity + (up ? 0.05f : -0.05f), 0.05f, 1.0f);
    dt_conf_set_float("plugins/darkroom/masks/opacity", opacity);
    dt_toast_log(_("opacity: %d%%"), (int)(opacity * 100.0f));
    dt_dev_masks_list_change(darktable.develop);
    dt_control_queue_redraw_center();
    return 1;
  }
  return 0;
}

// clear accumulated points, mask preview, and iterative refinement state.
// must not be called while a decode is RUNNING (the caller's guard ensures
// that); a finished-but-unpublished result is drained here so a posthumous
// publication cannot resurrect the mask we are about to clear
static void _clear_selection(dt_masks_form_gui_t *gui)
{
  _object_data_t *d = _get_data(gui);
  if(!d)
    return;

  // an edit session that cleared its points can never report a no-op
  // again, even if the user clicks back to the same count: the replayed
  // chain is gone
  if(d->edit_valid)
    d->edit_dirty = TRUE;

  _decode_drain(gui);

  if(gui->guipoints)
    dt_masks_dynbuf_reset(gui->guipoints);
  if(gui->guipoints_payload)
    dt_masks_dynbuf_reset(gui->guipoints_payload);
  gui->guipoints_count = 0;
  _marks_resize(d, 0);

  g_free(d->mask);
  d->mask = NULL;
  d->mask_w = d->mask_h = 0;

  if(d->seg)
    dt_seg_reset_prev_mask(d->seg);

  // reset selection and preview state
  d->has_selection = FALSE;
  _free_preview_forms(d);

  dt_control_queue_redraw_center();
}

// the no-op criterion of an edit session: re-capturing the session right
// now would yield the very recipe snapshotted at the end of the replay,
// byte for byte. same recipe = same fingerprint = same file = nothing to
// commit, so the session closes without creating a duplicate mask, a
// duplicate file and a history item on every open/close.
//
// a memcmp of the blob, deliberately: the recipe IS the description of
// what the session produces, so anything that would change the produced
// mask changes it -- points, decode boundaries, thresholds, vectorisation
// parameters, models, encode geometry -- and any field a later version
// adds is covered without touching this code. enumerating the mutators
// instead (the point count this replaced) leaves every parameter changed
// without a click silently discarded
static gboolean _edit_recipe_unchanged(_object_data_t *d,
                                       dt_masks_form_gui_t *gui)
{
  if(!d->edit_baseline_valid)
    return FALSE;
  dt_rf_recipe_t now;
  // a session the recipe cannot describe is never a no-op: it has no
  // stable identity to compare, so let it commit.
  // the stage is re-resolved the way the baseline resolved it (through the
  // session accessor, at this instant): the memcmp below then covers the
  // matting fields like every other one, with no field named here
  const _matte_session_t now_matte = _session_matte(d);
  if(!_capture_recipe(d, gui, &now_matte, &now))
    return FALSE;
  return memcmp(&now, &d->edit_baseline_recipe, sizeof(now)) == 0;
}

static int _object_events_button_pressed(dt_iop_module_t *module,
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
  (void)pressure;
  (void)parentid;
  (void)index;
  if(type == GDK_2BUTTON_PRESS || type == GDK_3BUTTON_PRESS)
    return 1;
  if(!gui)
    return 0;

  _object_data_t *d = _get_data(gui);

  // the edit machine holds the session frozen between its opening and the
  // end of the replay: every pointer gesture is refused with a discreet
  // message, the machine in post_expose is the only actor. structural: a
  // clear would tear down the very state the replay computes on, a click
  // would start a decode on the replay's ONNX context. the parameter
  // mutators (scroll, panel sliders) carry the same test.
  //
  // there is NO cancel gesture during the freeze -- the masks flow has no
  // Escape handler. the exits are the generic ones (module refocus, image
  // change, view change), all of which reach dt_masks_clear_form_gui ->
  // _free_data, which ends the bookkeeping at once and defers the
  // destruction until the replay thread joins. bounded by encode + replay
  if(gui->creation && d && _edit_frozen(d))
  {
    dt_control_log(_("reloading the recorded mask..."));
    return 1;
  }

  if(gui->creation && which == 1
     && dt_modifier_is(state, GDK_CONTROL_MASK | GDK_SHIFT_MASK))
  {
    // ctrl+shift+click: clear selection (only after first selection),
    // blocked while a compute runs
    if(d && d->has_selection
       && g_atomic_int_get(&d->encode_state) == ENCODE_READY
       && g_atomic_int_get(&d->decode_state) != DECODE_RUNNING)
    {
      _clear_selection(gui);
      if(darktable.develop->proxy.masks.module)
        darktable.develop->proxy.masks.list_change(
          darktable.develop->proxy.masks.module);
    }
    else if(d && d->has_selection
            && g_atomic_int_get(&d->decode_state) == DECODE_RUNNING)
      dt_control_log(_("mask still computing, try again in a moment"));
    return 1;
  }
  else if(gui->creation && which == 1)
  {
    // need valid scratchpad and completed encoding
    if(!d || d->encode_state != ENCODE_READY)
      return 1;

    // dismiss the "ready" hint now that the user is interacting
    dt_control_log_ack_all();

    // start drag tracking, resolved as click on button release
    float wd, ht, iwidth, iheight;
    dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

    d->dragging = TRUE;
    d->drag_start_x = pzx * wd;
    d->drag_start_y = pzy * ht;
    return 1;
  }
  else if(gui->creation && which == 3)
  {
    // a session opened with no sink closes into no sink, whatever the
    // darkroom did since: unfolding a module mid-session refocuses from
    // NULL, which skips the focus-loss teardown, so dev->gui_module (the
    // module this handler is called with) may now be a module the user
    // never asked to receive the mask. the decision belongs to the
    // session, taken once at its opening -- never re-derived here.
    // covers the four closing paths below in one place
    if(d && d->edit_no_sink)
      module = NULL;

    // don't exit while background threads are running, and never
    // finalise/vectorise while a decode is in flight or unpublished: the
    // result would silently ignore the user's latest clicks
    if(d && (g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING
             || g_atomic_int_get(&d->decode_state) != DECODE_IDLE
             || d->decode_pending))
    {
      dt_control_log(_("mask still computing, try again in a moment"));
      return 1;
    }

    // an edit session closed with no change commits NOTHING: the float
    // round-trip of the re-captured coordinates would otherwise produce a
    // recipe differing in the last bits -- a fake new mask, a new file and
    // a new history item on every open/close. covers both closing
    // gestures; a clear set edit_dirty, so "clicked back to the same
    // state" never passes as unchanged even if the recipe matched
    if(d && d->edit_valid
       && g_atomic_int_get(&d->edit_pending) == EDIT_ACTIVE
       && !d->edit_dirty
       && _edit_recipe_unchanged(d, gui))
    {
      dt_control_log(_("mask unchanged, nothing to apply"));
      gui->creation = FALSE;
      gui->creation_continuous = FALSE;
      gui->creation_continuous_module = NULL;
      _free_data(gui);   // ends the edit bookkeeping too
      dt_masks_dynbuf_free(gui->guipoints);
      dt_masks_dynbuf_free(gui->guipoints_payload);
      gui->guipoints = NULL;
      gui->guipoints_payload = NULL;
      gui->guipoints_count = 0;
      dt_control_hinter_message("");
      dt_masks_change_form_gui(NULL);
      dt_control_queue_redraw_center();
      return 1;
    }

    // shift+right-click: finalise a precise raster mask at native
    // resolution, on a worker job. separate from the vector path on purpose:
    // this one has no editable points, and says so by being a distinct gesture
    if(d && d->has_selection && d->mask
       && dt_modifier_is(state, GDK_SHIFT_MASK))
    {
      if(!_launch_native_finalize(d, gui, module, FALSE))
        return 1;   // busy: keep the session, the user can retry

      // leave creation mode right away: the mask will be wired into the
      // module automatically when the job lands, there is nothing left to
      // do here. mirrors the vector path's exit sequence below.
      gui->creation = FALSE;
      gui->creation_continuous = FALSE;
      gui->creation_continuous_module = NULL;
      _free_data(gui);
      dt_masks_dynbuf_free(gui->guipoints);
      dt_masks_dynbuf_free(gui->guipoints_payload);
      gui->guipoints = NULL;
      gui->guipoints_payload = NULL;
      gui->guipoints_count = 0;
      dt_control_hinter_message("");
      dt_masks_change_form_gui(NULL);
      dt_control_queue_redraw_center();
      return 1;
    }

    // right-click: precise paths -- same native finalisation as the raster
    // gesture, traced by potrace at 1:1 instead of written to a file. falls
    // back to the classic working-grid vectorisation when the job is busy
    // or when there is no refined mask to work from
    if(d && d->has_selection && d->mask
       && _launch_native_finalize(d, gui, module, TRUE))
    {
      gui->creation = FALSE;
      gui->creation_continuous = FALSE;
      gui->creation_continuous_module = NULL;
      _free_data(gui);
      dt_masks_dynbuf_free(gui->guipoints);
      dt_masks_dynbuf_free(gui->guipoints_payload);
      gui->guipoints = NULL;
      gui->guipoints_payload = NULL;
      gui->guipoints_count = 0;
      dt_control_hinter_message("");
      dt_masks_change_form_gui(NULL);
      dt_control_queue_redraw_center();
      return 1;
    }

    // classic fallback: vectorize the working-grid mask immediately
    dt_masks_form_t *new_grp = NULL;
    if(d && d->preview_forms)
      new_grp = _finalize_from_preview(module, gui);
    else if(gui->guipoints_count > 0)
      new_grp = _finalize_mask(module, form, gui);

    // add the new group to the module's blend mask group
    if(new_grp)
    {
      dt_develop_t *dev = darktable.develop;
      if(module)
      {
        dt_masks_form_t *mod_grp
          = dt_masks_get_from_id(dev, module->blend_params->mask_id);
        if(!mod_grp)
        {
          mod_grp = dt_masks_create(DT_MASKS_GROUP);
          gchar *module_label = dt_history_item_get_name(module);
          snprintf(mod_grp->name, sizeof(mod_grp->name),
                   _("group '%s'"), module_label);
          g_free(module_label);
          dev->forms = g_list_append(dev->forms, mod_grp);
          module->blend_params->mask_id = mod_grp->formid;
        }
        dt_masks_point_group_t *grpt = dt_masks_group_add_form(mod_grp, new_grp);
        if(grpt)
          grpt->opacity = dt_conf_get_float("plugins/darkroom/masks/opacity");
      }
      dt_dev_add_masks_history_item(dev, module, TRUE);
    }

    // cleanup and exit creation mode
    gui->creation = FALSE;
    gui->creation_continuous = FALSE;
    gui->creation_continuous_module = NULL;

    _free_data(gui);

    dt_masks_dynbuf_free(gui->guipoints);
    dt_masks_dynbuf_free(gui->guipoints_payload);
    gui->guipoints = NULL;
    gui->guipoints_payload = NULL;
    gui->guipoints_count = 0;

    dt_control_hinter_message("");

    // exit creation mode and select the new group,
    // dt_masks_set_edit_mode requires a non-NULL module (it returns
    // immediately otherwise), so clear the form directly when module
    // is NULL (standalone mask creation)
    if(module)
    {
      dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
      dt_masks_iop_update(module);
    }
    else
    {
      dt_masks_change_form_gui(NULL);
    }
    dt_control_queue_redraw_center();
    return 1;
  }

  return 0;
}

static int _object_events_button_released(dt_iop_module_t *module,
                                          const float pzx,
                                          const float pzy,
                                          const int which,
                                          const uint32_t state,
                                          dt_masks_form_t *form,
                                          const dt_imgid_t parentid,
                                          dt_masks_form_gui_t *gui,
                                          const int index)
{
  (void)module;
  (void)pzx;
  (void)pzy;
  (void)form;
  (void)parentid;
  (void)index;

  if(!gui || which != 1)
    return 0;

  _object_data_t *d = _get_data(gui);
  if(!d || !d->dragging)
    return 0;

  d->dragging = FALSE;

  if(!gui->guipoints)
    gui->guipoints = dt_masks_dynbuf_init(200000, "object guipoints");
  if(!gui->guipoints)
    return 1;
  if(!gui->guipoints_payload)
    gui->guipoints_payload = dt_masks_dynbuf_init(100000,
                                                  "object guipoints_payload");
  if(!gui->guipoints_payload)
    return 1;

  // the provenance recipe describes at most DT_RF_RECIPE_MAX_POINTS clicks;
  // one more and the session could never be regenerated from its recipe.
  // refuse the point outright -- a hard, explained limit beats a mask that
  // silently lost its provenance (and 32 prompts is far beyond any real
  // refinement session; ctrl+shift+click restarts from zero)
  if(gui->guipoints_count >= DT_RF_RECIPE_MAX_POINTS)
  {
    dt_control_log(_("maximum of %d points reached, "
                     "ctrl+shift+click to start over"),
                   DT_RF_RECIPE_MAX_POINTS);
    return 1;
  }

  // click: foreground point, shift+click: background point (only
  // after first selection)
  const float label = (d->has_selection && dt_modifier_is(state, GDK_SHIFT_MASK))
    ? 0.0f : 1.0f;
  dt_masks_dynbuf_add_2(gui->guipoints, d->drag_start_x, d->drag_start_y);
  dt_masks_dynbuf_add(gui->guipoints_payload, label);
  gui->guipoints_count++;
  _marks_resize(d, gui->guipoints_count);   // new click: no decode yet
  d->has_selection = TRUE;

  // coalescing decided inside _launch_decode (single source of truth); the
  // vectorization preview updates at publication time (_decode_finish)
  _launch_decode(gui);

  // refresh mask properties panel so sliders update for
  // the current creation step (size vs cleanup/smoothing)
  if(darktable.develop->proxy.masks.module)
    darktable.develop->proxy.masks.list_change(darktable.develop->proxy.masks.module);

  dt_control_queue_redraw_center();
  return 1;
}

static int _object_events_mouse_moved(dt_iop_module_t *module,
                                      const float pzx,
                                      const float pzy,
                                      const double pressure,
                                      const int which,
                                      const float zoom_scale,
                                      dt_masks_form_t *form,
                                      const dt_imgid_t parentid,
                                      dt_masks_form_gui_t *gui,
                                      const int index)
{
  (void)module;
  (void)pressure;
  (void)which;
  (void)zoom_scale;
  (void)form;
  (void)parentid;
  (void)index;

  if(!gui)
    return 0;

  gui->form_selected = FALSE;
  gui->border_selected = FALSE;
  gui->source_selected = FALSE;
  gui->feather_selected = -1;
  gui->point_selected = -1;
  gui->seg_selected = -1;
  gui->point_border_selected = -1;

  if(gui->creation)
    dt_control_queue_redraw_center();

  return 1;
}

// timer callback: periodically redraw center so +/- cursor tracks shift key
static gboolean _modifier_poll(gpointer data)
{
  (void)data;
  // the same 100 ms tick carries the shape-button lock. this timer is
  // created with the encode (and with any decode that outlives it) and
  // removed with the session data, so it ticks over every instant a
  // compute can own the session. re-deriving the sensitivity here instead
  // of pairing it with an inverse call is what makes a stuck button
  // impossible: a module whose blend gui is built or rebuilt in the middle
  // of a compute is caught by the next tick, without anybody knowing
  dt_masks_update_shapes_sensitivity();
  dt_control_queue_redraw_center();
  return G_SOURCE_CONTINUE;
}

static void _object_events_post_expose(cairo_t *cr,
                                       const float zoom_scale,
                                       dt_masks_form_gui_t *gui,
                                       const int index,
                                       const int num_points)
{
  (void)index;
  (void)num_points;
  if(!gui)
    return;
  if(!gui->creation)
    return;

  // ensure scratchpad exists
  _object_data_t *d = _get_data(gui);
  if(!d)
  {
    d = g_new0(_object_data_t, 1);
    d->preview_cleanup = dt_conf_get_int(CONF_OBJECT_CLEANUP_KEY);
    d->preview_smoothing = dt_conf_get_float(CONF_OBJECT_SMOOTHING_KEY);
    d->preview_feather = dt_conf_get_float(CONF_OBJECT_FEATHER_KEY);
    d->preview_refine = dt_conf_key_exists(CONF_OBJECT_REFINE_BOUNDARY_KEY)
                        && dt_conf_get_bool(CONF_OBJECT_REFINE_BOUNDARY_KEY);
    // no edit session on this data (edit sessions pre-seed their own
    // scratchpad in dt_object_mask_edit_begin); NAN marks the threshold
    // override as absent
    d->edit_threshold = NAN;

    // restore persistent model (stays loaded across mask sessions)
    // if the active model changed in preferences, discard the old one
    {
      dt_ai_seg_t *ps = &darktable.ai_seg;
      char *active = dt_ai_models_get_active_for_task("mask");
      const char *persistent_id = dt_seg_get_model_id(ps->ctx);
      if(ps->ctx && active
         && g_strcmp0(active, persistent_id) != 0)
      {
        dt_print(DT_DEBUG_AI,
                 "[object mask] model changed (%s -> %s), "
                 "discarding persistent model",
                 persistent_id, active);
        dt_seg_free(ps->ctx);
        ps->ctx = NULL;
        dt_ai_env_destroy(ps->env);
        ps->env = NULL;
        ps->model_loaded = FALSE;
      }
      g_free(active);
      d->env = ps->env;
      d->seg = ps->ctx;
      d->model_loaded = ps->model_loaded;
      ps->env = NULL;
      ps->ctx = NULL;
      ps->model_loaded = FALSE;

      // connect view-change signal once to free model on darkroom exit
      if(!ps->signal_connected)
      {
        DT_CONTROL_SIGNAL_CONNECT(DT_SIGNAL_VIEWMANAGER_VIEW_CHANGED,
                                  _on_view_changed, NULL);
        ps->signal_connected = TRUE;
      }
    }

    gui->scratchpad = d;
    gui->scratchpad_cleanup = _free_data;
  }

  // detect distortion changes (crop/rotate on same image):
  // reset encoding so the image is re-analyzed
  const dt_imgid_t cur_imgid = darktable.develop->image_storage.id;
  const int cur_state = g_atomic_int_get(&d->encode_state);
  if((cur_state == ENCODE_READY || cur_state == ENCODE_ERROR)
     // deferred while a decode runs: it computes on the old geometry and
     // must land (and be drained) before the teardown. clicks during the
     // deferral are transiently misaligned and repaired by this reset
     && g_atomic_int_get(&d->decode_state) != DECODE_RUNNING
     // likewise deferred while the edit machine holds the session frozen:
     // the replay thread computes on d->seg, this reset would race it.
     // the machine's exit (ACTIVE or FAILED) re-runs this check
     && !_edit_frozen(d)
     && (d->encoded_imgid != cur_imgid
         || d->encoded_distort_hash != _compute_distort_hash(darktable.develop)))
  {
    // an edit session whose image or geometry just moved has lost its
    // replayed state: say so -- the reset below is otherwise completely
    // silent, the points simply vanish -- and end the edit bookkeeping.
    // called UNCONDITIONALLY: past this branch the tool IS a plain
    // creation session on the new state, so a remanent EDIT_FAILED left by
    // an earlier failure must not survive to describe it
    if(d->edit_valid)
      dt_control_log(_("edit session dropped: the geometry changed"));
    _edit_session_end(d, EDIT_NONE);
    // no result computed on the old geometry may be published past this
    _decode_drain(gui);
    if(d->encode_thread)
    {
      g_thread_join(d->encode_thread);
      d->encode_thread = NULL;
    }
    // a model loaded BY ID served the edit session that just ended: drop
    // it through the one function that knows how (safe here, the encode
    // thread is joined and no decode runs) so the fresh encode below
    // reloads the ACTIVE model at the current cap, like any plain session
    if(!_edit_drop_pinned_encode(d) && d->seg)
      dt_seg_reset_encoding(d->seg);
    g_free(d->mask);
    d->mask = NULL;
    d->mask_w = d->mask_h = 0;
    d->encode_w = d->encode_h = 0;
    d->encode_state = ENCODE_IDLE;
    // reset selection, preview, and point state so the new image starts fresh
    d->has_selection = FALSE;
    _free_preview_forms(d);
    if(gui->guipoints)
      dt_masks_dynbuf_reset(gui->guipoints);
    if(gui->guipoints_payload)
      dt_masks_dynbuf_reset(gui->guipoints_payload);
    gui->guipoints_count = 0;
    _marks_resize(d, 0);
  }

  // eager encoding: load model and encode image as soon as tool opens
  if(d->encode_state == ENCODE_IDLE)
  {
    dt_control_log(g_atomic_int_get(&d->edit_pending) == EDIT_WAIT_ENCODE
                   ? _("reloading the recorded mask...")
                   : _("object mask: analyzing image..."));
    d->encode_state = ENCODE_MSG_SHOWN;
    dt_control_queue_redraw_center();
    return;
  }

  if(d->encode_state == ENCODE_MSG_SHOWN)
  {
    // frame 2: launch background thread to render and encode the image.
    // the thread creates a temporary export pipe at high resolution
    // instead of using the low-res preview backbuf.
    // flush history to database so the encode thread's dt_dev_load_image
    // sees the current edits (crop/rotate may not be flushed yet)
    dt_dev_write_history(darktable.develop);

    const dt_hash_t cur_hash = _compute_distort_hash(darktable.develop);

    _encode_thread_data_t *td = g_new0(_encode_thread_data_t, 1);
    td->d = d;
    td->imgid = cur_imgid;
    td->history_end = darktable.develop->history_end;
    td->distort_hash = cur_hash;
    // the render cap this encoding runs under, decided HERE and nowhere
    // else: the thread never re-reads the preference, and the capture
    // records what the encoding really used
    td->render_size = _conf_render_size();
    // an edit session encodes PINNED: the recorded model by id, the
    // recorded dimensions, the recorded cap -- snapshotted here, the
    // thread never reads the live session fields
    if(d->edit_valid
       && g_atomic_int_get(&d->edit_pending) == EDIT_WAIT_ENCODE)
    {
      td->pinned = TRUE;
      g_strlcpy(td->pin_model, d->edit_recipe.seg_model,
                sizeof(td->pin_model));
      td->pin_w = d->edit_recipe.encode_w;
      td->pin_h = d->edit_recipe.encode_h;
      td->render_size = d->edit_render_size;
      td->pin_recipe = d->edit_recipe;
    }

    d->encoded_imgid = cur_imgid;
    d->encoded_distort_hash = cur_hash;
    d->encoded_render_size = td->render_size;
    d->encode_state = ENCODE_RUNNING;
    // start poll timer BEFORE the thread, it will detect completion
    // and also tracks modifier keys once encoding is ready
    if(!d->modifier_poll_id)
      d->modifier_poll_id = g_timeout_add(100, _modifier_poll, NULL);
    d->encode_thread = g_thread_new("ai-mask-encode", _encode_thread_func, td);
    return;
  }

  if(g_atomic_int_get(&d->encode_state) == ENCODE_RUNNING)
  {
    // keep the message visible while the thread is working. the pinned
    // encode is the longest phase of an edit session: say what it is, or
    // the user who clicked "edit mask" believes he started a creation
    if(g_atomic_int_get(&d->edit_pending) == EDIT_WAIT_ENCODE)
      dt_control_log(_("reloading the recorded mask..."));
    else
      dt_control_log(_("object mask: analyzing image..."));
    return;
  }

  if(g_atomic_int_get(&d->encode_state) == ENCODE_READY && d->encode_thread)
  {
    // thread finished (detected by poll timer redraw), join it
    g_thread_join(d->encode_thread);
    d->encode_thread = NULL;
    dt_control_log_ack_all();
    if(g_atomic_int_get(&d->edit_pending) == EDIT_WAIT_ENCODE)
      dt_control_log(_("reloading the recorded mask..."));
    else
      dt_control_log(_("click on object to create mask"));
  }

  if(g_atomic_int_get(&d->encode_state) == ENCODE_ERROR)
  {
    if(d->encode_thread)
    {
      g_thread_join(d->encode_thread);
      d->encode_thread = NULL;
      // log only once when the thread is first joined
      dt_control_log(_("object mask preparation failed"));
      // an edit session cannot start without its encode: end the edit
      // bookkeeping on the spot -- the tool session stays, dead, exactly
      // as the plain path leaves it. the purge call is the rule (every
      // EDIT_FAILED exit calls it) and a no-op here by design: nothing was
      // pinned successfully, and resetting a DEAD encode would only spin a
      // relaunch loop
      if(g_atomic_int_get(&d->edit_pending) == EDIT_WAIT_ENCODE)
      {
        _edit_session_end(d, EDIT_FAILED);
        _edit_drop_pinned_encode(d);
      }
    }
    return;
  }

  if(d->encode_state != ENCODE_READY)
    return;

  // --- ai edit session machine: WAIT_ENCODE -> REPLAYING -> ACTIVE ---
  // runs BEFORE the publication machine, and returns early while the
  // session is frozen: no decode can be in flight then and none may be
  // consumed, no overlay may draw a half-injected state
  {
    const int ep = g_atomic_int_get(&d->edit_pending);
    if(ep == EDIT_WAIT_ENCODE)
    {
      _edit_replay_start(gui, d);
      return;
    }
    if(ep == EDIT_REPLAYING)
    {
      _edit_replay_finish(gui, d);
      if(g_atomic_int_get(&d->edit_pending) == EDIT_REPLAYING)
        return;   // still replaying: keep the session frozen
    }
  }

  // --- asynchronous decode: publication machine ---
  // placed after the encode machine and after the invalidation branch, so a
  // publication can never see a state the invalidation just tore down in
  // the same expose. the edit machine's early returns above are the gate
  // demanded by its freeze contract: while WAIT_ENCODE or REPLAYING this
  // point is never reached, so no DECODE_READY can be consumed here
  {
    const int dst = g_atomic_int_get(&d->decode_state);
    if(dst == DECODE_READY || dst == DECODE_ERROR)
    {
      // the invalidation branch above may have deferred its teardown while
      // this compute ran, and the compute may have landed since it looked:
      // its result was made on the torn-down geometry, drain it -- the next
      // expose performs the teardown
      const gboolean stale
        = d->encoded_imgid != cur_imgid
          || d->encoded_distort_hash != _compute_distort_hash(darktable.develop);
      if(stale)
        _decode_drain(gui);
      else
        _decode_publish(gui);
    }
    else if(dst == DECODE_RUNNING)
      // deliberately a toast rather than the mouse-move hint: it stays
      // visible while the pointer is idle, and control.c dedups the
      // repeated message
      dt_control_log(_("computing mask..."));
  }

  float wd, ht, iwidth, iheight;
  dt_masks_get_image_size(&wd, &ht, &iwidth, &iheight);

  // --- Draw red overlay of current mask ---
  if(d->mask && d->mask_w > 0 && d->mask_h > 0)
  {
    const int mw = d->mask_w;
    const int mh = d->mask_h;
    const int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, mw);
    unsigned char *buf = g_try_malloc0((size_t)stride * mh);
    if(buf)
    {
      const float mask_thresh = _session_threshold(d);
      for(int y = 0; y < mh; y++)
      {
        unsigned char *row = buf + y * stride;
        for(int x = 0; x < mw; x++)
        {
          const float val = d->mask[y * mw + x];
          if(val > mask_thresh)
          {
            const unsigned char alpha = 80;
            row[x * 4 + 0] = 0;     // B
            row[x * 4 + 1] = 0;     // G
            row[x * 4 + 2] = alpha; // R (premultiplied)
            row[x * 4 + 3] = alpha; // A
          }
        }
      }

      cairo_surface_t *surface = cairo_image_surface_create_for_data
        (buf, CAIRO_FORMAT_ARGB32, mw, mh, stride);

      if(surface)
      {
        cairo_save(cr);
        cairo_scale(cr, wd / mw, ht / mh);
        cairo_set_source_surface(cr, surface, 0, 0);
        cairo_paint(cr);
        cairo_restore(cr);
        cairo_surface_destroy(surface);
      }
      g_free(buf);
    }
  }

  // draw vectorization preview (real path style with anchor dots)
  if(d->preview_forms)
  {
    const float msx = (d->mask_w > 0) ? wd / (float)d->mask_w : 1.0f;
    const float msy = (d->mask_h > 0) ? ht / (float)d->mask_h : 1.0f;

    for(GList *fl = d->preview_forms; fl; fl = g_list_next(fl))
    {
      dt_masks_form_t *f = fl->data;
      GList *pts = f->points;
      if(!pts) continue;

      dt_masks_point_path_t *first_pt = pts->data;
      cairo_move_to(cr,
                    first_pt->corner[0] * msx,
                    first_pt->corner[1] * msy);

      // cairo_curve_to(c1, c2, end) expects:
      //   c1 = outgoing handle of previous point (prev.ctrl2)
      //   c2 = incoming handle of this point (this.ctrl1)
      dt_masks_point_path_t *prev_pt = first_pt;
      for(GList *p = g_list_next(pts); p; p = g_list_next(p))
      {
        dt_masks_point_path_t *pt = p->data;
        cairo_curve_to(cr,
                       prev_pt->ctrl2[0] * msx, prev_pt->ctrl2[1] * msy,
                       pt->ctrl1[0] * msx, pt->ctrl1[1] * msy,
                       pt->corner[0] * msx, pt->corner[1] * msy);
        prev_pt = pt;
      }

      // close path back to first point
      cairo_curve_to(cr,
                     prev_pt->ctrl2[0] * msx, prev_pt->ctrl2[1] * msy,
                     first_pt->ctrl1[0] * msx, first_pt->ctrl1[1] * msy,
                     first_pt->corner[0] * msx, first_pt->corner[1] * msy);

      dt_masks_line_stroke(cr, FALSE, FALSE, FALSE, zoom_scale);

      for(GList *p = pts; p; p = g_list_next(p))
      {
        dt_masks_point_path_t *pt = p->data;
        dt_masks_draw_anchor(cr, FALSE, zoom_scale,
                             pt->corner[0] * msx, pt->corner[1] * msy);
      }
    }
  }

  // query pointer position and modifier state directly from GDK so the
  // cursor is drawn at the correct location even before the first
  // mouse_moved event fires.
  GtkWidget *cw = dt_ui_center(darktable.gui->ui);
  GdkWindow *win = gtk_widget_get_window(cw);
  GdkDevice *pointer = gdk_seat_get_pointer
    (gdk_display_get_default_seat(gdk_display_get_default()));
  GdkModifierType mod = 0;
  int dev_x = 0, dev_y = 0;
  if(win && pointer)
    gdk_window_get_device_position(win, pointer, &dev_x, &dev_y, &mod);

  // skip indicator when pointer is over a window above us (e.g. prefs)
  if(pointer
     && gdk_device_get_window_at_position(pointer, NULL, NULL) != win)
    return;
  const gboolean has_sel = d && d->has_selection;
  const gboolean ctrl_shift_held
    = has_sel
      && (mod & (GDK_CONTROL_MASK | GDK_SHIFT_MASK))
           == (GDK_CONTROL_MASK | GDK_SHIFT_MASK);
  const gboolean shift_held
    = has_sel && !ctrl_shift_held
      && (mod & GDK_SHIFT_MASK) != 0;

  // convert device coordinates to preview pipe pixel space
  {
    float pzx, pzy, zs;
    dt_dev_get_pointer_zoom_pos(&darktable.develop->full,
                                (float)dev_x, (float)dev_y,
                                &pzx, &pzy, &zs);
    gui->posx = pzx * wd;
    gui->posy = pzy * ht;
  }

  // draw cursor indicator for click interaction
  if(gui->posx >= 0.0f && gui->posx <= wd
     && gui->posy >= 0.0f && gui->posy <= ht)
  {
    const float r = DT_PIXEL_APPLY_DPI(8.0f) / zoom_scale;
    const float lw = DT_PIXEL_APPLY_DPI(2.0f) / zoom_scale;
    cairo_set_line_width(cr, lw);

    if(ctrl_shift_held)
    {
      // clear mode: draw undo/revert arrow above cursor
      cairo_set_source_rgba(cr, 0.9, 0.9, 0.9, 0.9);
      const float s = r * 0.7f; // icon size
      const float cx = gui->posx;
      const float cy = gui->posy - s * 1.8f; // above cursor

      cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
      cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);

      // arrowhead pointing left
      const float ax = cx - s * 0.8f;
      const float ay = cy - s;
      cairo_move_to(cr, ax + s * 0.65f, ay - s * 0.6f);
      cairo_line_to(cr, ax, ay);
      cairo_line_to(cr, ax + s * 0.65f, ay + s * 0.6f);
      cairo_stroke(cr);

      // horizontal line from arrow tip to half-circle top
      cairo_move_to(cr, ax, ay);
      cairo_line_to(cr, cx, ay);

      // half-circle curving right and down
      cairo_arc(cr, cx, cy, s, -G_PI * 0.5f, G_PI * 0.5f);

      // small horizontal tail at bottom going left
      cairo_line_to(cr, cx - s * 0.5f, cy + s);
      cairo_stroke(cr);
    }
    else
    {
      cairo_set_source_rgba(cr, 0.9, 0.9, 0.9, 0.9);
      // horizontal line (common to both + and -)
      cairo_move_to(cr, gui->posx - r, gui->posy);
      cairo_line_to(cr, gui->posx + r, gui->posy);
      cairo_stroke(cr);
      if(!shift_held)
      {
        // add mode: vertical line to form "+"
        cairo_move_to(cr, gui->posx, gui->posy - r);
        cairo_line_to(cr, gui->posx, gui->posy + r);
        cairo_stroke(cr);
      }
    }
  }

}

// --- stub functions (object is transient -- result is path masks) ---

static GSList *_object_setup_mouse_actions
  (const struct dt_masks_form_t *const form)
{
  (void)form;
  GSList *lm = NULL;
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT,
    0,
    _("[OBJECT] select / add foreground point"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT,
    GDK_SHIFT_MASK,
    _("[OBJECT] add background point"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_LEFT,
    GDK_CONTROL_MASK | GDK_SHIFT_MASK,
    _("[OBJECT] clear selection"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_RIGHT,
    0,
    _("[OBJECT] apply mask"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_SCROLL,
    0,
    _("[OBJECT] change smoothing"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_SCROLL,
    GDK_SHIFT_MASK,
    _("[OBJECT] change cleanup"));
  lm = dt_mouse_action_create_simple(
    lm,
    DT_MOUSE_ACTION_SCROLL,
    GDK_CONTROL_MASK,
    _("[OBJECT] change opacity"));
  return lm;
}

static void _object_set_form_name(dt_masks_form_t *const form,
                                  const size_t nb)
{
  snprintf(form->name, sizeof(form->name), _("object #%d"), (int)nb);
}

static void _object_set_hint_message(const dt_masks_form_gui_t *const gui,
                                     const dt_masks_form_t *const form,
                                     const int opacity,
                                     char *const restrict msgbuf,
                                     const size_t msgbuf_len)
{
  (void)form;
  if(gui->creation)
  {
    const _object_data_t *d = _get_data((dt_masks_form_gui_t *)gui);
    if(!d || d->encode_state != ENCODE_READY)
      return;  // no hints while encoding
    if(_edit_frozen(d))
    {
      // the machine is replaying the recorded session: gestures are
      // refused, say what is happening instead of promising clicks
      g_snprintf(msgbuf, msgbuf_len, _("reloading the recorded mask..."));
      return;
    }
    if(d->has_selection)
      g_snprintf(msgbuf,
                 msgbuf_len,
                 _("<b>add</b>: click, <b>subtract</b>: shift+click, "
                   "<b>clear</b>: ctrl+shift+click, "
                   "<b>apply</b>: right-click, "
                   "<b>apply+save raster</b>: shift+right-click\n"
                   "<b>smoothing</b>: scroll (%3.2f), "
                   "<b>cleanup</b>: shift+scroll (%d), "
                   "<b>opacity</b>: ctrl+scroll (%d%%)"),
                 d->preview_smoothing, d->preview_cleanup, opacity);
    else
      g_snprintf(msgbuf,
                 msgbuf_len,
                 _("<b>select</b>: click on object, "
                   "<b>opacity</b>: ctrl+scroll (%d%%)"),
                 opacity);
  }
}

static void _object_modify_property(dt_masks_form_t *const form,
                                    const dt_masks_property_t prop,
                                    const float old_val,
                                    const float new_val,
                                    float *sum,
                                    int *count,
                                    float *min,
                                    float *max)
{
  (void)form;

  dt_masks_form_gui_t *gui = darktable.develop->form_gui;
  _object_data_t *d = gui ? _get_data(gui) : NULL;

  if(!gui || !gui->creation) return;

  // the edit machine holds the session frozen between its opening and the
  // end of the replay: the sliders keep REPORTING the parameters seeded
  // from the recipe (the framework hides them at *count == 0) but may not
  // move them -- a drag would clobber the seeded values behind a preview
  // that cannot even redraw, breaking the promise of edit_begin that a
  // re-capture with no new click reproduces them. no toast: a slider drag
  // emits dozens of these calls
  const gboolean frozen = _edit_frozen(d);
  // an edit session owns its parameters: they came from the recipe, so the
  // increment starts from the SESSION value and never from the global
  // preference (which would make the first touch JUMP from the recorded
  // value to the preference), and the preference is never written (it is
  // the user's persistent choice -- the rule _session_threshold states)
  const gboolean in_session = d && d->edit_valid;

  // always increment *count - the framework hides the slider when
  // count==0 (see libs/masks.c gtk_widget_set_visible)
  switch(prop)
  {
    case DT_MASKS_PROPERTY_SIZE:
      break; // no size slider for click-based interaction
    case DT_MASKS_PROPERTY_CLEANUP:
    {
      int cleanup = in_session ? d->preview_cleanup
                               : dt_conf_get_int(CONF_OBJECT_CLEANUP_KEY);
      if(!frozen)
      {
        cleanup = CLAMP(cleanup + (int)(new_val - old_val), 0, 100);
        if(!in_session)
          dt_conf_set_int(CONF_OBJECT_CLEANUP_KEY, cleanup);
        if(d)
        {
          d->preview_cleanup = cleanup;
          _update_preview(d);
        }
      }
      *sum += cleanup;
      ++*count;
      break;
    }
    case DT_MASKS_PROPERTY_SMOOTHING:
    {
      float smoothing = in_session
        ? d->preview_smoothing
        : dt_conf_get_float(CONF_OBJECT_SMOOTHING_KEY);
      if(!frozen)
      {
        smoothing = CLAMP(smoothing + (new_val - old_val), 0.0f, 1.3f);
        if(!in_session)
          dt_conf_set_float(CONF_OBJECT_SMOOTHING_KEY, smoothing);
        if(d)
        {
          d->preview_smoothing = smoothing;
          _update_preview(d);
        }
      }
      *sum += smoothing;
      ++*count;
      break;
    }
    case DT_MASKS_PROPERTY_FEATHER:
    {
      const float ratio = (!old_val || !new_val) ? 1.0f : new_val / old_val;
      float feather = in_session ? d->preview_feather
                                 : dt_conf_get_float(CONF_OBJECT_FEATHER_KEY);
      if(!frozen)
      {
        if(feather < 0.0005f && ratio > 1.0f)
          feather = 0.001f; // bootstrap from zero on increase
        feather = CLAMP(feather * ratio, 0.0f, 1.0f);
        if(feather < 0.0002f) feather = 0.0f; // snap to a hard edge
        if(!in_session)
          dt_conf_set_float(CONF_OBJECT_FEATHER_KEY, feather);
        if(d)
        {
          d->preview_feather = feather;
          _update_preview(d);
        }
      }
      *sum += feather + feather; // both borders (same as path)
      *max = fminf(*max, 1.0f / feather);
      *min = fmaxf(*min, 0.0005f / feather);
      *count += 2; // both borders (same as path)
      break;
    }
    case DT_MASKS_PROPERTY_REFINE:
    {
      gboolean enabled = in_session
        ? d->preview_refine
        : dt_conf_get_bool(CONF_OBJECT_REFINE_BOUNDARY_KEY);
      if(!frozen && new_val != old_val)
      {
        // toggle applies on the next decoder run, not immediately
        enabled = new_val > 0.5f;
        if(!in_session)
          dt_conf_set_bool(CONF_OBJECT_REFINE_BOUNDARY_KEY, enabled);
        if(d) d->preview_refine = enabled;
      }
      *sum += enabled ? 1.0f : 0.0f;
      ++*count;
      break;
    }
    // the matting stage and its band scale act ONLY on the native
    // finalisation pass, never on the interactive preview: nothing
    // recomputes when they move, and their labels say "at finalisation"
    // for that reason. Outside a session they are preferences; inside one
    // they are session parameters seeded from the replayed recipe, exactly
    // like the threshold and the render cap -- which is what lets a
    // reopened mask be re-finalised with the stage it was MADE with
    // instead of whatever the preferences drifted to. Both branches go
    // through the same accessors the render and the recipe capture use, so
    // what these widgets report and what the finalisation runs cannot
    // disagree.
    //
    // Either control still leaves *count at zero -- and so DISAPPEARS (the
    // framework hides a property at count == 0, see the note above) -- in
    // the one situation where showing it would be a lie: no operator to
    // run. That is the SHIPPED state as long as the default preference
    // names a line the table does not carry: a visible checkbox would
    // tick, write the preference, leave the stage inert, capture a recipe
    // recording no matting, and give the user not one word about any of
    // it. The resolution goes through dt_matte_find, the same door the
    // finalisation and the replay use.
    case DT_MASKS_PROPERTY_MATTING:
    {
      // in a session: the operator the recipe recorded, or -- for a recipe
      // that recorded no stage -- the one the preference names, so that
      // turning matting ON mid-session is possible at all. outside: the
      // preference's, as ever
      const _matte_session_t cur = _session_matte(d);
      const dt_matte_op_t *const op
        = (in_session && cur.op) ? cur.op : _matte_conf_op();
      if(!op) break;
      gboolean enabled = in_session ? d->edit_matte_enabled
                                    : dt_conf_get_bool(CONF_MATTE_ENABLED_KEY);
      if(!frozen && new_val != old_val)
      {
        // applies to the next finalisation, nothing recomputes now
        enabled = new_val > 0.5f;
        if(in_session)
        {
          // the SESSION, never the user's persistent choice: the rule
          // _session_threshold states for every control here. the operator
          // is pinned at the same time, so a session that switches the
          // stage on records the line it actually ran
          d->edit_matte_enabled = enabled;
          d->edit_matte_op = op;
        }
        else
          dt_conf_set_bool(CONF_MATTE_ENABLED_KEY, enabled);
      }
      *sum += enabled ? 1.0f : 0.0f;
      ++*count;
      break;
    }
    case DT_MASKS_PROPERTY_MATTING_BAND:
    {
      const _matte_session_t cur = _session_matte(d);
      const dt_matte_op_t *const op
        = (in_session && cur.op) ? cur.op : _matte_conf_op();
      if(!op) break;
      // and the operator must actually READ the band scale. leaving the
      // control up for a line that ignores it is not a harmless spare
      // knob: the value travels in the recipe, the recipe is hashed
      // verbatim to name a content-addressed file, so a drag would spend
      // a model load and a full inference to file a bit-identical mask
      // under a second name. one render, two names -- the mirror image of
      // the defect the version field exists to prevent. counting nothing
      // here leaves *count at 0, which is exactly how the masks panel
      // decides a property has no widget (libs/masks.c)
      if(!(op->caps & DT_MATTE_USES_BAND)) break;
      float band = in_session ? d->edit_matte_band
                              : (float)dt_conf_get_float(CONF_MATTE_BAND_KEY);
      if(!frozen)
      {
        // no preview at the drag, by design: the widened band is derived
        // by the native pass of a line that declares DT_MATTE_USES_BAND,
        // and computing it here would either lie about the result or cost
        // seconds per pointer event
        band = CLAMPF(band + (new_val - old_val),
                      DT_MATTE_BAND_SCALE_MIN, DT_MATTE_BAND_SCALE_MAX);
        if(in_session)
          d->edit_matte_band = band;
        else
          dt_conf_set_float(CONF_MATTE_BAND_KEY, band);
      }
      *sum += band;
      ++*count;
      break;
    }
    default:;
  }
}

// the function table for object masks
const dt_masks_functions_t dt_masks_functions_object = {
  .point_struct_size = sizeof(struct dt_masks_point_object_t),
  .sanitize_config = NULL,
  .setup_mouse_actions = _object_setup_mouse_actions,
  .set_form_name = _object_set_form_name,
  .set_hint_message = _object_set_hint_message,
  .modify_property = _object_modify_property,
  .duplicate_points = NULL,
  .initial_source_pos = NULL,
  .get_distance = NULL,
  .get_points = NULL,
  .get_points_border = NULL,
  .get_mask = NULL,
  .get_mask_roi = NULL,
  .get_area = NULL,
  .get_source_area = NULL,
  .mouse_moved = _object_events_mouse_moved,
  .mouse_scrolled = _object_events_mouse_scrolled,
  .button_pressed = _object_events_button_pressed,
  .button_released = _object_events_button_released,
  .post_expose = _object_events_post_expose
};

gboolean dt_masks_object_available(void)
{
  if(!dt_ai_registry_is_enabled())
    return FALSE;
  char *model_id = dt_ai_models_get_active_for_task("mask");
  dt_ai_model_t *model = dt_ai_models_get_by_id(model_id);
  g_free(model_id);
  const gboolean available = model && model->status == DT_AI_MODEL_DOWNLOADED;
  dt_ai_model_free(model);
  return available;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
