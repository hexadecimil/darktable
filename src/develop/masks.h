/*
    This file is part of darktable,
    Copyright (C) 2013-2026 darktable developers.

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
#include "common/opencl.h"
#include "common/rasterfile_recipe.h"
#include "develop/pixelpipe.h"
#include "dtgtk/button.h"
#include "dtgtk/gradientslider.h"
#include "gui/gtk.h"

#include <assert.h>

#define DEVELOP_MASKS_VERSION (6)

G_BEGIN_DECLS

/**forms types */
typedef enum dt_masks_type_t
{
  DT_MASKS_NONE = 0, // keep first
  DT_MASKS_CIRCLE = 1 << 0,
  DT_MASKS_PATH = 1 << 1,
  DT_MASKS_GROUP = 1 << 2,
  DT_MASKS_CLONE = 1 << 3,
  DT_MASKS_GRADIENT = 1 << 4,
  DT_MASKS_ELLIPSE = 1 << 5,
  DT_MASKS_BRUSH = 1 << 6,
  DT_MASKS_NON_CLONE = 1 << 7,
#ifdef HAVE_AI
  DT_MASKS_OBJECT = 1 << 8,
#endif
  // a persistent shape whose content is not a geometry but a reference to
  // a raster mask file (an AI precise mask or a hand-picked file). bit 8
  // stays reserved for DT_MASKS_OBJECT on every build, HAVE_AI or not:
  // the two types must never collide across build variants
  DT_MASKS_RASTER = 1 << 9,
} dt_masks_type_t;

/**masts states */
typedef enum dt_masks_state_t
{
  DT_MASKS_STATE_NONE = 0,
  DT_MASKS_STATE_USE = 1 << 0,
  DT_MASKS_STATE_SHOW = 1 << 1,
  DT_MASKS_STATE_INVERSE = 1 << 2,
  DT_MASKS_STATE_UNION = 1 << 3,
  DT_MASKS_STATE_INTERSECTION = 1 << 4,
  DT_MASKS_STATE_DIFFERENCE = 1 << 5,
  DT_MASKS_STATE_EXCLUSION = 1 << 6,
  DT_MASKS_STATE_SUM = 1 << 7,
  DT_MASKS_STATE_OP = DT_MASKS_STATE_UNION
                    | DT_MASKS_STATE_INTERSECTION
                    | DT_MASKS_STATE_DIFFERENCE
                    | DT_MASKS_STATE_SUM
                    | DT_MASKS_STATE_EXCLUSION
} dt_masks_state_t;

typedef enum dt_masks_property_t
{
  DT_MASKS_PROPERTY_OPACITY,
  DT_MASKS_PROPERTY_SIZE,
  DT_MASKS_PROPERTY_HARDNESS,
  DT_MASKS_PROPERTY_FEATHER,
  DT_MASKS_PROPERTY_ROTATION,
  DT_MASKS_PROPERTY_CURVATURE,
  DT_MASKS_PROPERTY_COMPRESSION,
  DT_MASKS_PROPERTY_CLEANUP,
  DT_MASKS_PROPERTY_SMOOTHING,
  DT_MASKS_PROPERTY_REFINE,
  // AI object mask, applied at finalisation only: the matting stage and
  // its single degree of freedom. appended, like every property before
  // them -- the panel builds one widget per entry and each form type
  // decides which ones it answers for, so a new entry is invisible to
  // every shape that does not handle it
  DT_MASKS_PROPERTY_MATTING,
  DT_MASKS_PROPERTY_MATTING_BAND,
  DT_MASKS_PROPERTY_LAST
} dt_masks_property_t;

typedef enum dt_masks_points_states_t
{
  DT_MASKS_POINT_STATE_NORMAL = 1,
  DT_MASKS_POINT_STATE_USER = 2
} dt_masks_points_states_t;

typedef enum dt_masks_gradient_states_t
{
  DT_MASKS_GRADIENT_STATE_LINEAR = 1,
  DT_MASKS_GRADIENT_STATE_SIGMOIDAL = 2
} dt_masks_gradient_states_t;

typedef enum dt_masks_edit_mode_t
{
  DT_MASKS_EDIT_OFF = 0,
  DT_MASKS_EDIT_FULL = 1,
  DT_MASKS_EDIT_RESTRICTED = 2
} dt_masks_edit_mode_t;

typedef enum dt_masks_pressure_sensitivity_t
{
  DT_MASKS_PRESSURE_OFF = 0,
  DT_MASKS_PRESSURE_HARDNESS_REL = 1,
  DT_MASKS_PRESSURE_HARDNESS_ABS = 2,
  DT_MASKS_PRESSURE_OPACITY_REL = 3,
  DT_MASKS_PRESSURE_OPACITY_ABS = 4,
  DT_MASKS_PRESSURE_BRUSHSIZE_REL = 5
} dt_masks_pressure_sensitivity_t;

typedef enum dt_masks_ellipse_flags_t
{
  DT_MASKS_ELLIPSE_EQUIDISTANT = 0,
  DT_MASKS_ELLIPSE_PROPORTIONAL = 1
} dt_masks_ellipse_flags_t;

typedef enum dt_masks_source_pos_type_t
{
  DT_MASKS_SOURCE_POS_RELATIVE = 0,
  DT_MASKS_SOURCE_POS_RELATIVE_TEMP = 1,
  DT_MASKS_SOURCE_POS_ABSOLUTE = 2
} dt_masks_source_pos_type_t;

/* selected Bézier control point for path*/
typedef enum dt_masks_path_ctrl_t
{
  DT_MASKS_PATH_CRTL_NONE = 0,
  DT_MASKS_PATH_CTRL1 = 1,
  DT_MASKS_PATH_CTRL2 = 2

} dt_masks_path_ctrl_t;

/* restrictions on moving Bézier control points */
typedef enum dt_masks_path_edit_mode_t
{
  DT_MASKS_BEZIER_NONE = 0,        // preserve angle & scale
  DT_MASKS_BEZIER_SINGLE = 1,      // no restriction
  DT_MASKS_BEZIER_SYMMETRIC = 2,   // force full symmetry
  DT_MASKS_BEZIER_SING_SYMM = 3    // SINGLE && SYMMETRIC => force angle symmetry only
} dt_masks_path_edit_mode_t;

/** structure used to store 1 point for a circle */
typedef struct dt_masks_point_circle_t
{
  float center[2];
  float radius;
  float border;
} dt_masks_point_circle_t;

/** structure used to store 1 point for an ellipse */
typedef struct dt_masks_point_ellipse_t
{
  float center[2];
  float radius[2];
  float rotation;
  float border;
  dt_masks_ellipse_flags_t flags;
} dt_masks_point_ellipse_t;

#ifdef HAVE_AI
/** structure used to store 1 point for an object (AI segmentation) form */
typedef struct dt_masks_point_object_t
{
  float anchor[2]; // click position (normalized image coords)
  int label;       // 1 = foreground, 0 = background
} dt_masks_point_object_t;
#endif

/** structure used to store 1 point for a path form */
typedef struct dt_masks_point_path_t
{
  float corner[2];
  float ctrl1[2];
  float ctrl2[2];
  float border[2];
  dt_masks_points_states_t state;
} dt_masks_point_path_t;

/** structure used to store 1 point for a brush form */
typedef struct dt_masks_point_brush_t
{
  float corner[2];
  float ctrl1[2];
  float ctrl2[2];
  float border[2];
  float density;
  float hardness;
  dt_masks_points_states_t state;
} dt_masks_point_brush_t;

/** structure used to store anchor for a gradient */
typedef struct dt_masks_point_gradient_t
{
  float anchor[2];
  float rotation;
  float compression;
  float steepness;
  float curvature;
  dt_masks_gradient_states_t state;
} dt_masks_point_gradient_t;

/** structure used to store all forms's id for a group */
typedef struct dt_masks_point_group_t
{
  dt_mask_id_t formid;
  dt_mask_id_t parentid;
  int state;
  float opacity;
} dt_masks_point_group_t;

// ---- ai provenance trailer for groups of path forms ----
//
// a group of paths produced by the ai object mask carries the provenance
// recipe of the session that created it -- the exact prompt clicks, decode
// boundaries and model versions (see common/rasterfile_recipe.h) -- so the
// group can later be re-opened for ai editing from its original prompts.
// the trailer lives in dt_masks_form_t by value and is persisted by
// appending it to the group's point blob in the masks history: a reader
// that does not know the layout (an upstream darktable) reads exactly
// nb * sizeof(dt_masks_point_group_t) bytes, ignores the excess and drops
// it on its next write -- the mask itself stays intact either way.
//
// known exception, documented and accepted (no format change): the XMP
// import path for sidecars older than v3 (_add_mask_entries_to_db,
// exif.cc) strictly validates the group blob size and rejects the whole
// mask on mismatch. that path is unreachable for masks written by this
// build, which always produces v3+ sidecars whose blobs are read verbatim.
// watch upstream: if that strict validation is ever generalized to the
// v3+ path, the trailer must migrate to its own storage
#define DT_MASKS_AI_TRAILER_MAGIC 0x44544154u  // "DTAT"; 0 = no recipe
#define DT_MASKS_AI_TRAILER_VERSION 1
// the session was seeded with a synthetic prev_mask (context
// rasterisation): its recipe alone cannot reproduce the starting mask, a
// re-edit must seed a context again instead of replaying the clicks
#define DT_MASKS_AI_TRAILER_CONTEXT_SEEDED (1u << 0)

typedef struct dt_masks_ai_trailer_t
{
  uint32_t magic;        // DT_MASKS_AI_TRAILER_MAGIC, 0 = no recipe
  uint32_t version;      // trailer layout, independent of DEVELOP_MASKS_VERSION
  uint32_t flags;        // bit 0 = CONTEXT_SEEDED
  // low 32 bits of an image identity hash (basename without extension,
  // sensor dimensions, capture datetime -- the same ingredients as the
  // rasterfile fingerprint, cf. rasterfile_recipe.h): a history pasted
  // onto ANOTHER image carries clicks recorded on the original photo, and
  // a re-edit must detect that and seed a context instead of replaying
  // them. mirrored by outil/verifier_trailer.py
  uint32_t image_hash;
  // hash of the group content at capture time: the group's own serialized
  // blob (trailer excluded) followed by each child's point blob in list
  // order. covers child states (union/difference), opacities, order and
  // membership -- any manual retouch changes it, and a re-edit then knows
  // the recorded clicks no longer describe the current shapes
  int64_t paths_hash;
  dt_rf_recipe_t recipe; // the very struct the rasterfile params embed
} dt_masks_ai_trailer_t;

// appended to serialized blobs and copied around by value: no implicit
// padding allowed, layout frozen -- any change bumps the trailer version,
// and an unknown version reads as no trailer at all.
// deliberate lifecycle choices: dt_masks_form_duplicate does NOT copy the
// trailer (a duplicate is a new, hand-owned group and degrades to context
// seeding; may be revisited with the mask-edit work). size note: each
// trailer adds ~1.1 KB to the XMP packet of exported JPEGs, whose APP1
// segment caps at 64 KB -- marginal against the existing mask data, but
// part of that budget.
G_STATIC_ASSERT(sizeof(dt_masks_ai_trailer_t) == 1096);

// ---- the raster shape: a mask that IS a file ----
//
// the single serialized "point" of a DT_MASKS_RASTER form. not a geometry:
// a reference to the mask file that renders the shape, plus the provenance
// recipe that can regenerate that file (see common/rasterfile_recipe.h).
// when the recipe is valid it IS the reference -- the file name is derived
// content-addressed from its fingerprint under the local mask root, exactly
// like iop/rasterfile.c commit_params does; `file` is only read when the
// recipe is absent (sequential "<basename>_mask_N.png" output or a future
// hand-picked file) and names a leaf under the mask root.
// POD and layout frozen: the blob travels verbatim through DB and XMP and
// is hashed for pipe invalidation (dt_masks_group_hash), so no implicit
// padding and every reserved byte zeroed. an unknown version renders the
// form inert (never a wrong render). image identity is deliberately NOT
// stored: it is re-derived from the pipe's image at render time, the same
// semantics as the rasterfile module
#define DT_MASKS_RASTER_POINT_MAGIC 0x4454524Du  // "DTRM"; 0 = invalid
#define DT_MASKS_RASTER_POINT_VERSION 1

typedef struct dt_masks_point_raster_t
{
  uint32_t magic;      // DT_MASKS_RASTER_POINT_MAGIC
  uint32_t version;    // DT_MASKS_RASTER_POINT_VERSION
  uint32_t flags;      // DT_MASKS_RASTER_FLAG_*, zero for a plain file
  int32_t _pad;        // explicit, keep zeroed (the blob is hashed)
  char file[256];      // fallback leaf name when recipe.magic == 0
  dt_rf_recipe_t recipe;
} dt_masks_point_raster_t;

// 16 + 256 + 1072: any drift breaks every stored history of this type
G_STATIC_ASSERT(sizeof(dt_masks_point_raster_t) == 1344);

// the shape renders the COMPLEMENT of its file: a "background" is the
// subject detection, same recipe and same content-addressed file, read
// the other way round. carried by the shape itself and not by a group
// member's DT_MASKS_STATE_INVERSE, so it holds in the shape library, in
// every group the shape is later added to, and through a duplicate --
// the member state stays available on top of it, as for any shape.
// older blobs have the field zeroed: a plain file, unchanged
#define DT_MASKS_RASTER_FLAG_INVERT (1u << 0)

/** structure used to store pointers to the functions implementing operations on a mask shape */
/** plus a few per-class descriptive data items */
typedef struct dt_masks_functions_t
{
  int point_struct_size;   // sizeof(struct dt_masks_point_*_t)
  void (*sanitize_config)(dt_masks_type_t type_flags);
  GSList *(*setup_mouse_actions)(const struct dt_masks_form_t *const form);
  void (*set_form_name)(struct dt_masks_form_t *const form, const size_t nb);
  void (*set_hint_message)(const struct dt_masks_form_gui_t *const gui,
                           const struct dt_masks_form_t *const form,
                           const int opacity,
                           char *const __restrict__ msgbuf,
                           const size_t msgbuf_len);
  void (*modify_property)(struct dt_masks_form_t *const form,
                          dt_masks_property_t prop,
                          const float old_val,
                          const float new_val,
                          float *sum,
                          int *count,
                          float *min,
                          float *max);
  // grow/shrink (outset/inset) a shape to a signed absolute amount in the given
  // unit (use_percent: TRUE = % of shape size, FALSE = image pixels), measured
  // from a baseline captured the first time the shape is resized. Positive grows,
  // negative shrinks, 0 restores the baseline. Results are cached per offset, so
  // re-requesting a value is lossless. Returns TRUE if a usable shape resulted.
  // Currently only implemented by path masks.
  gboolean (*resize)(struct dt_masks_form_t *const form,
                     const int amount,
                     const gboolean use_percent);
  // report the resize offset currently applied to the shape, in the requested
  // unit, so a UI control can mirror it. Returns FALSE (amount 0) if no resize is
  // active. Currently only implemented by path masks.
  gboolean (*resize_get)(struct dt_masks_form_t *const form,
                         const gboolean use_percent,
                         float *amount);
  void (*duplicate_points)(dt_develop_t *const dev,
                           struct dt_masks_form_t *base,
                           struct dt_masks_form_t *dest);
  void (*initial_source_pos)(const float iwd,
                             const float iht,
                             float *x,
                             float *y);
  void (*get_distance)(const float x,
                       const float y,
                       const float as,
                       struct dt_masks_form_gui_t *gui,
                       const int index,
                       const int num_points,
                       gboolean *inside,
                       gboolean *inside_border,
                       int *near,
                       gboolean *inside_source,
                       float *dist);
  int (*get_points)(dt_develop_t *dev,
                    const float x,
                    const float y,
                    const float radius_a,
                    const float radius_b,
                    const float rotation,
                    float **points,
                    int *points_count);
  int (*get_points_border)(dt_develop_t *dev,
                           struct dt_masks_form_t *form,
                           float **points,
                           int *points_count,
                           float **border,
                           int *border_count,
                           const int source,
                           const dt_iop_module_t *const module);
  int (*get_mask)(const dt_iop_module_t *const module,
                  const dt_dev_pixelpipe_iop_t *const piece,
                  struct dt_masks_form_t *const form,
                  float **buffer,
                  int *width,
                  int *height,
                  int *posx,
                  int *posy);
  int (*get_mask_roi)(const dt_iop_module_t *const fmodule,
                      const dt_dev_pixelpipe_iop_t *const piece,
                      struct dt_masks_form_t *const form,
                      const dt_iop_roi_t *roi,
                      float *buffer);
  int (*get_area)(const dt_iop_module_t *const module,
                  const dt_dev_pixelpipe_iop_t *const piece,
                  struct dt_masks_form_t *const form,
                  int *width,
                  int *height,
                  int *posx,
                  int *posy);
  int (*get_source_area)(dt_iop_module_t *module,
                         dt_dev_pixelpipe_iop_t *piece,
                         struct dt_masks_form_t *form,
                         int *width,
                         int *height,
                         int *posx,
                         int *posy);
  int (*mouse_moved)(dt_iop_module_t *module,
                     float pzx,
                     float pzy,
                     const double pressure,
                     const int which,
                     const float zoom_scale,
                     struct dt_masks_form_t *form,
                     const dt_imgid_t parentid,
                     struct dt_masks_form_gui_t *gui,
                     const int index);
  int (*mouse_scrolled)(dt_iop_module_t *module,
                        float pzx,
                        float pzy,
                        const gboolean up,
                        uint32_t state,
                        struct dt_masks_form_t *form,
                        const dt_imgid_t parentid,
                        struct dt_masks_form_gui_t *gui,
                        const int index);
  int (*button_pressed)(dt_iop_module_t *module,
                        float pzx,
                        float pzy,
                        const double pressure,
                        const int which,
                        const int type,
                        const uint32_t state,
                        struct dt_masks_form_t *form,
                        const dt_imgid_t parentid,
                        struct dt_masks_form_gui_t *gui,
                        const int index);
  int (*button_released)(dt_iop_module_t *module,
                         float pzx,
                         float pzy,
                         const int which,
                         const uint32_t state,
                         struct dt_masks_form_t *form,
                         const dt_imgid_t parentid,
                         struct dt_masks_form_gui_t *gui,
                         const int index);
  void (*post_expose)(cairo_t *cr,
                      const float zoom_scale,
                      struct dt_masks_form_gui_t *gui,
                      const int index,
                      const int num_points);
} dt_masks_functions_t;

/** structure used to define a form */
typedef struct dt_masks_form_t
{
  GList *points; // list of point structures
  dt_masks_type_t type;
  const dt_masks_functions_t *functions;

  // position of the source (used only for clone). [0]=dx, [1]=dy, [2]=angle
  float source[3];
  // name of the form
  char name[128];
  // id used to store the form
  dt_mask_id_t formid;
  // version of the form
  int version;
  // ai provenance trailer, meaningful for groups of ai-produced paths only
  // (magic == 0 otherwise). in-memory it travels by value through form
  // duplication and the history deep copies; on disk it is appended to the
  // group's point blob, see dt_masks_write_masks_history_item
  dt_masks_ai_trailer_t ai_trailer;
} dt_masks_form_t;

typedef struct dt_masks_form_gui_points_t
{
  float *points;
  int points_count;
  float *border;
  int border_count;
  float *source;
  int source_count;
  gboolean clockwise;
} dt_masks_form_gui_points_t;

/** structure for dynamic buffers */
typedef struct dt_masks_dynbuf_t
{
  float *buffer;
  char tag[128];
  size_t pos;
  size_t size;
} dt_masks_dynbuf_t;

typedef struct dt_masks_intbuf_t
{
  int *buffer;
  char tag[128];
  size_t pos;
  size_t size;
} dt_masks_intbuf_t;


/** structure used to display a form */
typedef struct dt_masks_form_gui_t
{
  // points used to draw the form
  GList *points; // list of dt_masks_form_gui_points_t

  // points used to sample mouse moves
  dt_masks_dynbuf_t *guipoints, *guipoints_payload;
  int guipoints_count;

  // values for mouse positions, etc...
  float posx, posy, dx, dy, scrollx, scrolly, posx_source, posy_source;
  // TRUE if mouse has leaved the center window
  gboolean form_selected;
  gboolean border_selected;
  gboolean source_selected;
  gboolean source_rotating;
  gboolean counter_rotate_source;
  // joint rotation grabbed from the source shape: the mouse circles the source,
  // so its angular sweep must be measured about the source centroid (not the
  // destination centroid) to keep the rotation gain symmetric with grabbing the
  // target. The applied angle is identical either way; only the pivot used to
  // read the mouse motion differs.
  gboolean rotate_about_source;
  gboolean pivot_selected;
  gboolean select_only_border;
  dt_masks_edit_mode_t edit_mode;
  int point_selected;
  int point_edited;
  int feather_selected;
  dt_masks_path_ctrl_t bezier_ctrl; // For paths, this selects a Bézier control point.
  int seg_selected;
  int point_border_selected;
  int source_pos_type;

  gboolean form_dragging;
  gboolean source_dragging;
  gboolean form_rotating;
  gboolean border_toggling;
  gboolean gradient_toggling;
  int point_dragging;
  int feather_dragging;
  int seg_dragging;
  int point_border_dragging;

  dt_masks_path_edit_mode_t bezier_mode;  // Bézier editing with shift or ctrl
  float bezier_ctrl_angle;  // angle between ctrl1 and ctrl2
  float bezier_ctrl_scale;  // length of ctrl2 relative to ctrl1

  int group_edited;
  int group_selected;

  guint show_all_feathers;

  gboolean creation;
  gboolean creation_continuous;
  gboolean creation_closing_form;
  dt_iop_module_t *creation_module;
  dt_iop_module_t *creation_continuous_module;

  dt_masks_pressure_sensitivity_t pressure_sensitivity;

  // ids
  dt_mask_id_t formid;
  dt_hash_t pipe_hash;

  // opaque per-type data (e.g. segmentation context for object masks)
  void *scratchpad;
  void (*scratchpad_cleanup)(struct dt_masks_form_gui_t *gui);
} dt_masks_form_gui_t;

/** special value to indicate an invalid or uninitialized coordinate */
/** (replaces former use of NAN and isnan() by the most negative float) **/
#define DT_INVALID_COORDINATE (-FLT_MAX)

/** the shape-specific function tables */
extern const dt_masks_functions_t dt_masks_functions_circle;
extern const dt_masks_functions_t dt_masks_functions_ellipse;
extern const dt_masks_functions_t dt_masks_functions_brush;
extern const dt_masks_functions_t dt_masks_functions_path;
extern const dt_masks_functions_t dt_masks_functions_gradient;
extern const dt_masks_functions_t dt_masks_functions_group;
extern const dt_masks_functions_t dt_masks_functions_raster;
#ifdef HAVE_AI
extern const dt_masks_functions_t dt_masks_functions_object;
/** check if AI object mask model is downloaded and AI is enabled */
gboolean dt_masks_object_available(void);
/** TRUE while an AI computation owns the object tool session: the eager
 * encode, an interactive decode, or the replay of a reopened recipe */
gboolean dt_masks_object_session_busy(void);

/** a row of the static detector table (common/ai/detectors.h) */
struct dt_detector_t;

/** what a catalogue entry for a promptless detector can offer NOW. the
 * catalogue writes the reason into the entry's label (GTK3 delivers no
 * tooltip to an insensitive item), so every state maps to one wording */
typedef enum dt_masks_object_detect_state_t
{
  DT_MASKS_OBJECT_DETECT_READY = 0,   // model installed: a click detects
  DT_MASKS_OBJECT_DETECT_DOWNLOAD,    // a click downloads, then detects
  DT_MASKS_OBJECT_DETECT_DOWNLOADING, // a download is already in flight
  DT_MASKS_OBJECT_DETECT_UNAVAILABLE, // nothing installed or downloadable
  DT_MASKS_OBJECT_DETECT_AI_OFF,      // AI disabled in preferences
} dt_masks_object_detect_state_t;

/** state of a detector's catalogue entry: the exact mirror of what
 * dt_masks_object_detect_launch would accept, so the label never
 * promises what the launch refuses. never cache it -- installs and
 * preference flips move it */
dt_masks_object_detect_state_t
dt_masks_object_detect_state(const struct dt_detector_t *detector);

/** GUI thread, darkroom only: run the one-shot promptless detection of
 * `detector` as a cancellable background job -- no interactive session,
 * no canvas freeze. the job downloads the task's model first when the
 * state said DOWNLOAD, renders the image, runs the detector, finalises
 * through the native precise-mask pass and applies the produced raster
 * shape to `module` (nullable). the job writes the promptless v2
 * provenance recipe, so the mask is regenerable like a clicked one.
 * FALSE when nothing was launched (state refuses, or another
 * finalisation holds the token) */
gboolean dt_masks_object_detect_launch(const struct dt_detector_t *detector,
                                       dt_iop_module_t *module);
#endif

/** the validated point of a raster shape: NULL when the form is not one
 * or its serialized blob has a layout this build does not know */
dt_masks_point_raster_t *dt_masks_raster_point(const dt_masks_form_t *form);
/** the mask file path a raster point names for `img`: content-addressed
 * from the recipe fingerprint under the local mask root, or the fallback
 * leaf under that same root. NULL when it names nothing. caller frees */
gchar *dt_masks_raster_resolve_path(const dt_masks_point_raster_t *pt,
                                    const dt_image_t *img);
/** sample the file a raster point names at `n` level-0 file coordinates
 * (x, y pairs in the sampler's corner indexing, the render's own frame)
 * into `out`, through the shared decoded cache; `step` is the file
 * distance between adjacent samples and picks the minification level
 * the render would pick. FALSE when the file cannot be read */
gboolean dt_masks_raster_sample(const dt_masks_point_raster_t *pt,
                                const dt_image_t *img,
                                const float *file_pts,
                                const size_t n,
                                const float step,
                                float *out);

/** THE question every "add shape" entry point asks before mutating
 * anything: would starting a new shape destroy work in progress? starting
 * one goes through dt_masks_change_form_gui, which clears the form gui and
 * with it the per-type scratchpad of the tool currently computing. keep
 * this a pure predicate: it is read from gestures AND from the widget
 * refresh below, which must not have side effects */
gboolean dt_masks_shapes_locked(void);

/** re-derive the sensitivity (and the explanatory tooltip) of every "add
 * shape" button of every module from the predicate above. assignment, never
 * a toggle: no exit path can leave a button stuck, the last call wins and
 * the truth is re-read each time. cheap and idempotent, made to be called
 * from a timer as well as from the state transitions */
void dt_masks_update_shapes_sensitivity(void);

/** init dt_masks_form_gui_t struct with default values */
void dt_masks_init_form_gui(dt_masks_form_gui_t *gui);

/** get points in real space with respect of distortion dx and dy are
 * used to eventually move the center of the circle */
int dt_masks_get_points_border(dt_develop_t *dev,
                               dt_masks_form_t *form,
                               float **points,
                               int *points_count,
                               float **border,
                               int *border_count,
                               const int source,
                               const dt_iop_module_t *module);

/** get the rectangle which include the form and his border */
int dt_masks_get_area(const dt_iop_module_t *module,
                      const dt_dev_pixelpipe_iop_t *piece,
                      dt_masks_form_t *form,
                      int *width,
                      int *height,
                      int *posx,
                      int *posy);
int dt_masks_get_source_area(dt_iop_module_t *module,
                             dt_dev_pixelpipe_iop_t *piece,
                             dt_masks_form_t *form,
                             int *width,
                             int *height,
                             int *posx,
                             int *posy);
/** get the transparency mask of the form and his border */
static inline int dt_masks_get_mask(const dt_iop_module_t *const module,
                                    const dt_dev_pixelpipe_iop_t *const piece,
                                    dt_masks_form_t *const form,
                                    float **buffer,
                                    int *width,
                                    int *height,
                                    int *posx,
                                    int *posy)
{
  return (form->functions && form->functions->get_mask)
    ? form->functions->get_mask(module, piece, form, buffer, width, height, posx, posy)
    : 0;
}

static inline int dt_masks_get_mask_roi(const dt_iop_module_t *const module,
                                        const dt_dev_pixelpipe_iop_t *const piece,
                                        dt_masks_form_t *const form,
                                        const dt_iop_roi_t *roi,
                                        float *buffer)
{
  return (form->functions && form->functions->get_mask_roi)
    ? form->functions->get_mask_roi(module, piece, form, roi, buffer)
    : 0;
}

int dt_masks_group_render(dt_iop_module_t *module,
                          dt_dev_pixelpipe_iop_t *piece,
                          dt_masks_form_t *form,
                          float **buffer,
                          int *roi,
                          const float scale);
int dt_masks_group_render_roi(dt_iop_module_t *module,
                              dt_dev_pixelpipe_iop_t *piece,
                              dt_masks_form_t *form,
                              const dt_iop_roi_t *roi,
                              float *buffer);

// returns current masks version
int dt_masks_version(void);

// update masks from older versions
int dt_masks_legacy_params(dt_develop_t *dev,
                           void *params,
                           const int old_version,
                           const int new_version);
/*
 * TODO:
 *
 * int
 * dt_masks_legacy_params(
 *   dt_develop_t *dev,
 *   const void *const old_params, const int old_version,
 *   void *new_params,             const int new_version);
 */

/** we create a completely new form. */
dt_masks_form_t *dt_masks_create(dt_masks_type_t type);
/** we create a completely new form and add it to darktable.develop->allforms. */
dt_masks_form_t *dt_masks_create_ext(dt_masks_type_t type);
/** replace dev->forms with forms */
void dt_masks_replace_current_forms(dt_develop_t *dev, GList *forms);
/** returns a form with formid == id from a list of forms */
dt_masks_form_t *dt_masks_get_from_id_ext(GList *forms, dt_mask_id_t id);
/** returns a form with formid == id from dev->forms */
dt_masks_form_t *dt_masks_get_from_id(const dt_develop_t *dev, dt_mask_id_t id);
/** check if a form is used by a given module (directly or as a child of its group) */
gboolean dt_masks_is_in_module(dt_mask_id_t maskid, const struct dt_iop_module_t *module);

/** whether this group still carries the name darktable wrote into it when the
 *  module took it, rather than one a photographer typed. the mask manager asks
 *  before repeating on a row the module name that name already contains */
gboolean dt_masks_group_name_is_default(const dt_masks_form_t *grp,
                                        const struct dt_iop_module_t *module);
/** register forms into the mask manager */
void dt_masks_register_forms(dt_develop_t *dev,
                             GList *forms);

/** read the forms from the db */
void dt_masks_read_masks_history(dt_develop_t *dev, const dt_imgid_t imgid);
/** write the forms into the db */
void dt_masks_write_masks_history_item(const dt_imgid_t imgid,
                                       const int num,
                                       const dt_masks_form_t *form);
void dt_masks_free_form(dt_masks_form_t *form);
void dt_masks_cleanup_unused(dt_develop_t *dev);

/** function used to manipulate forms for masks */
void dt_masks_change_form_gui(dt_masks_form_t *newform);
void dt_masks_clear_form_gui(const dt_develop_t *dev);

/** the shape shown in yellow over the photograph, the way a module's
    "display mask" shows its own: one at a time, rendered at the end of
    the full pipe by iop/gamma.c. NO_MASKID puts it away. the module's
    yellow and this one share the photograph, so lighting one puts the
    other out */
void dt_masks_preview_shape(dt_develop_t *dev, const dt_mask_id_t formid);
/** is `formid` the shape shown that way, on the image on screen */
gboolean dt_masks_preview_is(const dt_develop_t *dev, const dt_mask_id_t formid);
void dt_masks_reset_form_gui(void);
void dt_masks_reset_show_masks_icons(void);

gboolean dt_masks_events_mouse_moved(struct dt_iop_module_t *module,
                                     const float x,
                                     const float y,
                                     const double pressure,
                                     const int which,
                                     const float zoom_scale);
gboolean dt_masks_events_button_released(struct dt_iop_module_t *module,
                                         const float x,
                                         const float y,
                                         const int which,
                                         const uint32_t state,
                                         const float zoom_scale);
gboolean dt_masks_events_button_pressed(struct dt_iop_module_t *module,
                                        const float x,
                                        const float y,
                                        const double pressure,
                                        const int which,
                                        const int type,
                                        const uint32_t state);
gboolean dt_masks_events_mouse_scrolled(struct dt_iop_module_t *module,
                                        const float x,
                                        const float y,
                                        const gboolean up,
                                        const uint32_t state);
// Return TRUE if scrolling over the center view should adjust the visible
// mask (size/border/opacity) instead of zoom/pan. Returns FALSE while drawing
// a path, since path creation has no scroll-adjustable parameter.
gboolean dt_masks_scroll_over_mask(void);
void dt_masks_events_post_expose(const struct dt_iop_module_t *module,
                                 cairo_t *cr,
                                 const int32_t width,
                                 const int32_t height,
                                 const float pointerx,
                                 const float pointery,
                                 const float zoom_scale);
gboolean dt_masks_events_mouse_leave(struct dt_iop_module_t *module);
gboolean dt_masks_events_mouse_enter(struct dt_iop_module_t *module);

/** functions used to manipulate gui data */
void dt_masks_gui_form_create(dt_masks_form_t *form,
                              dt_masks_form_gui_t *gui,
                              const int index,
                              const struct dt_iop_module_t *module);
void dt_masks_gui_form_remove(dt_masks_form_t *form,
                              dt_masks_form_gui_t *gui,
                              const int index);
// Constrain a drag target (in preview/processed-pipe pixel coords, wd/ht =
// processed image size) so it stays within the image expanded by
// DT_MASKS_MOVE_MARGIN. Used when translating a whole form / its anchor / clone
// source so the dragged control point stays within the image or reasonably
// close, instead of being movable to an arbitrary distance where the shape
// would be lost.
void dt_masks_clamp_move_pts(float *pts, const float wd, const float ht);
void dt_masks_gui_form_test_create(dt_masks_form_t *form,
                                   dt_masks_form_gui_t *gui,
                                   const struct dt_iop_module_t *module);
/** how the next drawn shape combines with the ones already in its mask, armed
 *  before anything is drawn. DT_MASKS_STATE_NONE clears it, and any bit
 *  outside DT_MASKS_STATE_OP is dropped. it outlives the form on purpose: a
 *  run of three subtractions is one click and not three. the base is never
 *  affected -- it carries no operator and cannot be given one.
 *  `module` is the module whose mask it was armed for, and it is honoured for
 *  that one only: every drawn shape in darktable is saved through the same
 *  funnel, including shapes drawn from another module's blending panel and the
 *  clone circles iop/spots.c builds when it converts a legacy edit. */
void dt_masks_set_next_operator(const dt_masks_state_t op,
                                struct dt_iop_module_t *module);

void dt_masks_gui_form_save_creation(dt_develop_t *dev,
                                     struct dt_iop_module_t *module,
                                     dt_masks_form_t *form,
                                     dt_masks_form_gui_t *gui);
void dt_masks_group_ungroup(dt_masks_form_t *dest_grp, dt_masks_form_t *grp);
void dt_masks_group_update_name(dt_iop_module_t *module);
dt_masks_point_group_t *dt_masks_group_add_form(dt_masks_form_t *grp,
                                                const dt_masks_form_t *form);

void dt_masks_iop_edit_toggle_callback(GtkToggleButton *togglebutton,
                                       struct dt_iop_module_t *module);
void dt_masks_iop_value_changed_callback(GtkWidget *widget,
                                         struct dt_iop_module_t *module);
dt_masks_edit_mode_t dt_masks_get_edit_mode(void);
gboolean dt_masks_is_restricted_mode(void);
void dt_masks_set_edit_mode(struct dt_iop_module_t *module,
                            const dt_masks_edit_mode_t value);
void dt_masks_set_edit_mode_single_form(struct dt_iop_module_t *module,
                                        const dt_mask_id_t formid,
                                        const dt_masks_edit_mode_t value);
void dt_masks_iop_update(struct dt_iop_module_t *module);
void dt_masks_iop_combo_populate(GtkWidget *w,
                                 struct dt_iop_module_t **m);
void dt_masks_iop_use_same_as(struct dt_iop_module_t *module,
                              struct dt_iop_module_t *src);
gboolean dt_masks_iop_add_exist(struct dt_iop_module_t *module,
                                const dt_mask_id_t formid);
dt_hash_t dt_masks_group_hash(dt_hash_t hash, dt_masks_form_t *form);

void dt_masks_form_remove(struct dt_iop_module_t *module,
                          dt_masks_form_t *grp,
                          dt_masks_form_t *form);
float dt_masks_form_change_opacity(dt_masks_form_t *form,
                                   const dt_imgid_t parentid,
                                   const float amount);
void dt_masks_form_move(dt_masks_form_t *grp,
                        const dt_mask_id_t formid,
                        const gboolean up);
int dt_masks_form_duplicate(dt_develop_t *dev,
                            const dt_mask_id_t formid);
/* returns a duplicate tof form, including the formid */
dt_masks_form_t *dt_masks_dup_masks_form(const dt_masks_form_t *form);
/* duplicate the list of forms, replace item in the list with form with the same formid */
GList *dt_masks_dup_forms_deep(GList *forms, dt_masks_form_t *form);

/** utils functions */
gboolean dt_masks_point_in_form_exact(const float x,
                                      const float y,
                                      const float *points,
                                      const int points_start,
                                      const int points_count);
gboolean dt_masks_point_in_form_near(const float x,
                                     const float y,
                                     const float *points,
                                     const int points_start,
                                     const int points_count,
                                     const float distance,
                                     int *near);
float dt_masks_drag_factor(dt_masks_form_gui_t *gui,
                           const int index,
                           const int k,
                           const gboolean border);

float dt_masks_change_size(const gboolean up,
                           const float value,
                           const float min,
                           const float max);

float dt_masks_change_rotation(const gboolean up,
                               const float value,
                               const gboolean is_degree);

/** allow to select a shape inside an iop */
void dt_masks_select_form(struct dt_iop_module_t *module,
                          const dt_masks_form_t *sel);

/** utils for selecting the source of a clone mask while creating it */
void dt_masks_draw_clone_source_pos(cairo_t *cr,
                                    const float zoom_scale,
                                    const float x,
                                    const float y);
void dt_masks_set_source_pos_initial_state(dt_masks_form_gui_t *gui,
                                           const uint32_t state,
                                           const float pzx,
                                           const float pzy);
void dt_masks_set_source_pos_initial_value(dt_masks_form_gui_t *gui,
                                           const int mask_type,
                                           dt_masks_form_t *form,
                                           const float pzx,
                                           const float pzy);
void dt_masks_calculate_source_pos_value(const dt_masks_form_gui_t *gui,
                                         const int mask_type,
                                         const float initial_xpos,
                                         const float initial_ypos,
                                         const float xpos,
                                         const float ypos,
                                         float *px,
                                         float *py,
                                         const int adding);

/** detail mask support */
float *dt_masks_calc_scharr_mask(struct dt_dev_pixelpipe_t *pipe,
                                 float *src,
                                 const int width,
                                 const int height,
                                 const gboolean rawmode);
float *dt_masks_calc_detail_mask(struct dt_dev_pixelpipe_iop_t *piece,
                                 const float threshold,
                                 const gboolean detail);
void dt_masks_calc_detail_blend(float *const src,
                                float *out,
                                const size_t msize,
                                const float threshold,
                                const gboolean detail);


/** return the list of possible mouse actions */
GSList *dt_masks_mouse_actions(const dt_masks_form_t *form);

void dt_group_events_post_expose(cairo_t *cr,
                                 const float zoom_scale,
                                 dt_masks_form_t *form,
                                 dt_masks_form_gui_t *gui);


/******************************************************
 * code for dynamic handling of intermediate buffers
 * buffer for floats
 */
static inline gboolean _dt_masks_dynbuf_growto(dt_masks_dynbuf_t *a,
                                               const size_t newsize)
{
  float *newbuf = dt_alloc_align_float(newsize);
  if (!newbuf)
  {
    // not much we can do here except emit an error message
    dt_print(DT_DEBUG_ALWAYS,
             "critical: out of memory for dynbuf '%s' with size request %zu!",
             a->tag, newsize);
    return FALSE;
  }
  if (a->buffer)
  {
    memcpy(newbuf, a->buffer, a->size * sizeof(float));
    dt_print(DT_DEBUG_MASKS, "[masks dynbuf '%s'] grows to size %lu (is %p, was %p)",
             a->tag,
             (unsigned long)a->size, newbuf, a->buffer);
    dt_free_align(a->buffer);
  }
  a->size = newsize;
  a->buffer = newbuf;
  return TRUE;
}

static inline
dt_masks_dynbuf_t *dt_masks_dynbuf_init(const size_t size, const char *tag)
{
  assert(size > 0);
  dt_masks_dynbuf_t *a = (dt_masks_dynbuf_t *)calloc(1, sizeof(dt_masks_dynbuf_t));

  if(a != NULL)
  {
    g_strlcpy(a->tag, tag, sizeof(a->tag)); //only for debugging purposes
    a->pos = 0;
    if(_dt_masks_dynbuf_growto(a, size))
      dt_print(DT_DEBUG_MASKS, "[masks dynbuf '%s'] with initial size %lu (is %p)",
               a->tag,
               (unsigned long)a->size, a->buffer);
    if(a->buffer == NULL)
    {
      free(a);
      a = NULL;
    }
  }
  return a;
}

static inline
void dt_masks_dynbuf_add(dt_masks_dynbuf_t *a, const float value)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos == a->size, 0))
  {
    if (a->size == 0 || !_dt_masks_dynbuf_growto(a, 2 * a->size))
      return;
  }
  a->buffer[a->pos++] = value;
}

static inline
void dt_masks_dynbuf_add_2(dt_masks_dynbuf_t *a, const float value1, const float value2)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos + 2 >= a->size, 0))
  {
    if (a->size == 0 || !_dt_masks_dynbuf_growto(a, 2 * (a->size+1)))
      return;
  }
  a->buffer[a->pos++] = value1;
  a->buffer[a->pos++] = value2;
}

// Return a pointer to N floats past the current end of the dynbuf's
// contents, marking them as already in use.  The caller should then
// fill in the reserved elements using the returned pointer.
static inline
float *dt_masks_dynbuf_reserve_n(dt_masks_dynbuf_t *a, const int n)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos + n >= a->size, 0))
  {
    if(a->size == 0) return NULL;
    size_t newsize = a->size;
    while(a->pos + n >= newsize) newsize *= 2;
    if (!_dt_masks_dynbuf_growto(a, newsize))
    {
      return NULL;
    }
  }
  // get the current end of the (possibly reallocated) buffer, then
  // mark the next N items as in-use
  float *reserved = a->buffer + a->pos;
  a->pos += n;
  return reserved;
}

static inline
void dt_masks_dynbuf_add_zeros(dt_masks_dynbuf_t *a, const int n)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos + n >= a->size, 0))
  {
    if(a->size == 0) return;
    size_t newsize = a->size;
    while(a->pos + n >= newsize) newsize *= 2;
    if (!_dt_masks_dynbuf_growto(a, newsize))
    {
      return;
    }
  }
  // now that we've ensured a sufficiently large buffer add N zeros to
  // the end of the existing data
  memset(a->buffer + a->pos, 0, n * sizeof(float));
  a->pos += n;
}


static inline
float dt_masks_dynbuf_get(dt_masks_dynbuf_t *a, const int offset)
{
  assert(a != NULL);
  // offset: must be negative distance relative to end of buffer
  assert(offset < 0);
  assert((long)a->pos + offset >= 0);
  return (a->buffer[a->pos + offset]);
}

static inline
float dt_masks_dynbuf_get_absolute(dt_masks_dynbuf_t *a, const int position)
{
  assert(a != NULL);
  assert(position >= 0);
  assert((long)a->pos > position);
  return (a->buffer[position]);
}

static inline
void dt_masks_dynbuf_set(dt_masks_dynbuf_t *a, const int offset, const float value)
{
  assert(a != NULL);
  // offset: must be negative distance relative to end of buffer
  assert(offset < 0);
  assert((long)a->pos + offset >= 0);
  a->buffer[a->pos + offset] = value;
}

static inline
void dt_masks_dynbuf_set_absolute(dt_masks_dynbuf_t *a, const int position, const float value)
{
  assert(a != NULL);
  assert(position >= 0);
  assert((long)a->pos > position);
  a->buffer[position] = value;
}

static inline
float *dt_masks_dynbuf_buffer(dt_masks_dynbuf_t *a)
{
  assert(a != NULL);
  return a->buffer;
}

static inline
size_t dt_masks_dynbuf_position(dt_masks_dynbuf_t *a)
{
  assert(a != NULL);
  return a->pos;
}

static inline
void dt_masks_dynbuf_reset_position(dt_masks_dynbuf_t *a, const size_t newpos)
{
  assert(a != NULL);
  assert(newpos <= a->pos);
  a->pos = newpos;
}

static inline
void dt_masks_dynbuf_reset(dt_masks_dynbuf_t *a)
{
  assert(a != NULL);
  a->pos = 0;
}

static inline
float *dt_masks_dynbuf_harvest(dt_masks_dynbuf_t *a)
{
  // take out data buffer and make dynamic buffer obsolete
  if(a == NULL) return NULL;
  float *r = a->buffer;
  a->buffer = NULL;
  a->pos = a->size = 0;
  return r;
}

static inline
void dt_masks_dynbuf_free(dt_masks_dynbuf_t *a)
{
  if(a == NULL) return;
  dt_print(DT_DEBUG_MASKS, "[masks dynbuf '%s'] freed (was %p)", a->tag,
          a->buffer);
  dt_free_align(a->buffer);
  free(a);
}

// Dump buffer to file for debugging.
static inline
void dt_masks_dynbuf_debug_print(dt_masks_dynbuf_t *a, gboolean to_stdout)
{
  if(a == NULL) return;
  if (to_stdout)
  {
    printf("'%s' buffer: ", a->tag);
    for (size_t i = 0; i < a->pos; i += 2)
    {
      printf("(%f %f), ", a->buffer[i], a->buffer[i+1]);
    }
    printf("\n");
  }
  else
  {
    FILE *f;
    char filename[255] = { 0 };
    sprintf(filename, "debug-%ld-%s", time(NULL), a->tag);
    f = g_fopen(filename, "w");
    for (size_t i = 0; i < a->pos; i += 2)
    {
      fprintf(f, "%f %f\n", a->buffer[i], a->buffer[i+1]);
    }
    fclose(f);
  }
}

/******************************************************
 * code for dynamic handling of intermediate buffers
 * buffer for ints
 */
static inline gboolean _dt_masks_intbuf_growto(dt_masks_intbuf_t *a,
                                               const size_t newsize)
{
  int *newbuf = dt_alloc_align_int(newsize);
  if (!newbuf)
  {
    // not much we can do here except emit an error message
    dt_print(DT_DEBUG_ALWAYS,
             "critical: out of memory for intbuf '%s' with size request %zu!",
             a->tag, newsize);
    return FALSE;
  }
  if (a->buffer)
  {
    memcpy(newbuf, a->buffer, a->size * sizeof(int));
    dt_print(DT_DEBUG_MASKS, "[masks intbuf '%s'] grows to size %lu (is %p, was %p)",
             a->tag,
             (unsigned long)a->size, newbuf, a->buffer);
    dt_free_align(a->buffer);
  }
  a->size = newsize;
  a->buffer = newbuf;
  return TRUE;
}


static inline
dt_masks_intbuf_t *dt_masks_intbuf_init(const size_t size, const char *tag)
{
  assert(size > 0);
  dt_masks_intbuf_t *a = (dt_masks_intbuf_t *)calloc(1, sizeof(dt_masks_intbuf_t));

  if(a != NULL)
  {
    g_strlcpy(a->tag, tag, sizeof(a->tag)); //only for debugging purposes
    a->pos = 0;
    if(_dt_masks_intbuf_growto(a, size))
      dt_print(DT_DEBUG_MASKS, "[masks intbuf '%s'] with initial size %lu (is %p)",
               a->tag,
               (unsigned long)a->size, a->buffer);
    if(a->buffer == NULL)
    {
      free(a);
      a = NULL;
    }
  }
  return a;
}


static inline
void dt_masks_intbuf_add2(dt_masks_intbuf_t *a, const float value1, const float value2)
{
  assert(a != NULL);
  assert(a->pos <= a->size);
  if(__builtin_expect(a->pos + 2 >= a->size, 0))
  {
    if (a->size == 0 || !_dt_masks_intbuf_growto(a, 2 * (a->size+1)))
      return;
  }
  a->buffer[a->pos++] = value1;
  a->buffer[a->pos++] = value2;
}

static inline
size_t dt_masks_intbuf_position(dt_masks_intbuf_t *a)
{
  assert(a != NULL);
  return a->pos;
}

static inline
void dt_masks_intbuf_free(dt_masks_intbuf_t *a)
{
  if(a == NULL) return;
  dt_print(DT_DEBUG_MASKS, "[masks intbuf '%s'] freed (was %p)", a->tag,
          a->buffer);
  dt_free_align(a->buffer);
  free(a);
}

// Dump buffer to file for debugging.
/*
static inline
void dt_masks_intnbuf_debug_print(dt_masks_intbuf_t *a)
{
  if(a == NULL) return;
  FILE *f;
  char filename[255] = { 0 };
  sprintf(filename, "debug-%ld-%s", time(NULL), a->tag);
  f = g_fopen(filename, "w");
  for (size_t i = 0; i < a->pos; i += 2)
  {
    fprintf(f, "%d %d\n", a->buffer[i], a->buffer[i+1]);
  }
  fclose(f);
}
*/

/* End of dynamic buffer code
 ******************************************************/

static inline
int dt_masks_roundup(const int num, const int mult)
{
  const int rem = num % mult;

  return (rem == 0) ? num : num + mult - rem;
}

#define DT_MASKS_CONF(type, shape, param) \
  (type & (DT_MASKS_CLONE | DT_MASKS_NON_CLONE) \
   ? "plugins/darkroom/spots/" #shape "_" #param \
   : "plugins/darkroom/masks/" #shape "/" #param)

void dt_masks_draw_anchor(cairo_t *cr,
                          const gboolean selected,
                          const float zoom_scale,
                          const float x,
                          const float y);

/* draw the small control point for selected anchor in path & brush */
void dt_masks_draw_ctrl(cairo_t *cr,
                        const float x,
                        const float y,
                        const float zoom_scale,
                        const gboolean selected);

/* find the closest to point (px, py) in points array.
   nb_ctrl is the number of points (control points) to
   skip at the start of points.
*/
void dt_masks_closest_point(const int count,
                            const int nb_ctrl,
                            const float *points,
                            const float px,
                            const float py,
                            float *x,
                            float *y);

/* Rotate the control points of a path/brush outline in screen space and project
   them back to normalized image coordinates. `gpt_points` is the gui display
   buffer (interleaved x,y) whose first `nb*3` pairs are the control points,
   stored per node as ctrl1, corner, ctrl2; `points_count` is its number of
   (x,y) pairs. Each control point is rotated by (cos_a, sin_a) around the screen
   pivot (cx, cy), back-transformed through the pipe in a single batch, and
   written to `out` (normalized, same interleaving, nb*6 floats). Shared by the
   path and brush rotate gestures. */
void dt_masks_rotate_ctrl_points(dt_develop_t *dev,
                                 const float *const gpt_points,
                                 const int points_count,
                                 const int nb,
                                 const float cx,
                                 const float cy,
                                 const float cos_a,
                                 const float sin_a,
                                 const float iwidth,
                                 const float iheight,
                                 float *const out);

/* draw a line from -> to with an arrow at the end.
   if touch_dest is true then the arrow will be at the
   (to_x, to_y) location, otherwise a small space will
   be left.
*/
void dt_masks_draw_arrow(cairo_t *cr,
                         const float from_x,
                         const float from_y,
                         const float to_x,
                         const float to_y,
                         const float zoom_scale,
                         const gboolean touch_dest);

/* stroke the arrow on cr depending on selection */
void dt_masks_stroke_arrow(cairo_t *cr,
                           const dt_masks_form_gui_t *gui,
                           const int group,
                           const float zoom_scale);

/* set line width for the mask drawing depending on the status
   border, source & selected
*/
void dt_masks_line_stroke(cairo_t *cr,
                          const gboolean border,
                          const gboolean source,
                          const gboolean selected,
                          const float zoom_scale);

static inline float dt_masks_sensitive_dist(const float zoom_scale)
{
  return DT_PIXEL_APPLY_DPI(7) / zoom_scale;
}

static inline void dt_masks_get_image_size(float *width,
                                           float *height,
                                           float *iwidth,
                                           float *iheight)
{
  // iwidth/iheight must match preview->iwidth/iheight (= pipe->iwidth/iheight used
  // by _path_get_pts_border to scale corner coordinates before distort_transform).
  // width/height must match preview->processed_width/height, which is what both
  // dt_dev_get_preview_size() and dt_view_paint_surface FALLBACK use as canvas size.
  const dt_develop_t *dev = darktable.develop;
  const dt_dev_pixelpipe_t *preview = dev->preview_pipe;
  const float iscale = preview->iscale > 0.f ? preview->iscale : 1.f;

  // Use preview pipe's actual processed dimensions, not full.pipe/iscale.
  // The two differ by up to 1 pixel due to independent integer truncations
  // in each pipeline (e.g. after crop), causing a systematic mask overlay shift.
  // dt_dev_get_preview_size() uses the same value, so both are consistent.
  if(preview->processed_width > 0)
  {
    if(width  ) *width   = preview->processed_width;
    if(height ) *height  = preview->processed_height;
  }
  else if(dev->full.pipe && dev->full.pipe->processed_width > 0)
  {
    if(width  ) *width   = dev->full.pipe->processed_width  / iscale;
    if(height ) *height  = dev->full.pipe->processed_height / iscale;
  }
  else
  {
    if(width  ) *width   = preview->backbuf_width;
    if(height ) *height  = preview->backbuf_height;
  }

  // iwidth/iheight must equal pipe->iwidth/iheight (the pipeline input dimensions
  // used to scale corners in _path_get_pts_border / other mask get_points_border
  // functions), so that backtransform(corner * pipe->iwidth) / iwidth = corner.
  if(iwidth ) *iwidth  = preview->iwidth;
  if(iheight) *iheight = preview->iheight;

}

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
