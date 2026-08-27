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

// only glib and stddef: this contract is shared by the mask finalisation
// (which lives behind HAVE_AI) and by operators that are plain C with no
// network at all, so it must not drag either side into the other

#include <glib.h>
#include <stddef.h>

G_BEGIN_DECLS

// the matting stage of the precise raster mask, as a TABLE OF OPERATORS.
//
// the native finalisation ends inside a band around the contour: a guided
// filter re-derives sub-pixel coverage there, and the result is blended
// back into the soft hint outside it. everything that could ever replace
// that filter -- the same filter at a wider radius, a closed-form matte, a
// matting network -- reads the same inputs and writes the same plane, so it
// is registered as a LINE OF A TABLE rather than an `if` in the render
// core: {id, version, task, caps, run}.
//
// `id` and `version` are what the provenance recipe records. `version` is
// an ALGORITHM revision, bumped by any numeric change (a constant counts),
// because the recipe is hashed verbatim to name a content-addressed cache
// file: the same fingerprint must never cover two different renders. a
// replay therefore looks the operator up BY ID and fails on an unknown one
// instead of falling back to another line.
//
// THE STAGE CONTRACT carries the band, never a trimap: `hint_bin` (the
// binarised mask that drives the band logic and the filter), `hint_soft`
// (the continuous one, which keeps genuine partial coverage away from the
// contour) and the per-pixel band radius `R(x,y)`. Operators that want a
// trimap derive one from the band with dt_matte_trimap_from_band(): one
// view of one band, no second distance transform and no second threshold.

typedef enum dt_matte_caps_t
{
  DT_MATTE_CAP_NONE     = 0,
  // wants a trimap; the caller builds one from the band on demand
  DT_MATTE_NEEDS_TRIMAP = 1 << 0,
  // works the frame in overlapping tiles rather than in one shot
  DT_MATTE_TILED        = 1 << 1,
  // needs a model file: a missing or refused model DISABLES the stage,
  // it never silently selects another operator
  DT_MATTE_NEEDS_MODEL  = 1 << 2,
  // has a CPU path: a slow path is still a path, so no execution
  // provider check belongs in the render core
  DT_MATTE_CPU_OK       = 1 << 3,
  // READS `stage->band_scale`. The widening law that turns the user's
  // slider into a per-pixel radius belongs to an operator, not to the
  // render core: a line that does not declare this bit sees the very band
  // the plain stage measures, whatever the slider says, and moving the
  // slider would re-fingerprint the recipe -- a reload of the model, a
  // full inference and a new cache name -- for a bit-identical plane.
  // "One render, two names" is the same defect as "two renders, one
  // name", so the GUI HIDES the control until a line declares the bit.
  // No line of this table declares it today; the adaptive operator that
  // would is not part of this build
  DT_MATTE_USES_BAND    = 1 << 4
} dt_matte_caps_t;

// one invocation of the stage, on the native grid of the finalisation.
// every plane is width*height and owned by the caller; `out` is written,
// nothing else is
typedef struct dt_matte_stage_t
{
  const float *guide;      // width*height*4, the native render, [0,1]
  const float *hint_bin;   // width*height, 0/1 -- drives the band
  const float *hint_soft;  // width*height, [0,1] -- the continuous mask
  // R(x,y), width*height, in [r_base, r_max]. NULL means the uniform
  // r_base of the current stage, which is what every v1 operator gets
  const float *radius;
  // trimap view of the band, width*height values in {0, 0.5, 1}, built by
  // the CALLER with dt_matte_trimap_from_band() and ONLY for a line that
  // declares DT_MATTE_NEEDS_TRIMAP -- NULL for every other one, and an
  // operator that wants it must check. it is a stage field rather than
  // something the operator derives for itself because the band it views
  // is the caller's: the render core owns the summed-area table the band
  // weight comes from, and a second derivation in an operator is exactly
  // the drift this contract exists to prevent
  const float *trimap;
  int width, height;
  int r_base;              // band radius of the plain stage, in pixels
  int r_max;               // ceiling of R(x,y); == r_base when radius is NULL
  int w_gf;                // guided-filter radius of the plain stage
  float band_scale;        // the single user degree of freedom, [0.5, 2.0]
  // polled at the expensive steps, both nullable. after a FALSE return --
  // cancelled or failed, the two are not told apart here -- `out` is
  // INDETERMINATE and must not be consumed: an operator may use the
  // caller's plane as its own accumulator (the tiled one does, to avoid a
  // second full-region buffer), so what is left there is neither the
  // previous content nor a finished result. FALSE means "no plane", never
  // "the plane as it was"
  gboolean (*keep_going)(void *);
  void *user;
} dt_matte_stage_t;

typedef struct dt_matte_op_t dt_matte_op_t;

typedef gboolean (*dt_matte_run_t)(const dt_matte_op_t *const op,
                                   const dt_matte_stage_t *const stage,
                                   float *const out);

struct dt_matte_op_t
{
  const char *id;       // recipe id, fits DT_RF_RECIPE_MATTING_ID_LEN
  const char *version;  // revision, fits DT_RF_RECIPE_MATTING_VERSION_LEN
  const char *task;     // AI registry task; NULL for the pure-C operators
  // registry id of the model this line loads, BY ID and never "the active
  // model of the task" (the dt_refine_load lesson, detect.h:37). NULL
  // unless DT_MATTE_NEEDS_MODEL. it is table data and not a preference so
  // that the UX surfaces can ask whether the model of a RECORDED operator
  // is installed without loading anything
  const char *model;
  int caps;             // dt_matte_caps_t bitmask
  // the weight this line's plane carries in the composition -- the
  // `wmatte` of dt_matte_compose_px, a per-SESSION scalar and never a
  // per-pixel field. TABLE data, not something the render core computes:
  // the witness line leaves it 0, and that zero is what makes routing the
  // plain stage through the table an identity BY CONSTRUCTION. The core
  // reads it before doing anything, so a zero-weight line costs no second
  // plane, no second filter run and no evaluation of the operator term --
  // the short circuit dt_matte_compose_px documents, decided here once
  // rather than trusted per pixel
  float wmatte;
  dt_matte_run_t run;
};

/** The operator registered under `id`, or NULL when this build has no
 *  such line -- on NULL a replay FAILS, it never falls back. */
const dt_matte_op_t *dt_matte_find(const char *id);

/** The witness line: the band-limited guided filter the finalisation has
 *  always run, at the very constants it has always run it with. Pure C,
 *  no model, always present. */
extern const dt_matte_op_t dt_matte_op_gf_band;

#ifdef HAVE_MATTE_VITMATTE
/** The ViTMatte line -- PERSONAL BUILDS ONLY (USE_MATTE_VITMATTE), never
 *  part of an upstream package: the model weights derive from a dataset
 *  whose agreement forbids commercial use, so no darktable release may
 *  carry, fetch or advertise them. Declared behind the same guard as the
 *  file that defines it, so a build without the option does not merely
 *  fail to register the line -- it does not know the symbol exists. */
extern const dt_matte_op_t dt_matte_op_vitmatte;
#endif

/** Run an operator. FALSE means only "no output was produced", and it
 *  covers two situations a caller must not confuse: there was no operator
 *  to run (a NULL table entry, or a line with no `run` -- the stage is
 *  disabled, which is a normal state, not an error), or an operator ran
 *  and failed. This helper deliberately answers the same FALSE to both,
 *  because the distinction is not its to make; a caller that reports
 *  failures tests `op` itself and treats a NULL one as a silent fallback
 *  to the plain chain. Turning a disabled stage into a "filter failed"
 *  message is the mistake this note exists to prevent. */
static inline gboolean dt_matte_run(const dt_matte_op_t *const op,
                                    const dt_matte_stage_t *const stage,
                                    float *const out)
{
  return (op && op->run) ? op->run(op, stage, out) : FALSE;
}

/** Trimap view of the band: unknown wherever the band weight is nonzero,
 *  foreground/background from `hint_bin` outside it. `trimap` receives
 *  npix values in {0, 0.5, 1}.
 *
 *  ONE view of ONE band. `wband` is the very weight the composition then
 *  blends with -- no second distance transform, no second threshold, and
 *  in particular no Euclidean erosion/dilation pair: "unknown" here is
 *  the CHEBYSHEV band the summed-area window already measures, which is
 *  the (2R+1)^2 box holding both classes. That box is strictly wider than
 *  a Euclidean disc of the same radius, so this trimap hands the operator
 *  a slightly larger unknown region than a distance-transform trimap
 *  would; the pixels it adds are precisely those where `wband` is
 *  smallest, so the composition damps whatever the operator answers there
 *  towards the soft hint. Measured on the bench's fur case: +27.9% unknown
 *  pixels, median wband 0.032 over them, 0.0046 MAE against the
 *  distance-transform reference inside its 8 px error band.
 *
 *  DEFINED next to the table rather than with an operator: the CALLER
 *  builds the trimap (see DT_MATTE_NEEDS_TRIMAP) and the caller is the
 *  render core, which is compiled whether or not any line asks for one. */
void dt_matte_trimap_from_band(const float *const hint_bin,
                               const float *const wband,
                               float *const trimap,
                               const size_t npix);

/** The generic three-term composition of the stage, per pixel:
 *
 *    alpha = wband * (wmatte * alpha_op + (1 - wmatte) * alpha_gf)
 *            + (1 - wband) * hint_soft
 *
 *  `wband` fades the refined coverage out to the soft hint at the edge of
 *  the band; `wmatte` fades the plain guided filter over to the operator
 *  inside it. With `wmatte` identically zero the inner term is NOT
 *  evaluated at all, and what is left is exactly the two-term composition
 *  the finalisation has always computed, in the same order, with the same
 *  roundings. That short circuit is the whole point: the identity holds
 *  BY CONSTRUCTION, not by trusting 0*x + 1*y == y for whatever `x` an
 *  operator returned -- the literal product would propagate a NaN or an
 *  infinity, the branch discards the value whatever it is.
 *
 *  It neutralises a VALUE, not a pointer. `alpha_op` is taken by value, so
 *  the operator plane has already been read by the time this runs: keeping
 *  a possibly absent plane unread is the CALLER's job, not this function's.
 *  The canonical form of that guard is the ternary at the composition loop
 *  in object.c -- `alpha_op ? CLAMPF(alpha_op[k], 0.f, 1.f) : refined` --
 *  which reads nothing when there is no plane and hands over a value this
 *  function ignores anyway while `wmatte` is zero. The scalar signature is
 *  deliberate: it folds away completely at this call site. */
static inline float dt_matte_compose_px(const float hint_soft,
                                        const float alpha_gf,
                                        const float alpha_op,
                                        const float wband,
                                        const float wmatte)
{
  const float op = (wmatte > 0.0f)
                   ? (wmatte * alpha_op + (1.0f - wmatte) * alpha_gf)
                   : alpha_gf;
  return wband * op + (1.0f - wband) * hint_soft;
}

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
