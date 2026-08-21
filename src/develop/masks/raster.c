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

#include "develop/imageop.h"
#include "develop/masks.h"

// the raster shape: a persistent mask form whose content is not a geometry
// but a reference to a raster mask file (see dt_masks_point_raster_t in
// masks.h). the form renders by sampling the decoded file, so it has no
// editable anchors and no creation gesture: instances are built by code
// (the precise-mask finalisation) and behave as regular group members
// everywhere else -- combination, opacity, duplication, removal

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
  .get_mask_roi = NULL,
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
