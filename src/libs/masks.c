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
#include "common/gdk_event_utils.h"

#include "develop/masks.h"
#include "bauhaus/bauhaus.h"
#include "common/darktable.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "gui/accelerators.h"
#include "gui/draw.h"
#include "gui/gtk.h"
#include "gui/preferences.h"
#include "libs/lib.h"
#include "libs/lib_api.h"

DT_MODULE(1)

#pragma GCC diagnostic ignored "-Wshadow"

static void _lib_masks_recreate_list(dt_lib_module_t *self);
static void _lib_masks_update_list(dt_lib_module_t *self);
static void _lib_masks_selection_change(dt_lib_module_t *self,
                                        struct dt_iop_module_t *module,
                                        const dt_mask_id_t selectid);
static gboolean _lib_masks_selection_change_r(GtkTreeModel *model,
                                              GtkTreeSelection *selection,
                                              GtkTreeIter *iter,
                                              struct dt_iop_module_t *module,
                                              const dt_mask_id_t selectid,
                                              const int level);
static void _lib_masks_get_values(GtkTreeModel *model,
                                  GtkTreeIter *iter,
                                  dt_iop_module_t **module,
                                  dt_mask_id_t *groupid,
                                  dt_mask_id_t *formid);
static gboolean
_update_foreach(GtkTreeModel *model, GtkTreePath *path, GtkTreeIter *iter, gpointer data);

// three states, computed once per library row by _shape_scope() below
typedef enum dt_masks_shape_scope_t
{
  DT_MASKS_SCOPE_MODULE = 0,  // at least one module renders it
  DT_MASKS_SCOPE_GROUP_ONLY,  // only filed in groups no module renders
  DT_MASKS_SCOPE_ORPHAN       // nothing references it at all
} dt_masks_shape_scope_t;

static dt_masks_shape_scope_t _shape_scope(const dt_mask_id_t formid,
                                           char *groups,
                                           const size_t groups_length);

typedef struct dt_lib_masks_t
{
  /* vbox with managed history items */
  GtkWidget *bt_circle, *bt_path, *bt_gradient, *bt_ellipse, *bt_brush;
#ifdef HAVE_AI
  GtkWidget *bt_object;
#endif
  // the manager shows two lists now. `treeview` holds the masks: the groups,
  // with their members in the order they are applied. `library` holds the
  // shapes, each exactly once, as itself -- where a shape is named, duplicated
  // and deleted for good. a shape used by three modules is ONE library row and
  // three mask rows. two views mean two stores and two selections, so
  // everything that used to read "the" selection reads _masks_active_view()
  GtkWidget *treeview, *library;
  // the view the last selection change or right-click happened in. the context
  // menu is built now and its entries read the selection back when they fire,
  // so which list they act on has to be decided when the menu opens, once
  GtkWidget *active_view;
  // the library's column and name cell, so "rename" can open the in-place
  // editor on the row the photographer pointed at
  GtkTreeViewColumn *lib_col;
  GtkCellRenderer *lib_name_cell;
  // caption under the library, shown only when at least one shape is not
  // linked to a module: it names exactly the set the cleanup is about
  GtkWidget *lib_unlinked;
  dt_gui_collapsible_section_t cs;
  GtkWidget *property[DT_MASKS_PROPERTY_LAST];
  GtkWidget *pressure, *smoothing;
  float last_value[DT_MASKS_PROPERTY_LAST];
  GtkWidget *none_label;
  // path-only shrink/grow (outset/inset) control. The slider is a signed offset
  // measured from the shape's baseline (captured by the path mask the first time
  // it is resized); 0 restores it. The unit (px / %) is a toggle in the slider's
  // quad. The baseline and the per-offset result cache live in the path mask
  // code, so the slider just sets and mirrors the current offset.
  GtkWidget *resize_box, *resize_amount;
  guint resize_timer;       // debounce source id (0 = none)
  gboolean resize_updating; // guard: programmatic slider change, don't commit

  GdkPixbuf *ic_inverse, *ic_union, *ic_intersection;
  GdkPixbuf *ic_difference, *ic_sum, *ic_exclusion, *ic_used;

  // a selection requested (e.g. right after creating a shape) before the tree
  // had the matching row: re-applied once gui_update rebuilds the tree. 0 = none.
  dt_mask_id_t pending_selectid;
  struct dt_iop_module_t *pending_selmodule;

  // structure hash of the tree as last built; when unchanged (e.g. a slider edit
  // that only alters shape parameters) gui_update refreshes rows in place instead
  // of recreating the store, so the panel scroll never moves. See _forms_structure_hash.
  guint tree_hash;
  gboolean tree_hash_valid;

  // the creation bar: one row, under both lists, saying where the next drawn
  // shape is about to go and letting it be called off. arm_module is this
  // panel's claim on the creation in flight, not a second source of truth for
  // it: a shape button in a module's own blending panel arms the very same
  // canvas and reaches the very same proxy, and only this tells the two apart
  GtkWidget *creation_bar, *creation_label;
  struct dt_iop_module_t *arm_module;
} dt_lib_masks_t;

static void _resize_update(dt_lib_masks_t *d);
static void _creation_bar_update(dt_lib_masks_t *d);
static void _creation_end_continuous(void);

#define DT_MASKS_NVIEWS 2

// 0 = the masks, 1 = the shape library. an index rather than two named fields
// so that "do it to both" stays a one-line loop everywhere below
static GtkWidget *_masks_view(dt_lib_masks_t *lm, const int v)
{
  return v == 0 ? lm->treeview : lm->library;
}

// the view an action applies to. pinned by every right-click before the menu
// exists, and by every selection that is not a deselection
static GtkWidget *_masks_active_view(dt_lib_masks_t *lm)
{
  return lm->active_view ? lm->active_view : lm->treeview;
}

const char *name(dt_lib_module_t *self)
{
  return _("mask manager");
}

const char *description(dt_lib_module_t *self)
{
  return _("manipulate the drawn shapes used\n"
           "for masks on the processing modules");
}

dt_view_type_flags_t views(dt_lib_module_t *self)
{
  return DT_VIEW_DARKROOM;
}

uint32_t container(dt_lib_module_t *self)
{
  return DT_UI_CONTAINER_PANEL_LEFT_CENTER;
}

int position(const dt_lib_module_t *self)
{
  return 10;
}

static gboolean _selected_masks_are_used(GtkTreeModel *model,
                                         GList *selected,
                                         dt_iop_module_t *module)
{
  for(GList *item = selected; item; item = g_list_next(item))
  {
    GtkTreePath *ipath = (GtkTreePath *)item->data;
    GtkTreeIter iter;
    if(gtk_tree_model_get_iter(model, &iter, ipath))
    {
      dt_mask_id_t id = INVALID_MASKID;
      _lib_masks_get_values(model, &iter, NULL, NULL, &id);

      if(dt_is_valid_maskid(id)
         && dt_masks_is_in_module(id, module))
      {
        return TRUE;
      }
    }
  }

  return FALSE;
}

void expanded_state(dt_lib_module_t *self,
                    const gboolean expanded)
{
  // if not expanded we may want to disable the mask

  if (!expanded)
  {
    dt_develop_t *dev = darktable.develop;
    dt_iop_module_t *mod = dev->gui_module;
    const dt_iop_gui_blend_data_t *bd = mod ? mod->blend_data : NULL;

    /*
      We hide the masks if:
      - If the active module is not enabled
      - If the active module is enabled but has no mask
      - There is some selected masks in mask manage
      - If the active module is enabled but the selected masks are not used by the
        module.
    */

    dt_lib_masks_t *lm = self->data;

    // the selection can be in either list; the module keeps its mask shown if
    // what is selected -- wherever it is selected -- is used by that module.
    // the list of rows was never freed here either
    gboolean sel_used = FALSE;

    for(int v = 0; v < DT_MASKS_NVIEWS; v++)
    {
      GtkWidget *view = _masks_view(lm, v);
      if(!view) continue;

      GtkTreeModel *model = NULL;
      GList *selected = gtk_tree_selection_get_selected_rows
        (gtk_tree_view_get_selection(GTK_TREE_VIEW(view)), &model);
      if(selected && _selected_masks_are_used(model, selected, mod)) sel_used = TRUE;
      g_list_free_full(selected, (GDestroyNotify)gtk_tree_path_free);
    }

    if (!(mod
          && mod->enabled
          && (mod->blend_params->mask_mode & DEVELOP_MASK_MASK)
          && bd->masks_shown != DT_MASKS_EDIT_OFF
          && sel_used))
    {
      // a continuous run is not part of form_gui's own cleanup: dropping the
      // form alone would leave it set, and the next shape drawn from anywhere
      // would silently chain on it. this ends whoever's run is up, the same
      // way the line below ends whoever's creation is in flight
      _creation_end_continuous();
      dt_masks_change_form_gui(NULL);
      dt_control_queue_redraw_center();
    }
  }
  else
  {
    // when the mask manager is opened, reflect the shape currently selected on
    // the canvas so its controls show right away, instead of the user having to
    // click the shape in the tree first.
    dt_develop_t *dev = darktable.develop;
    if(dt_is_valid_maskid(dev->mask_form_selected_id))
      _lib_masks_selection_change(self, dev->gui_module, dev->mask_form_selected_id);
  }
}

typedef enum dt_masks_tree_cols_t
{
  TREE_TEXT = 0,
  TREE_MODULE,
  TREE_GROUPID,
  TREE_FORMID,
  TREE_EDITABLE,
  TREE_IC_OP,
  TREE_IC_OP_VISIBLE,
  TREE_IC_INVERSE,
  TREE_IC_INVERSE_VISIBLE,
  TREE_IC_USED,
  TREE_IC_USED_VISIBLE,
  TREE_USED_TEXT,
  // rank of the shape in the application order of its module's mask, ""
  // where that order carries no meaning. TREE_BASE holds _("base") on the
  // shape that lays the buffer down, "" everywhere else. both are derived,
  // never persisted, and written by _set_iter_name only
  TREE_NUM,
  TREE_BASE,
  // library rows only: one word when no module renders the shape, "" on every
  // other row. derived like the two above, and written in the same single
  // place, _set_iter_name -- which does walk dev->iop for it, but only on the
  // rows that can carry it: the library holds one row per shape, nothing
  // nested, and every other row in either store leaves before that walk
  TREE_LINK,
  TREE_COUNT
} dt_masks_tree_cols_t;

// the store's column types, written once and indexed by the enum itself.
// gtk_tree_store_new() takes a positional variadic list, which is a second list
// to keep in step with dt_masks_tree_cols_t by hand: a mismatch there is a
// runtime fault, not a build error -- and with more than one store the list
// would have to be copied, which is exactly how two copies drift apart.
// designated initializers remove both problems: adding a column is one line
// next to the enum entry it belongs to, in one place.
// filled on the spot rather than declared `static const`: GDK_TYPE_PIXBUF is a
// function call, not a constant expression, so C will not let this be static
// initializer data.
static GtkTreeStore *_masks_store_new(void)
{
  GType types[TREE_COUNT] =
    {
      [TREE_TEXT] = G_TYPE_STRING,
      [TREE_MODULE] = G_TYPE_POINTER,
      [TREE_GROUPID] = G_TYPE_INT,
      [TREE_FORMID] = G_TYPE_INT,
      [TREE_EDITABLE] = G_TYPE_BOOLEAN,
      [TREE_IC_OP] = GDK_TYPE_PIXBUF,
      [TREE_IC_OP_VISIBLE] = G_TYPE_BOOLEAN,
      [TREE_IC_INVERSE] = GDK_TYPE_PIXBUF,
      [TREE_IC_INVERSE_VISIBLE] = G_TYPE_BOOLEAN,
      [TREE_IC_USED] = GDK_TYPE_PIXBUF,
      [TREE_IC_USED_VISIBLE] = G_TYPE_BOOLEAN,
      [TREE_USED_TEXT] = G_TYPE_STRING,
      [TREE_NUM] = G_TYPE_STRING,
      [TREE_BASE] = G_TYPE_STRING,
      [TREE_LINK] = G_TYPE_STRING,
    };

  return gtk_tree_store_newv(TREE_COUNT, types);
}

// boolean = TRUE renders as a checkbox; min/max/relative are unused
const struct
{
  gchar *name;
  gchar *format;
  float min, max;
  gboolean relative;
  gboolean boolean;
} _masks_properties[DT_MASKS_PROPERTY_LAST]
  = { [ DT_MASKS_PROPERTY_OPACITY] = {N_("opacity"), "%", 0, 1, FALSE, FALSE },
      [ DT_MASKS_PROPERTY_SIZE] = { N_("size"), "%", 0.0001, 1, TRUE, FALSE },
      [ DT_MASKS_PROPERTY_HARDNESS] = { N_("hardness"), "%", 0.0001, 1, TRUE, FALSE },
      [ DT_MASKS_PROPERTY_FEATHER] = { N_("feather"), "%", 0.0001, 1, TRUE, FALSE },
      [ DT_MASKS_PROPERTY_ROTATION] = { N_("rotation"), "°", 0, 360, FALSE, FALSE },
      [ DT_MASKS_PROPERTY_CURVATURE] = { N_("curvature"), "%", -1, 1, FALSE, FALSE },
      [ DT_MASKS_PROPERTY_COMPRESSION] = { N_("compression"), "%", 0.0001, 1, TRUE, FALSE },
      [ DT_MASKS_PROPERTY_CLEANUP] = { N_("cleanup"), "", 0, 100, FALSE, FALSE },
      [ DT_MASKS_PROPERTY_SMOOTHING] = { N_("smoothing"), "", 0, 1.3, FALSE, FALSE },
      [ DT_MASKS_PROPERTY_REFINE] = { N_("refine mask boundary"), "", 0, 1, FALSE, TRUE },
};

gboolean _timeout_show_all_feathers(gpointer userdata)
{
  dt_masks_form_gui_t *gui = userdata;
  gui->show_all_feathers = 0;
  dt_control_queue_redraw_center();
  return G_SOURCE_REMOVE;
}

static void _property_changed(GtkWidget *widget, dt_masks_property_t prop)
{
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  dt_lib_masks_t *d = self->data;
  dt_develop_t *dev = darktable.develop;
  dt_masks_form_t *form = dev->form_visible;
  dt_masks_form_gui_t *gui = dev->form_gui;
  if(!form || !gui)
  {
    gtk_widget_hide(widget);
    return;
  }

  const gboolean is_bool = _masks_properties[prop].boolean;
  const float value = is_bool
    ? (float)gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget))
    : dt_bauhaus_slider_get(widget);

  // a size/feather/rotation edit reshapes the path and drops its shrink/grow
  // baseline (see path.c); remember that so we can reset the slider to 0 below.
  const gboolean reshaped = (value != d->last_value[prop]) &&
                            (prop == DT_MASKS_PROPERTY_SIZE || prop == DT_MASKS_PROPERTY_FEATHER ||
                             prop == DT_MASKS_PROPERTY_ROTATION);

  DT_ENTER_GUI_UPDATE();
  int count = 0, pos = 0;
  float sum = 0, min = _masks_properties[prop].min, max = _masks_properties[prop].max;
  if(!is_bool)
  {
    if(_masks_properties[prop].relative)
    {
      max /= min;
      min /= _masks_properties[prop].max;
    }
    else
    {
      max -= min;
      min -= _masks_properties[prop].max;
    }
  }

  if(prop == DT_MASKS_PROPERTY_OPACITY && gui->creation)
  {
    float opacity = dt_conf_get_float("plugins/darkroom/masks/opacity");
    opacity = CLAMP(opacity + value - d->last_value[prop], 0.05f, 1.0f);
    dt_conf_set_float("plugins/darkroom/masks/opacity", opacity);
    sum += opacity;
    ++count;
  }
  else if(!(form->type & DT_MASKS_GROUP)
          && form->functions
          && form->functions->modify_property)
  {
    form->functions->modify_property(form, prop, d->last_value[prop],
                                     value, &sum, &count, &min, &max);
    if(!gui->creation && value != d->last_value[prop])
      dt_masks_gui_form_create(form, gui, pos, dev->gui_module);
  }
  else
  {
    for(GList *fpts = form->points; fpts; fpts = g_list_next(fpts), pos++)
    {
      dt_masks_point_group_t *fpt = fpts->data;
      dt_masks_form_t *sel = dt_masks_get_from_id(darktable.develop, fpt->formid);
      if(!sel
         || (dev->mask_form_selected_id && dev->mask_form_selected_id != sel->formid))
        continue;;

      if(prop == DT_MASKS_PROPERTY_OPACITY && dt_is_valid_maskid(fpt->parentid))
      {
        const float new_opacity = dt_masks_form_change_opacity(sel, fpt->parentid,
                                                         value - d->last_value[prop]);
        sum += new_opacity;
        max = fminf(max, 1.0f - new_opacity);
        min = fmaxf(min, .05f - new_opacity);
        ++count;
      }
      else
      {
        const int saved_count = count;

        if(sel->functions && sel->functions->modify_property)
          sel->functions->modify_property(sel, prop, d->last_value[prop],
                                          value, &sum, &count, &min, &max);

        if(count != saved_count
           && value != d->last_value[prop])
        {
          // we recreate the form points
          dt_masks_gui_form_create(sel, gui, pos, dev->gui_module);
        }
      }
    }
  }

  gtk_widget_set_visible(widget, count != 0);
  if(count)
  {
    if(value != d->last_value[prop]
       && sum / count != d->last_value[prop]
       && prop != DT_MASKS_PROPERTY_OPACITY
       && !gui->creation)
    {
      if(gui->show_all_feathers)
        g_source_remove(gui->show_all_feathers);

      gui->show_all_feathers = g_timeout_add_seconds(2, _timeout_show_all_feathers, gui);

      // we save the new parameters
      dt_dev_add_masks_history_item(darktable.develop, dev->gui_module, TRUE);
    }

    if(is_bool)
    {
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(widget),
                                   (sum / count) > 0.5f);
      d->last_value[prop] =
        (float)gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    }
    else
    {
      if(_masks_properties[prop].relative)
      {
        max *= sum / count;
        min *= sum / count;
      }
      else
      {
        max += sum / count;
        min += sum / count;
      }

      if(dt_isnan(min)) min = _masks_properties[prop].min;
      if(dt_isnan(max)) max = _masks_properties[prop].max;
      dt_bauhaus_slider_set_soft_range(widget, min, max);

      dt_bauhaus_slider_set(widget, sum / count);
      d->last_value[prop] = dt_bauhaus_slider_get(widget);
    }

    gtk_widget_hide(d->none_label);
    dt_control_queue_redraw_center();
  }

  DT_LEAVE_GUI_UPDATE();

  // the recreate-list triggered above is skipped while we hold the gui-update
  // guard, so refresh the shrink/grow slider here: a reshape reset the path's
  // baseline, so this reads back 0.
  if(reshaped)
    _resize_update(d);
}

// Quad for the shrink/grow slider's unit toggle: always shows "%" inside a
// button-like square frame. Its active state (drawn brighter by bauhaus) tells
// whether % mode is engaged; the slider label spells out the unit. Rendered
// with the shared bauhaus label font; the caller has set the source color.
static void _paint_resize_unit(
  cairo_t *cr, const gint x, const gint y, const gint w, const gint h, const gint flags, void *data)
{
  const char *txt = "%";
  cairo_save(cr);

  // square frame fitting the quad area, stroke kept inside via a half-line inset
  const double lw = DT_PIXEL_APPLY_DPI(1.0);
  const double side = MIN(w, h) - lw;
  const double fx = x + lw * 0.5;
  const double fy = y + (h - MIN(w, h)) / 2.0 + lw * 0.5;
  const double r = DT_PIXEL_APPLY_DPI(2.0);
  cairo_new_sub_path(cr);
  cairo_arc(cr, fx + side - r, fy + r, r, -M_PI_2, 0.0);
  cairo_arc(cr, fx + side - r, fy + side - r, r, 0.0, M_PI_2);
  cairo_arc(cr, fx + r, fy + side - r, r, M_PI_2, M_PI);
  cairo_arc(cr, fx + r, fy + r, r, M_PI, 1.5 * M_PI);
  cairo_close_path(cr);
  cairo_set_line_width(cr, lw);
  cairo_stroke(cr);

  // text scaled to fit the padded interior, centred
  PangoLayout *layout = pango_cairo_create_layout(cr);
  if(darktable.bauhaus->pango_font_desc)
    pango_layout_set_font_description(layout, darktable.bauhaus->pango_font_desc);
  pango_layout_set_text(layout, txt, -1);
  int tw = 0, th = 0;
  pango_layout_get_pixel_size(layout, &tw, &th);

  const double pad = DT_PIXEL_APPLY_DPI(1.0);
  const double avail = side - 2.0 * pad;
  const double scale = (tw > 0 && th > 0) ? fmin(avail / tw, avail / th) : 1.0;
  cairo_translate(cr, fx + (side - tw * scale) / 2.0, fy + (side - th * scale) / 2.0);
  cairo_scale(cr, scale, scale);
  pango_cairo_show_layout(cr, layout);
  g_object_unref(layout);
  cairo_restore(cr);
}

// The single path the shrink/grow slider acts on, or NULL. We require an
// unambiguous target: a standalone path, the path explicitly selected inside a
// group, or - when nothing is explicitly selected - the group's only path.
// *out_index receives the form's position in the group (0 for a standalone
// path) for dt_masks_gui_form_create.
static dt_masks_form_t *_selected_single_path(int *out_index)
{
  if(out_index)
    *out_index = 0;
  dt_develop_t *dev = darktable.develop;
  dt_masks_form_t *form = dev->form_visible;
  if(!form)
    return NULL;

  if(!(form->type & DT_MASKS_GROUP))
    return (form->type & DT_MASKS_PATH) ? form : NULL;

  // walk the group's shapes, tracking the sole path as we go: mask_form_selected_id
  // is only set once a shape is clicked, so right after creating or selecting a
  // single path it is still unset - fall back to that path if it is the only one.
  const gboolean has_selection = dt_is_valid_maskid(dev->mask_form_selected_id);
  dt_masks_form_t *single = NULL;
  int single_pos = 0;
  int npaths = 0;
  int pos = 0;
  for(GList *fpts = form->points; fpts; fpts = g_list_next(fpts), pos++)
  {
    dt_masks_point_group_t *fpt = fpts->data;
    dt_masks_form_t *sel = dt_masks_get_from_id(dev, fpt->formid);
    if(!sel || !(sel->type & DT_MASKS_PATH))
      continue;

    // an explicit selection wins immediately
    if(has_selection && fpt->formid == dev->mask_form_selected_id)
    {
      if(out_index)
        *out_index = pos;
      return sel;
    }

    npaths++;
    single = sel;
    single_pos = pos;
  }

  // a specific shape was selected but it is not a path in this group
  if(has_selection)
    return NULL;

  if(npaths == 1)
  {
    if(out_index)
      *out_index = single_pos;
    return single;
  }
  return NULL;
}

static void _resize_cancel_pending(dt_lib_masks_t *d)
{
  if(d->resize_timer)
  {
    g_source_remove(d->resize_timer);
    d->resize_timer = 0;
  }
}

// Set the shape to the slider's absolute offset and commit one history item. The
// path mask owns the baseline and a per-offset result cache, so the morph runs
// (at most) once per distinct value and 0 restores the baseline.
static void _resize_commit(dt_lib_masks_t *d)
{
  dt_develop_t *dev = darktable.develop;
  dt_masks_form_gui_t *gui = dev->form_gui;
  int idx = 0;
  dt_masks_form_t *form = _selected_single_path(&idx);
  if(!form || !gui || !form->functions || !form->functions->resize)
    return;

  const int amount = (int)roundf(dt_bauhaus_slider_get(d->resize_amount));
  const gboolean pct = dt_bauhaus_widget_get_quad_active(d->resize_amount);

  if(!form->functions->resize(form, amount, pct) && amount < 0)
    dt_control_log(_("shrink amount too large: the path would disappear"));

  dt_masks_gui_form_create(form, gui, idx, dev->gui_module);
  dt_dev_add_masks_history_item(dev, dev->gui_module, TRUE);
  dt_control_queue_redraw_center();
}

static gboolean _resize_timeout(gpointer data)
{
  dt_lib_masks_t *d = data;
  d->resize_timer = 0;
  _resize_commit(d);
  return G_SOURCE_REMOVE;
}

// Debounce: morphing is expensive, so commit ~180 ms after the last change
// rather than on every slider tick.
static void _resize_schedule_commit(dt_lib_masks_t *d)
{
  if(d->resize_updating)
    return;
  if(d->resize_timer)
    g_source_remove(d->resize_timer);
  d->resize_timer = g_timeout_add(180, _resize_timeout, d);
}

static void _resize_amount_changed(GtkWidget *w, dt_lib_masks_t *d)
{
  _resize_schedule_commit(d);
}

// Reflect the current unit in the slider's value suffix (e.g. "5 px" / "5 %").
// Changing a lib widget's *label* at runtime re-registers its action and
// crashes (the first set_label moves the widget's module to its leaf action),
// so we use the value format, which is safe to update repeatedly. hard_max is
// 1000 (> 10) so "%" is a plain suffix, not the percentage auto-scaling case.
static void _resize_sync_unit(dt_lib_masks_t *d)
{
  const gboolean pct = dt_bauhaus_widget_get_quad_active(d->resize_amount);
  dt_bauhaus_slider_set_format(d->resize_amount, pct ? " %" : " px");
}

// the unit toggle lives in the slider's quad; bauhaus flips the active flag
// before emitting "quad-pressed", so the new state is read directly
static void _resize_unit_quad(GtkWidget *w, dt_lib_masks_t *d)
{
  const gboolean pct = dt_bauhaus_widget_get_quad_active(w);
  // keep the shared resize unit (also used by the scroll gesture) in sync,
  // storing the untranslated enum strings the path resize code matches against
  dt_conf_set_string("masks/path_resize_unit", pct ? "% of path size" : "pixels");
  _resize_sync_unit(d);
  _resize_schedule_commit(d);
}

// Refresh the shrink/grow controls for the current selection: show them only for
// a single path, and mirror the offset the path mask currently has applied (0 for
// a fresh shape, or whatever a scroll-wheel resize left). Querying the path mask
// keeps the slider truthful across both gestures without a second baseline here.
static void _resize_update(dt_lib_masks_t *d)
{
  int idx = 0;
  dt_masks_form_t *path = _selected_single_path(&idx);

  if(path)
  {
    const gboolean pct = dt_bauhaus_widget_get_quad_active(d->resize_amount);
    float amount = 0.0f;
    if(path->functions && path->functions->resize_get)
      path->functions->resize_get(path, pct, &amount);

    // reflect the current offset without triggering a (re)commit
    _resize_cancel_pending(d);
    d->resize_updating = TRUE;
    dt_bauhaus_slider_set(d->resize_amount, roundf(amount));
    d->resize_updating = FALSE;
  }
  gtk_widget_set_visible(d->resize_box, path != NULL);
}

static void _update_all_properties(dt_lib_masks_t *self)
{
  gtk_widget_show(self->none_label);

  for(int i = 0; i < DT_MASKS_PROPERTY_LAST; i++)
    _property_changed(self->property[i], i);

  dt_masks_form_t *form = darktable.develop->form_visible;
  gboolean drawing_brush = form && form->type & DT_MASKS_BRUSH;

  gtk_widget_set_visible(self->pressure, drawing_brush && darktable.gui->have_pen_pressure);
  gtk_widget_set_visible(self->smoothing, drawing_brush);

  // shrink/grow applies only to a single path shape
  _resize_update(self);

  // ... and the row that says where the next drawn shape is going
  _creation_bar_update(self);
}

static void _lib_masks_get_values(GtkTreeModel *model,
                                  GtkTreeIter *iter,
                                  dt_iop_module_t **module,
                                  dt_mask_id_t *groupid,
                                  dt_mask_id_t *formid)
{
  // returns module & groupid & formid if requested
  if(module ) gtk_tree_model_get(model, iter, TREE_MODULE, module, -1);
  if(groupid) gtk_tree_model_get(model, iter, TREE_GROUPID, groupid, -1);
  if(formid ) gtk_tree_model_get(model, iter, TREE_FORMID, formid, -1);
}

static void _lib_masks_inactivate_icons(dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;

  // we set the add shape icons inactive
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lm->bt_circle), FALSE);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lm->bt_ellipse), FALSE);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lm->bt_path), FALSE);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lm->bt_gradient), FALSE);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lm->bt_brush), FALSE);
#ifdef HAVE_AI
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(lm->bt_object), FALSE);
#endif
}

/* -------------------------------------------------------------------------
   "new mask": one entry point that names the module the shape is for.

   until now a shape created from this panel took its target from whatever row
   happened to be selected, silently, and got NO target at all when nothing was
   selected -- that is where an orphan shape comes from on this side. the rule
   does not change; what changes is that it is written down before anything is
   drawn, and that a creation with no target is refused instead of performed.
   ------------------------------------------------------------------------- */

// a module whose mask gui is actually there. blend_data is only allocated for
// a module whose blending gui was built, and masks_support already carries
// !IOP_FLAGS_NO_MASKS, so this says what the two-flag test at line 1241 says
// plus the one thing that matters here: blend_data exists. the
// continuous-creation tail dereferences creation_module->blend_data with no
// guard at all (masks/circle.c:363 and its four siblings), so a target without
// it is a crash on ctrl+creation, not a cosmetic problem.
static gboolean _mask_target_has_gui(const dt_iop_module_t *m)
{
  const dt_iop_gui_blend_data_t *bd = m ? m->blend_data : NULL;
  return bd && bd->masks_support && bd->masks_inited;
}

// ... and that can take a drawn mask right now. drawn and raster masking are
// exclusive in blend_params, and _blendop_masks_modes_toggle() refuses the
// switch outright (blend_gui.c: mask_mode & DEVELOP_MASK_RASTER -> FALSE). this
// is the one "that module cannot borrow this" the code really enforces; the
// pipeline-order rule everyone expects exists for raster masks only
// (_raster_combo_populate), a drawn shape being geometry distorted at the
// position of whoever consumes it.
static gboolean _mask_target_ok(const dt_iop_module_t *m)
{
  return _mask_target_has_gui(m)
    && !(m->blend_params->mask_mode & DEVELOP_MASK_RASTER);
}

// ... and worth *offering* in a list. two more refusals, deliberately not
// applied to a target the user already points at: a module the current pipe
// does not run (iop_order == INT_MAX) and a deprecated one that is not already
// part of this edit. same trio as libs/modulegroups.c:770.
static gboolean _mask_target_listed(const dt_iop_module_t *m)
{
  return _mask_target_has_gui(m)
    && !dt_iop_is_hidden(m)
    && m->iop_order != INT_MAX
    && (m->enabled || !(m->flags() & IOP_FLAGS_DEPRECATED));
}

// how many shapes a module's drawn mask holds. 0 both when it has no group and
// when its group is empty: to the photographer the two read the same
static int _mask_target_shapes(const dt_iop_module_t *m)
{
  const dt_masks_form_t *grp =
    dt_masks_get_from_id(darktable.develop, m->blend_params->mask_id);
  return (grp && (grp->type & DT_MASKS_GROUP)) ? (int)g_list_length(grp->points) : 0;
}

// a module named the way darktable names it everywhere else, multi-instance
// included, plus what the click is about to run into. gtk3 delivers no event,
// hence no tooltip, to an insensitive item: whatever an entry cannot do -- or
// will do besides -- has to be readable in the entry itself.
static gchar *_mask_target_label(const dt_iop_module_t *m)
{
  gchar *name = dt_history_item_get_name(m);
  const int shapes = _mask_target_shapes(m);

  const gchar *note = NULL;
  if(m->blend_params->mask_mode & DEVELOP_MASK_RASTER)
    note = _("uses a raster mask");
  else if(!m->enabled)
    note = _("off");

  gchar *count = (shapes > 0)
    ? g_strdup_printf(ngettext("%d shape", "%d shapes", shapes), shapes)
    : NULL;

  gchar *label;
  if(count && note) label = g_strdup_printf("%s (%s, %s)", name, count, note);
  else if(count)    label = g_strdup_printf("%s (%s)", name, count);
  else if(note)     label = g_strdup_printf("%s (%s)", name, note);
  else              label = g_strdup(name);

  g_free(count);
  g_free(name);
  return label;
}

// is this pointer still one of the modules of the current pipe? a menu item
// carries a module pointer for as long as the menu is open, and a module can
// be destroyed meanwhile (instance removed, image changed, history compressed)
static gboolean _mask_target_alive(const dt_iop_module_t *m)
{
  if(!m) return FALSE;
  for(const GList *l = darktable.develop->iop; l; l = g_list_next(l))
    if(l->data == (gconstpointer)m) return TRUE;
  return FALSE;
}

// where a new shape goes when nothing said otherwise, in decreasing order of
// evidence:
//   1. the module of the selected row -- this is the rule _tree_add_shape has
//      always applied, it was simply never said out loud;
//   2. the module open in the right panel (dev->gui_module);
//   3. the last module of the pipe that already carries a drawn mask (dev->iop
//      is sorted, so the last match is the latest one).
// NULL when nothing qualifies. the catalogue prints the answer, which is the
// whole reason three rules can coexist without surprising anyone.
static dt_iop_module_t *_mask_default_target(dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  dt_iop_module_t *module = NULL;

  // deliberately the masks view and not the active one: TREE_MODULE is only
  // written on rows that belong to a module's mask, and the library is flat and
  // module-less by construction -- reading its selection here would always
  // answer NULL and silently demote rule 1 to rule 2
  GtkTreeModel *model = NULL;
  GList *selected = gtk_tree_selection_get_selected_rows
    (gtk_tree_view_get_selection(GTK_TREE_VIEW(lm->treeview)), &model);
  if(selected)
  {
    GtkTreeIter iter;
    if(gtk_tree_model_get_iter(model, &iter, selected->data))
      _lib_masks_get_values(model, &iter, &module, NULL, NULL);
    g_list_free_full(selected, (GDestroyNotify)gtk_tree_path_free);
  }
  if(_mask_target_ok(module)) return module;

  module = darktable.develop->gui_module;
  if(_mask_target_ok(module)) return module;

  dt_iop_module_t *last = NULL;
  for(const GList *l = darktable.develop->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    if(_mask_target_listed(m) && _mask_target_ok(m) && _mask_target_shapes(m) > 0)
      last = m;
  }
  return last;
}

/* -------------------------------------------------------------------------
   the creation bar: the row that says where the next shape is going.

   the catalogue names the target once, in a menu that is gone by the time the
   shape is drawn. between that click and the first click on the image there is
   nothing on screen saying what was armed, and no way back short of drawing
   something and undoing it. this is that missing line.
   ------------------------------------------------------------------------- */

// a continuous run outlives the form it belongs to: dt_masks_clear_form_gui()
// resets `creation` and `creation_module` but not these two, so every place
// that ends a creation from outside the canvas has to end the run as well --
// which is what a right-click on the image does, in every shape's
// _*_events_button_pressed(), before dropping the form
static void _creation_end_continuous(void)
{
  dt_masks_form_gui_t *fg = darktable.develop->form_gui;
  if(!fg) return;
  fg->creation_continuous = FALSE;
  fg->creation_continuous_module = NULL;
}

// the one place the bar is decided, reached from _update_all_properties(), so
// from all four refresh paths of this panel. what is armed is read back from
// form_gui rather than mirrored here: the bar then promises exactly what the
// next click on the image will do, and the shape code stays free to end a
// creation without telling us -- which is what it does.
static void _creation_bar_update(dt_lib_masks_t *d)
{
  const dt_masks_form_gui_t *fg = darktable.develop->form_gui;

  // a creation is in flight while form_gui carries either half of it, and only
  // then. `creation_continuous` deliberately does not count: nothing generic
  // ever resets it -- not dt_masks_clear_form_gui(), so not a focus change, a
  // tree click or an image change either -- and reading it as proof of an
  // armament is how this row would end up naming a module long after the
  // canvas stopped listening. the two halves take turns instead:
  //   `creation` alone, target still to be filled in, is where a continuous
  //   run sits while it chains the next shape; `creation_module` alone is
  //   where a shape sits between being saved and that chain restarting
  const gboolean pending = fg && (fg->creation || fg->creation_module);

  // ... and it is ours until someone else names a target, which is what a
  // module's own blending panel does a moment after taking the focus
  const gboolean stolen =
    fg && fg->creation_module && fg->creation_module != d->arm_module;

  if(d->arm_module
     && (!pending || stolen || !_mask_target_alive(d->arm_module)))
  {
    // the creation this panel armed is over, so the run it belonged to is too.
    // it is ended here because here is where we learn of it: a right-click on
    // the image ends its own run, but a focus change or a new image simply
    // drops the form, and the next shape drawn from anywhere would chain
    if(!pending && fg && fg->creation_continuous_module == d->arm_module)
      _creation_end_continuous();
    d->arm_module = NULL;
  }

  if(d->arm_module)
  {
    // the module alone, without the "(2 shapes)" the catalogue adds to pick
    // between entries: inside a sentence that annotation reads as something
    // the shape about to be drawn is going to do
    gchar *name = dt_history_item_get_name(d->arm_module);
    gchar *text = g_strdup_printf(_("next shape goes to %s"), name);
    gtk_label_set_text(GTK_LABEL(d->creation_label), text);
    // the left panel is narrow, the row already carries a button and the
    // module name is what the sentence ends on: the one word the bar exists
    // to give is the first one the ellipsis takes
    gtk_widget_set_tooltip_text(d->creation_label, text);
    g_free(text);
    g_free(name);
  }

  gtk_widget_set_visible(d->creation_bar, d->arm_module != NULL);
}

// call it off without having to find the image first. these are the calls a
// right-click on the canvas makes; the disarming itself is left to
// _creation_bar_update(), which change_form_gui() reaches through the
// selection proxy, so the bar goes down in one place whatever took it down
static void _creation_bar_cancel(GtkButton *button, dt_lib_masks_t *d)
{
  dt_iop_module_t *module = d->arm_module;

  _creation_end_continuous();
  if(_mask_target_alive(module))
  {
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
    dt_masks_iop_update(module);
  }
  else
    dt_masks_change_form_gui(NULL);

  dt_control_queue_redraw_center();
}

// the one creation path of this panel. the icon row, the catalogue and the
// tree context menu all land here, so none of them can produce a shape with no
// module behind it and all of them refuse with the same words.
static gboolean _start_creation(dt_lib_module_t *self,
                                dt_iop_module_t *module,
                                const dt_masks_type_t type,
                                const gboolean continuous)
{
  if(!self) return FALSE;

#ifdef HAVE_AI
  if(type == DT_MASKS_OBJECT && !dt_masks_object_available())
  {
    dt_control_log(_("AI model is not available. Check preferences > AI"));
    _lib_masks_inactivate_icons(self);
    return FALSE;
  }
#endif

  // the module may have gone away between the click that opened a menu and the
  // click that picked a shape
  if(!_mask_target_alive(module) || !_mask_target_ok(module))
  {
    // a shape with no module is a shape nothing renders: it lands in the
    // manager, in no pipe, and stays there until "delete unused shapes" is
    // found. do not create it
    dt_control_log(_("no module can take a drawn mask"));
    _lib_masks_inactivate_icons(self);
    return FALSE;
  }

  // the target has to be listening. this is what a module's own shape button
  // does before creating anything, and it is not decoration: nothing in
  // develop/masks/*.c ever writes mask_mode, so a shape attached to a module
  // still on DEVELOP_MASK_DISABLED would be drawn and do nothing. the call
  // takes the focus, sets the mask indicator and adds the history item that
  // enables the module -- all of it inside blend_gui.c, where it belongs
  dt_iop_gui_enable_drawn_mask(module);
  // enable_drawn_mask is a no-op when drawn masking is already on, and the
  // focus has to be taken in that case too
  dt_iop_request_focus(module);

  // we create the new form
  dt_masks_form_t *spot = dt_masks_create(type);
  dt_masks_change_form_gui(spot);

  dt_masks_form_gui_t *gui = darktable.develop->form_gui;
  gui->creation_module = module;
  gui->group_selected = 0;
  // the new form must be editable
  gui->edit_mode = DT_MASKS_EDIT_FULL;

  // stated both ways round, unlike the shape buttons of a blending panel: a
  // run left over from an earlier ctrl+click survives dt_masks_clear_form_gui()
  // and would turn this plain click into a series nobody asked for
  gui->creation_continuous = continuous;
  gui->creation_continuous_module = continuous ? module : NULL;

  // the one place the armament is recorded, and all it records is that this
  // panel is what armed the canvas: where the shape goes is form_gui's answer
  // just above, read back by _creation_bar_update()
  dt_lib_masks_t *d = self->data;
  d->arm_module = module;
  _creation_bar_update(d);

  _lib_masks_inactivate_icons(self);
  dt_control_queue_redraw_center();
  return TRUE;
}

// menu-item adapter. the catalogue and the context menu write the module they
// mean on the item; an item without one falls back to the default rule, so a
// path that forgets to set it degrades to today's behaviour minus the orphan
static void _tree_add_shape(GtkWidget *widget, gpointer shape)
{
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  if(!self) return;

  dt_iop_module_t *module = g_object_get_data(G_OBJECT(widget), "target");
  if(!module) module = _mask_default_target(self);

  _start_creation(self, module, GPOINTER_TO_INT(shape), FALSE);
}

static void _bt_add_shape_cb(GtkGestureSingle *gesture, int n_press, double x, double y, gpointer shape)
{
  if(dt_gui_current_button(gesture) != GDK_BUTTON_PRIMARY) return;

  // proxy.masks.module was dereferenced here without a guard; a gesture can
  // fire while the panel is being torn down
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  if(!self) return;

  // the icons stay the fast path for someone who already knows where the shape
  // goes -- and they keep their shortcuts, which live at the action path
  // "shapes/add *" and would be silently invalidated if the row were replaced.
  // what changes is that they resolve their target exactly as the catalogue
  // does, so the two entry points can never disagree, and that they say so when
  // there is nowhere to put the shape instead of quietly making an orphan
  _start_creation(self, _mask_default_target(self), GPOINTER_TO_INT(shape),
                  dt_modifier_is(dt_gui_current_state(gesture), GDK_CONTROL_MASK));
}

// the shape types, in the order of the icon row so the two entry points read
// the same way. the labels are the ones already used by the context menu and by
// the icon tooltips -- no new string for translators. DT_MASKS_OBJECT is the
// one type that can be unavailable at runtime, and the icon row has always had
// it while the context menu never did: the catalogue settles it
static const struct
{
  dt_masks_type_t type;
  const char *label;
} _new_mask_shapes[] =
{
  { DT_MASKS_BRUSH,    N_("add brush")     },
  { DT_MASKS_CIRCLE,   N_("add circle")    },
  { DT_MASKS_ELLIPSE,  N_("add ellipse")   },
  { DT_MASKS_PATH,     N_("add path")      },
  { DT_MASKS_GRADIENT, N_("add gradient")  },
#ifdef HAVE_AI
  { DT_MASKS_OBJECT,   N_("add AI object") },
#endif
};

// one row per shape type, wired to the callback the icon row and the context
// menu already use. the module travels on the item, so the same function fills
// the top level of the catalogue, every per-module submenu, and the context menu
static void _new_mask_shape_items(GtkMenuShell *menu, dt_iop_module_t *target)
{
  for(int i = 0; i < (int)G_N_ELEMENTS(_new_mask_shapes); i++)
  {
    const gchar *reason = NULL;

#ifdef HAVE_AI
    if(_new_mask_shapes[i].type == DT_MASKS_OBJECT && !dt_masks_object_available())
      reason = _("AI model not available");
#endif
    if(!target) reason = _("no module to attach it to");

    // gtk3 delivers no event, hence no tooltip, to an insensitive widget:
    // whatever an entry cannot do has to be readable in the entry itself
    gchar *label = reason
      ? g_strdup_printf("%s (%s)", _(_new_mask_shapes[i].label), reason)
      : g_strdup(_(_new_mask_shapes[i].label));

    GtkWidget *item = gtk_menu_item_new_with_label(label);
    g_free(label);

    gtk_widget_set_sensitive(item, reason == NULL);
    g_object_set_data(G_OBJECT(item), "target", target);
    g_signal_connect(item, "activate", G_CALLBACK(_tree_add_shape),
                     GINT_TO_POINTER(_new_mask_shapes[i].type));
    gtk_menu_shell_append(menu, item);
  }
}

// the target, spelled out and not clickable. the manager, unlike a blending
// panel, has no module of its own: where the shape is about to land has to be
// said before it is drawn. that single line is what makes a shape attached to
// nothing impossible to create by accident
static void _new_mask_target_header(GtkMenuShell *menu, const dt_iop_module_t *target)
{
  gchar *name = target
    ? _mask_target_label(target)
    : g_strdup(_("none"));
  gchar *header = g_strdup_printf(_("target: %s"), name);

  GtkWidget *item = gtk_menu_item_new_with_label(header);
  gtk_widget_set_sensitive(item, FALSE);
  gtk_menu_shell_append(menu, item);

  g_free(header);
  g_free(name);
  gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
}

// every other module that could take the shape, each with the same list of
// types one level down: picking there picks the target AND the shape in one
// gesture, so no target has to be remembered between two openings of the menu.
// dev->iop is sorted by iop_order, so two passes give the modules that already
// carry a drawn mask first -- that is where a second shape usually goes -- then
// the rest, each pass in pipe order, so the list reads like the module list on
// the right
static gboolean _new_mask_other_modules(GtkMenuShell *menu,
                                        const dt_iop_module_t *current)
{
  gboolean any = FALSE, any_masked = FALSE, separated = FALSE;

  for(int pass = 0; pass < 2; pass++)
  {
    for(const GList *l = darktable.develop->iop; l; l = g_list_next(l))
    {
      dt_iop_module_t *m = l->data;
      if(m == current || !_mask_target_listed(m)) continue;

      const gboolean masked = _mask_target_shapes(m) > 0;
      if(masked != (pass == 0)) continue;

      if(masked) any_masked = TRUE;
      else if(any_masked && !separated)
      {
        gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
        separated = TRUE;
      }

      gchar *label = _mask_target_label(m);
      GtkWidget *item = gtk_menu_item_new_with_label(label);
      g_free(label);

      if(_mask_target_ok(m))
      {
        GtkWidget *sub = gtk_menu_new();
        _new_mask_shape_items(GTK_MENU_SHELL(sub), m);
        gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), sub);
      }
      else
        // on a raster mask: _blendop_masks_modes_toggle() would refuse the
        // switch, so the entry stays and says so rather than disappearing
        gtk_widget_set_sensitive(item, FALSE);

      gtk_menu_shell_append(menu, item);
      any = TRUE;
    }
  }

  return any;
}

// built at click, never in gui_update: _forms_structure_hash() mixes
// dev->gui_module in but gui_update only runs on dt_dev_masks_list_change, so a
// menu built there would show a stale target after a mere change of focus
static void _new_mask_clicked(GtkButton *button, dt_lib_module_t *self)
{
  dt_iop_module_t *target = _mask_default_target(self);

  GtkMenuShell *menu = GTK_MENU_SHELL(gtk_menu_new());

  _new_mask_target_header(menu, target);
  _new_mask_shape_items(menu, target);

  GtkWidget *others = gtk_menu_new();
  if(_new_mask_other_modules(GTK_MENU_SHELL(others), target))
  {
    gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
    GtkWidget *item = gtk_menu_item_new_with_label(_("on another module"));
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), others);
    gtk_menu_shell_append(menu, item);
  }
  else
  {
    // no other module qualifies, so no menu item ever took this submenu and
    // nothing ever sank its floating reference. gtk_widget_destroy() only runs
    // dispose, which would leave the object alive at one reference: sink it
    // first, exactly as dt_gui_menu_popup does with the menu it is handed
    g_object_ref_sink(others);
    g_object_unref(others);
  }

  // dt_gui_menu_popup takes the floating ref and drops it on "deactivate"
  dt_gui_menu_popup(GTK_MENU(menu), GTK_WIDGET(button),
                    GDK_GRAVITY_SOUTH_WEST, GDK_GRAVITY_NORTH_WEST);
}

static void _tree_add_exist(GtkButton *button, dt_masks_form_t *grp)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return;
  // we get the new formid
  const dt_mask_id_t id = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(button), "formid"));
  dt_iop_module_t *module = g_object_get_data(G_OBJECT(button), "module");

  // we add the form in this group
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(form && dt_masks_group_add_form(grp, form))
  {
    // we save the group
    dt_dev_add_masks_history_item(darktable.develop, NULL, FALSE);

    // and we apply the change
    dt_masks_iop_update(module);
    dt_dev_masks_selection_change(darktable.develop, NULL, grp->formid);
  }
}

static void _tree_group(GtkButton *button, dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  // we create the new group
  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  snprintf(grp->name, sizeof(grp->name), _("group #%d"),
           g_list_length(darktable.develop->forms));

  // we add all selected forms to this group
  // the list the menu was opened from: a grouping asked for in the library must
  // not read the masks zone's selection, and the other way round
  GtkWidget *view = _masks_active_view(lm);
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));

  int pos = 0;
  GList *items = gtk_tree_selection_get_selected_rows(selection, NULL);
  for(GList *items_iter = items; items_iter; items_iter = g_list_next(items_iter))
  {
    GtkTreePath *item = (GtkTreePath *)items_iter->data;
    GtkTreeIter iter;
    if(gtk_tree_model_get_iter(model, &iter, item))
    {
      dt_mask_id_t id = INVALID_MASKID;
      _lib_masks_get_values(model, &iter, NULL, NULL, &id);

      if(dt_is_valid_maskid(id))
      {
        dt_masks_point_group_t *fpt = malloc(sizeof(dt_masks_point_group_t));
        fpt->formid = id;
        fpt->parentid = grp->formid;
        fpt->opacity = 1.0f;
        fpt->state = DT_MASKS_STATE_USE;
        if(pos > 0) fpt->state |= DT_MASKS_STATE_UNION;
        grp->points = g_list_append(grp->points, fpt);
        pos++;
      }
    }
  }
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);

  // we add this group to the general list
  darktable.develop->forms = g_list_append(darktable.develop->forms, grp);

  // add we save
  dt_dev_add_masks_history_item(darktable.develop, NULL, FALSE);
  _lib_masks_recreate_list(self);
  // dt_masks_change_form_gui(grp);
}

// where a shape sits in the application order of its group. index 0 is the
// base: it lays the buffer down, and it is the only member allowed to carry
// no operator -- group.c reads a missing operator as "overwrite everything
// applied so far" (final `else` of _group_get_mask).
// returns -1 when the group or the shape is unknown.
//
// everything that used to be decided from a row's position on screen is
// decided here instead: the tree is a projection of this list, and after
// this commit that projection is no longer a mirror.
static int _group_point_index(const dt_masks_form_t *grp,
                              const dt_mask_id_t formid)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return -1;

  int pos = 0;
  for(const GList *pts = grp->points; pts; pts = g_list_next(pts))
  {
    const dt_masks_point_group_t *pt = pts->data;
    // we stop at the first match, exactly like dt_masks_form_move() and
    // _tree_operation, so display and reordering always agree on which
    // occurrence they mean
    if(pt->formid == formid) return pos;
    pos++;
  }
  return -1;
}

// formid of the shape at `index` in the application order, INVALID_MASKID
// when there is none.
static dt_mask_id_t _group_point_id(const dt_masks_form_t *grp,
                                    const int index)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP) || index < 0)
    return INVALID_MASKID;

  const dt_masks_point_group_t *pt =
    g_list_nth_data(grp->points, (guint)index);
  return pt ? pt->formid : INVALID_MASKID;
}

// DT_MASKS_STATE_SHOW is not "this shape is the base": dt_masks_group_add_form
// sets it on every point it creates. it is read in exactly two places, both in
// this file (the operator-icon lookups), and means "draw the operator glyph".
// the base must not draw one, so when a move or a deletion changes which shape
// sits at index 0 we hand that flag over: the incoming base drops SHOW, the
// outgoing one gets it back plus a UNION if it had no operator at all --
// without which group.c would treat it as a fresh buffer and silently drop
// every shape applied before it.
//
// pass INVALID_MASKID as old_base_id when the outgoing base is on its way out
// (a deletion): there is nothing to hand back to.
//
// both ids come from grp->points, never from a row position.
static void _handover_base_state(dt_masks_form_t *grp,
                                 const dt_mask_id_t new_base_id,
                                 const dt_mask_id_t old_base_id)
{
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return;

  for(const GList *pts = grp->points; pts; pts = g_list_next(pts))
  {
    dt_masks_point_group_t *pt = pts->data;

    if(pt->formid == new_base_id)
      pt->state &= ~DT_MASKS_STATE_SHOW;
    else if(pt->formid == old_base_id)
    {
      // ensure an operator is defined, as we are going to show one
      if((pt->state & DT_MASKS_STATE_OP) == DT_MASKS_STATE_NONE)
        pt->state |= DT_MASKS_STATE_UNION;
      pt->state |= DT_MASKS_STATE_SHOW;
    }
  }
}

static void _set_iter_name(dt_lib_masks_t *lm,
                           dt_masks_form_t *form,
                           const int state,
                           const float opacity,
                           GtkTreeModel *model,
                           GtkTreeIter *iter)
{
  if(!form) return;

  // TREE_TEXT must stay exactly form->name on an editable row:
  // _tree_cell_edited copies the displayed string straight back into
  // form->name. TREE_EDITABLE is (grp_id == 0), and a root row is always built
  // with opacity 1.0f, so no "%" suffix can reach it either. the rank and the
  // base marker live in columns of their own and never in TREE_TEXT
  char str[256] = "";
  g_strlcat(str, form->name, sizeof(str));

  if(opacity != 1.0f)
  {
    char str2[256] = "";
    g_strlcpy(str2, str, sizeof(str2));
    snprintf(str, sizeof(str), "%s %d%%", str2, (int)(opacity * 100));
  }

  // the shapes of a mask are stacked: the first lays the base, the next ones
  // combine onto it. number them so a module's mask reads like a recipe --
  // only there: at the root, and inside a stand-alone group no module uses,
  // the order means nothing on screen and gets no number.
  // the base MARKER is not tied to that: group.c forbids an operator on index
  // 0 of *every* group and the context menu greys the five "mode:" entries
  // accordingly, so the marker has to appear wherever that rule bites --
  // otherwise the menu is disabled without saying why.
  // TREE_MODULE and TREE_GROUPID are already set on the row by the time we are
  // called, so we read them back instead of growing the signature
  dt_iop_module_t *module = NULL;
  dt_mask_id_t grid = INVALID_MASKID;
  dt_mask_id_t id = INVALID_MASKID;
  _lib_masks_get_values(model, iter, &module, &grid, &id);

  char num[8] = "";
  const char *base = "";

  if(dt_is_valid_maskid(grid))
  {
    const int rank =
      _group_point_index(dt_masks_get_from_id(darktable.develop, grid), id);

    if(rank >= 0)
    {
      if(module) snprintf(num, sizeof(num), "%d", rank + 1);
      // the base carries no operator and cannot be given one; say so rather
      // than leave an unexplained empty operator slot and a greyed menu
      if(rank == 0) base = _("base");
    }
  }

  // a library row -- the only row in either store that is a shape with no
  // parent group, and the only place a shape exists as itself. what the "used"
  // badge could never say is whether a module actually renders it: a word, not
  // a colour and not a glyph, because it has to translate and to follow a theme
  // change, which the pixbufs rasterised once in gui_init do not.
  // the walk this costs is paid on library rows only, and dev->forms is the
  // list a human drew by hand
  const char *link = "";

  if(!dt_is_valid_maskid(grid)
     && !(form->type & DT_MASKS_GROUP)
     && _shape_scope(form->formid, NULL, 0) != DT_MASKS_SCOPE_MODULE)
    link = _("no module");

  const gboolean show = state & DT_MASKS_STATE_SHOW;

  GdkPixbuf *icop = NULL;
  GdkPixbuf *icinv = NULL;

  if(state & DT_MASKS_STATE_UNION)
    icop = lm->ic_union;
  else if(state & DT_MASKS_STATE_INTERSECTION)
    icop = lm->ic_intersection;
  else if(state & DT_MASKS_STATE_DIFFERENCE)
    icop = lm->ic_difference;
  else if(state & DT_MASKS_STATE_SUM)
    icop = lm->ic_sum;
  else if(state & DT_MASKS_STATE_EXCLUSION)
    icop = lm->ic_exclusion;

  if(state & DT_MASKS_STATE_INVERSE)
    icinv = lm->ic_inverse;

  gtk_tree_store_set(GTK_TREE_STORE(model), iter,
                     TREE_TEXT, str,
                     TREE_NUM, num,
                     TREE_BASE, base,
                     TREE_LINK, link,
                     TREE_IC_OP, icop,
                     TREE_IC_OP_VISIBLE, (icop != NULL) && show,
                     TREE_IC_INVERSE, icinv,
                     TREE_IC_INVERSE_VISIBLE, (icinv != NULL),
                     -1);
}

static void _tree_cleanup(GtkButton *button, dt_lib_module_t *self)
{
  dt_masks_cleanup_unused(darktable.develop);
  _lib_masks_recreate_list(self);
}

static void _add_masks_history_item(dt_lib_masks_t *lm)
{
  DT_ENTER_GUI_UPDATE();
  dt_dev_add_masks_history_item(darktable.develop, NULL, FALSE);
  DT_LEAVE_GUI_UPDATE();
}


static void _tree_operation(GtkButton *button, gpointer user_data)
{
  dt_masks_state_t change_state = GPOINTER_TO_INT(user_data);
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  dt_lib_masks_t *lm = self->data;

  // now we go through all selected nodes
  // the list the menu was opened from: an operator change asked for in the
  // library must not read the masks zone's selection, and the other way round
  GtkWidget *view = _masks_active_view(lm);
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
  gboolean change = FALSE;
  GList *items = gtk_tree_selection_get_selected_rows(selection, NULL);

  for(const GList *items_iter = items;
      items_iter;
      items_iter = g_list_next(items_iter))
  {
    GtkTreePath *item = (GtkTreePath *)items_iter->data;
    GtkTreeIter iter;

    if(gtk_tree_model_get_iter(model, &iter, item))
    {
      dt_mask_id_t grid = INVALID_MASKID;
      dt_mask_id_t id = INVALID_MASKID;
      _lib_masks_get_values(model, &iter, NULL, &grid, &id);

      dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, grid);
      if(grp && (grp->type & DT_MASKS_GROUP))
      {
        // we search the entry to inverse
        for(const GList *pts = grp->points; pts; pts = g_list_next(pts))
        {
          dt_masks_point_group_t *pt = pts->data;
          if(pt->formid == id)
          {
            if(change_state == DT_MASKS_STATE_INVERSE
               || (pt->state & DT_MASKS_STATE_OP
                   && !(pt->state & change_state)))
            {
              if(change_state != DT_MASKS_STATE_INVERSE)
                pt->state &= ~DT_MASKS_STATE_OP;
              pt->state ^= change_state;
              _set_iter_name(lm, dt_masks_get_from_id(darktable.develop, id),
                             pt->state, pt->opacity, model,
                             &iter);
              change = TRUE;
            }
            break;
          }
        }
      }
    }
  }
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);

  if(change)
    _add_masks_history_item(lm);
}

static void _add_tree_operation(GtkMenuShell *menu,
                                gchar *label,
                                dt_masks_state_t state,
                                dt_masks_state_t selected_states,
                                gboolean sensitive)
{
  GtkWidget *item = gtk_check_menu_item_new_with_label(label);
  gtk_widget_set_sensitive(item, sensitive);
  if(selected_states & state)
    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(item), TRUE);
  g_signal_connect(item, "activate", G_CALLBACK(_tree_operation),
                    GINT_TO_POINTER(state));
  gtk_menu_shell_append(menu, item);
}

// reordering is a per-shape operation. the old code looped over the whole
// selection and moved each row against a tree model already stale after the
// first move; it also called gtk_tree_model_iter_next() without checking its
// return value and then read the resulting iter, so acting on the bottom row
// could hand garbage ids to the state swap. we act on the first selected row
// only, from grp->points, and the menu keeps both entries insensitive unless
// exactly one row is selected.
static void _tree_move_shape(dt_lib_module_t *self, const gboolean later)
{
  dt_lib_masks_t *lm = self->data;

  // rank and base only exist in the masks zone -- the library is flat and
  // carries neither -- but the entry that leads here is only offered under
  // `from_group`, so the view is read the same way as every other action
  GtkWidget *view = _masks_active_view(lm);
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
  GList *items = gtk_tree_selection_get_selected_rows(selection, NULL);
  if(!items) return;

  GtkTreeIter iter;
  if(gtk_tree_model_get_iter(model, &iter, items->data))
  {
    dt_mask_id_t grid = INVALID_MASKID;
    dt_mask_id_t id = INVALID_MASKID;
    _lib_masks_get_values(model, &iter, NULL, &grid, &id);

    dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, grid);
    const int rank = _group_point_index(grp, id);
    const int nb_points = grp ? (int)g_list_length(grp->points) : 0;

    // "earlier" means closer to the base, i.e. a lower index in grp->points;
    // dt_masks_form_move()'s `up` argument means "one step later in the list".
    // the two are opposites, and after the display flip "earlier" is also the
    // one that moves the row up on screen.
    // rank < 0 (root row, unknown group) fails both tests.
    const gboolean can_move = later ? (rank >= 0 && rank + 1 < nb_points)
                                    : (rank > 0);

    if(can_move)
    {
      dt_masks_clear_form_gui(darktable.develop);

      // crossing index 0 hands the base over to another shape. both ids are
      // resolved BEFORE the move, and _handover_base_state only touches state
      // bits, never positions
      if(later && rank == 0)
        _handover_base_state(grp, _group_point_id(grp, 1), id);
      else if(!later && rank == 1)
        _handover_base_state(grp, id, _group_point_id(grp, 0));

      dt_masks_form_move(grp, id, later);

      dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
      _lib_masks_recreate_list(self);
    }
  }
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);
}

static void _tree_apply_earlier(GtkButton *button, dt_lib_module_t *self)
{
  _tree_move_shape(self, FALSE);
}

static void _tree_apply_later(GtkButton *button, dt_lib_module_t *self)
{
  _tree_move_shape(self, TRUE);
}

// dt_masks_form_remove() walks dev->iop and dev->forms, and nothing else
// (src/develop/masks/masks.c): a shape deleted for good while it is also filed
// in a group no module renders leaves a dt_masks_point_group_t behind, pointing
// at an id that no longer resolves. nothing crashes -- _lib_masks_list_recurs
// skips it through its `if(f)` -- but the group silently loses a member and the
// xmp keeps the corpse. so a real deletion takes those references out itself.
// only groups NO module renders are touched: a group a module renders is
// dt_masks_form_remove()'s business, and pruning it here would change what that
// module renders. a group no module renders produces no pixel, by definition,
// so this changes nothing on screen -- and keeps the group coherent for the day
// it is handed to a module.
// a group left empty is deliberately kept: dropping it would mean removing a
// form from dev->forms while walking it, an empty group is a state the manager
// already displays, and "delete unused shapes" already collects it.
static void _detach_from_unused_groups(const dt_mask_id_t formid)
{
  if(!dt_is_valid_maskid(formid)) return;

  for(const GList *forms = darktable.develop->forms;
      forms;
      forms = g_list_next(forms))
  {
    dt_masks_form_t *grp = forms->data;
    if(!(grp->type & DT_MASKS_GROUP)) continue;
    // recursive: a sub-group nested in a module's mask answers MODULE here and
    // is left alone, which is exactly what we want
    if(_shape_scope(grp->formid, NULL, 0) == DT_MASKS_SCOPE_MODULE) continue;

    GList *pts = grp->points;
    while(pts)
    {
      GList *next = g_list_next(pts);
      dt_masks_point_group_t *pt = pts->data;

      if(pt->formid == formid)
      {
        // dropping the base of a group hands the base over, exactly as a move
        // or a deletion inside a module's mask does
        if(_group_point_index(grp, formid) == 0)
          _handover_base_state(grp, _group_point_id(grp, 1), INVALID_MASKID);

        grp->points = g_list_remove_link(grp->points, pts);
        free(pt);
        g_list_free_1(pts);
      }
      pts = next;
    }
  }
}

static void _tree_delete_shape(GtkButton *button, dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;

  dt_masks_clear_form_gui(darktable.develop);

  // now we go through all selected nodes
  // the list the menu was opened from: a deletion asked for in the library must
  // not read the masks zone's selection, and the other way round
  GtkWidget *view = _masks_active_view(lm);
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
  dt_iop_module_t *module = NULL;

  GList *items = gtk_tree_selection_get_selected_rows(selection, NULL);

  for(const GList *items_iter = items;
      items_iter;
      items_iter = g_list_next(items_iter))
  {
    GtkTreePath *item = (GtkTreePath *)items_iter->data;
    GtkTreeIter iter;
    if(gtk_tree_model_get_iter(model, &iter, item))
    {
      GtkTreeIter *prev_iter = gtk_tree_iter_copy(&iter);
      GtkTreeIter *next_iter = gtk_tree_iter_copy(&iter);
      const gboolean has_previous = gtk_tree_model_iter_previous(model, prev_iter);
      const gboolean has_next = gtk_tree_model_iter_next(model, next_iter);
      dt_mask_id_t grid = INVALID_MASKID;
      dt_mask_id_t id = INVALID_MASKID;
      _lib_masks_get_values(model, &iter, &module, &grid, &id);

      // moving the selection to a neighbouring row is interface comfort, not
      // persisted state: it stays as it was, flip or no flip
      if(has_previous)
        gtk_tree_selection_select_iter(selection, prev_iter);
      else if(has_next)
        gtk_tree_selection_select_iter(selection, next_iter);

      // historical quirk, kept deliberately: `module` is re-read from the row
      // above when there is one, overwriting the one just read from the row
      // itself. gtk_tree_model_iter_previous() stays among siblings, so for a
      // row inside a group this is the same module; changing it would alter
      // which module dt_masks_form_remove() logs its history item against
      if(has_previous)
        _lib_masks_get_values(model, prev_iter, &module, NULL, NULL);

      // deleting the base promotes the next shape in the application order; it
      // must stop showing an operator icon, exactly as when a move hands the
      // base over. its operator bit is deliberately left alone: clearing it
      // would change what the mask renders, which a UI commit must never do.
      // read from the live grp->points, which earlier turns of this loop have
      // already amputated -- not from a row position, which is what made the
      // old code promote the row *below* once the display stopped being a mirror
      dt_masks_form_t *pgrp = dt_masks_get_from_id(darktable.develop, grid);
      if(_group_point_index(pgrp, id) == 0)
        _handover_base_state(pgrp, _group_point_id(pgrp, 1), INVALID_MASKID);
      gtk_tree_iter_free(prev_iter);
      gtk_tree_iter_free(next_iter);

      // a row with no group is a real deletion, not a detach -- TREE_GROUPID is
      // 0 on a root row, and that is the very test the menu used to choose
      // between "delete everywhere" and "remove from <module>"
      if(!dt_is_valid_maskid(grid))
        _detach_from_unused_groups(id);

      dt_masks_form_remove(module, dt_masks_get_from_id(darktable.develop, grid),
                           dt_masks_get_from_id(darktable.develop, id));
    }
  }
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);

  dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
  _lib_masks_recreate_list(self);
}

static void _tree_duplicate_shape(GtkButton *button, dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;

  // we get the selected node
  // the list the menu was opened from: a duplication asked for in the library
  // must not read the masks zone's selection, and the other way round
  GtkWidget *view = _masks_active_view(lm);
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
  GList *items = gtk_tree_selection_get_selected_rows(selection, NULL);
  if(!items) return;
  GtkTreePath *item = (GtkTreePath *)items->data;
  GtkTreeIter iter;
  if(gtk_tree_model_get_iter(model, &iter, item))
  {
    dt_mask_id_t id = INVALID_MASKID;
    _lib_masks_get_values(model, &iter, NULL, NULL, &id);

    const dt_mask_id_t nid = dt_masks_form_duplicate(darktable.develop, id);
    if(dt_is_valid_maskid(nid))
    {
      dt_dev_masks_selection_change(darktable.develop, NULL, nid);
      //_lib_masks_recreate_list(self);
    }
  }
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);
}

// a shape is named in one place -- its library row -- and nowhere else:
// TREE_EDITABLE is (grp_id == 0), and a root row is a library row now. double
// clicking still opens the editor; this entry is what makes it findable
static void _tree_rename(GtkButton *button, dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  GtkWidget *view = _masks_active_view(lm);
  if(view != lm->library || !lm->lib_col || !lm->lib_name_cell) return;

  GList *items = gtk_tree_selection_get_selected_rows
    (gtk_tree_view_get_selection(GTK_TREE_VIEW(view)), NULL);
  if(!items) return;

  gtk_tree_view_set_cursor_on_cell(GTK_TREE_VIEW(view), items->data,
                                   lm->lib_col, lm->lib_name_cell, TRUE);
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);
}

static void _tree_cell_edited(GtkCellRendererText *cell,
                              gchar *path_string,
                              gchar *new_text,
                              GtkWidget *view)
{
  // the renderer belongs to one view; resolving the path against any other
  // would rename whatever row happens to sit at the same index
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  GtkTreeIter iter;
  if(!gtk_tree_model_get_iter_from_string(model, &iter, path_string)) return;

  dt_mask_id_t id = INVALID_MASKID;
  _lib_masks_get_values(model, &iter, NULL, NULL, &id);
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(!form) return;

  // we want to make sure that the new name is not an empty
  // string. else this would convert in the xmp file into "<rdf:li/>"
  // which produces problems. we use a single whitespace as the pure
  // minimum text.
  gchar *text = strlen(new_text) == 0 ? " " : new_text;

  // first, we need to update the mask name

  g_strlcpy(form->name, text, sizeof(form->name));
  dt_dev_add_masks_history_item(darktable.develop, NULL, FALSE);
}

static void _tree_selection_change(GtkTreeSelection *selection, dt_lib_masks_t *self)
{
  DT_GUARD_GUI_UPDATE();

  // gtk_tree_selection_get_tree_view() is the getter: no side table needed
  GtkWidget *view = GTK_WIDGET(gtk_tree_selection_get_tree_view(selection));
  const int nb = gtk_tree_selection_count_selected_rows(selection);

  // two lists, one canvas: the list that just spoke owns both, the other one
  // lets go. without this two rows stay highlighted and the last view to emit
  // wins the canvas. the unselect comes straight back here and the guard above
  // returns at once, so there is no loop -- and it happens BEFORE form_visible
  // is rebuilt below, from this view only. a deselection claims nothing
  if(nb > 0)
  {
    self->active_view = view;

    DT_ENTER_GUI_UPDATE();
    for(int v = 0; v < DT_MASKS_NVIEWS; v++)
    {
      GtkWidget *other = _masks_view(self, v);
      if(other && other != view)
        gtk_tree_selection_unselect_all
          (gtk_tree_view_get_selection(GTK_TREE_VIEW(other)));
    }
    DT_LEAVE_GUI_UPDATE();
  }

  // we reset all "show mask" icon of iops
  dt_masks_reset_show_masks_icons();

  // else, we create a new form group with the selection and display it
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  dt_masks_form_t *grp = dt_masks_create(DT_MASKS_GROUP);
  GList *items = gtk_tree_selection_get_selected_rows(selection, NULL);

  for(const GList *items_iter = items;
      items_iter;
      items_iter = g_list_next(items_iter))
  {
    GtkTreePath *item = (GtkTreePath *)items_iter->data;
    GtkTreeIter iter;

    if(gtk_tree_model_get_iter(model, &iter, item))
    {
      dt_mask_id_t grid = INVALID_MASKID;
      dt_mask_id_t id = INVALID_MASKID;
      _lib_masks_get_values(model, &iter, NULL, &grid, &id);

      dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
      if(form)
      {
        dt_masks_point_group_t *fpt = malloc(sizeof(dt_masks_point_group_t));
        fpt->formid = id;
        fpt->parentid = grid;
        fpt->state = DT_MASKS_STATE_USE;
        fpt->opacity = 1.0f;
        grp->points = g_list_append(grp->points, fpt);
        // we eventually set the "show masks" icon of iops
        if(nb == 1 && (form->type & DT_MASKS_GROUP))
        {
          dt_iop_module_t *module = NULL;
          _lib_masks_get_values(model, &iter, &module, NULL, NULL);

          if(module && module->blend_data
             && (module->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
             && !(module->flags() & IOP_FLAGS_NO_MASKS))
          {
            dt_iop_gui_blend_data_t *bd = module->blend_data;
            bd->masks_shown = DT_MASKS_EDIT_FULL;
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(bd->masks_edit), TRUE);
            gtk_widget_queue_draw(bd->masks_edit);
          }
        }
      }
    }
  }
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);

  dt_masks_form_t *grp2 = dt_masks_create(DT_MASKS_GROUP);
  grp2->formid = NO_MASKID;
  dt_masks_group_ungroup(grp2, grp);

  // don't call dt_masks_change_form_gui because it triggers a selection change again
  dt_masks_clear_form_gui(darktable.develop);
  darktable.develop->form_visible = grp2;

  // update sticky accels window
  if(darktable.view_manager->accels_window.window
     && darktable.view_manager->accels_window.sticky)
    dt_view_accels_refresh(darktable.view_manager);

  darktable.develop->form_gui->edit_mode = DT_MASKS_EDIT_FULL;
  dt_control_queue_redraw_center();

  _update_all_properties(self);
}

// number of modules that would actually lose the shape if it was deleted
// from the mask manager. this walks exactly like dt_masks_form_remove()
// does: same IOP_FLAGS_SUPPORTS_BLENDING filter, same two ways a module
// can hold a shape, so the number written in a destructive label is the
// one that deletion will act on. a module counts once, however many times
// the shape appears in its group.
// standalone groups are deliberately ignored: they are not modules, and
// dt_masks_form_remove() does not walk them either -- hence "module" and
// never "place" in the label.
static int _shape_use_count(const dt_mask_id_t formid)
{
  if(!dt_is_valid_maskid(formid)) return 0;

  int nb = 0;

  for(const GList *modules = darktable.develop->iop;
      modules;
      modules = g_list_next(modules))
  {
    dt_iop_module_t *m = modules->data;
    if(!(m->flags() & IOP_FLAGS_SUPPORTS_BLENDING)) continue;

    if(m->blend_params->mask_id == formid)
    {
      nb++;
      continue;
    }

    dt_masks_form_t *grp =
      dt_masks_get_from_id(darktable.develop, m->blend_params->mask_id);
    if(!(grp && (grp->type & DT_MASKS_GROUP))) continue;

    for(const GList *pts = grp->points; pts; pts = g_list_next(pts))
    {
      const dt_masks_point_group_t *pt = pts->data;
      if(pt->formid == formid)
      {
        nb++;
        break;
      }
    }
  }

  return nb;
}

// the context menu of a tree row, built at click. it was inline in
// _tree_button_pressed_cb, which made that function 345 lines of which the
// gesture handling was ten: the menu is one subject, the hit-testing of a click
// is another, and the next changes to this panel touch one or the other, never
// both. `view` is the list the click landed in -- the menu acts on that view's
// selection and nothing else.
// `mouse_path` is borrowed: valid for the call, freed by the caller, which is
// where it was hit-tested and where it outlives this function. it used to be
// freed here, on the one branch that reselects the row, and leaked on every
// other click -- and now that the two sides live in different functions, who
// frees it has to be written down rather than read off the code.
static void _tree_context_menu(dt_lib_module_t *self,
                               GtkWidget *view,
                               GtkTreeSelection *selection,
                               GtkTreeModel *model,
                               GtkTreePath *mouse_path,
                               const gboolean on_row,
                               dt_iop_module_t *module)
{
  dt_lib_masks_t *lm = self->data;
  GtkTreeIter iter;

  GdkModifierType state;
  gtk_get_current_event_state(&state);
  // if we are already inside the selection, no change
  if(on_row
     && !gtk_tree_selection_path_is_selected(selection, mouse_path))
  {
    if(!dt_modifier_is(state, GDK_CONTROL_MASK))
      gtk_tree_selection_unselect_all(selection);

    gtk_tree_selection_select_path(selection, mouse_path);
  }

  // and we display the context-menu
  GtkMenuShell *menu = GTK_MENU_SHELL(gtk_menu_new());
  GtkWidget *item;

  // we get all infos from selection
  const int nb = gtk_tree_selection_count_selected_rows(selection);
  gboolean from_group = FALSE;

  // read from grp->points, not from the row's position: the base is index 0,
  // the shape applied last is index n-1
  gboolean is_base_row = FALSE;
  gboolean is_last_applied = FALSE;
  dt_masks_state_t selected_states = DT_MASKS_STATE_NONE;

  // despite its name, grpid receives TREE_FORMID: the id of the selected
  // row itself. parent_grid is the one holding TREE_GROUPID, i.e. the
  // group that row belongs to (NO_MASKID for a top-level row)
  int grpid = NO_MASKID;
  dt_mask_id_t parent_grid = NO_MASKID;
  // TREE_MODULE of the *selected* row -- unlike `module` above, which
  // comes from the row under the pointer and is NULL on a blank click
  dt_iop_module_t *sel_module = NULL;
  int depth = 0;
  dt_masks_form_t *grp = NULL;

  if(nb > 0)
  {
    GList *selected = gtk_tree_selection_get_selected_rows(selection, NULL);
    GtkTreePath *it0 = (GtkTreePath *)selected->data;
    depth = gtk_tree_path_get_depth(it0);
    if(nb == 1)
    {
      // before freeing the list of selected rows, we check if the
      // form is a group or not
      if(gtk_tree_model_get_iter(model, &iter, it0))
      {
        _lib_masks_get_values(model, &iter, &sel_module, &parent_grid, &grpid);
        grp = dt_masks_get_from_id(darktable.develop, grpid);
      }

      // where the selected row sits in its group's application order. the
      // base is the one shape that must not be given an operator, and the two
      // ends of the list are the ones that cannot move any further. computed
      // for a single selection only. parent_grid holds TREE_GROUPID and grpid
      // holds TREE_FORMID -- a root row has parent_grid == NO_MASKID, so its
      // rank is -1 and both flags stay FALSE
      dt_masks_form_t *parent_grp =
        dt_masks_get_from_id(darktable.develop, parent_grid);
      const int rank = _group_point_index(parent_grp, grpid);
      const int nb_points = (parent_grp && (parent_grp->type & DT_MASKS_GROUP))
        ? (int)g_list_length(parent_grp->points) : 0;

      is_base_row = (rank == 0);
      is_last_applied = (rank >= 0) && (rank == nb_points - 1);
    }

    for(const GList *items_iter = selected;
        items_iter;
        items_iter = g_list_next(items_iter))
    {
      GtkTreePath *item = (GtkTreePath *)items_iter->data;

      if(gtk_tree_model_get_iter(model, &iter, item))
      {
        dt_mask_id_t grid = INVALID_MASKID;
        dt_mask_id_t id = INVALID_MASKID;
        _lib_masks_get_values(model, &iter, NULL, &grid, &id);

        dt_masks_form_t *grp2 = dt_masks_get_from_id(darktable.develop, grid);
        if(grp2 && (grp2->type & DT_MASKS_GROUP))
        {
          for(const GList *pts = grp2->points; pts; pts = g_list_next(pts))
          {
            dt_masks_point_group_t *pt = pts->data;
            if(pt->formid == id) selected_states |= pt->state;
          }
        }
      }
    }

    g_list_free_full(selected, (GDestroyNotify)gtk_tree_path_free);
  }

  if(depth > 1)
    from_group = TRUE;

  if(nb == 0 || (grp && grp->type & DT_MASKS_GROUP))
  {
    // right-clicking inside a module's group is itself the answer to "which
    // module?": no target line, we already know. a right-click on empty space
    // answers nothing, so the target is resolved and then written down,
    // exactly as the catalogue does it
    dt_iop_module_t *ctx = _mask_target_ok(sel_module) ? sel_module : NULL;
    if(!ctx)
    {
      ctx = _mask_default_target(self);
      _new_mask_target_header(menu, ctx);
    }
    _new_mask_shape_items(menu, ctx);
  }

  if(grp && grp->type & DT_MASKS_GROUP)
  {
    // existing forms
    gboolean has_unused_shapes = FALSE;
    GtkWidget *menu0 = gtk_menu_new();

    for(GList *forms = darktable.develop->forms;
        forms;
        forms = g_list_next(forms))
    {
      dt_masks_form_t *form = forms->data;
      if((form->type & (DT_MASKS_CLONE|DT_MASKS_NON_CLONE)) || form->formid == grpid)
      {
        continue;
      }
      char str[10000] = "";
      g_strlcat(str, form->name, sizeof(str));
      int nbuse = 0;

      // we search were this form is used
      for(const GList *modules = darktable.develop->iop;
          modules;
          modules = g_list_next(modules))
      {
        dt_iop_module_t *m = modules->data;
        dt_masks_form_t *grp = dt_masks_get_from_id(m->dev, m->blend_params->mask_id);
        if(grp && (grp->type & DT_MASKS_GROUP))
        {
          for(const GList *pts = grp->points; pts; pts = g_list_next(pts))
          {
            dt_masks_point_group_t *pt = pts->data;
            if(pt->formid == form->formid)
            {
              if(m == module)
              {
                nbuse = -1;
                break;
              }
              if(nbuse == 0) g_strlcat(str, " (", sizeof(str));
              g_strlcat(str, " ", sizeof(str));
              gchar *module_label = dt_history_item_get_name(m);
              g_strlcat(str, module_label, sizeof(str));
              g_free(module_label);
              nbuse++;
            }
          }
        }
      }
      if(nbuse != -1)
      {
        if(nbuse > 0) g_strlcat(str, " )", sizeof(str));

        // we add the menu entry
        item = gtk_menu_item_new_with_label(str);
        g_object_set_data(G_OBJECT(item), "formid", GUINT_TO_POINTER(form->formid));
        g_object_set_data(G_OBJECT(item), "module", module);
        g_signal_connect(G_OBJECT(item), "activate", G_CALLBACK(_tree_add_exist), grp);
        gtk_menu_shell_append(GTK_MENU_SHELL(menu0), item);
        has_unused_shapes = TRUE;
      }
    }

    if(has_unused_shapes)
    {
      item = gtk_menu_item_new_with_label(_("add existing shape"));
      gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), menu0);
      gtk_menu_shell_append(menu, item);
    }
  }

  if(!from_group && nb > 0)
  {
    dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, grpid);
    if(!(grp && (grp->type & DT_MASKS_GROUP)))
    {
      if(nb == 1)
      {
        // a root row is a library row: the masks zone only holds groups at
        // depth 1. renaming is only offered where a shape has a name of its
        // own, and only there is TREE_EDITABLE true
        if(view == lm->library)
        {
          item = gtk_menu_item_new_with_label(_("rename"));
          g_signal_connect(item, "activate", G_CALLBACK(_tree_rename), self);
          gtk_menu_shell_append(menu, item);
        }

        item = gtk_menu_item_new_with_label(_("duplicate this shape"));
        g_signal_connect(item, "activate", G_CALLBACK(_tree_duplicate_shape), self);
        gtk_menu_shell_append(menu, item);
      }
      // this does not only remove the row: the shape is dropped from
      // every module using it. say so, and say how many modules are
      // concerned -- but only when that number is both defined and
      // meaningful. it is computed for a single selection only, and a
      // multiple selection may mix shapes and groups, so every other
      // case keeps the historical wording rather than claiming a scope
      // that cannot be backed
      const int used = (nb == 1) ? _shape_use_count(grpid) : 0;

      if(used > 0)
      {
        gchar *label =
          g_strdup_printf(ngettext("delete everywhere (%d module)",
                                   "delete everywhere (%d modules)", used), used);
        item = gtk_menu_item_new_with_label(label);
        g_free(label);
      }
      else
        item = gtk_menu_item_new_with_label(_("delete this shape"));

      g_signal_connect(item, "activate", G_CALLBACK(_tree_delete_shape), self);
      gtk_menu_shell_append(menu, item);
    }
    else
    {
      item = gtk_menu_item_new_with_label(_("delete group (shapes are kept)"));
      g_signal_connect(item, "activate", G_CALLBACK(_tree_delete_shape), self);
      gtk_menu_shell_append(menu, item);
    }
  }
  else if(nb > 0 && depth < 3)
  {
    // here the shape is only detached from the group of that row: it
    // stays in the mask manager and in every other module. name the
    // group we are leaving so the difference with a deletion is
    // readable. parent_grid and sel_module are only filled for a single
    // selection, so anything else falls back to the plain wording
    dt_masks_form_t *parent = dt_masks_get_from_id(darktable.develop, parent_grid);
    gchar *scope = NULL;

    if(parent)
    {
      // a group owned by a module is already named after it
      // ("group `exposure'"), the plain module name reads better here
      if(sel_module && parent->formid == sel_module->blend_params->mask_id)
        scope = dt_history_item_get_name(sel_module);
      else if(*parent->name)
        scope = g_strdup(parent->name);
    }

    if(scope)
    {
      gchar *label = g_strdup_printf(_("remove from %s"), scope);
      item = gtk_menu_item_new_with_label(label);
      g_free(label);
      g_free(scope);
    }
    else
      item = gtk_menu_item_new_with_label(_("remove from group"));

    g_signal_connect(item, "activate", G_CALLBACK(_tree_delete_shape), self);
    gtk_menu_shell_append(menu, item);
  }

  if(nb > 1 && !from_group)
  {
    gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
    item = gtk_menu_item_new_with_label(_("group the forms"));
    g_signal_connect(item, "activate", G_CALLBACK(_tree_group), self);
    gtk_menu_shell_append(menu, item);
  }

  if(from_group && depth < 3)
  {
    gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
    _add_tree_operation(menu, _("use inverted shape"),
                        DT_MASKS_STATE_INVERSE, selected_states, TRUE);

    gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
    _add_tree_operation(menu, _("mode: union"),
                        DT_MASKS_STATE_UNION, selected_states, !is_base_row);
    _add_tree_operation(menu, _("mode: intersection"),
                        DT_MASKS_STATE_INTERSECTION, selected_states, !is_base_row);
    _add_tree_operation(menu, _("mode: difference"),
                        DT_MASKS_STATE_DIFFERENCE, selected_states, !is_base_row);
    _add_tree_operation(menu, _("mode: sum"),
                        DT_MASKS_STATE_SUM, selected_states, !is_base_row);
    _add_tree_operation(menu, _("mode: exclusion"),
                        DT_MASKS_STATE_EXCLUSION, selected_states, !is_base_row);

    gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
    // this is time, not space: the list is the order the shapes are applied
    // in. offered for a single row only -- see _tree_move_shape
    item = gtk_menu_item_new_with_label(_("apply earlier"));
    gtk_widget_set_sensitive(item, nb == 1 && !is_base_row);
    g_signal_connect(item, "activate", G_CALLBACK(_tree_apply_earlier), self);
    gtk_menu_shell_append(menu, item);

    item = gtk_menu_item_new_with_label(_("apply later"));
    gtk_widget_set_sensitive(item, nb == 1 && !is_last_applied);
    g_signal_connect(item, "activate", G_CALLBACK(_tree_apply_later), self);
    gtk_menu_shell_append(menu, item);
  }

  gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
  item = gtk_menu_item_new_with_label(_("delete unused shapes"));
  g_signal_connect(item, "activate", G_CALLBACK(_tree_cleanup), self);
  gtk_menu_shell_append(menu, item);

  gtk_widget_show_all(GTK_WIDGET(menu));

  GdkEvent *event = gtk_get_current_event();
  gtk_menu_popup_at_pointer(GTK_MENU(menu), event);
  gdk_event_free(event);
}

// what the click hit, and with which button -- nothing else. the right button
// hands over to _tree_context_menu() above; everything that used to be built
// here is there now
static void _tree_button_pressed_cb(GtkGestureSingle *gesture,
                                    int n_press,
                                    double x,
                                    double y,
                                    dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  GtkWidget *treeview = dt_gui_get_widget(gesture);
  // the menu is built now and its entries read the selection back when they
  // fire: point them at the list the menu was opened on. a right-click inside
  // an existing selection never reaches _tree_selection_change, so it is set
  // here too, and before anything can pop up
  lm->active_view = treeview;
  // we first need to adjust selection
  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(treeview));
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(treeview));

  GtkTreePath *mouse_path = NULL;
  GtkTreeIter iter;
  dt_iop_module_t *module = NULL;
  gboolean on_row = FALSE;
  if(gtk_tree_view_get_path_at_pos(GTK_TREE_VIEW(treeview),
                                   (gint)x, (gint)y, &mouse_path, NULL,
                                   NULL, NULL))
  {
    on_row = TRUE;
    // we retrieve the iter and module from path
    if(gtk_tree_model_get_iter(model, &iter, mouse_path))
    {
      _lib_masks_get_values(model, &iter, &module, NULL, NULL);
    }
  }

  /* single click with the right mouse button? */
  const guint button = gtk_gesture_single_get_current_button(gesture);
  if(button == GDK_BUTTON_PRIMARY)
  {
    // if click on a blank space, then deselect all
    if(!on_row)
    {
      gtk_tree_selection_unselect_all(selection);
    }
  }
  else if(button == GDK_BUTTON_SECONDARY)
  {
    _tree_context_menu(self, treeview, selection, model,
                       mouse_path, on_row, module);
  }

  // ours since gtk_tree_view_get_path_at_pos succeeded, on every button and
  // whatever the menu did with it -- the menu only borrows it
  if(mouse_path) gtk_tree_path_free(mouse_path);
}

static gboolean _tree_restrict_select(GtkTreeSelection *selection,
                                      GtkTreeModel *model,
                                      GtkTreePath *path,
                                      const gboolean path_currently_selected,
                                      gpointer data)
{
  DT_GUARD_GUI_UPDATE(TRUE);

  // if the change is SELECT->UNSELECT no pb
  if(path_currently_selected) return TRUE;

  // if selection is empty, no pb
  if(gtk_tree_selection_count_selected_rows(selection) == 0) return TRUE;

  // now we unselect all members of selection with not the same parent node
  // idem for all those with a different depth
  int *indices = gtk_tree_path_get_indices(path);
  const int depth = gtk_tree_path_get_depth(path);

  GList *items = gtk_tree_selection_get_selected_rows(selection, NULL);
  GList *items_iter = items;
  while(items_iter)
  {
    GtkTreePath *item = (GtkTreePath *)items_iter->data;
    const int dd = gtk_tree_path_get_depth(item);
    int *ii = gtk_tree_path_get_indices(item);
    int ok = 1;
    if(dd != depth)
      ok = 0;
    else if(dd == 1)
      ok = 1;
    else if(ii[dd - 2] != indices[dd - 2])
      ok = 0;
    if(!ok)
    {
      gtk_tree_selection_unselect_path(selection, item);
      g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);
      items_iter = items = gtk_tree_selection_get_selected_rows(selection, NULL);
      continue;
    }
    items_iter = g_list_next(items_iter);
  }
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);
  return TRUE;
}

static gboolean _tree_query_tooltip(GtkWidget *widget,
                                    gint x,
                                    gint y,
                                    const gboolean keyboard_tip,
                                    GtkTooltip *tooltip,
                                    gpointer data)
{
  GtkTreeIter iter;
  GtkTreeView *tree_view = GTK_TREE_VIEW(widget);
  GtkTreeModel *model = gtk_tree_view_get_model(tree_view);
  GtkTreePath *path = NULL;
  gchar *tmp = NULL;
  gboolean show = FALSE;

  if(!gtk_tree_view_get_tooltip_context(tree_view, &x, &y,
                                        keyboard_tip, &model, &path, &iter))
    return FALSE;

  gtk_tree_model_get(model, &iter, TREE_USED_TEXT, &tmp, -1);
  // it used to be tied to the "used" badge being visible, so the one row that
  // most needs a word -- a shape nothing references -- was the one row that
  // stayed silent
  show = tmp && *tmp;
  if(show)
  {
    // plain text, not markup: this string carries group names typed by the
    // photographer, and a single "&" in one of them blanked the whole tooltip
    gtk_tooltip_set_text(tooltip, tmp);
    gtk_tree_view_set_tooltip_row(tree_view, tooltip, path);
  }

  gtk_tree_path_free(path);
  g_free(tmp);

  return show;
}

static void _is_form_used(const dt_mask_id_t formid,
                          dt_masks_form_t *grp,
                          char *text,
                          const size_t text_length,
                          int *nb)
{
  if(!grp)
  {
    for(const GList *forms = darktable.develop->forms;
        forms;
        forms = g_list_next(forms))
    {
      dt_masks_form_t *form = forms->data;
      if(form->type & DT_MASKS_GROUP) _is_form_used(formid, form, text, text_length, nb);
    }
  }
  else if(grp->type & DT_MASKS_GROUP)
  {
    for(const GList *points = grp->points;
        points;
        points = g_list_next(points))
    {
      dt_masks_point_group_t *point = points->data;
      dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, point->formid);
      if(form)
      {
        if(point->formid == formid)
        {
          (*nb)++;
          if(*nb > 1) g_strlcat(text, "\n", text_length);
          g_strlcat(text, grp->name, text_length);
        }

        if(form->type & DT_MASKS_GROUP)
          _is_form_used(formid, form, text, text_length, nb);
      }
    }
  }
}

// three states -- and three meanings of "used" already live in this file, none
// of which coincide:
//   _shape_use_count()      counts MODULES exactly the way dt_masks_form_remove()
//                           walks them: NOT recursive, and it must stay that way,
//                           it backs the "delete everywhere (n modules)" label.
//   _is_form_used()         counts memberships in ANY group of dev->forms,
//                           stand-alone groups included. it feeds the "used"
//                           badge and its tooltip, and is left untouched.
//   _masks_cleanup_unused() keeps whatever is reachable, recursively, from a
//                           history item's blend_params->mask_id.
// the split a library row needs is "does a module render this", so MODULE is
// answered by dt_masks_is_in_module(), which IS recursive: _shape_use_count()
// misses a shape buried in a sub-group of a module's mask and would call a
// rendered shape unlinked.
// the union of the two other states is what "delete unused shapes" is about,
// which is why the caption under the library says "not linked to a module" and
// never "unused": a shape sitting in a stand-alone group is not unused, and the
// cleanup takes it away all the same.
// `groups`, when given, receives the group names behind a GROUP_ONLY verdict.
// deliberately not cached, though a rebuild asks this of each shape four times:
// twice in gui_update's two-pass ordering, once in _lib_masks_list_recurs for
// the tooltip, once in _set_iter_name for the word on the row. each call walks
// dev->iop and then, on failure, dev->forms -- a few hundred pointer
// comparisons over lists a human drew by hand, against a rebuild that only
// happens when the structure changed. a cache would have to be invalidated
// wherever a module's mask_id or a group's contents move, which is the kind of
// bookkeeping that goes stale in silence and shows a wrong word on the row.
static dt_masks_shape_scope_t _shape_scope(const dt_mask_id_t formid,
                                           char *groups,
                                           const size_t groups_length)
{
  if(groups && groups_length) groups[0] = '\0';
  if(!dt_is_valid_maskid(formid)) return DT_MASKS_SCOPE_ORPHAN;

  for(const GList *modules = darktable.develop->iop;
      modules;
      modules = g_list_next(modules))
  {
    dt_iop_module_t *m = modules->data;
    if((m->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
       && !(m->flags() & IOP_FLAGS_NO_MASKS)
       && dt_masks_is_in_module(formid, m))
      return DT_MASKS_SCOPE_MODULE;
  }

  // no module renders it. is it filed anywhere at all? _is_form_used walks every
  // group of dev->forms, stand-alone ones included -- which is what makes it the
  // wrong answer to the module question and the right one to this one. it counts
  // a nested group twice; we only read "> 0", so that long-standing quirk cannot
  // reach the screen through here
  char str[1000] = "";
  int nb = 0;
  _is_form_used(formid, NULL, str, sizeof(str), &nb);

  if(nb > 0)
  {
    if(groups) g_strlcpy(groups, str, groups_length);
    return DT_MASKS_SCOPE_GROUP_ONLY;
  }
  return DT_MASKS_SCOPE_ORPHAN;
}

static void _lib_masks_list_recurs(GtkTreeStore *treestore,
                                   GtkTreeIter *toplevel,
                                   dt_masks_form_t *form,
                                   const int grp_id,
                                   dt_iop_module_t *module,
                                   const int gstate,
                                   const float opacity,
                                   dt_lib_masks_t *lm)
{
  if(form->type & (DT_MASKS_CLONE|DT_MASKS_NON_CLONE)) return;
  // we create the text entry
  char str[256] = "";
  g_strlcat(str, form->name, sizeof(str));
  // we get the right pixbufs
  GdkPixbuf *icop = NULL;
  GdkPixbuf *icinv = NULL;
  GdkPixbuf *icuse = NULL;

  const gboolean show = gstate & DT_MASKS_STATE_SHOW;

  if(gstate & DT_MASKS_STATE_UNION)
    icop = lm->ic_union;
  else if(gstate & DT_MASKS_STATE_INTERSECTION)
    icop = lm->ic_intersection;
  else if(gstate & DT_MASKS_STATE_DIFFERENCE)
    icop = lm->ic_difference;
  else if(gstate & DT_MASKS_STATE_SUM)
    icop = lm->ic_sum;
  else if(gstate & DT_MASKS_STATE_EXCLUSION)
    icop = lm->ic_exclusion;

  if(gstate & DT_MASKS_STATE_INVERSE)
    icinv = lm->ic_inverse;

  char str2[1000] = "";
  int nbuse = 0;

  if(grp_id == 0)
  {
    // the "used" badge and its tooltip keep their historical meaning, quirks
    // included: this shape is filed in at least one group, here are their names
    _is_form_used(form->formid, NULL, str2, sizeof(str2), &nbuse);
    if(nbuse > 0) icuse = lm->ic_used;

    if(!(form->type & DT_MASKS_GROUP))
    {
      // a library row. the word itself goes in TREE_LINK, written by
      // _set_iter_name like every other derived column; what is settled here is
      // the sentence behind it, which needs the names of the groups holding the
      // shape and so cannot be had from the scope alone. short on the row,
      // spelled out under the list and in this tooltip: a library where every
      // row carries a sentence is a library nobody reads
      char groups[1000] = "";
      const dt_masks_shape_scope_t scope =
        _shape_scope(form->formid, groups, sizeof(groups));

      if(scope != DT_MASKS_SCOPE_MODULE)
      {
        if(scope == DT_MASKS_SCOPE_GROUP_ONLY)
          // "unused" would be a lie in one direction and a trap in the other:
          // no module renders it, yet the group holding it is real
          snprintf(str2, sizeof(str2),
                   _("no module uses this shape\n"
                     "it is only filed in:\n%s"), groups);
        else
          g_strlcpy(str2, _("no module and no group uses this shape"),
                    sizeof(str2));
      }
    }
  }

  if(!(form->type & DT_MASKS_GROUP))
  {
    // we just add it to the tree
    GtkTreeIter child;

    if(toplevel)
    {
      // inside a group: rows follow grp->points, so the shape that lays the
      // base comes first and the list reads top-down in the order the shapes
      // are applied. TREE_MODULE / TREE_GROUPID are written just below, and
      // _set_iter_name reads them back -- do not move that call up
      gtk_tree_store_append(treestore, &child, toplevel);
    }
    else
    {
      // the library store holds shapes and nothing else, so there is no run of
      // groups to step over any more: rows land in the order gui_update feeds
      // them, the ones a module renders first. the block that counted leading
      // group rows went out with the groups
      gtk_tree_store_append(treestore, &child, NULL);
    }

    gtk_tree_store_set(treestore, &child,
                       TREE_TEXT, str,
                       TREE_MODULE, module,
                       TREE_GROUPID, grp_id,
                       TREE_FORMID, form->formid,
                       TREE_EDITABLE, (grp_id == 0),
                       TREE_IC_OP, icop,
                       TREE_IC_OP_VISIBLE, (icop != NULL) && show,
                       TREE_IC_INVERSE, icinv,
                       TREE_IC_INVERSE_VISIBLE, (icinv != NULL),
                       TREE_IC_USED, icuse,
                       TREE_IC_USED_VISIBLE, (nbuse > 0),
                       TREE_USED_TEXT, str2,
                       -1);
    _set_iter_name(lm, form, gstate, opacity, GTK_TREE_MODEL(treestore), &child);
  }
  else
  {
    // we first check if it's a "module" group or not
    if(grp_id == 0 && !module)
    {
      for(const GList *iops = darktable.develop->iop; iops; iops = g_list_next(iops))
      {
        dt_iop_module_t *iop = iops->data;
        if((iop->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
           && !(iop->flags() & IOP_FLAGS_NO_MASKS)
           && iop->blend_params->mask_id == form->formid)
        {
          module = iop;
          break;
        }
      }
    }

    // we add the group node to the tree
    GtkTreeIter child;
    if(toplevel)
      // a group nested in another group is a member of its parent's
      // application order like any shape: same rule as above
      gtk_tree_store_append(treestore, &child, toplevel);
    else
      // at the root there is no application order to show. keep the historical
      // stacking so root rows -- and _tree_group, which builds a new group in
      // the visual order of the selection and is only offered for root rows --
      // are untouched
      gtk_tree_store_prepend(treestore, &child, NULL);
    gtk_tree_store_set(treestore, &child,
                       TREE_TEXT, str,
                       TREE_MODULE, module,
                       TREE_GROUPID, grp_id,
                       TREE_FORMID, form->formid,
                       TREE_EDITABLE, (grp_id == 0),
                       TREE_IC_OP, icop,
                       TREE_IC_OP_VISIBLE, (icop != NULL) && show,
                       TREE_IC_INVERSE, icinv,
                       TREE_IC_INVERSE_VISIBLE, (icinv != NULL),
                       TREE_IC_USED, icuse,
                       TREE_IC_USED_VISIBLE, (nbuse > 0),
                       TREE_USED_TEXT, str2,
                       -1);
    _set_iter_name(lm, form, gstate, opacity, GTK_TREE_MODEL(treestore), &child);

    // we add all nodes to the tree
    for(const GList *forms = form->points; forms; forms = g_list_next(forms))
    {
      dt_masks_point_group_t *grpt = forms->data;
      dt_masks_form_t *f = dt_masks_get_from_id(darktable.develop, grpt->formid);
      if(f)
        _lib_masks_list_recurs(treestore, &child, f,
                               form->formid, module, grpt->state, grpt->opacity, lm);
    }
  }
}

gboolean _find_mask_iter_by_values(GtkTreeModel *model,
                                   GtkTreeIter *iter,
                                   const dt_iop_module_t *module,
                                   const dt_mask_id_t formid,
                                   const int level)
{
  do
  {
    dt_mask_id_t fid = INVALID_MASKID;
    dt_iop_module_t *mod;
    _lib_masks_get_values(model, iter, &mod, NULL, &fid);
    gboolean found = (fid == formid)
      && ((level == 1)
          || (module == NULL || (mod && dt_iop_module_is(module, mod->op))));
    if(found) return found;

    GtkTreeIter child, parent = *iter;
    if(gtk_tree_model_iter_children(model, &child, &parent))
    {
      found = _find_mask_iter_by_values(model, &child, module, formid, level + 1);
      if(found)
      {
        *iter = child;
        return found;
      }
    }
  } while(gtk_tree_model_iter_next(model, iter));

  return FALSE;
}

GList *_lib_masks_get_selected(GtkWidget *view)
{
  GList *res = NULL;

  // a selection belongs to one view: saved from that view, restored into that
  // view's new store. carrying a row over to the other list would answer a
  // different question than the one that was asked
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));

  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));

  GList *items = gtk_tree_selection_get_selected_rows(selection, &model);

  for(GList *items_iter = items;
      items_iter;
      items_iter = g_list_next(items_iter))
  {
    GtkTreePath *item = (GtkTreePath *)items_iter->data;
    GtkTreeIter iter;
    if(gtk_tree_model_get_iter(model, &iter, item))
    {
      dt_mask_id_t fid = INVALID_MASKID;
      dt_mask_id_t gid = INVALID_MASKID;
      dt_iop_module_t *mod;
      _lib_masks_get_values(model, &iter, &mod, &gid, &fid);
      res = g_list_prepend(res, GINT_TO_POINTER(fid));
      res = g_list_prepend(res, GINT_TO_POINTER(gid));
      res = g_list_prepend(res, (void *)(mod));
    }
  }

  g_list_foreach(items, (GFunc)gtk_tree_path_free, NULL);
  g_list_free(items);

  return res;
}

// consumes `selectids` (triples module/groupid/formid, as built above) and
// re-selects in `model` what it can still find. a row that is gone is simply not
// restored. returns TRUE if anything was selected, so the caller can make that
// view the active one
static gboolean _restore_selection(GtkWidget *view,
                                   GtkTreeModel *model,
                                   GList *selectids)
{
  gboolean any = FALSE;
  if(!selectids) return FALSE;

  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));

  for(GList *ids = selectids; ids; )
  {
    dt_iop_module_t *mod = ids->data;
    ids = g_list_next(ids);
    // const int gid = GPOINTER_TO_INT(ids->data); // not needed, skip it
    ids = g_list_next(ids);
    const int fid = GPOINTER_TO_INT(ids->data);
    ids = g_list_next(ids);

    GtkTreeIter iter;
    if(gtk_tree_model_get_iter_first(model, &iter)
       && _find_mask_iter_by_values(model, &iter, mod, fid, 1))
    {
      GtkTreePath *path = gtk_tree_model_get_path(model, &iter);
      gtk_tree_view_expand_to_path(GTK_TREE_VIEW(view), path);
      gtk_tree_view_scroll_to_cell(GTK_TREE_VIEW(view), path, NULL,
                                   TRUE, 0.5, 0.5);
      gtk_tree_path_free(path);
      gtk_tree_selection_select_iter(selection, &iter);
      any = TRUE;
    }
  }
  g_list_free(selectids);
  return any;
}

// A hash of everything that determines the tree's *structure* (which rows exist
// and how they nest): the active module, and every form's id/type plus each
// group member's id and parent. Deliberately excludes per-shape parameters -
// opacity, state, size, feather, rotation - so editing them (e.g. dragging a
// slider) leaves the hash unchanged and gui_update can refresh the rows in place
// instead of recreating the store, which would reset the panel scroll.
static guint _forms_structure_hash(void)
{
  dt_develop_t *dev = darktable.develop;
  guint h = 2166136261u; // FNV-1a
#define _MIX(v)                                                                                    \
  do                                                                                               \
  {                                                                                                \
    h = (h ^ (guint)(v)) * 16777619u;                                                              \
  } while(0)
  const size_t mod = (size_t)dev->gui_module;
  _MIX(mod);
  _MIX(mod >> 32);
  for(const GList *l = dev->forms; l; l = g_list_next(l))
  {
    const dt_masks_form_t *f = l->data;
    _MIX(f->formid);
    _MIX(f->type);
    if(f->type & DT_MASKS_GROUP)
      for(const GList *p = f->points; p; p = g_list_next(p))
      {
        const dt_masks_point_group_t *pt = p->data;
        _MIX(pt->formid);
        _MIX(pt->parentid);
      }
  }
  // which module wears which mask is part of what the tree shows: the library
  // is ordered by it and each row states it. it lives in blend_params, not in
  // dev->forms, so without this the early-out in gui_update would keep serving
  // a library saying "no module" about a shape a module has just picked up.
  // the price is a rebuild -- and a scroll reset -- on mask assignment, which
  // is rare and genuinely structural
  for(const GList *l = dev->iop; l; l = g_list_next(l))
  {
    const dt_iop_module_t *m = l->data;
    if(m->flags() & IOP_FLAGS_SUPPORTS_BLENDING) _MIX(m->blend_params->mask_id);
  }
#undef _MIX
  return h;
}

void gui_update(dt_lib_module_t *self)
{
  /* first destroy all buttons in list */
  dt_lib_masks_t *lm = self->data;
  if(!lm) return;

  DT_TRY_GUI_UPDATE();

  // if the tree structure is unchanged (e.g. this update was triggered by a slider
  // edit, which only alters shape parameters), do not recreate the store - that
  // resets the panel scroll and makes the view jump under the cursor. Just refresh
  // the existing rows in place (text/icons), leaving scroll and selection untouched.
  const guint newhash = _forms_structure_hash();
  if((lm->treeview || lm->library) && lm->tree_hash_valid
     && newhash == lm->tree_hash
     && !dt_is_valid_maskid(lm->pending_selectid))
  {
    // both models: a parameter edit refreshes whichever rows show it
    for(int v = 0; v < DT_MASKS_NVIEWS; v++)
    {
      GtkWidget *view = _masks_view(lm, v);
      GtkTreeModel *model =
        view ? gtk_tree_view_get_model(GTK_TREE_VIEW(view)) : NULL;
      if(model)
        gtk_tree_model_foreach(model, _update_foreach, lm);
    }
    DT_LEAVE_GUI_UPDATE();
    return;
  }

  // if a treeview is already present, let's get the currently selected items
  // as we are going to recreate the tree.
  // two views now, so each one answers for itself: testing lm->treeview and
  // then reading lm->library through it is one field deciding for another
  GList *selectids[DT_MASKS_NVIEWS] = { NULL, NULL };

  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
  {
    GtkWidget *view = _masks_view(lm, v);
    if(view) selectids[v] = _lib_masks_get_selected(view);
  }

  _lib_masks_inactivate_icons(self);

  // we store : text ; *module ; groupid ; formid -- see _masks_store_new()
  GtkTreeStore *store[DT_MASKS_NVIEWS];
  for(int v = 0; v < DT_MASKS_NVIEWS; v++) store[v] = _masks_store_new();

  // top list: the masks, i.e. the groups, with their shapes nested in the order
  // they are applied. unchanged -- including the prepend that keeps the
  // historical stacking of root groups, which _tree_group still agrees with
  for(const GList *forms = darktable.develop->forms;
      forms;
      forms = g_list_next(forms))
  {
    dt_masks_form_t *form = forms->data;
    if(form->type & DT_MASKS_GROUP)
      _lib_masks_list_recurs(store[0], NULL, form, 0, NULL, 0, 1.0, lm);
  }

  // bottom list: the library every mask draws from -- each shape exactly once.
  // two passes, so the ones no module renders end up together at the bottom,
  // right above the caption that names them. an ordering, deliberately, and not
  // a header row: a row in this store that is not a shape would be walked by
  // _remove_foreach, _update_foreach, _lib_masks_selection_change_r and the
  // depth arithmetic of the context menu, each of which would have to learn to
  // skip it
  int unlinked = 0;
  for(int pass = 0; pass < 2; pass++)
  {
    for(const GList *forms = darktable.develop->forms;
        forms;
        forms = g_list_next(forms))
    {
      dt_masks_form_t *form = forms->data;
      if(form->type & DT_MASKS_GROUP) continue;
      // clones never reach the manager (_lib_masks_list_recurs drops them);
      // filtered here too so the count below matches the rows on screen
      if(form->type & (DT_MASKS_CLONE | DT_MASKS_NON_CLONE)) continue;

      const gboolean linked =
        _shape_scope(form->formid, NULL, 0) == DT_MASKS_SCOPE_MODULE;
      if(linked != (pass == 0)) continue;
      if(!linked) unlinked++;

      _lib_masks_list_recurs(store[1], NULL, form, 0, NULL, 0, 1.0, lm);
    }
  }

  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
  {
    GtkWidget *view = _masks_view(lm, v);
    if(!view) continue;
    gtk_tree_view_set_model(GTK_TREE_VIEW(view), GTK_TREE_MODEL(store[v]));
    // select the rows as selected in the previous tree
    if(_restore_selection(view, GTK_TREE_MODEL(store[v]), selectids[v]))
      lm->active_view = view;
  }

  // apply a selection that was requested before this row existed (e.g. a shape
  // that was just created): now that the tree is rebuilt, select its row so the
  // new shape shows up as selected in the mask manager.
  if(dt_is_valid_maskid(lm->pending_selectid))
  {
    // a shape created with a module focused lands in that module's mask; one
    // created from the manager lands in the library. try the masks first
    gboolean found = FALSE;
    GtkWidget *holder = NULL;
    for(int v = 0; v < DT_MASKS_NVIEWS && !found; v++)
    {
      GtkWidget *view = _masks_view(lm, v);
      if(!view) continue;
      GtkTreeModel *model = GTK_TREE_MODEL(store[v]);
      GtkTreeSelection *selection =
        gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
      GtkTreeIter iter;
      if(!gtk_tree_model_get_iter_first(model, &iter)) continue;

      gtk_tree_view_expand_all(GTK_TREE_VIEW(view));
      found = _lib_masks_selection_change_r(model, selection, &iter,
                                            lm->pending_selmodule,
                                            lm->pending_selectid, 1);
      if(found)
      {
        holder = view;
        lm->active_view = view;

        // make the just-created shape the active one so the properties reflect
        // it, rather than the previously selected shape: mask_form_selected_id
        // is otherwise only set once a shape is clicked on the canvas.
        darktable.develop->mask_form_selected_id = lm->pending_selectid;
        _update_all_properties(lm);

        GList *rows = gtk_tree_selection_get_selected_rows(selection, NULL);
        if(rows)
        {
          // a genuinely new shape: reveal its row
          gtk_tree_view_scroll_to_cell(GTK_TREE_VIEW(view), rows->data,
                                       NULL, TRUE, 0.5, 0.5);
          g_list_free_full(rows, (GDestroyNotify)gtk_tree_path_free);
        }
      }
    }

    // one list highlighted, never two: the restore above may have put a row
    // back in the list the pending id does not live in. that list is then left
    // showing nothing, so it also goes back to being folded -- searching it
    // expanded it whole, and an empty tree splayed open is not a state the user
    // asked for. the list that holds the pending id keeps whatever the search
    // opened, since that is what reveals the row.
    for(int v = 0; v < DT_MASKS_NVIEWS; v++)
    {
      GtkWidget *view = _masks_view(lm, v);
      if(!view || view == holder) continue;
      GtkTreeSelection *selection =
        gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
      if(holder) gtk_tree_selection_unselect_all(selection);
      if(gtk_tree_selection_count_selected_rows(selection) == 0)
        gtk_tree_view_collapse_all(GTK_TREE_VIEW(view));
    }
    lm->pending_selectid = NO_MASKID;
    lm->pending_selmodule = NULL;
  }

  // the caption sits directly under the run it describes and says what is true
  // of that run, not what the cleanup promises to do about it
  if(unlinked > 0)
  {
    gchar *t = g_strdup_printf(ngettext("%d shape not linked to a module",
                                        "%d shapes not linked to a module",
                                        unlinked), unlinked);
    gtk_label_set_text(GTK_LABEL(lm->lib_unlinked), t);
    g_free(t);
  }
  gtk_widget_set_visible(lm->lib_unlinked, unlinked > 0);

  for(int v = 0; v < DT_MASKS_NVIEWS; v++) g_object_unref(store[v]);

  // remember the structure we just built so a later parameter-only update can be
  // served in place (see the early-out at the top of this function).
  lm->tree_hash = newhash;
  lm->tree_hash_valid = TRUE;

  DT_LEAVE_GUI_UPDATE();

  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
  {
    GtkWidget *view = _masks_view(lm, v);
    if(view) dt_gui_widget_reallocate_now(view);
  }
}

static void _lib_masks_recreate_list(dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  dt_lib_gui_queue_update(self);

  DT_TRY_GUI_UPDATE();

  _update_all_properties(lm);

  DT_LEAVE_GUI_UPDATE();

}

static gboolean _update_foreach(GtkTreeModel *model,
                                GtkTreePath *path,
                                GtkTreeIter *iter,
                                gpointer data)
{
  if(!iter) return 0;

  // we retrieve the ids
  dt_mask_id_t grid = INVALID_MASKID;
  dt_mask_id_t id = INVALID_MASKID;
  _lib_masks_get_values(model, iter, NULL, &grid, &id);

  // we retrieve the forms
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(!form) return 0;
  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, grid);

  // and the values
  int state = 0;
  float opacity = 1.0f;

  if(grp && (grp->type & DT_MASKS_GROUP))
  {
    for(const GList *pts = grp->points; pts; pts = g_list_next(pts))
    {
      dt_masks_point_group_t *pt = pts->data;
      if(pt->formid == id)
      {
        state = pt->state;
        opacity = pt->opacity;
        break;
      }
    }
  }

  _set_iter_name(data, form, state, opacity, model, iter);
  return 0;
}

static void _lib_masks_update_list(dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  // for each node , we refresh the string
  // both models, always: a shape shown in one list and stale in the other is
  // the same shape saying two things at once
  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
  {
    GtkWidget *view = _masks_view(lm, v);
    GtkTreeModel *model =
      view ? gtk_tree_view_get_model(GTK_TREE_VIEW(view)) : NULL;
    if(!model) continue;

    gtk_tree_model_foreach(model, _update_foreach, lm);
  }
}

static gboolean _remove_foreach(GtkTreeModel *model,
                                GtkTreePath *path,
                                GtkTreeIter *iter,
                                gpointer data)
{
  if(!iter) return 0;
  GList **rl = (GList **)data;
  const dt_mask_id_t refid = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(model), "formid"));
  const dt_mask_id_t refgid = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(model), "groupid"));

  dt_mask_id_t grid = INVALID_MASKID;
  dt_mask_id_t id = INVALID_MASKID;
  _lib_masks_get_values(model, iter, NULL, &grid, &id);

  if(grid == refgid && id == refid)
  {
    GtkTreeRowReference *rowref = gtk_tree_row_reference_new(model, path);
    *rl = g_list_append(*rl, rowref);
  }
  return 0;
}

static void _lib_masks_remove_item(dt_lib_module_t *self,
                                   const dt_mask_id_t formid,
                                   const dt_mask_id_t parentid)
{
  dt_lib_masks_t *lm = self->data;
  // for each node , we refresh the string
  // both models, always. a shape removed from one and left in the other is a
  // row pointing at a freed form -- and develop/masks/path.c calls
  // dt_dev_masks_list_remove() directly, from the canvas
  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
  {
    GtkWidget *view = _masks_view(lm, v);
    GtkTreeModel *model =
      view ? gtk_tree_view_get_model(GTK_TREE_VIEW(view)) : NULL;
    if(!model) continue;

    GList *rl = NULL;
    g_object_set_data(G_OBJECT(model), "formid", GUINT_TO_POINTER(formid));
    g_object_set_data(G_OBJECT(model), "groupid", GUINT_TO_POINTER(parentid));
    gtk_tree_model_foreach(model, _remove_foreach, &rl);

    for(const GList *rlt = rl; rlt; rlt = g_list_next(rlt))
    {
      GtkTreeRowReference *rowref = (GtkTreeRowReference *)rlt->data;
      GtkTreePath *path = gtk_tree_row_reference_get_path(rowref);
      gtk_tree_row_reference_free(rowref);
      if(path)
      {
        GtkTreeIter iter;
        if(gtk_tree_model_get_iter(model, &iter, path))
        {
          gtk_tree_store_remove(GTK_TREE_STORE(model), &iter);
        }
        gtk_tree_path_free(path);
      }
    }
    g_list_free(rl);
  }
}

static gboolean _lib_masks_selection_change_r(GtkTreeModel *model,
                                              GtkTreeSelection *selection,
                                              GtkTreeIter *iter,
                                              struct dt_iop_module_t *module,
                                              const dt_mask_id_t selectid,
                                              const int level)
{
  gboolean found = FALSE;

  GtkTreeIter i = *iter;
  do
  {
    dt_mask_id_t id = INVALID_MASKID;
    dt_iop_module_t *mod;
    _lib_masks_get_values(model, &i, &mod, NULL, &id);

    if((id == selectid)
       && ((level == 1)
           || (module == NULL || (mod && dt_iop_module_is(module, mod->op)))))
    {
      gtk_tree_selection_select_iter(selection, &i);
      found = TRUE;
      break;
    }

    // check for children if any
    GtkTreeIter child, parent = i;
    if(gtk_tree_model_iter_children(model, &child, &parent))
    {
      found = _lib_masks_selection_change_r(model, selection,
                                            &child, module, selectid, level + 1);
      if(found)
      {
        break;
      }
    }
  } while(gtk_tree_model_iter_next(model, &i) == TRUE);

  return found;
}

static void _lib_masks_selection_change(dt_lib_module_t *self,
                                        struct dt_iop_module_t *module,
                                        const dt_mask_id_t selectid)
{
  dt_lib_masks_t *lm = self->data;
  if(!lm->treeview || !lm->library) return;

  DT_ENTER_GUI_UPDATE();

  // clear both, then answer in the masks first: a request carrying a module
  // means a row inside that module's mask, and the library row for the same
  // shape would be a different answer to the same question
  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
    gtk_tree_selection_unselect_all
      (gtk_tree_view_get_selection(GTK_TREE_VIEW(_masks_view(lm, v))));

  gboolean found = FALSE;
  for(int v = 0; v < DT_MASKS_NVIEWS && !found; v++)
  {
    GtkWidget *view = _masks_view(lm, v);
    GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
    GtkTreeIter iter;
    if(!model || !gtk_tree_model_get_iter_first(model, &iter)) continue;

    GtkTreeSelection *selection =
      gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
    gtk_tree_view_expand_all(GTK_TREE_VIEW(view));
    found = _lib_masks_selection_change_r(model, selection, &iter, module, selectid, 1);
    if(found) lm->active_view = view;
    else gtk_tree_view_collapse_all(GTK_TREE_VIEW(view));
  }

  // a shape just created is not yet in the (still to be rebuilt) tree, so the
  // row can't be selected now; remember it and let gui_update apply it once the
  // tree is rebuilt. Otherwise the request is satisfied, so drop any pending one.
  if(!found && dt_is_valid_maskid(selectid))
  {
    lm->pending_selectid = selectid;
    lm->pending_selmodule = module;
  }
  else
  {
    lm->pending_selectid = NO_MASKID;
    lm->pending_selmodule = NULL;
  }

  DT_LEAVE_GUI_UPDATE();

  _update_all_properties(lm);

  // a just-created shape is not in the tree yet (pending): rebuild the list now
  // instead of waiting for the next lazy panel redraw. Otherwise the new row -
  // and the panel reflow it causes - only appears the first time the user drags
  // a property slider, making the sliders visibly jump. gui_update applies (and
  // clears) the pending selection. dt_lib_gui_update is a no-op unless a rebuild
  // was already queued (dt_dev_masks_list_change, which creation triggers).
  if(dt_is_valid_maskid(lm->pending_selectid))
    dt_lib_gui_update(self);
}

static GdkPixbuf *_get_pixbuf_from_cairo(DTGTKCairoPaintIconFunc paint,
                                         const int width,
                                         const int height)
{
  cairo_surface_t *cst = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
  cairo_t *cr = cairo_create(cst);
  dt_gui_gtk_set_source_rgba(cr, DT_GUI_COLOR_BUTTON_FG, 1.0);
  paint(cr, 0, 0, width, height, 0, NULL);
  cairo_destroy(cr);
  guchar *data = cairo_image_surface_get_data(cst);
  dt_draw_cairo_to_gdk_pixbuf(data, width, height);
  return gdk_pixbuf_new_from_data(data, GDK_COLORSPACE_RGB, TRUE, 8, width, height,
                                  cairo_image_surface_get_stride(cst), NULL, NULL);
}

// both lists are the same widget fed different rows. keeping them built by one
// function is the whole point: the moment they diverge, every fix has to be
// written twice and one of the two will be forgotten.
// NOTE the renderer stack below is the one place renderers are packed.
static void _build_masks_view(dt_lib_module_t *self,
                              GtkWidget *view,
                              const gboolean library)
{
  dt_lib_masks_t *d = self->data;

  GtkTreeViewColumn *col = gtk_tree_view_column_new();
  gtk_tree_view_column_set_title(col, "shapes");
  gtk_tree_view_append_column(GTK_TREE_VIEW(view), col);
  GtkCellRenderer *renderer;

  if(!library)
  {
    // the application rank, first thing on the row: read the column top-down and
    // you read the order the shapes are applied in. right-aligned and two
    // characters wide so a group reaching ten shapes does not shift every name
    // sideways. small and insensitive: the theme greys it for us -- dark theme
    // and light theme alike -- and no colour is hardcoded.
    // a constant empty gutter reads as a margin; a column that moves reads as a
    // bug -- which is why the library, where no row is ever ranked, carries none
    renderer = gtk_cell_renderer_text_new();
    g_object_set(renderer,
                 "xalign", 1.0,
                 "xpad", (guint)DT_PIXEL_APPLY_DPI(2),
                 "width-chars", 2,
                 "scale", PANGO_SCALE_SMALL,
                 "sensitive", FALSE,
                 NULL);
    gtk_tree_view_column_pack_start(col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(col, renderer, "text", TREE_NUM);

    renderer = gtk_cell_renderer_pixbuf_new();
    gtk_tree_view_column_pack_start(col, renderer, FALSE);
    gtk_tree_view_column_set_attributes(col, renderer, "pixbuf", TREE_IC_OP, NULL);
    gtk_tree_view_column_add_attribute(col, renderer, "visible", TREE_IC_OP_VISIBLE);

    // "base" sits where the operator glyph would be, on the one row that has no
    // operator and cannot be given one. a word rather than a glyph: it has to
    // translate, and it has to follow a theme change -- which the icons,
    // rasterised once above, do not
    renderer = gtk_cell_renderer_text_new();
    g_object_set(renderer,
                 "xalign", 0.0,
                 "xpad", (guint)DT_PIXEL_APPLY_DPI(1),
                 "scale", PANGO_SCALE_SMALL,
                 "sensitive", FALSE,
                 NULL);
    gtk_tree_view_column_pack_start(col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(col, renderer, "text", TREE_BASE);

    renderer = gtk_cell_renderer_pixbuf_new();
    gtk_tree_view_column_pack_start(col, renderer, FALSE);
    gtk_tree_view_column_set_attributes(col, renderer, "pixbuf", TREE_IC_INVERSE, NULL);
    gtk_tree_view_column_add_attribute(col, renderer, "visible", TREE_IC_INVERSE_VISIBLE);
  }

  renderer = gtk_cell_renderer_text_new();
  g_object_set(renderer, "ellipsize", PANGO_ELLIPSIZE_MIDDLE, NULL);
  gtk_tree_view_column_pack_start(col, renderer, TRUE);
  gtk_tree_view_column_add_attribute(col, renderer, "text", TREE_TEXT);
  gtk_tree_view_column_add_attribute(col, renderer, "editable", TREE_EDITABLE);
  g_signal_connect(renderer, "edited", G_CALLBACK(_tree_cell_edited), view);
  dt_gui_commit_on_focus_loss(renderer, NULL);
  if(library)
  {
    d->lib_col = col;
    d->lib_name_cell = renderer;
  }

  // packed from the right edge inwards: the badge first, so it keeps the very
  // place it has today
  renderer = gtk_cell_renderer_pixbuf_new();
  gtk_tree_view_column_pack_end(col, renderer, FALSE);
  gtk_tree_view_column_set_attributes(col, renderer, "pixbuf", TREE_IC_USED, NULL);
  gtk_tree_view_column_add_attribute(col, renderer, "visible", TREE_IC_USED_VISIBLE);

  if(library)
  {
    // and just left of it, what the badge cannot say: no module renders this
    // shape. same treatment as the rank column -- small, insensitive, greyed by
    // the theme, no colour in the C. no ellipsizing: the name cell expands and
    // gives way, a truncated statement would read as a different statement
    renderer = gtk_cell_renderer_text_new();
    g_object_set(renderer,
                 "xalign", 1.0,
                 "xpad", (guint)DT_PIXEL_APPLY_DPI(2),
                 "scale", PANGO_SCALE_SMALL,
                 "sensitive", FALSE,
                 NULL);
    gtk_tree_view_column_pack_end(col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(col, renderer, "text", TREE_LINK);

    // the library is flat by construction: no expander gutter to indent it
    gtk_tree_view_set_show_expanders(GTK_TREE_VIEW(view), FALSE);
  }

  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
  gtk_tree_selection_set_mode(selection, GTK_SELECTION_MULTIPLE);
  gtk_tree_selection_set_select_function(selection, _tree_restrict_select, d, NULL);
  gtk_tree_view_set_headers_visible(GTK_TREE_VIEW(view), FALSE);
  gtk_widget_set_has_tooltip(view, TRUE);
  g_signal_connect(view, "query-tooltip", G_CALLBACK(_tree_query_tooltip), NULL);
  g_signal_connect(selection, "changed", G_CALLBACK(_tree_selection_change), d);
  dt_gui_connect_click_all(view, _tree_button_pressed_cb, NULL, self);
}

void gui_init(dt_lib_module_t *self)
{
  /* initialize ui widgets */
  dt_lib_masks_t *d = g_malloc0(sizeof(dt_lib_masks_t));
  self->data = (void *)d;

  // initialise all masks pixbuf. This is needed for the "automatic"
  // cell renderer of the treeview
  const int bs2 = DT_PIXEL_APPLY_DPI(13);
  d->ic_inverse = _get_pixbuf_from_cairo(dtgtk_cairo_paint_masks_inverse, bs2, bs2);
  d->ic_used = _get_pixbuf_from_cairo(dtgtk_cairo_paint_masks_used, bs2, bs2);
  d->ic_union = _get_pixbuf_from_cairo(dtgtk_cairo_paint_masks_union, bs2 * 2, bs2);
  d->ic_intersection =
    _get_pixbuf_from_cairo(dtgtk_cairo_paint_masks_intersection, bs2 * 2, bs2);
  d->ic_difference =
    _get_pixbuf_from_cairo(dtgtk_cairo_paint_masks_difference, bs2 * 2, bs2);
  d->ic_sum =
    _get_pixbuf_from_cairo(dtgtk_cairo_paint_masks_sum, bs2 * 2, bs2);
  d->ic_exclusion =
    _get_pixbuf_from_cairo(dtgtk_cairo_paint_masks_exclusion, bs2 * 2, bs2);

  // initialise widgets
  d->bt_gradient = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_gradient, 0, NULL);
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("add gradient"),
                   d->bt_gradient, &dt_action_def_toggle);
  g_object_set_data(G_OBJECT(d->bt_gradient), DT_ACTION_GESTURE_KEY,
                    dt_gui_connect_click(d->bt_gradient, _bt_add_shape_cb, NULL,
                                         GINT_TO_POINTER(DT_MASKS_GRADIENT)));
  gtk_widget_set_tooltip_text(d->bt_gradient, _("add gradient"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->bt_gradient), FALSE);

  d->bt_path = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_path, 0, NULL);
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("add path"),
                   d->bt_path, &dt_action_def_toggle);
  g_object_set_data(G_OBJECT(d->bt_path), DT_ACTION_GESTURE_KEY,
                    dt_gui_connect_click(d->bt_path, _bt_add_shape_cb, NULL,
                                         GINT_TO_POINTER(DT_MASKS_PATH)));
  gtk_widget_set_tooltip_text(d->bt_path, _("add path"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->bt_path), FALSE);

  d->bt_ellipse = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_ellipse, 0, NULL);
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("add ellipse"),
                   d->bt_ellipse, &dt_action_def_toggle);
  g_object_set_data(G_OBJECT(d->bt_ellipse), DT_ACTION_GESTURE_KEY,
                    dt_gui_connect_click(d->bt_ellipse, _bt_add_shape_cb, NULL,
                                         GINT_TO_POINTER(DT_MASKS_ELLIPSE)));
  gtk_widget_set_tooltip_text(d->bt_ellipse, _("add ellipse"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->bt_ellipse), FALSE);

  d->bt_circle = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_circle, 0, NULL);
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("add circle"),
                   d->bt_circle, &dt_action_def_toggle);
  g_object_set_data(G_OBJECT(d->bt_circle), DT_ACTION_GESTURE_KEY,
                    dt_gui_connect_click(d->bt_circle, _bt_add_shape_cb, NULL,
                                         GINT_TO_POINTER(DT_MASKS_CIRCLE)));
  gtk_widget_set_tooltip_text(d->bt_circle, _("add circle"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->bt_circle), FALSE);

  d->bt_brush = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_brush, 0, NULL);
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("add brush"),
                   d->bt_brush, &dt_action_def_toggle);
  g_object_set_data(G_OBJECT(d->bt_brush), DT_ACTION_GESTURE_KEY,
                    dt_gui_connect_click(d->bt_brush, _bt_add_shape_cb, NULL,
                                         GINT_TO_POINTER(DT_MASKS_BRUSH)));
  gtk_widget_set_tooltip_text(d->bt_brush, _("add brush"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->bt_brush), FALSE);

#ifdef HAVE_AI
  d->bt_object = dtgtk_togglebutton_new(dtgtk_cairo_paint_masks_object, 0, NULL);
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("add object"),
                   d->bt_object, &dt_action_def_toggle);
  g_object_set_data(G_OBJECT(d->bt_object), DT_ACTION_GESTURE_KEY,
                    dt_gui_connect_click(d->bt_object, _bt_add_shape_cb, NULL,
                                         GINT_TO_POINTER(DT_MASKS_OBJECT)));
  gtk_widget_set_tooltip_text(d->bt_object, _("add AI object"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->bt_object), FALSE);
#endif

  d->treeview = gtk_tree_view_new();
  _build_masks_view(self, d->treeview, FALSE);

  d->library = gtk_tree_view_new();
  // the ground is what separates the two zones, and the name is the whole of it:
  // the shade is derived from the theme's own background in
  // data/themes/darktable.css, next to the delete-dialog precedent, so light and
  // dark both follow and no colour is written here
  gtk_widget_set_name(d->library, "masks-library");
  _build_masks_view(self, d->library, TRUE);
  d->active_view = d->treeview;

  // the row used to open on a decorative "created shapes" label; it becomes the
  // explicit entry point instead. same row, same height, one more thing that
  // can be clicked -- and the label was redundant with the tree right below it,
  // which is what shows the shapes that exist. the icons stay to its right:
  // they carry the "shapes/add *" action paths and user shortcuts are persisted
  // by path, so removing them would silently invalidate existing bindings.
  // dt_action_button_new already sets hexpand, dt_gui_expand was redundant
  GtkWidget *bt_new = dt_action_button_new
    (self, N_("new mask"), _new_mask_clicked, self,
     _("pick a shape and the module it belongs to\n"
       "right-click to cancel while drawing"), 0, 0);

  GtkWidget *shape_buttons = dt_gui_hbox
    (bt_new,
     d->bt_brush, d->bt_circle, d->bt_ellipse, d->bt_path, d->bt_gradient);
#ifdef HAVE_AI
  dt_gui_box_add(shape_buttons, d->bt_object);
#endif

  d->lib_unlinked = dt_ui_label_new("");
  // dt_ui_label_new sets ellipsize END, and gtk_label_ensure_layout picks
  // ellipsize over wrap: the caption would be cut instead of wrapped, and it is
  // the one place "not linked to a module" is spelled out. turn ellipsize off
  // so the wrap below is the one that applies -- the left panel is narrow and
  // this sentence has to survive it whole
  gtk_label_set_ellipsize(GTK_LABEL(d->lib_unlinked), PANGO_ELLIPSIZE_NONE);
  gtk_label_set_line_wrap(GTK_LABEL(d->lib_unlinked), TRUE);
  // the one place the cleanup's own definition is stated, and the only place a
  // promise about it is hedged: it keeps whatever a history step still refers
  // to, which this classification does not measure
  gtk_widget_set_tooltip_text
    (d->lib_unlinked,
     _("no module uses them\n"
       "\"delete unused shapes\" removes those no history step refers to either"));

  // the creation bar. it belongs to the panel and not to either resize wrapper:
  // a row added inside one of them would change what
  // "plugins/darkroom/masks/heightview" measures, and the list would come back
  // one row shorter at the next start
  d->creation_label = dt_ui_label_new("");
  GtkWidget *cancel = dtgtk_button_new(dtgtk_cairo_paint_cancel, 0, NULL);
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("cancel creation"),
                   cancel, &dt_action_def_button);
  gtk_widget_set_tooltip_text(cancel, _("do not create the shape"));
  g_signal_connect(G_OBJECT(cancel), "clicked",
                   G_CALLBACK(_creation_bar_cancel), d);
  d->creation_bar = dt_gui_hbox(dt_gui_expand(d->creation_label), cancel);

  // the masks on top, the shapes they are drawn from below. a row in the top
  // list belongs to a mask -- deleting it detaches it. a row in the library IS
  // the shape. that is the whole point of the split.
  // both lists stay in place even when empty: a right-click on blank space is
  // how "add brush/circle/..." and "delete unused shapes" are reached, and that
  // path must not disappear with the last row
  self->widget = dt_gui_vbox
    (shape_buttons,
     dt_ui_resize_wrap(d->treeview, 200, "plugins/darkroom/masks/heightview"),
     dt_ui_section_label_new(C_("section", "shape library")),
     dt_ui_resize_wrap(d->library, 120, "plugins/darkroom/masks/heightlibrary"),
     d->lib_unlinked,
     d->creation_bar);

  // gui_update decides whether the caption is up. show_all first, then
  // no_show_all and an explicit hide, or a later panel show_all brings it back
  // -- same pattern as the shrink/grow slider further down
  gtk_widget_show_all(d->lib_unlinked);
  gtk_widget_set_no_show_all(d->lib_unlinked, TRUE);
  gtk_widget_hide(d->lib_unlinked);

  // same for the bar: show_all reaches the label and the cross once, then
  // no_show_all keeps a later panel-wide show_all from putting the row back
  // up. hidden, a box child takes no height at all, which is the whole point
  gtk_widget_show_all(d->creation_bar);
  gtk_widget_set_no_show_all(d->creation_bar, TRUE);
  gtk_widget_hide(d->creation_bar);

  dt_gui_new_collapsible_section
    (&d->cs,
     "plugins/darkroom/masks/expand_properties",
     _("properties"),
     GTK_BOX(self->widget),
     DT_ACTION(self));
  d->none_label = dt_ui_label_new(_("no shapes selected"));
  dt_gui_box_add(d->cs.container, d->none_label);
  gtk_widget_show_all(GTK_WIDGET(d->cs.container));
  gtk_widget_set_no_show_all(GTK_WIDGET(d->cs.container), TRUE);

  for(int i = 0; i < DT_MASKS_PROPERTY_LAST; i++)
  {
    GtkWidget *w;
    if(_masks_properties[i].boolean)
    {
      w = gtk_check_button_new_with_label(_(_masks_properties[i].name));
      dt_action_define(DT_ACTION(self), N_("properties"),
                       _masks_properties[i].name, w, &dt_action_def_toggle);
      d->last_value[i] = (float)gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(w));
      g_signal_connect(G_OBJECT(w), "toggled",
                       G_CALLBACK(_property_changed), GINT_TO_POINTER(i));
    }
    else
    {
      w = dt_bauhaus_slider_new_action(DT_ACTION(self),
                                       _masks_properties[i].min,
                                       _masks_properties[i].max,
                                       0, 0.0, 2);
      dt_bauhaus_widget_set_label(w, N_("properties"),
                                  _masks_properties[i].name);
      dt_bauhaus_slider_set_format(w, _masks_properties[i].format);
      dt_bauhaus_slider_set_digits(w, 2);
      if(_masks_properties[i].relative)
        dt_bauhaus_slider_set_log_curve(w);
      d->last_value[i] = dt_bauhaus_slider_get(w);
      g_signal_connect(G_OBJECT(w), "value-changed",
                       G_CALLBACK(_property_changed), GINT_TO_POINTER(i));
    }
    d->property[i] = w;
    dt_gui_box_add(d->cs.container, w);
  }

  d->pressure = dt_gui_preferences_enum(DT_ACTION(self), "pressure_sensitivity");
  dt_bauhaus_widget_set_label(d->pressure, N_("properties"), N_("pressure"));
  d->smoothing = dt_gui_preferences_enum(DT_ACTION(self), "brush_smoothing");
  dt_bauhaus_widget_set_label(d->smoothing, N_("properties"), N_("smoothing"));
  dt_gui_box_add(d->cs.container, d->pressure, d->smoothing);

  // path-only shrink/grow (outset/inset) control: a single signed slider whose
  // quad toggles the unit (px / %). Unlike "size" (a live centroid scaling),
  // this rasterizes and re-vectorizes the outline. The value is a signed offset
  // measured from the shape's baseline; 0 restores it. It commits (debounced) on
  // change. The soft range is ±20 but the hard range is wide so an explicit value
  // can be typed in.
  d->resize_amount = dt_bauhaus_slider_new_action(DT_ACTION(self), -1000, 1000, 1, 0.0, 0);
  dt_bauhaus_widget_set_label(d->resize_amount, N_("properties"), N_("shrink or grow"));
  dt_bauhaus_slider_set_soft_range(d->resize_amount, -20, 20);
  dt_bauhaus_slider_set_format(d->resize_amount, "");
  gtk_widget_set_tooltip_text(d->resize_amount,
                              _("grow (positive) or shrink (negative) the selected path,\n"
                                "relative to its shape when selected; 0 restores the original"));
  g_signal_connect(
    G_OBJECT(d->resize_amount), "value-changed", G_CALLBACK(_resize_amount_changed), d);

  // unit (px / %) toggle in the slider's quad
  dt_bauhaus_widget_set_quad_paint(d->resize_amount, _paint_resize_unit, 0, NULL);
  dt_bauhaus_widget_set_quad_toggle(d->resize_amount, TRUE);
  {
    const char *unit = dt_conf_get_string_const("masks/path_resize_unit");
    dt_bauhaus_widget_set_quad_active(d->resize_amount, !g_strcmp0(unit, "% of path size"));
  }
  dt_bauhaus_widget_set_quad_tooltip(
    d->resize_amount, _("shrink/grow unit: image pixels (px) or % of path size — click to toggle"));
  g_signal_connect(G_OBJECT(d->resize_amount), "quad-pressed", G_CALLBACK(_resize_unit_quad), d);
  // initial value suffix reflects the stored unit
  _resize_sync_unit(d);

  d->resize_box = d->resize_amount;
  dt_gui_box_add(d->cs.container, d->resize_amount);
  gtk_widget_show_all(d->resize_amount);
  gtk_widget_set_no_show_all(d->resize_amount, TRUE);

  // "size" scales the shape live; "shrink or grow" insets/outsets its outline.
  // They are the two resize controls, so keep the grow/shrink slider right below
  // "size" instead of at the end of the property list.
  {
    GList *kids = gtk_container_get_children(GTK_CONTAINER(d->cs.container));
    const gint size_pos = g_list_index(kids, d->property[DT_MASKS_PROPERTY_SIZE]);
    if(size_pos >= 0)
      gtk_box_reorder_child(d->cs.container, d->resize_amount, size_pos + 1);
    g_list_free(kids);
  }

  // set proxy functions
  darktable.develop->proxy.masks.module = self;
  darktable.develop->proxy.masks.list_change = _lib_masks_recreate_list;
  darktable.develop->proxy.masks.list_update = _lib_masks_update_list;
  darktable.develop->proxy.masks.list_remove = _lib_masks_remove_item;
  darktable.develop->proxy.masks.selection_change = _lib_masks_selection_change;
}

void gui_cleanup(dt_lib_module_t *self)
{
  dt_lib_masks_t *d = self->data;
  if(d && d->resize_timer)
    g_source_remove(d->resize_timer);
  g_free(self->data);
  self->data = NULL;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
