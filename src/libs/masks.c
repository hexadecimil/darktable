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

// the five composition operators, in the order they have always been listed
// in. ONE table: the context menu entries, the glyph a row shows, the pixbuf
// rasterised at start-up and the operator a click sets all read it -- four
// parallel lists before this, and four places to forget.
// the labels are the menu's own strings, unchanged: building the entries from
// a shorter word and a "mode: " prefix would be a new msgid, and the context
// menu would fall back to English in every locale until the translators catch
// up.
// the order is also the order the state bits used to be tested in, so a point
// that somehow carries two of them resolves to the same operator it always did
static const struct
{
  dt_masks_state_t state;
  const char *label;              // the context menu entry
  const char *name;               // the operator alone, for the cell tooltip
  DTGTKCairoPaintIconFunc paint;  // the glyph drawn on the row
} _masks_operators[] =
{
  { DT_MASKS_STATE_UNION,        N_("mode: union"),        N_("union"),
    dtgtk_cairo_paint_masks_union },
  { DT_MASKS_STATE_INTERSECTION, N_("mode: intersection"), N_("intersection"),
    dtgtk_cairo_paint_masks_intersection },
  { DT_MASKS_STATE_DIFFERENCE,   N_("mode: difference"),   N_("difference"),
    dtgtk_cairo_paint_masks_difference },
  { DT_MASKS_STATE_SUM,          N_("mode: sum"),          N_("sum"),
    dtgtk_cairo_paint_masks_sum },
  { DT_MASKS_STATE_EXCLUSION,    N_("mode: exclusion"),    N_("exclusion"),
    dtgtk_cairo_paint_masks_exclusion },
};

// which operator a point carries, as an index into the table above, -1 for
// none. index 0 of a group legitimately has none -- group.c reads a missing
// operator as "overwrite everything applied so far"; anywhere else it is
// stale data no current code path creates.
// >= 0 is exactly `state & DT_MASKS_STATE_OP`, that constant being the OR of
// the five states listed above and nothing else -- but it is the table, and
// not the constant, that the rest of this file may then index with
static int _op_index(const dt_masks_state_t state)
{
  for(int i = 0; i < (int)G_N_ELEMENTS(_masks_operators); i++)
    if(state & _masks_operators[i].state) return i;
  return -1;
}

// the shape types, in one table read in two orders. the table itself is in the
// order the row at the top of the panel draws them -- which is also the order
// every module's blending panel draws them in: blend_gui.c packs its six with
// gtk_box_pack_end, so its row reads brush, circle, ellipse, path, gradient,
// object from the left. two rows in the same column of the screen reading the
// same way round is the whole of what makes that duplication bearable.
// `menu_rank` is the other order: a row is scanned at a glance and its order
// hardly matters, a menu is read from the top and the entry reached for most
// should not be the last one, so the catalogue still opens on the circle and
// ends on the brush.
// `label` is what a menu entry and a button tooltip say and `ctrl` what the
// tooltip adds about ctrl+click -- the strings the context menu and every
// blending panel already used, no new one for translators. `action` is the id
// the shortcut path is built from and it is frozen: those paths outlived this
// row once already, and "add object" is what it registered, tooltip and menu
// saying "add AI object" or not. DT_MASKS_OBJECT is the one type that can be
// unavailable at runtime; the refusal is _start_creation's, in one place
static const struct
{
  dt_masks_type_t type;
  const char *label;   // the menu entry and the button tooltip
  const char *action;  // the action id -- never rename, never retranslate
  const char *ctrl;    // what ctrl+click does, NULL if it does nothing else
  DTGTKCairoPaintIconFunc paint;  // the glyph on the button
  int menu_rank;       // where this type sits in a menu, see above
} _new_mask_shapes[] =
{
  { DT_MASKS_BRUSH,    N_("add brush"),     N_("add brush"),
    N_("add multiple brush strokes"),
    dtgtk_cairo_paint_masks_brush,    4 },
  { DT_MASKS_CIRCLE,   N_("add circle"),    N_("add circle"),
    N_("add multiple circles"),
    dtgtk_cairo_paint_masks_circle,   0 },
  { DT_MASKS_ELLIPSE,  N_("add ellipse"),   N_("add ellipse"),
    N_("add multiple ellipses"),
    dtgtk_cairo_paint_masks_ellipse,  1 },
  { DT_MASKS_PATH,     N_("add path"),      N_("add path"),
    N_("add multiple paths"),
    dtgtk_cairo_paint_masks_path,     3 },
  { DT_MASKS_GRADIENT, N_("add gradient"),  N_("add gradient"),
    N_("add multiple gradients"),
    dtgtk_cairo_paint_masks_gradient, 2 },
#ifdef HAVE_AI
  // no ctrl line: a blending panel does not offer one either, an object being
  // picked out of the image and not drawn
  { DT_MASKS_OBJECT,   N_("add AI object"), N_("add object"),
    NULL,
    dtgtk_cairo_paint_masks_object,   5 },
#endif
};

// which of the six kinds a shape is, as an index into the table above, -1 for
// none -- the same contract as _op_index() and the same reason for a table
// rather than a switch. form->type is a bitmask that also carries
// DT_MASKS_GROUP and the clone bits, so it is tested against the table and
// never compared to it. -1 is a real answer: a group is not a kind of shape,
// and DT_MASKS_OBJECT leaves the table entirely in a build without the AI,
// where an object shape read back from an XMP written by one still has to draw
static int _type_index(const dt_masks_type_t type)
{
  for(int i = 0; i < (int)G_N_ELEMENTS(_new_mask_shapes); i++)
    if(type & _new_mask_shapes[i].type) return i;
  return -1;
}

typedef struct dt_lib_masks_t
{
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
  // the masks zone's operator column, and the whole of the click target: a
  // click is "on the operator" when GTK hands back THIS column, never when a
  // pixel offset falls inside a range. one field for both lists, and it only
  // ever names a column of the masks zone: the library builds none, so a
  // hit-test run there hands back one of ITS columns and cannot match -- which
  // is what makes one pointer comparison enough to tell the two views apart.
  // written under if(!library) in _build_masks_view: written unconditionally
  // the second call would leave the library's column here, and the gesture
  // would be dead in one list and live in the other
  GtkTreeViewColumn *op_col;
  // caption under the library, shown only when at least one shape is not
  // linked to a module: it names exactly the set the cleanup is about
  GtkWidget *lib_unlinked;
  // what the empty state swaps out. the two boxes and not the two views:
  // dt_ui_resize_wrap hands back the event box wrapping the scrolled window,
  // and hiding a view alone would leave that window's min_content_height
  // holding the space open under the message
  GtkWidget *masks_box, *lib_box;
  GtkWidget *lib_label;    // the "shape library" heading, above its list
  // the empty-state sentences, shown only while both lists are empty. two
  // widgets and not one box because M1 reads sentence, button, sentence and
  // the button between them is the one at the top of the panel
  GtkWidget *empty_title, *empty_hint;
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

  GdkPixbuf *ic_inverse, *ic_used;
  // the operator glyphs, indexed by _masks_operators: same table, same order,
  // so a sixth operator is one line there and nothing at all here
  GdkPixbuf *ic_op[G_N_ELEMENTS(_masks_operators)];
  // the kind glyphs, indexed by _new_mask_shapes: same table, same order, so a
  // seventh kind is one line there and nothing at all here
  GdkPixbuf *ic_type[G_N_ELEMENTS(_new_mask_shapes)];

  // a selection requested (e.g. right after creating a shape) before the tree
  // had the matching row: re-applied once gui_update rebuilds the tree. 0 = none.
  dt_mask_id_t pending_selectid;
  struct dt_iop_module_t *pending_selmodule;

  // structure hash of the tree as last built; when unchanged (e.g. a slider edit
  // that only alters shape parameters) gui_update refreshes rows in place instead
  // of recreating the store, so the panel scroll never moves. See _forms_structure_hash.
  guint tree_hash;
  gboolean tree_hash_valid;

  // the top row of M2, in two rows and not one: six buttons and a module name
  // side by side ask for ~310 px, twice the panel's configured floor
  // (min_panel_width, 150); stacked they ask for ~170 px, which is what the
  // panel already asks for today. bt_new is the named button of M1: an empty
  // panel is a screen that teaches and needs words, a populated one is a
  // screen that works and needs glyphs. it is hidden, never destroyed --
  // dt_action_button_new registered the action ON it
  GtkWidget *bt_new;
  GtkWidget *target_row, *target_label, *bt_target;
  GtkWidget *shape_row;
  GtkWidget *bt_shape[G_N_ELEMENTS(_new_mask_shapes)];

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
static void _target_row_update(dt_lib_masks_t *d);
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

// M1: an image that carries no mask at all gets a sentence instead of two
// empty lists. read back from the models rather than from dev->forms so it
// agrees with the rows on screen -- _lib_masks_list_recurs drops clone and
// retouch shapes, and those must not keep the panel looking occupied
static void _empty_state_update(dt_lib_masks_t *lm)
{
  // cs.expander is the last widget gui_init builds: no expander, no panel yet
  if(!lm->cs.expander) return;

  gboolean empty = TRUE;
  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
  {
    GtkWidget *view = _masks_view(lm, v);
    GtkTreeModel *model =
      view ? gtk_tree_view_get_model(GTK_TREE_VIEW(view)) : NULL;
    if(model && gtk_tree_model_iter_n_children(model, NULL) > 0) empty = FALSE;
  }

  gtk_widget_set_visible(lm->empty_title, empty);
  gtk_widget_set_visible(lm->empty_hint, empty);
  // M1 shows one named button between the two sentences; M2 shows the two rows
  // instead. six mute glyphs say nothing to someone who has never drawn a
  // mask, and a named button is a line of prose in a screen that works
  gtk_widget_set_visible(lm->bt_new, empty);
  gtk_widget_set_visible(lm->target_row, !empty);
  gtk_widget_set_visible(lm->shape_row, !empty);
  gtk_widget_set_visible(lm->masks_box, !empty);
  gtk_widget_set_visible(lm->lib_label, !empty);
  gtk_widget_set_visible(lm->lib_box, !empty);
  // "properties" goes too. it is the only other thing packed into the panel,
  // M1 is a screen with one action on it, and with nothing to select the
  // section can say nothing but "no shapes selected"
  gtk_widget_set_visible(lm->cs.expander, !empty);
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
  // what kind of shape the row is -- circle, ellipse, gradient, path, brush,
  // object -- as a pixbuf out of ic_type[]. derived like the three above and
  // written in the same single place. a mask holding exactly ONE shape carries
  // that shape's icon: it is the only case where a group has a kind at all,
  // and the only one a mask row can state without lying
  TREE_IC_TYPE,
  TREE_IC_TYPE_VISIBLE,
  // the opacity this shape has IN THIS MASK, as "85%", and "" at 100%: the
  // column states what is not the default and stays quiet about what is.
  // opacity belongs to the membership and not to the shape -- the same
  // gradient is 85% in one mask and 100% in the next -- so it is written on
  // group members and nowhere else. derived, written by _set_iter_name only
  TREE_OPACITY,
  // the module this mask serves, as "-> exposure", on the mask rows and
  // nowhere else: a shape inside a mask does not belong to a module, the mask
  // does. derived, written by _set_iter_name only
  TREE_TARGET,
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
      [TREE_IC_TYPE] = GDK_TYPE_PIXBUF,
      [TREE_IC_TYPE_VISIBLE] = G_TYPE_BOOLEAN,
      [TREE_OPACITY] = G_TYPE_STRING,
      [TREE_TARGET] = G_TYPE_STRING,
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

  // ... and the line above the shape buttons, which says the same thing
  // before anything is armed at all
  _target_row_update(self);
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

/* -------------------------------------------------------------------------
   "new mask": one entry point that names the module the shape is for.

   until now a shape created from this panel took its target from whatever row
   happened to be selected, silently, and got NO target at all when nothing was
   selected -- that is where an orphan shape comes from on this side. that rule
   is kept and extended until it always answers; what changes is that the
   answer is written down before anything is drawn, next to the way to change
   it, and that a creation with no target is refused instead of performed.
   ------------------------------------------------------------------------- */

// a module whose mask gui is actually there. blend_data is only allocated for
// a module whose blending gui was built, and masks_support already carries
// !IOP_FLAGS_NO_MASKS, so this says what the SUPPORTS_BLENDING && !NO_MASKS
// pair says elsewhere in this file plus the one thing that matters here:
// blend_data exists. the
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
    // with a context of its own. the bare "off" msgid is a combobox value
    // everywhere else in darktable, and the translations written for that are
    // capitalised and inflected: "exposure (2 shapes, Desactive(e))" is what
    // this line reads like in french once it borrows one
    note = C_("module", "off");

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
//      is sorted, so the last match is the latest one);
//   4. exposure, switched on or not. the first three rules are all empty on an
//      image that was just opened -- dt_dev_change_image() drops the focus --
//      and a rule that answers nothing there leaves the one button of this
//      panel opening on a menu with nothing to click. exposure is the module
//      every image has, the one darktable already singles out (the histogram
//      drags it through dev->proxy.exposure), and a drawn mask on it is what
//      dodging and burning is;
//   5. failing even that -- exposure carrying a raster mask -- the last module
//      of the pipe that is switched on and could take the shape. no rule falls
//      back to a module that is off and unnamed: on a jpeg not one switched-on
//      module supports blending, which is why rule 4 comes first and does not
//      ask whether exposure is on.
// nothing here is decided silently: the menu prints the answer and offers to
// change it before a shape is drawn. NULL only when no module of this pipe can
// take a drawn mask at all.
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
  // TREE_MODULE holds a raw pointer and the store outlives the pipe it was
  // built from by one refresh: dt_dev_masks_list_change() only queues the
  // rebuild, and this rule is read on the way there. checked against dev->iop
  // before it is dereferenced -- the alive test only compares pointers
  if(_mask_target_alive(module) && _mask_target_ok(module)) return module;

  module = darktable.develop->gui_module;
  if(_mask_target_ok(module)) return module;

  // rules 3 to 5 read the pipe once. `preferred` is the exposure instance
  // darktable itself works through, so a multi-instance edit answers the same
  // module here as it does everywhere else; without it the first instance in
  // pipe order does
  const dt_iop_module_t *preferred = darktable.develop->proxy.exposure.module;
  dt_iop_module_t *masked = NULL, *exposure = NULL, *switched_on = NULL;

  for(const GList *l = darktable.develop->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    if(!_mask_target_listed(m) || !_mask_target_ok(m)) continue;

    if(_mask_target_shapes(m) > 0) masked = m;
    if(m->enabled) switched_on = m;
    if(!g_strcmp0(m->op, "exposure") && (!exposure || m == preferred))
      exposure = m;
  }

  if(masked) return masked;
  return exposure ? exposure : switched_on;
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
    // the creation this panel armed is over, so the run it belonged to is too
    // -- whichever of the three ways above told us. it is ended here because
    // here is where we learn of it: a right-click on the image ends its own
    // run, but a focus change, a new image, or a shape button in a module's
    // own panel all just drop the form, and nothing generic ever resets the
    // run. left standing it would turn the next plain click anywhere into a
    // series, on a module pointer that may not exist any more
    if(fg && fg->creation_continuous_module == d->arm_module)
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

  // and the six toggles follow the very same answer: lit while this panel's
  // creation is in flight, and only the type actually being drawn. upstream
  // switched all six off by hand from two call sites (_lib_masks_inactivate_
  // icons); this is the one place that knows, so it is the only one that
  // touches them -- and it can also light the armed one, which upstream could
  // not, its buttons being left to flip themselves. no re-entry guard: a dtgtk
  // togglebutton wires "toggled" to gtk_widget_queue_draw and nothing else,
  // its click living in the gesture controller gui_init installs
  const dt_masks_form_t *drawn =
    d->arm_module ? darktable.develop->form_visible : NULL;
  for(int i = 0; i < (int)G_N_ELEMENTS(_new_mask_shapes); i++)
    gtk_toggle_button_set_active
      (GTK_TOGGLE_BUTTON(d->bt_shape[i]),
       drawn && (drawn->type & _new_mask_shapes[i].type));
}

// the target row: where the next shape goes, written down before anything is
// drawn. it reads the very rule the menu opens on, so the line and the menu
// cannot contradict each other.
//
// refreshed from _update_all_properties(), so from all four refresh paths of
// this panel -- which covers every change of focus that had a module in focus
// before: dt_iop_request_focus() calls dt_masks_reset_form_gui(), landing in
// dt_dev_masks_selection_change() with dev->gui_module already set to the new
// module. what it does not cover is a focus taken while nothing was focused,
// the first module opened after an image change: darktable raises no signal
// for a focus change and nothing else here can hear it. the line is then one
// rule behind until anything at all moves, and both the menu and the creation
// itself re-read the rule at the moment of the gesture
static void _target_row_update(dt_lib_masks_t *d)
{
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  const dt_iop_module_t *target = self ? _mask_default_target(self) : NULL;

  if(target)
  {
    // the module alone, without the "(2 shapes)" the menu adds to tell two
    // entries apart: there is one line here and nothing to tell it apart from.
    // the arrow carries no msgid -- it is a glyph and a module name
    gchar *name = dt_history_item_get_name(target);
    gchar *text = g_strdup_printf("→ %s", name);
    gtk_label_set_text(GTK_LABEL(d->target_label), text);
    g_free(text);

    // the panel is narrow and the module name is all this line carries, so it
    // is also the first thing the ellipsis takes: the tooltip gives it back
    // whole, exactly as the creation bar below does with its own sentence
    gchar *tip = g_strdup_printf(_("where the next shape goes: %s"), name);
    gtk_widget_set_tooltip_text(d->target_label, tip);
    g_free(tip);
    g_free(name);
  }
  else
  {
    gtk_label_set_text(GTK_LABEL(d->target_label),
                       _("no module can take a shape"));
    gtk_widget_set_tooltip_text(d->target_label,
                                _("where the next shape goes"));
  }
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

// the one creation path of this panel. the catalogue, the tree context menu
// and the shape shortcuts all land here, so none of them can produce a shape
// with no module behind it and all of them refuse with the same words.
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
    return FALSE;
  }
#endif

  // the module may have gone away between the click that opened a menu and the
  // click that picked a shape
  if(!_mask_target_alive(module) || !_mask_target_ok(module))
  {
    // a shape with no module is a shape nothing renders: it lands in the
    // library, in no pipe, and stays there until "delete unused shapes" is
    // found. do not create it -- and say what to do rather than what failed,
    // since both halves of the sentence are a way out: a selected mask answers
    // for its module, an open module answers for itself
    dt_control_log
      (_("select a mask or open a module to choose where the shape goes"));
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

  dt_control_queue_redraw_center();
  return TRUE;
}

// the press on a shape button is taken before the button itself sees it, and
// taken for good. a GtkToggleButton flips its own state on RELEASE, which
// would land after the press handler below and undo what
// _creation_bar_update() just decided; claiming the sequence in the capture
// phase stops that flip from ever happening. every module's blending panel
// wires the same six buttons the same way (dt_iop_togglebutton_new), and it is
// the contract _action_process_toggle() writes down for a shortcut replayed
// through DT_ACTION_GESTURE_KEY: the callback manages the button states itself
static void _bt_shape_claim_cb(GtkGesture *gesture,
                               GdkEventSequence *sequence,
                               gpointer user_data)
{
  gtk_gesture_set_sequence_state(gesture, sequence, GTK_EVENT_SEQUENCE_CLAIMED);
}

// the six toggles of the top row, restored. everything upstream wrote by hand
// here -- the AI guard, the creation, the ctrl run, then putting the icons
// back down -- is one call now: _start_creation() carries the guard, the
// target and the run, and _creation_bar_update() decides what the row shows.
//
// ctrl+click draws shape after shape. that is the fallback the six actions
// lost when they became commands, and it comes back with the toggles:
// dt_action_def_toggle maps ctrl to DT_ACTION_EFFECT_TOGGLE_CTRL, which
// _action_process_toggle replays through this very gesture -- hence the
// DT_ACTION_GESTURE_KEY set on each button in gui_init
static void _bt_add_shape_cb(GtkGestureSingle *gesture,
                             int n_press,
                             double x,
                             double y,
                             gpointer shape)
{
  if(dt_gui_current_button(gesture) != GDK_BUTTON_PRIMARY) return;

  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  if(!self) return;

  const gboolean continuous =
    dt_modifier_is(dt_gui_current_state(gesture), GDK_CONTROL_MASK);

  // taken or refused, the six buttons end up showing what is armed and
  // nothing else. only the refusal needs the call: _start_creation() returns
  // before touching the row on that path, and has already said why in the same
  // words every other entry point uses
  if(!_start_creation(self, _mask_default_target(self),
                      GPOINTER_TO_INT(shape), continuous))
    _creation_bar_update(self->data);
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

// one row per shape type, wired to the callback the context menu and the shape
// shortcuts already use. the module travels on the item, so the same function
// fills the top level of the catalogue, every per-module submenu, and the
// context menu
static void _new_mask_shape_items(GtkMenuShell *menu, dt_iop_module_t *target)
{
  // a shape list without a target is a list of dead ends, so there is no such
  // list: every caller resolves one first. what is left below is written per
  // entry because it really is per entry -- one model that is not installed
  if(!target) return;

  // by menu_rank and not by table order: the table is the row's order, and a
  // menu that opened on "add brush" because the row does would have moved the
  // entry reached for most to the bottom. the ranks are a permutation, so the
  // scan finds one entry per pass -- and the type left out of a build without
  // the AI is the one holding the last rank
  for(int rank = 0; rank < (int)G_N_ELEMENTS(_new_mask_shapes); rank++)
  {
    int i = 0;
    while(i < (int)G_N_ELEMENTS(_new_mask_shapes)
          && _new_mask_shapes[i].menu_rank != rank) i++;
    if(i == (int)G_N_ELEMENTS(_new_mask_shapes)) continue;

    const gchar *reason = NULL;

#ifdef HAVE_AI
    if(_new_mask_shapes[i].type == DT_MASKS_OBJECT && !dt_masks_object_available())
      reason = _("AI model not available");
#endif

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

// picking a module from the "..." of the target row: it says where the next
// shape goes, and stops there. taking the focus IS the answer, rule 2 of
// _mask_default_target() being dev->gui_module, so the line re-reads what this
// just set instead of being told twice.
//
// two calls and not one. dt_iop_request_focus() notifies nothing when the
// module asked for is the one already focused, and nothing either when nothing
// was focused before -- it only resets the mask gui on the way OUT of a
// module. and rule 1, the module of a selected mask row, outranks the focus:
// left alone it would keep answering for this line and the menu would look
// inert. the second call is the reset the first one makes when it does change
// focus, so making it unconditionally settles all three cases the same way,
// and leaves the canvas and the two lists agreeing as a focus change does
static void _target_set_cb(GtkMenuItem *item, gpointer module)
{
  // a menu outlives nothing, but the pipe under it can be rebuilt while it is
  // open -- same guard _start_creation() puts on the module it is handed
  if(!_mask_target_alive(module)) return;

  dt_iop_request_focus(module);
  dt_masks_change_form_gui(NULL);
}

// every module that could take the shape. `flat` picks which of the two menus
// this is: in the catalogue each entry carries the same list of types one
// level down, so picking there picks the target AND the shape in one gesture;
// under the "..." of the target row there is no such list, because the six
// shapes are the row 20 px underneath and a menu repeating them would be a
// second, slower copy of it. `current` is the module to leave out: it is the
// one already named on the line this list hangs from
static gboolean _new_mask_other_modules(GtkMenuShell *menu,
                                        const dt_iop_module_t *current,
                                        const gboolean flat)
{
  gboolean any = FALSE;
  int last_rank = -1;

  // three ranks: the modules that already carry a drawn mask -- where a second
  // shape usually goes -- then the ones this edit has switched on, then the
  // rest, which is most of them. inside a rank the walk is backwards, because
  // dev->iop is sorted by iop_order and the right panel is stacked from its
  // end (views/darkroom.c walks g_list_last() to g_list_previous()): read this
  // way the list is in the order the modules are on screen, and not upside
  // down from it
  for(int rank = 0; rank < 3; rank++)
  {
    for(const GList *l = g_list_last(darktable.develop->iop);
        l;
        l = g_list_previous(l))
    {
      dt_iop_module_t *m = l->data;
      if(m == current || !_mask_target_listed(m)) continue;

      const int r = (_mask_target_shapes(m) > 0) ? 0 : (m->enabled ? 1 : 2);
      if(r != rank) continue;

      if(last_rank >= 0 && r != last_rank)
        gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
      last_rank = r;

      gchar *label = _mask_target_label(m);
      GtkWidget *item = gtk_menu_item_new_with_label(label);
      g_free(label);

      if(!_mask_target_ok(m))
        // on a raster mask: _blendop_masks_modes_toggle() would refuse the
        // switch, so the entry stays and says so rather than disappearing
        gtk_widget_set_sensitive(item, FALSE);
      else if(flat)
        g_signal_connect(item, "activate", G_CALLBACK(_target_set_cb), m);
      else
      {
        GtkWidget *sub = gtk_menu_new();
        _new_mask_shape_items(GTK_MENU_SHELL(sub), m);
        gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), sub);
      }

      gtk_menu_shell_append(menu, item);
      any = TRUE;
    }
  }

  return any;
}

// the line the menu opens on: where the shape is about to land, and the way to
// send it elsewhere. the manager, unlike a blending panel, has no module of
// its own, so this is the only place that can say it -- and it says it before
// anything is drawn rather than after.
//
// "apply to" and not "target": everywhere else in darktable a target is a
// value to reach (target black luminance, target gamma, target color), never a
// destination. one line for both jobs and not two, because naming the module
// and changing it are the same question asked twice
static void _new_mask_target_header(GtkMenuShell *menu,
                                    const dt_iop_module_t *target)
{
  gchar *name = _mask_target_label(target);
  gchar *header = g_strdup_printf(_("apply to: %s"), name);
  g_free(name);

  GtkWidget *item = gtk_menu_item_new_with_label(header);
  g_free(header);

  GtkWidget *others = gtk_menu_new();
  if(_new_mask_other_modules(GTK_MENU_SHELL(others), target, FALSE))
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), others);
  else
  {
    // nothing else in this pipe is even worth listing, so the line states the
    // target and stops there. no item ever took this submenu and nothing ever
    // sank its floating reference: gtk_widget_destroy() only runs dispose,
    // which would leave the object alive at one reference, so sink it first,
    // as dt_gui_menu_popup does with the menu it is handed
    g_object_ref_sink(others);
    g_object_unref(others);
    gtk_widget_set_sensitive(item, FALSE);
  }

  gtk_menu_shell_append(menu, item);
  gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
}

// built at click, never in gui_update: _forms_structure_hash() mixes
// dev->gui_module in but gui_update only runs on dt_dev_masks_list_change, so a
// menu built there would show a stale target after a mere change of focus
static void _new_mask_clicked(GtkButton *button, dt_lib_module_t *self)
{
  dt_iop_module_t *target = _mask_default_target(self);

  // this is the button of the empty state, and a menu with nothing to click is
  // what it must never open on. _mask_default_target() answers for any pipe
  // holding a module that can take a drawn mask, so getting here means this
  // one holds none: say it once and open nothing, the way darkroom.c and
  // export.c refuse a style list with no style in it
  if(!target)
  {
    dt_control_log
      (_("select a mask or open a module to choose where the shape goes"));
    return;
  }

  GtkMenuShell *menu = GTK_MENU_SHELL(gtk_menu_new());

  // where it lands, then what to draw. the shapes are at the top level and
  // clickable the moment the menu opens -- on an image just opened as on any
  // other -- and the line above them is both the answer to "where?" and the
  // way to change it
  _new_mask_target_header(menu, target);
  _new_mask_shape_items(menu, target);

  // dt_gui_menu_popup takes the floating ref and drops it on "deactivate"
  dt_gui_menu_popup(GTK_MENU(menu), GTK_WIDGET(button),
                    GDK_GRAVITY_SOUTH_WEST, GDK_GRAVITY_NORTH_WEST);
}

// the "..." of the target row: send the next shape somewhere else. the modules
// and only the modules, one level, no shapes -- the row of six sits right
// under the line this hangs from, so an entry per shape here would be the same
// six targets reached the slow way. picking a module is the whole gesture, and
// the shape after it is one click on the row.
//
// built at click for the same reason the catalogue is: the answer moves with
// the focus and nothing signals a focus change to this panel
static void _target_menu_clicked(GtkButton *button, dt_lib_module_t *self)
{
  const dt_iop_module_t *target = _mask_default_target(self);

  GtkMenuShell *menu = GTK_MENU_SHELL(gtk_menu_new());
  if(!_new_mask_other_modules(menu, target, TRUE))
  {
    // one module in the whole pipe can hold a drawn mask and it is the one
    // already on the line: nothing to open. sink the floating reference first,
    // as dt_gui_menu_popup would have
    g_object_ref_sink(menu);
    g_object_unref(menu);
    dt_control_log(_("no other module can take a shape"));
    return;
  }

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

// the group member for `formid`, and where it sits in the application order.
// NULL when the group is unknown, is not a group, or does not hold that shape.
//
// index 0 is the base: it lays the buffer down, and it is the only member
// allowed to carry no operator -- group.c reads a missing operator as
// "overwrite everything applied so far" (final `else` of _group_get_mask_roi,
// which is also the branch that initialises a buffer nothing else zeroes).
//
// everything that used to be decided from a row's position on screen is
// decided here instead: the tree is a projection of this list, and since the
// rank commit that projection is no longer a mirror. one walker, so the rank
// printed on a row, the glyph drawn next to it and the cell that reacts to a
// click can never be computed three slightly different ways.
//
// const on the way out as well as on the way in: _tree_operation() is the one
// place a member's state changes, and it walks grp->points for itself, behind
// the check that keeps an operator off the base.
static const dt_masks_point_group_t *
_group_point_get(const dt_masks_form_t *grp,
                 const dt_mask_id_t formid,
                 int *index)
{
  if(index) *index = -1;
  if(!grp || !(grp->type & DT_MASKS_GROUP)) return NULL;

  int pos = 0;
  for(const GList *pts = grp->points; pts; pts = g_list_next(pts))
  {
    const dt_masks_point_group_t *pt = pts->data;
    // we stop at the first match, exactly like dt_masks_form_move() and
    // _tree_operation, so display and reordering always agree on which
    // occurrence they mean
    if(pt->formid == formid)
    {
      if(index) *index = pos;
      return pt;
    }
    pos++;
  }
  return NULL;
}

// where a shape sits in the application order of its group, -1 when the group
// or the shape is unknown. see _group_point_get above for the contract.
static int _group_point_index(const dt_masks_form_t *grp,
                              const dt_mask_id_t formid)
{
  int pos = -1;
  _group_point_get(grp, formid, &pos);
  return pos;
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
// sets it on every point it creates and _tree_group() on none of them. it has
// no reader left -- what a row draws is decided from the application rank now,
// which is the rule group.c actually enforces. the handover below is kept for
// the one thing in it that changes what a mask renders: the outgoing base gets
// a UNION if it had no operator at all, without which group.c would treat it
// as a fresh buffer and silently drop every shape applied before it.
//
// the bit itself is still handed over, on purpose: dt_masks_point_group_t goes
// into the XMP as a raw blob (common/exif.cc), so every stored edit carries it
// and a darktable predating this commit reads it back and still draws its
// operator glyph from it. keeping it in step costs two lines here and stops a
// round trip through such a version from putting a glyph on the base.
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
      // this shape is not at index 0 any more, and group.c reads a missing
      // operator as "overwrite everything applied so far": it must have one
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

  // TREE_TEXT is exactly form->name, on every row and with nothing appended:
  // _tree_cell_edited copies the displayed string straight back into
  // form->name, so whatever else is shown here is what a rename would write
  // into the name. that used to hold by coincidence only -- the "%" suffix was
  // built right here, and TREE_EDITABLE being (grp_id == 0) simply never
  // coincided with a row carrying one. the rank, the base marker, the kind and
  // now the opacity live in columns of their own
  char str[256] = "";
  g_strlcat(str, form->name, sizeof(str));

  // 100% is the default and gets no text: a column that states the ordinary
  // case on every row states nothing at all
  char opac[8] = "";
  if(opacity != 1.0f)
    snprintf(opac, sizeof(opac), "%d%%", (int)(opacity * 100));

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

  // TREE_MODULE holds a raw pointer and the store outlives the pipe it was
  // built from by one refresh: dt_dev_masks_list_change() only queues the
  // rebuild and an in-place refresh runs on the way there. the rank below has
  // only ever tested this pointer for NULL, which is why nothing needed the
  // check until a column read THROUGH it. same test, and the same reason, as
  // _mask_default_target
  dt_iop_module_t *live = _mask_target_alive(module) ? module : NULL;

  int rank = -1;
  char num[8] = "";
  const char *base = "";

  if(dt_is_valid_maskid(grid))
  {
    rank =
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

  // M2: "what is this group for?" answered on the row that raises the
  // question. the mask rows only -- which is a root row carrying a module:
  // grp_id is 0 there, and a library row carries no module at all.
  // dt_history_item_get_name and the same "→ %s" the target line at the top
  // of the panel is built from: one says where a mask lives, the other where
  // the next shape goes, and they have to read as the same statement
  gchar *target = NULL;

  if(live && !dt_is_valid_maskid(grid))
  {
    gchar *mname = dt_history_item_get_name(live);
    target = g_strdup_printf("→ %s", mname);
    g_free(mname);
  }

  // the glyph is drawn exactly where the shape HAS an operator: inside a
  // group, past the base, with an operator bit set. that is the very test
  // _tree_operation() enforces, so what is shown is what can be changed.
  // _op_cell_at_bin() is stricter by one condition, depth 2 -- because the
  // context menu is -- so a shape inside a group nested in a group shows its
  // operator and neither gesture offers to change it. it is shown all the
  // same: group.c honours that operator whatever the nesting, and a row that
  // hid it would be the one row lying about what it renders.
  // it used to be DT_MASKS_STATE_SHOW, which is not that predicate:
  // _tree_group() never sets it, so every group built with "group the forms"
  // drew no operator at all although group.c was honouring one
  const int op = (rank > 0) ? _op_index(state) : -1;
  GdkPixbuf *icop = (op >= 0) ? lm->ic_op[op] : NULL;

  GdkPixbuf *icinv = NULL;
  if(state & DT_MASKS_STATE_INVERSE)
    icinv = lm->ic_inverse;

  // M2.2: every shape says what it is. a group has no kind of its own -- but a
  // group holding exactly one shape has that shape's, and a mask made of a
  // single gradient IS a gradient. one is the only count at which the icon
  // cannot come to mean something else the moment a second shape is added.
  // read from grp->points and not from the row's children: on a first pass the
  // row is being built and has none yet
  const dt_masks_form_t *tform = form;

  if(form->type & DT_MASKS_GROUP)
    tform = (g_list_length(form->points) == 1)
      ? dt_masks_get_from_id(darktable.develop, _group_point_id(form, 0))
      : NULL;

  const int ty = tform ? _type_index(tform->type) : -1;
  GdkPixbuf *ictype = (ty >= 0) ? lm->ic_type[ty] : NULL;

  gtk_tree_store_set(GTK_TREE_STORE(model), iter,
                     TREE_TEXT, str,
                     TREE_OPACITY, opac,
                     TREE_NUM, num,
                     TREE_BASE, base,
                     TREE_LINK, link,
                     TREE_IC_OP, icop,
                     TREE_IC_OP_VISIBLE, (icop != NULL),
                     TREE_IC_INVERSE, icinv,
                     TREE_IC_INVERSE_VISIBLE, (icinv != NULL),
                     TREE_IC_TYPE, ictype,
                     TREE_IC_TYPE_VISIBLE, (ictype != NULL),
                     TREE_TARGET, target,
                     -1);

  g_free(target);
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

// the five operator entries, shared verbatim by the context menu and by the
// operator column of a row: same table, same order, same labels, same rule
// about the base. inversion is NOT one of them -- DT_MASKS_STATE_OP is the OR
// of these five and nothing else, and the inverse glyph is a different cell.
// _tree_operation() reads the view's selection itself, so these items take no
// row argument: the caller's only duty is to have selected the right row
// before the menu is shown.
static void _add_tree_operations(GtkMenuShell *menu,
                                 const dt_masks_state_t selected_states,
                                 const gboolean is_base_row)
{
  for(int i = 0; i < (int)G_N_ELEMENTS(_masks_operators); i++)
    _add_tree_operation(menu, _(_masks_operators[i].label),
                        _masks_operators[i].state, selected_states,
                        !is_base_row);
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
    if(ctx)
      _new_mask_shape_items(menu, ctx);
    else if((ctx = _mask_default_target(self)))
    {
      // same two lines as the button, in the same order. no target at all
      // leaves the shapes out, which is harmless here and would not be in the
      // button: this menu has items of its own below
      _new_mask_target_header(menu, ctx);
      _new_mask_shape_items(menu, ctx);
    }
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
    _add_tree_operations(menu, selected_states, is_base_row);

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

// what the operator column has to say at a point, if anything
typedef enum dt_masks_op_hit_t
{
  DT_MASKS_OP_HIT_NONE = 0,  // not the operator column, or nothing to say
  DT_MASKS_OP_HIT_BASE,      // the base: says why it has none, not clickable
  DT_MASKS_OP_HIT_OPERATOR   // an operator a click may change
} dt_masks_op_hit_t;

// what sits under bin-window coords (bx,by) in `view`. the whole hit-test is
// the column comparison below: GTK hands us the column under the pointer and
// we compare it with the one we built. no pixel offset is computed anywhere --
// gui/preferences_ai.c runs the same test on the same GTK version, and reads a
// model column back the same way to decide whether the cell is live.
// `view` is a parameter and not lm->treeview: both lists share this file's
// handlers, and hit-testing one list against the other's rows and scroll
// offset is the kind of thing that only misbehaves once there is content.
// op_col only ever names a column of the masks zone, so hit-testing the
// library hands back one of its own columns and never matches.
//
// a row qualifies only where the context menu offers the five "mode:" entries
// AND offers them enabled: depth 2 (the menu's `from_group && depth < 3`), not
// the shape at index 0, and carrying an operator bit -- which is exactly what
// _tree_operation() will accept. clicking anywhere else in the column stays a
// plain selection, because a menu whose every entry does nothing is worse than
// no menu at all.
// the depth-2 test is also what keeps us off the expander: _build_masks_view
// puts the expander on the name column for that reason, see the note there.
// the operator test is spelled _op_index() >= 0 and not the DT_MASKS_STATE_OP
// mask _tree_operation() uses. same predicate today, the constant being the OR
// of the five table entries -- but this way the caller of *state_out is holding
// a state the table is known to answer for, rather than one a sixth operator
// added to the enum alone would leave it unable to index.
static dt_masks_op_hit_t _op_cell_at_bin(dt_lib_masks_t *lm,
                                         GtkWidget *view,
                                         const gint bx,
                                         const gint by,
                                         dt_masks_state_t *state_out)
{
  if(!lm->op_col) return DT_MASKS_OP_HIT_NONE;

  GtkTreeView *tv = GTK_TREE_VIEW(view);
  GtkTreeModel *model = gtk_tree_view_get_model(tv);
  GtkTreePath *path = NULL;
  GtkTreeViewColumn *column = NULL;
  dt_masks_op_hit_t hit = DT_MASKS_OP_HIT_NONE;

  if(model
     && gtk_tree_view_get_path_at_pos(tv, bx, by, &path, &column, NULL, NULL)
     && column == lm->op_col
     && gtk_tree_path_get_depth(path) == 2)
  {
    GtkTreeIter iter;
    if(gtk_tree_model_get_iter(model, &iter, path))
    {
      dt_mask_id_t grid = INVALID_MASKID;
      dt_mask_id_t id = INVALID_MASKID;
      _lib_masks_get_values(model, &iter, NULL, &grid, &id);

      // read from grp->points, never from the row's position: the same walk,
      // the same function, as the rank and the "base" marker on screen. a row
      // whose group no longer holds its shape yields NULL and the click stays
      // a plain selection
      int rank = -1;
      const dt_masks_point_group_t *pt =
        _group_point_get(dt_masks_get_from_id(darktable.develop, grid),
                         id, &rank);

      if(rank == 0)
        hit = DT_MASKS_OP_HIT_BASE;
      else if(pt && rank > 0 && _op_index(pt->state) >= 0)
      {
        if(state_out) *state_out = pt->state;
        hit = DT_MASKS_OP_HIT_OPERATOR;
      }
    }
  }

  if(path) gtk_tree_path_free(path);
  return hit;
}

// exactly the zone the click reacts to, and back to NULL as soon as we leave
// it -- a cursor set on the bin window and never reset stays a hand over the
// whole panel. the base gets no hand: nothing there opens
static void _tree_motion_cb(GtkEventControllerMotion *controller,
                            double x,
                            double y,
                            dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  GtkWidget *view = dt_gui_get_widget(controller);
  GdkWindow *bin = gtk_tree_view_get_bin_window(GTK_TREE_VIEW(view));
  if(!bin) return;

  gint bx, by;
  gtk_tree_view_convert_widget_to_bin_window_coords(GTK_TREE_VIEW(view),
                                                    (gint)x, (gint)y, &bx, &by);

  // this runs on every motion event over the list, so touch the window only
  // when the answer changes. nothing else puts a cursor on it, so its current
  // cursor is a faithful record of what we last decided
  const gboolean over =
    _op_cell_at_bin(lm, view, bx, by, NULL) == DT_MASKS_OP_HIT_OPERATOR;
  if(over == (gdk_window_get_cursor(bin) != NULL)) return;

  if(over)
  {
    GdkCursor *cursor =
      gdk_cursor_new_from_name(gdk_window_get_display(bin), "pointer");
    gdk_window_set_cursor(bin, cursor);
    if(cursor) g_object_unref(cursor);
  }
  else
    gdk_window_set_cursor(bin, NULL);
}

static void _tree_leave_cb(GtkEventControllerMotion *controller,
                           dt_lib_module_t *self)
{
  GtkWidget *view = dt_gui_get_widget(controller);
  GdkWindow *bin = gtk_tree_view_get_bin_window(GTK_TREE_VIEW(view));
  if(bin) gdk_window_set_cursor(bin, NULL);
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

  // gesture coordinates are relative to the widget allocation, while
  // gtk_tree_view_get_path_at_pos() wants bin-window ones. the two happen to
  // coincide here -- headers hidden, and dt_ui_resize_wrap sets the horizontal
  // policy to GTK_POLICY_NEVER so the view never scrolls sideways -- which is
  // why passing x/y straight through got away with it. a COLUMN hit-test
  // depends on the horizontal offset being right, so convert once and use the
  // result everywhere, as libs/collect.c, libs/geotagging.c and
  // gui/preferences_ai.c already do
  gint bin_x, bin_y;
  gtk_tree_view_convert_widget_to_bin_window_coords(GTK_TREE_VIEW(treeview),
                                                    (gint)x, (gint)y,
                                                    &bin_x, &bin_y);

  GtkTreePath *mouse_path = NULL;
  GtkTreeIter iter;
  dt_iop_module_t *module = NULL;
  gboolean on_row = FALSE;
  if(gtk_tree_view_get_path_at_pos(GTK_TREE_VIEW(treeview),
                                   bin_x, bin_y, &mouse_path, NULL,
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
    // dt_gui_current_state() and not gtk_get_current_event_state(): a press
    // emitted by a shortcut carries no current event, which leaves that call
    // writing nothing at all into its out parameter. the helper hands back the
    // effect's own modifiers there, and "no modifier" is the one answer we
    // must not get wrong -- it is what decides whether a menu opens
    const GdkModifierType mods = dt_gui_current_state(gesture);
    dt_masks_state_t op_state = DT_MASKS_STATE_NONE;

    // a plain left click on the operator column of a shape that carries one
    // opens the five modes right where the glyph is. a modified click still
    // belongs to the selection -- ctrl and shift build a multiple one, and it
    // would be unbuildable if this vertical strip swallowed those clicks. the
    // "pressed" signal fires once per press, so without n_press a double click
    // would pop the menu twice.
    // note what we do NOT do: dt_gui_claim(gesture). this handler has never
    // claimed the sequence and must not start: claiming cancels the
    // treeview's own gesture, and with it selection, the rename double click
    // and the expanders
    if(n_press == 1
       && mouse_path
       && dt_modifier_is(mods, 0)
       && _op_cell_at_bin(lm, treeview, bin_x, bin_y, &op_state)
          == DT_MASKS_OP_HIT_OPERATOR)
    {
      // _tree_operation() acts on the view's selection and not on a row handed
      // to it, so the selection has to BE the clicked row before the menu can
      // open. we do it here rather than trust the treeview's own gesture: GTK
      // defers collapsing a multiple selection onto the clicked row until the
      // button is released, and the menu's grab swallows that release
      if(gtk_tree_selection_count_selected_rows(selection) != 1
         || !gtk_tree_selection_path_is_selected(selection, mouse_path))
      {
        gtk_tree_selection_unselect_all(selection);
        gtk_tree_selection_select_path(selection, mouse_path);
      }

      // the pointer is about to be grabbed by the menu and GTK3 does not
      // reliably deliver a leave event for a grab crossing: drop the hand now
      // rather than leave it hanging over the panel until the menu closes
      GdkWindow *bin = gtk_tree_view_get_bin_window(GTK_TREE_VIEW(treeview));
      if(bin) gdk_window_set_cursor(bin, NULL);

      // the very entries of the context menu, from the very table: one row,
      // never the base, so all five are enabled and the one in force is ticked
      GtkMenuShell *menu = GTK_MENU_SHELL(gtk_menu_new());
      _add_tree_operations(menu, op_state, FALSE);
      // dt_gui_menu_popup takes the floating ref and drops it on "deactivate",
      // and pops at the pointer when handed no widget
      dt_gui_menu_popup(GTK_MENU(menu), NULL, 0, 0);
    }
    // if click on a blank space, then deselect all
    else if(!on_row)
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

  // the operator column answers for itself, on the CELL and not on the row:
  // what the glyph means and that a click changes it, and on the base why it
  // has none -- which is the whole compensation for a click there doing
  // nothing. in the non-keyboard branch gtk_tree_view_get_tooltip_context()
  // has just rewritten x/y into bin-window coordinates, which is what the
  // column lookup wants; a keyboard tooltip has no column under a pointer, so
  // there the row tooltip answers as it always did.
  // it answers first and returns, because two tooltips on one row erase each
  // other. in practice they never meet: TREE_USED_TEXT is only ever filled on
  // a root row, and this column only ever answers inside a group
  if(!keyboard_tip)
  {
    dt_lib_masks_t *lm = data;
    dt_masks_state_t op_state = DT_MASKS_STATE_NONE;
    const dt_masks_op_hit_t hit =
      _op_cell_at_bin(lm, widget, x, y, &op_state);

    if(hit == DT_MASKS_OP_HIT_BASE)
      gtk_tooltip_set_text(tooltip,
                           _("the shape the others are applied onto\n"
                             "it takes no operator"));
    else if(hit == DT_MASKS_OP_HIT_OPERATOR)
    {
      // the word behind the glyph, and the one thing a glyph cannot say
      gchar *text =
        g_strdup_printf("%s\n%s",
                        _(_masks_operators[_op_index(op_state)].name),
                        _("click to change how this shape combines"));
      gtk_tooltip_set_text(tooltip, text);
      g_free(text);
    }

    if(hit != DT_MASKS_OP_HIT_NONE)
    {
      // a hit means the column is lm->op_col, so there is nothing to look up
      gtk_tree_view_set_tooltip_cell(tree_view, tooltip, path,
                                     lm->op_col, NULL);
      gtk_tree_path_free(path);
      return TRUE;
    }
  }

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
  // no operator or inverse work here. both gtk_tree_store_set() below are
  // followed by _set_iter_name(), which derives every displayed column of the
  // row from the very same state -- this block computed some of them only to
  // be overwritten a few lines later. one writer for what a row displays, so
  // "refresh in place" and "full rebuild" cannot disagree
  GdkPixbuf *icuse = NULL;

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

  // last, so it reads the models the loop above installed. the early-out at the
  // top of this function cannot skip a change of state: _forms_structure_hash
  // mixes every form's id, so gaining or losing the last one always misses it
  _empty_state_update(lm);

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

  // this is the one path that empties the lists without going through
  // gui_update. its only caller adds a history item right after, which does
  // reach gui_update -- but relying on that would leave the panel showing two
  // empty lists the day someone calls the proxy on its own
  _empty_state_update(lm);
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
  GtkCellRenderer *renderer;

  if(!library)
  {
    // a column of its own, and the whole reason there are two: a click is "on
    // the operator" when GTK reports THIS column under the pointer -- no pixel
    // arithmetic, nothing to keep in step with a renderer's padding. the same
    // test gui/preferences_ai.c runs on its info column.
    // rank, operator, "base": everything that says where the shape sits in the
    // application order, in one strip that does not move with depth
    d->op_col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(d->op_col, "operator");
    gtk_tree_view_append_column(GTK_TREE_VIEW(view), d->op_col);

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
    gtk_tree_view_column_pack_start(d->op_col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(d->op_col, renderer, "text", TREE_NUM);

    // the operator, and the one thing in this strip left at full contrast: it
    // is the one thing here a click can change, and that difference in weight
    // is the whole hierarchy of the column
    renderer = gtk_cell_renderer_pixbuf_new();
    gtk_tree_view_column_pack_start(d->op_col, renderer, FALSE);
    gtk_tree_view_column_set_attributes(d->op_col, renderer,
                                        "pixbuf", TREE_IC_OP, NULL);
    gtk_tree_view_column_add_attribute(d->op_col, renderer,
                                       "visible", TREE_IC_OP_VISIBLE);

    // "base" sits where the operator glyph would be, on the one row that has no
    // operator and cannot be given one. a word rather than a glyph: it has to
    // translate, and it has to follow a theme change -- which the icons,
    // rasterised once in gui_init, do not.
    // it is in this column and not next to the name on purpose: the strip has
    // to read vertically. a click on the word is therefore a click "on the
    // operator", which is why the base is a case _op_cell_at_bin() answers for
    // rather than a case it ignores
    renderer = gtk_cell_renderer_text_new();
    g_object_set(renderer,
                 "xalign", 0.0,
                 "xpad", (guint)DT_PIXEL_APPLY_DPI(1),
                 "scale", PANGO_SCALE_SMALL,
                 "sensitive", FALSE,
                 NULL);
    gtk_tree_view_column_pack_start(d->op_col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(d->op_col, renderer, "text", TREE_BASE);
  }

  GtkTreeViewColumn *col = gtk_tree_view_column_new();
  gtk_tree_view_column_set_title(col, "shapes");
  // the leftover width goes to the names, not to the strip on their left:
  // without this the name cell stops giving way and the ellipsize below has no
  // boundary to ellipsize against. set on both lists, needed by one: in the
  // library the names ARE the only column, and the last column of a view is
  // handed the leftover width whether or not it is marked expanding
  gtk_tree_view_column_set_expand(col, TRUE);
  gtk_tree_view_append_column(GTK_TREE_VIEW(view), col);

  if(!library)
  {
    // the expander -- and the per-depth indentation with it -- belongs to the
    // name column, not to the strip. GTK would otherwise leave it in the first
    // column, and gtk_tree_view_get_path_at_pos() reports the expander area as
    // part of that column: on a group nested in a group, which is a depth-2
    // row and therefore a row carrying an operator, one click would both
    // unfold the row and open the operator menu, with nothing to tell the two
    // apart. it also means the ranks line up under each other instead of
    // stepping right with depth
    gtk_tree_view_set_expander_column(GTK_TREE_VIEW(view), col);

    renderer = gtk_cell_renderer_pixbuf_new();
    gtk_tree_view_column_pack_start(col, renderer, FALSE);
    gtk_tree_view_column_set_attributes(col, renderer,
                                        "pixbuf", TREE_IC_INVERSE, NULL);
    gtk_tree_view_column_add_attribute(col, renderer,
                                       "visible", TREE_IC_INVERSE_VISIBLE);
  }

  // the kind of shape, on both lists and in the same place on the row: after
  // the inverse marker, before the name. M3 reads a row left to right as one
  // sentence -- "intersect, with the inverse of this circle" -- and the kind is
  // the noun of it
  renderer = gtk_cell_renderer_pixbuf_new();
  gtk_tree_view_column_pack_start(col, renderer, FALSE);
  gtk_tree_view_column_set_attributes(col, renderer,
                                      "pixbuf", TREE_IC_TYPE, NULL);
  gtk_tree_view_column_add_attribute(col, renderer,
                                     "visible", TREE_IC_TYPE_VISIBLE);

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

  if(!library)
  {
    // the opacity of this shape in this mask, at the right edge of the row.
    // a fixed four-character gutter: "100%" is the widest string it could hold
    // and it never holds one, but a cell whose width follows its content moves
    // every name on every row the moment one shape leaves 100%. small and
    // insensitive, like the rank on the other side -- the theme greys it, dark
    // and light alike, and no colour is written here
    renderer = gtk_cell_renderer_text_new();
    g_object_set(renderer,
                 "xalign", 1.0,
                 "xpad", (guint)DT_PIXEL_APPLY_DPI(2),
                 "width-chars", 4,
                 "scale", PANGO_SCALE_SMALL,
                 "sensitive", FALSE,
                 NULL);
    gtk_tree_view_column_pack_end(col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(col, renderer, "text", TREE_OPACITY);

    // which module this mask serves, between the name and the opacity -- pack
    // order is right to left, so packing it after puts it on the left of it.
    // ellipsized by the end: a module name cut short is still the same
    // statement, where a cut short name is a different mask.
    // and capped, which is the whole of the priority rule. this cell does not
    // expand, so it asks for its natural width and a panel too narrow to grant
    // everything shares what is left: "diffuse or sharpen" would take its
    // sixteen characters out of the name. sixteen is the cap because it is
    // what M2 shows fitting -- "→ local contrast", uncut
    renderer = gtk_cell_renderer_text_new();
    g_object_set(renderer,
                 "xalign", 1.0,
                 "xpad", (guint)DT_PIXEL_APPLY_DPI(2),
                 "ellipsize", PANGO_ELLIPSIZE_END,
                 "max-width-chars", 16,
                 "scale", PANGO_SCALE_SMALL,
                 "sensitive", FALSE,
                 NULL);
    gtk_tree_view_column_pack_end(col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(col, renderer, "text", TREE_TARGET);
  }

  if(library)
  {
    // packed from the right edge inwards: the badge first, so it keeps the
    // very place it has today.
    // the library and nowhere else: on a mask row the badge answered "is this
    // filed in a group", which a mask row answers by existing. here the
    // question is real -- one row per shape, and the badge is the whole of
    // what says the shape is in use at all
    renderer = gtk_cell_renderer_pixbuf_new();
    gtk_tree_view_column_pack_end(col, renderer, FALSE);
    gtk_tree_view_column_set_attributes(col, renderer,
                                        "pixbuf", TREE_IC_USED, NULL);
    gtk_tree_view_column_add_attribute(col, renderer,
                                       "visible", TREE_IC_USED_VISIBLE);

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
  // `d` and no longer NULL: the tooltip has to know which column is the
  // operator one
  g_signal_connect(view, "query-tooltip", G_CALLBACK(_tree_query_tooltip), d);
  g_signal_connect(selection, "changed", G_CALLBACK(_tree_selection_change), d);
  // a click target inside a list does not announce itself; the pointer does.
  // masks zone only: the library has no operator column, so nothing to point at
  if(!library)
    dt_gui_connect_motion(view, _tree_motion_cb, NULL, _tree_leave_cb, self);
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
  // the operator glyphs, from the one table that also builds the menu entries
  // and decides which one a row shows. twice as wide as they are high, as they
  // have always been
  for(int i = 0; i < (int)G_N_ELEMENTS(_masks_operators); i++)
    d->ic_op[i] = _get_pixbuf_from_cairo(_masks_operators[i].paint,
                                         bs2 * 2, bs2);

  // the kind glyphs, from the table that also draws the six buttons above the
  // list: a row and the button that made it show the same drawing. square,
  // like the inverse and "used" badges -- only the operators are twice as wide
  for(int i = 0; i < (int)G_N_ELEMENTS(_new_mask_shapes); i++)
    d->ic_type[i] = _get_pixbuf_from_cairo(_new_mask_shapes[i].paint,
                                           bs2, bs2);

  // the six shape toggles are back, and with them the only registration of the
  // six "shapes/add ..." paths. there can be exactly one: dt_action_locate()
  // never duplicates a path, it reuses the node -- and dt_action_define() on a
  // node dt_action_register() already claimed overwrites its type to WIDGET
  // while leaving target pointing at the callback, because it only sets the
  // weak pointer when target is still NULL (accelerators.c). a shortcut a user
  // bound to one of these paths is unaffected by the COMMAND -> WIDGET change:
  // shortcutsrc is read after dt_lib_init, _action_find() resolves by path
  // alone, and a command wrote no element or effect token to drop
  for(int i = 0; i < (int)G_N_ELEMENTS(_new_mask_shapes); i++)
  {
    GtkWidget *w = dtgtk_togglebutton_new(_new_mask_shapes[i].paint, 0, NULL);
    dt_action_define(DT_ACTION(self), N_("shapes"), _new_mask_shapes[i].action,
                     w, &dt_action_def_toggle);

    // the gesture a bound key is replayed through, wired exactly as
    // dt_iop_togglebutton_new wires these same six in a blending panel:
    // capture phase and claimed, so the toggle's own release never fights
    // _creation_bar_update. without DT_ACTION_GESTURE_KEY on the button,
    // _action_process_toggle falls back to a synthetic GdkEvent this gesture
    // never sees, and ctrl + the bound key stops chaining shapes
    GtkGesture *gesture = gtk_gesture_multi_press_new(w);
    gtk_event_controller_set_propagation_phase(GTK_EVENT_CONTROLLER(gesture),
                                               GTK_PHASE_CAPTURE);
    dt_gui_add_controller(w, gesture);
    gtk_gesture_single_set_button(GTK_GESTURE_SINGLE(gesture), 0);
    g_signal_connect(gesture, "begin", G_CALLBACK(_bt_shape_claim_cb), NULL);
    g_signal_connect(gesture, "pressed", G_CALLBACK(_bt_add_shape_cb),
                     GINT_TO_POINTER(_new_mask_shapes[i].type));
    g_object_set_data(G_OBJECT(w), DT_ACTION_GESTURE_KEY, gesture);

    // the sentence dt_iop_togglebutton_new composes for these very six buttons
    // in a blending panel, from the same msgids and in the same words: the two
    // rows sit in the same column of the screen, and one of them saying what
    // ctrl+click does while the other keeps quiet about it is how a reader
    // concludes they are not the same button
    if(_new_mask_shapes[i].ctrl)
    {
      gchar *tip = g_strdup_printf(_("%s\nctrl+click to %s"),
                                   _(_new_mask_shapes[i].label),
                                   _(_new_mask_shapes[i].ctrl));
      gtk_widget_set_tooltip_text(w, tip);
      g_free(tip);
    }
    else
      gtk_widget_set_tooltip_text(w, _(_new_mask_shapes[i].label));

    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(w), FALSE);
    d->bt_shape[i] = w;
  }

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

  // the named button. it survives in the empty state and only there: a panel
  // with nothing in it is a screen that teaches and needs words, a populated
  // one is a screen that works and needs glyphs. it is hidden, never removed
  // -- dt_action_button_new registers the action ON the button, so a twin
  // would claim the same path "lib/masks/new mask", and a hidden widget still
  // answers its shortcut. dt_action_button_new already sets hexpand
  d->bt_new = dt_action_button_new
    (self, N_("new mask"), _new_mask_clicked, self,
     _("pick a shape and the module it belongs to\n"
       "right-click to cancel while drawing"), 0, 0);

  // R1: the module the next shape goes to, and the way to send it elsewhere.
  // a label and not a button: a button promises an action on its own text, and
  // this text is a state. the chevron and not the three bars of
  // dtgtk_cairo_paint_presets: libs/lib.c gives those to the presets button of
  // every panel, so this row would have carried the same glyph as the header
  // 20 px above it, for another job. a chevron is what darktable draws
  // wherever a control opens a list. same shape as the creation bar built
  // below, and as the rule rows of libs/collect.c
  d->target_label = dt_ui_label_new("");
  d->bt_target = dtgtk_button_new(dtgtk_cairo_paint_dropdown, 0, NULL);
  gtk_widget_set_tooltip_text(d->bt_target,
                              _("choose the module the next shape belongs to"));
  g_signal_connect(G_OBJECT(d->bt_target), "clicked",
                   G_CALLBACK(_target_menu_clicked), self);
  d->target_row = dt_gui_hbox(dt_gui_expand(d->target_label), d->bt_target);

  // R2: the six shapes, one click each. six buttons ask for about 148 px of
  // their own (1.15em min-width + 0.07em margin + 1px padding + 1px border,
  // six times over), 170 px inside the panel's 0.65em side padding -- against
  // 310 px for the same six with the module name on the same line, twice the
  // 150 px min_panel_width. hence two rows
  d->shape_row = dt_gui_hbox();
  for(int i = 0; i < (int)G_N_ELEMENTS(_new_mask_shapes); i++)
    dt_gui_box_add(d->shape_row, dt_gui_expand(d->bt_shape[i]));

  // M1: an image carrying no mask says so, and says what a mask is for, rather
  // than showing two empty lists. the sentences go either side of the button
  // above -- that is M1's reading order, and no second button is needed for it.
  // a twin would need one anyway: dt_action_button_new registers the action ON
  // the button, so it would claim the same path "lib/masks/new mask"
  d->empty_title = gtk_label_new(_("no masks on this image"));
  d->empty_hint = gtk_label_new(_("a mask limits an adjustment to an area"));
  gtk_label_set_line_wrap(GTK_LABEL(d->empty_title), TRUE);
  gtk_label_set_line_wrap(GTK_LABEL(d->empty_hint), TRUE);
  gtk_label_set_justify(GTK_LABEL(d->empty_title), GTK_JUSTIFY_CENTER);
  gtk_label_set_justify(GTK_LABEL(d->empty_hint), GTK_JUSTIFY_CENTER);
  dt_gui_add_class(d->empty_hint, "dt_dimmed");
  gtk_widget_set_margin_top(d->empty_title, DT_PIXEL_APPLY_DPI(12));
  gtk_widget_set_margin_bottom(d->empty_title, DT_PIXEL_APPLY_DPI(6));
  gtk_widget_set_margin_top(d->empty_hint, DT_PIXEL_APPLY_DPI(6));
  gtk_widget_set_margin_bottom(d->empty_hint, DT_PIXEL_APPLY_DPI(12));

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
  //
  // both lists go away together, and only when both are empty. the right-click
  // on blank space they used to protect offers, in that state and only that
  // state, the shape entries of the button above -- a strict subset of its menu
  // -- plus "delete unused shapes", which can then only reach what neither list
  // was showing: retouch and spot-removal shapes are held by a module and kept,
  // what is left is what an undone history step still refers to. below the last
  // row of a non-empty list that menu is untouched
  d->masks_box = dt_ui_resize_wrap(d->treeview, 200,
                                   "plugins/darkroom/masks/heightview");
  // xalign 0: dt_ui_section_label_new centres its text, and centred over a
  // full-width rule is how darktable draws a separator -- which is why this one
  // read as a footer under the list above instead of a heading for the list
  // below. the class stays, so colour, weight and rule still come from the
  // theme. same fix on "properties", built by the same helper
  d->lib_label = dt_ui_section_label_new(C_("section", "shape library"));
  gtk_label_set_xalign(GTK_LABEL(d->lib_label), 0.0f);
  d->lib_box = dt_ui_resize_wrap(d->library, 120,
                                 "plugins/darkroom/masks/heightlibrary");

  self->widget = dt_gui_vbox
    (d->empty_title,
     d->bt_new,
     d->empty_hint,
     d->target_row,
     d->shape_row,
     d->masks_box,
     d->lib_label,
     d->lib_box,
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

  // the message and the two lists are the two sides of one switch, so both
  // sides take the same treatment: raise the children once, then no_show_all,
  // or the panel-wide gtk_widget_show_all() of libs/lib.c decides this instead
  // of gui_update. the two sentences are leaves, nothing under them to raise
  gtk_widget_set_no_show_all(d->empty_title, TRUE);
  gtk_widget_set_no_show_all(d->empty_hint, TRUE);

  gtk_widget_show_all(d->masks_box);
  gtk_widget_show_all(d->lib_label);
  gtk_widget_show_all(d->lib_box);
  gtk_widget_set_no_show_all(d->masks_box, TRUE);
  gtk_widget_set_no_show_all(d->lib_label, TRUE);
  gtk_widget_set_no_show_all(d->lib_box, TRUE);

  // the top row and the named button are the two sides of the same switch, so
  // they take the same treatment: raise the children once -- show_all has to
  // reach the six buttons and their canvases -- then no_show_all, or the
  // panel-wide gtk_widget_show_all() of libs/lib.c decides this instead of
  // _empty_state_update
  gtk_widget_show_all(d->bt_new);
  gtk_widget_show_all(d->target_row);
  gtk_widget_show_all(d->shape_row);
  gtk_widget_set_no_show_all(d->bt_new, TRUE);
  gtk_widget_set_no_show_all(d->target_row, TRUE);
  gtk_widget_set_no_show_all(d->shape_row, TRUE);

  dt_gui_new_collapsible_section
    (&d->cs,
     "plugins/darkroom/masks/expand_properties",
     _("properties"),
     GTK_BOX(self->widget),
     DT_ACTION(self));
  // the helper builds its header with dt_ui_section_label_new too: left-align
  // it as well, or the panel contradicts itself two lines apart
  gtk_label_set_xalign(GTK_LABEL(d->cs.label), 0.0f);
  d->none_label = dt_ui_label_new(_("no shapes selected"));
  dt_gui_box_add(d->cs.container, d->none_label);
  gtk_widget_show_all(GTK_WIDGET(d->cs.container));
  gtk_widget_set_no_show_all(GTK_WIDGET(d->cs.container), TRUE);
  // the section is packed into self->widget by the helper, so it is on screen
  // in the empty state too unless gui_update is allowed to take it away. same
  // pattern: show_all raises the header and its arrow exactly as the panel's
  // own show_all would have, the container above keeps its own no_show_all,
  // and from here on the empty state decides
  gtk_widget_show_all(d->cs.expander);
  gtk_widget_set_no_show_all(d->cs.expander, TRUE);

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

  // last, and it has to be last: the panel opens on M1 or on M2 and never on
  // both, and until this runs the show_all above has the named button of one
  // up beside the two rows of the other. nothing has called gui_update yet --
  // libs/lib.c only queues it, on the first draw -- and _empty_state_update
  // returns until cs.expander exists, which it does from here on. the two
  // models are empty at this point, so this settles on M1, which is the right
  // screen for a panel built before any image is loaded
  _empty_state_update(d);

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
