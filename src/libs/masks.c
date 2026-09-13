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
#include "common/ai/detectors.h"
#include "common/darktable.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/masks/object_recipe.h"
#include "dtgtk/paint_cell.h"
#include "gui/accelerators.h"
#include "gui/drag_and_drop.h"
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

// defined with the other grp->points walkers below; the module list of the
// context menu needs the direct-membership answer before that point
static int _group_point_index(const dt_masks_form_t *grp,
                              const dt_mask_id_t formid);
static dt_iop_module_t *_mask_group_owner(const dt_mask_id_t formid);

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

// the three the bar arms, in M2's order. states and not indices, so a button
// resolves through _op_index() into the one table above and back out as the
// glyph the row will show once the shape is in: three entries here, and
// nothing at all to keep in step.
// the word on the button is M2's, and it is not the operator's name: a
// photographer adds and subtracts, and "union" and "difference" are what the
// result is called. the technical name is one hover away, from the msgid the
// row's own menu already uses, so the two vocabularies stay tied together.
// sum and exclusion are left out on purpose. M2 shows three, they are the
// three a photographer names, and the two others stay one right-click away on
// the row itself -- where they always were
static const struct
{
  dt_masks_state_t state;
  const char *label;  // M2's word for it, on the button
} _arm_operators[] =
{
  { DT_MASKS_STATE_UNION,        N_("add") },
  { DT_MASKS_STATE_DIFFERENCE,   N_("subtract") },
  { DT_MASKS_STATE_INTERSECTION, N_("intersect") },
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

// 0 = the masks, 1 = the shape library. declared here and not beside
// _masks_view() below because the panel keeps one field per list of a few
// things now, and an array indexed by this is what keeps "do it to both" a
// one-line loop
#define DT_MASKS_NVIEWS 2

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
  // the name column of each list and the cell inside it, indexed exactly like
  // _masks_view(). two uses, and both lists need them: renaming opens the
  // in-place editor on the row the photographer pointed at, and the double
  // click that starts it has to know it landed on the name and not on one of
  // the cells a single click already acts on
  GtkTreeViewColumn *name_col[DT_MASKS_NVIEWS];
  GtkCellRenderer *name_cell[DT_MASKS_NVIEWS];
  // did the editor actually open? a one-shot handshake between
  // _tree_start_rename() and "editing-started", read on the line after the
  // call that should have opened it and never anywhere else. the cell is
  // editable only while a rename lasts, and a cell left armed by an editor
  // that never opened is a cell the next plain click would edit -- the very
  // behaviour the arming exists to remove
  gboolean rename_started;
  // the drag in flight, as ids and the view it left from -- never a path, an
  // iter or a form pointer: gui_update swaps both stores under an open drag
  // without notice, so every drag-motion and the drop itself re-resolve these
  // against the model and grp->points of the moment. armed by a qualifying
  // press in _tree_button_pressed_cb, turned into a real drag by
  // _tree_motion_cb past the drag threshold, cleared by "drag-end"
  GtkWidget *drag_view;
  dt_mask_id_t drag_formid;
  dt_mask_id_t drag_groupid;
  gint drag_x, drag_y;
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
  // the masks zone's power column, hit-tested the same way and for the same
  // reason: one pointer comparison, no pixel arithmetic. it is a column of its
  // own and not a cell packed beside the name, because a click has to be able
  // to tell "switch the module off" from "select this mask", and a column is
  // the only boundary GTK reports back
  GtkTreeViewColumn *power_col;
  // and the show-mask column, hit-tested exactly like the two before it
  GtkTreeViewColumn *show_col;
  // caption under the library, shown only when at least one shape is not
  // linked to a module: it names exactly the set the cleanup is about
  GtkWidget *lib_unlinked;
  // the shape library's heading row: the section label, and the one button
  // that empties what the caption under the list describes. a heading and not
  // the creation row -- F3-4 asks for that distance, and it is all this panel
  // has. `unlinked` is what gui_update just counted, read back by the
  // confirmation so the dialog and the caption state one number
  GtkWidget *lib_row, *bt_cleanup;
  int unlinked;
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

  GdkPixbuf *ic_inverse;
  // the operator glyphs, indexed by _masks_operators: same table, same order,
  // so a sixth operator is one line there and nothing at all here
  GdkPixbuf *ic_op[G_N_ELEMENTS(_masks_operators)];
  // the kind glyphs, indexed by _new_mask_shapes: same table, same order, so a
  // seventh kind is one line there and nothing at all here
  GdkPixbuf *ic_type[G_N_ELEMENTS(_new_mask_shapes)];
  // the raster glyph, apart from ic_type on purpose: that array is
  // indexed by _new_mask_shapes, the table that also builds the six
  // creation buttons, and a raster mask is nothing one draws
  GdkPixbuf *ic_raster;

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

  // M2's header eye, built by gui_tool_box(). rebuilt with the expander on
  // every view change, so the field is weak-cleared on destroy there: its
  // sensitivity is the only thing about it that moves afterwards
  GtkWidget *bt_hide;

  // the hint row: one row, under both lists, saying what the panel is waiting
  // for -- where the next drawn shape is about to go, or that an operator is
  // armed and a shape has still to be picked -- and letting either be called
  // off. arm_module is this panel's claim on the creation in flight, not a
  // second source of truth for it: a shape button in a module's own blending
  // panel arms the very same canvas and reaches the very same proxy, and only
  // this tells the two apart. bt_creation_cancel is the cross, which stands
  // down for the one state of the row that states a fact instead of waiting
  GtkWidget *creation_bar, *creation_label, *bt_creation_cancel;
  struct dt_iop_module_t *arm_module;
  // ... and the same claim for a creation armed towards the shape library:
  // there is no module to hold then, so arm_module alone could not tell
  // "this panel armed a shape for no module" from "nothing armed". a
  // creation in flight with form_gui->creation_module == NULL is ours
  // exactly when this is set
  gboolean arm_library;

  // the module picked by hand in the menu of the target row, or NULL for the
  // shape library. this is the ONLY memory the panel keeps of where the next
  // shape goes: everything else is read at the moment of the gesture. it is
  // compared and validated (alive, able to take a drawn mask) before every
  // use, and forgotten with the image it was chosen on -- a module pointer
  // of one pipe means nothing in the next, and a freed address can be handed
  // back to a module of the next image
  struct dt_iop_module_t *pick_target;
  dt_imgid_t pick_imgid;

  // the operator bar: M2's contextual row, under both lists and above the
  // creation bar. it says how the NEXT shape will combine, before it is drawn
  // -- today that choice is made afterwards, shape by shape, in a menu that
  // has to be found. arm_op is this panel's copy of what
  // dt_masks_set_next_operator() holds, so the three buttons can show it;
  // DT_MASKS_STATE_NONE is disarmed.
  // arm_updating is the re-entry guard the toggles need: setting a
  // GtkToggleButton active emits "toggled", and the handler would take a
  // refresh for a click. same pattern, same reason, as resize_updating above
  GtkWidget *arm_bar, *arm_label;
  GtkWidget *bt_arm[G_N_ELEMENTS(_arm_operators)];
  dt_masks_state_t arm_op;
  // the module whose mask the armament was chosen for, kept beside it so a
  // different mask being selected puts the armament down instead of
  // inheriting it. compared, never dereferenced
  struct dt_iop_module_t *arm_target;
  gboolean arm_updating;
} dt_lib_masks_t;

static void _resize_update(dt_lib_masks_t *d);
static void _creation_bar_update(dt_lib_masks_t *d);
static void _arm_bar_update(dt_lib_masks_t *d);
static void _target_row_update(dt_lib_masks_t *d);
static void _creation_end_continuous(void);
#ifdef HAVE_AI
static void _object_button_menu(dt_lib_module_t *self, GtkWidget *button);
#endif

// the list an index names -- see DT_MASKS_NVIEWS above for which is which.
// an index rather than two named fields, so that "do it to both" stays a
// one-line loop everywhere below
static GtkWidget *_masks_view(dt_lib_masks_t *lm, const int v)
{
  return v == 0 ? lm->treeview : lm->library;
}

// ... and the way back, for the handlers a view hands itself to: -1 when the
// widget is neither list, which is what keeps an array lookup off a bad index
static int _masks_view_index(dt_lib_masks_t *lm, GtkWidget *view)
{
  if(!view) return -1;
  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
    if(_masks_view(lm, v) == view) return v;
  return -1;
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
  gtk_widget_set_visible(lm->lib_row, !empty);
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

// M2's header eye: take every drawn shape off the photograph, without having
// to find a blank strip in a list that fills its own height. that strip is
// what the photographer found by himself -- gtk_tree_selection_unselect_all()
// on a blank click -- and its one limit is real: a full list has no blank.
//
// a button and not a toggle, because the engine holds no such state. what is
// drawn IS dev->form_visible (views/darkroom.c reads it straight, and
// develop/masks/group.c walks its points with no per-shape flag), so "hidden"
// can only mean "nothing armed". a toggle would promise a state it cannot
// keep: the next click on a row, the next shape drawn, a module's own eye all
// put shapes back without telling this panel.
//
// the deselection goes to both lists at once and through
// _tree_selection_change(), the ONE place this panel writes form_visible, so
// the lists and the canvas cannot end up disagreeing, and it leaves the lists
// folded exactly as they were. the other branch cannot: dt_masks_change_form_
// gui(NULL) lands in _lib_masks_selection_change(), which collapses what it
// fails to find -- the price of reaching what no row of ours put on screen.
//
// it takes away the outlines and nothing else, which is what its tooltip says.
// the filled overlay of the show-mask cell is a module request that
// develop/blend.c only honours for the module holding the focus; clearing it
// would mean dt_iop_set_mask_display(), i.e. a focus change and a pipe
// recompute inside a button that is meant to cost nothing, and the one cell
// that can be lit is lit, one click away
static void _hide_all_shapes(dt_lib_module_t *self)
{
  dt_lib_masks_t *d = self->data;
  gboolean any = FALSE;

  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
  {
    GtkWidget *view = _masks_view(d, v);
    GtkTreeSelection *sel =
      view ? gtk_tree_view_get_selection(GTK_TREE_VIEW(view)) : NULL;
    if(sel && gtk_tree_selection_count_selected_rows(sel) > 0)
    {
      any = TRUE;
      gtk_tree_selection_unselect_all(sel);
    }
  }

  // and what no row of ours put there: a creation in flight, a form armed
  // while both lists showed nothing. the form is dropped outright then, which
  // is what folding the panel already does -- and a continuous run left
  // standing would chain the next shape drawn anywhere
  if(!any)
  {
    _creation_end_continuous();
    dt_masks_change_form_gui(NULL);
  }

  dt_masks_reset_show_masks_icons();
  dt_control_queue_redraw_center();
}

static void _bt_hide_cb(GtkButton *button, dt_lib_module_t *self)
{
  _hide_all_shapes(self);
}

// the one hook libs/lib.c offers a panel for a button of its own in the header
// (lib.c packs it beside the reset and the presets buttons, which is where M2
// draws this one); libs/collect.c is its only other user. called from
// dt_lib_gui_get_expander(), i.e. when the view is built, long after gui_init:
// self->data is there
GtkWidget *gui_tool_box(dt_lib_module_t *self)
{
  dt_lib_masks_t *d = self->data;

  // the glyph every module's blending panel already gives to "show and edit
  // mask elements": one drawing for one subject, on both sides of the screen
  d->bt_hide = dtgtk_button_new(dtgtk_cairo_paint_masks_eye, 0, NULL);
  // the outlines and not the filled overlay, which is the reach this button
  // has -- see _hide_all_shapes()
  gtk_widget_set_tooltip_text
    (d->bt_hide, _("hide the shape outlines drawn over the photograph"));
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("hide on the photograph"),
                   d->bt_hide, &dt_action_def_button);
  g_signal_connect(G_OBJECT(d->bt_hide), "clicked",
                   G_CALLBACK(_bt_hide_cb), self);
  // this button lives in the expander header, and views/view.c destroys the
  // whole expander on every view change while this panel's data survives.
  // without this the field would outlive the widget and the next refresh --
  // the selection proxy fires from outside the darkroom too -- would set the
  // sensitivity of freed memory
  g_signal_connect(G_OBJECT(d->bt_hide), "destroy",
                   G_CALLBACK(gtk_widget_destroyed), &d->bt_hide);
  return d->bt_hide;
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
  TREE_USED_TEXT,
  // rank of the shape in the application order of its module's mask, ""
  // where that order carries no meaning. derived, never persisted, and
  // written by _set_iter_name only
  TREE_NUM,
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
  // whether the module this row's mask serves is switched off. two booleans
  // and not one: a model column is bound to a cell property as it stands, with
  // no way to negate it on the way, and the two properties that read this want
  // opposite senses -- "strikethrough" on the name, "sensitive" around it
  TREE_MODULE_OFF,
  TREE_MODULE_ON,
  // whether this row carries the power cell at all: a mask row, whose module
  // is still in the pipe and has a switch a click may reach. it is written
  // here and read back by _op_cell_at_bin(), so the cell a click reacts to and
  // the cell that is drawn are one decision and not two -- the same thing
  // gui/preferences_ai.c does with its info column. derived, written by
  // _set_iter_name only
  TREE_POWER,
  // whether this row carries the show-mask cell, and whether that module's
  // mask is the one currently on screen. same pair of senses as
  // TREE_MODULE_OFF / TREE_MODULE_ON and for the same reason: "visible" and
  // "sensitive" are two cell properties and a model column cannot be negated
  // on the way in. derived, written by _set_iter_name only
  TREE_SHOW,
  TREE_SHOW_ON,
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
      [TREE_USED_TEXT] = G_TYPE_STRING,
      [TREE_NUM] = G_TYPE_STRING,
      [TREE_LINK] = G_TYPE_STRING,
      [TREE_IC_TYPE] = GDK_TYPE_PIXBUF,
      [TREE_IC_TYPE_VISIBLE] = G_TYPE_BOOLEAN,
      [TREE_OPACITY] = G_TYPE_STRING,
      [TREE_TARGET] = G_TYPE_STRING,
      [TREE_MODULE_OFF] = G_TYPE_BOOLEAN,
      [TREE_MODULE_ON] = G_TYPE_BOOLEAN,
      [TREE_POWER] = G_TYPE_BOOLEAN,
      [TREE_SHOW] = G_TYPE_BOOLEAN,
      [TREE_SHOW_ON] = G_TYPE_BOOLEAN,
    };

  return gtk_tree_store_newv(TREE_COUNT, types);
}

// boolean = TRUE renders as a checkbox; min/max/relative are unused.
//
// `defval` is the slider's rest position -- what a reset lands on, and the
// value read back as last_value before any shape has answered for the
// property. It MUST lie inside [min, max]: the bauhaus constructor
// normalises it without clamping (d->pos = (defval - min) / (max - min)),
// so a default below the minimum leaves the widget at a negative position
// and last_value below min, and the first drag then applies the whole gap
// as a delta. Every property whose range starts at (or just above) zero
// leaves it 0 by omission, which is what the shipped code passed for all
// of them; only a range that does NOT contain zero has to say so.
const struct
{
  gchar *name;
  gchar *format;
  float min, max;
  gboolean relative;
  gboolean boolean;
  float defval;
  gchar *tooltip;   // NULL: the label says it all
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
      // these two act on the full-resolution pass that computes the mask
      // file, never on the interactive preview: the tooltip is where a
      // user learns that the mask is recomputed when they move
      [ DT_MASKS_PROPERTY_MATTING] = { N_("soft edges for hair and fur"), "", 0, 1, FALSE, TRUE, 0,
        N_("recompute the mask with an alpha matting stage:\n"
           "semi-transparent edges on hair, fur and foliage instead of a hard cut.\n"
           "slower, and it needs the matting model") },
      // the ONE range in this table that does not contain zero, hence the
      // only entry that has to name its default: 1.0 is the calibrated
      // band width, the same number darktableconfig.xml gives the
      // preference this slider edits
      [ DT_MASKS_PROPERTY_MATTING_BAND] = { N_("soft edge width"), "", 0.5, 2.0, FALSE, FALSE, 1.0,
        N_("width of the band along the edge where the matting decides the transparency,\n"
           "relative to the calibrated width: wider for long hair and fur, narrower for clean edges.\n"
           "the mask is recomputed") },
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

  // ... and the hint row, which says where the next drawn shape is going.
  // first of the three: it is what settles whether a creation this panel armed
  // is still in flight, and the two below read that answer
  _creation_bar_update(self);

  // ... and the line above the shape buttons, which says the same thing
  // before anything is armed at all
  _target_row_update(self);

  // ... and the row that says how the next shape will combine once it lands.
  // after _target_row_update on purpose: it reads the very target that call
  // has just written down. last, and it raises or drops the hint row above as
  // its closing act -- that row also stands for an armed operator, and whether
  // one is still armed is settled inside this call
  _arm_bar_update(self);

  // ... and the header eye, which can only take away what is there. the empty
  // group _tree_selection_change() leaves behind is not NULL, so the test is
  // on the points and not on the form. a creation in flight counts too: it
  // holds no point until the first click on the image, and it is exactly what
  // the button drops when no row of ours put anything on screen. the two
  // halves of it are the ones _creation_bar_update() reads
  const dt_masks_form_gui_t *fg = darktable.develop->form_gui;
  const gboolean drawing = fg && (fg->creation || fg->creation_module);
  if(self->bt_hide)
    gtk_widget_set_sensitive(self->bt_hide,
                             (form && form->points != NULL) || drawing);

  // ... and the cleanup, which destroys a shape being drawn:
  // dt_masks_cleanup_unused() opens on dt_masks_change_form_gui(NULL). the
  // same answer, read the other way round
  if(self->bt_cleanup)
    gtk_widget_set_sensitive(self->bt_cleanup, !drawing);
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
   "new mask": one entry point that names where the shape is going.

   a shape created from this panel takes its target from the row that is
   selected, or from the module picked by hand in the target row's menu, and
   goes to the shape library -- upstream's behaviour, a shape linked to no
   module -- when neither said anything. what this panel adds is that the
   answer is written down before anything is drawn, next to the way to change
   it: the line above the shape buttons, the header of the catalogue and the
   hint row all read the same rule.
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

// a raster consumer: a module whose blending takes its mask from another
// module's raster output. the consumer is this panel's analogue of a
// mask, the source its analogue of a shape -- one row per consumer
static gboolean _raster_consumer(const dt_iop_module_t *m)
{
  return m && m->blend_params
    && (m->blend_params->mask_mode & DEVELOP_MASK_ENABLED)
    && (m->blend_params->mask_mode & DEVELOP_MASK_RASTER)
    && m->raster_mask.sink.source != NULL;
}

// what tells a raster row apart in either store: it names a module and
// no form. a mask row carries its group's formid, a library row its
// shape's, and the one root row without a module is the unattached group
static gboolean _raster_row(const dt_iop_module_t *module,
                            const dt_mask_id_t id)
{
  return module != NULL && !dt_is_valid_maskid(id);
}

// wire `target` to take its mask from `source`'s raster output, or, on
// source == NULL, unwire it and leave raster masking altogether. the
// exact sequence of the raster combo (_raster_value_changed_callback)
// and of the automatic path in develop/masks/object.c, announcement of a
// displaced drawn mask included. everything is re-read at click time:
// a menu outlives the pipe under it
static void _raster_set_source(dt_iop_module_t *target,
                               dt_iop_module_t *source,
                               const dt_mask_id_t id)
{
  if(!_mask_target_alive(target) || !_mask_target_has_gui(target))
    return;
  if(source && !_mask_target_alive(source)) return;

  if(target->raster_mask.sink.source)
    g_hash_table_remove
      (target->raster_mask.sink.source->raster_mask.source.users, target);

  dt_develop_blend_params_t *bp = target->blend_params;
  const uint32_t old_mode = bp->mask_mode;
  gboolean reprocess = FALSE;

  target->raster_mask.sink.source = source;
  target->raster_mask.sink.id = source ? id : INVALID_MASKID;

  if(source)
  {
    reprocess = !dt_iop_is_raster_mask_used(source, id);
    // insert and not add: the value is the mask id, as the commit writes
    // it -- readers only trust the keys, but no reason to feed the quirk
    g_hash_table_insert(source->raster_mask.source.users, target,
                        GINT_TO_POINTER(id));
    memcpy(bp->raster_mask_source, source->op,
           sizeof(bp->raster_mask_source));
    bp->raster_mask_instance = source->multi_priority;
    bp->raster_mask_id = id;
    bp->mask_mode = (old_mode & ~DEVELOP_MASK_MASK)
      | DEVELOP_MASK_ENABLED | DEVELOP_MASK_RASTER;
    if(old_mode & DEVELOP_MASK_MASK)
      dt_control_log(_("the drawn mask of %s was replaced by the raster"
                       " mask"), target->name());
  }
  else
  {
    memset(bp->raster_mask_source, 0, sizeof(bp->raster_mask_source));
    bp->raster_mask_instance = 0;
    bp->raster_mask_id = INVALID_MASKID;
    // deliberately not the combo's "no mask used", which stays in raster
    // mode and blends against a mask of zeros: removing the row removes
    // the mode, the conditional blend survives
    bp->mask_mode = old_mode & ~DEVELOP_MASK_RASTER;
  }

  dt_dev_add_history_item(darktable.develop, target, TRUE);
  if(target->gui_data) dt_iop_gui_update(target);
  dt_dev_masks_list_change(darktable.develop);
  if(reprocess)
    dt_dev_reprocess_all(darktable.develop);
}

static void _raster_assign_cb(GtkMenuItem *item, gpointer source)
{
  dt_iop_module_t *target = g_object_get_data(G_OBJECT(item), "target");
  const dt_mask_id_t id =
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "rasterid"));
  _raster_set_source(target, source, id);
}

static void _raster_detach_cb(GtkMenuItem *item, gpointer target)
{
  _raster_set_source(target, NULL, INVALID_MASKID);
}

// one entry per raster mask advertised by a module BEFORE `target` in
// the pipe -- the very walk and the very cut of the blending panel's
// raster combo (_raster_combo_populate): the pipe refuses a source at or
// past its consumer, so none is offered. the one already in use stays,
// greyed, and says so in the label, gtk3 giving insensitive items no
// tooltip
static gboolean _raster_source_items(GtkMenuShell *menu,
                                     dt_iop_module_t *target)
{
  gboolean any = FALSE;

  for(const GList *l = darktable.develop->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    if(m == target) break;

    GHashTableIter miter;
    gpointer key, value;
    g_hash_table_iter_init(&miter, m->raster_mask.source.masks);
    while(g_hash_table_iter_next(&miter, &key, &value))
    {
      const dt_mask_id_t id = GPOINTER_TO_INT(key);
      const gboolean in_use = (m == target->raster_mask.sink.source)
        && (id == target->raster_mask.sink.id);

      gchar *label = in_use
        ? g_strdup_printf("%s (%s)", (const char *)value, _("in use"))
        : g_strdup((const char *)value);
      GtkWidget *item = gtk_menu_item_new_with_label(label);
      g_free(label);

      if(in_use)
        gtk_widget_set_sensitive(item, FALSE);
      else
      {
        g_object_set_data(G_OBJECT(item), "target", target);
        g_object_set_data(G_OBJECT(item), "rasterid", GINT_TO_POINTER(id));
        g_signal_connect(item, "activate",
                         G_CALLBACK(_raster_assign_cb), m);
      }
      gtk_menu_shell_append(menu, item);
      any = TRUE;
    }
  }

  return any;
}

// the module of the row selected in the masks list, if that module is still in
// the pipe and can take a drawn mask -- nothing otherwise. this is the one
// answer that comes from a deliberate gesture: the four rules below it are
// inferences, and the operator bar needs to tell the two apart.
//
// deliberately the masks view and not the active one: TREE_MODULE is only
// written on rows that belong to a module's mask, and the library is flat and
// module-less by construction -- reading its selection here would always
// answer NULL and silently demote rule 1 to rule 2.
//
// TREE_MODULE holds a raw pointer and the store outlives the pipe it was built
// from by one refresh: dt_dev_masks_list_change() only queues the rebuild, and
// this is read on the way there. checked against dev->iop before it is
// dereferenced -- the alive test only compares pointers
static dt_iop_module_t *_mask_selected_target(dt_lib_masks_t *lm)
{
  dt_iop_module_t *module = NULL;
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

  return (_mask_target_alive(module) && _mask_target_ok(module))
    ? module : NULL;
}

// the module picked by hand in the target row's menu, if it still stands:
// same image, still in the pipe, still able to take a drawn mask. the pick is
// forgotten -- not merely ignored -- the moment any of the three fails, so a
// module that comes back to life later does not silently come back as the
// target with it, and so the panel never dereferences a pointer of a pipe
// that has been torn down. the image test is what forgets it on an image
// change: the panel hears of one through the list_change proxy, which lands
// here through _update_all_properties() -> _target_row_update(), before any
// gesture can read the stale pick
static dt_iop_module_t *_mask_picked_target(dt_lib_masks_t *lm)
{
  if(!lm->pick_target) return NULL;

  if(lm->pick_imgid != darktable.develop->image_storage.id
     || !_mask_target_alive(lm->pick_target)
     || !_mask_target_ok(lm->pick_target))
  {
    lm->pick_target = NULL;
    lm->pick_imgid = NO_IMGID;
    return NULL;
  }
  return lm->pick_target;
}

// where a new shape goes, from the two things the photographer actually said
// and nothing else:
//   1. the module of the selected row -- the rule _tree_add_shape has always
//      applied, now said out loud on the target row;
//   2. the module picked by hand in the menu of that row;
//   3. failing both, NULL: the shape library. the shape is saved on its own,
//      linked to no module, as upstream's mask manager has always done it
//      (_tree_add_shape with creation_module = NULL), and it can be sent to
//      a module afterwards from its row's context menu.
// nothing is inferred any more. this used to fall back on the focused module,
// then on the last masked one, then on exposure, then on the last module
// switched on -- and a "select subject" launched from this panel landed on
// exposure without anyone having asked for it. an implicit target is a
// surprise, and a surprise that writes history: the library is the one
// answer that does nothing behind the photographer's back. the target row,
// the catalogue header and the hint row all print this answer before a shape
// is drawn, and the target row's menu is where to change it
static dt_iop_module_t *_mask_default_target(dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;

  dt_iop_module_t *module = _mask_selected_target(lm);
  if(module) return module;

  return _mask_picked_target(lm);
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

// the hint row, first half: the one place a creation in flight is decided,
// reached from _update_all_properties(), so from all four refresh paths of
// this panel. what is armed is read back from form_gui rather than mirrored
// here: the row then promises exactly what the next click on the image will
// do, and the shape code stays free to end a creation without telling us --
// which is what it does.
//
// the row itself is raised or dropped by _arm_bar_update(), which runs after
// this and knows the other half -- an operator armed with nothing being drawn
// yet, which is F2-2's chip and states the same kind of thing
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
  //   where a shape sits between being saved and that chain restarting.
  // a creation armed for the library has only the first half to show:
  // creation_module stays NULL from start to end, which is what makes the
  // shape land in dev->forms alone -- see library_gap below for the moment
  // between two of its shapes
  const gboolean pending = fg && (fg->creation || fg->creation_module);

  // ... and it is ours until someone else names a target, which is what a
  // module's own blending panel does a moment after taking the focus. for a
  // library creation any module named at all is someone else
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

  // the library claim goes down by the same two of those three ways -- there
  // is no module to lose -- plus one moment it has to survive. a module run
  // sits between two shapes with `creation_module` alone set, see above; a
  // library run has no module to leave there, so between the save and the
  // chain restarting form_gui carries neither half. what it does carry, and
  // only there, is the id the save just wrote (dt_masks_gui_form_save_
  // creation sets formid, dt_masks_clear_form_gui() zeroes it) beside a run
  // naming no module. the formid is what tells that gap from a run left
  // standing after the form was dropped: a focus change, a row click or a
  // new image all go through clear_form_gui, and the id goes with them.
  // read under our own claim only -- a run naming no module is also what a
  // run that never started looks like, and the claim is the proof it did
  const gboolean library_gap =
    fg && !fg->creation && !fg->creation_module
    && fg->creation_continuous && !fg->creation_continuous_module
    && dt_is_valid_maskid(fg->formid);

  if(d->arm_library && ((!pending && !library_gap) || stolen))
  {
    if(fg && fg->creation_continuous && !fg->creation_continuous_module)
      _creation_end_continuous();
    d->arm_library = FALSE;
  }

  if(d->arm_module)
  {
    // the module alone, without the "(2 shapes)" the catalogue adds to pick
    // between entries: inside a sentence that annotation reads as something
    // the shape about to be drawn is going to do
    gchar *name = dt_history_item_get_name(d->arm_module);
    // the shape and not "it": the icon this sentence refers back to is a row
    // away, above two lists, and a pronoun that far from its antecedent reads
    // as if the module in brackets were a qualifier of the photograph
    gchar *text = g_strdup_printf(_("draw the shape on the photograph (%s)"),
                                  name);
    gtk_label_set_text(GTK_LABEL(d->creation_label), text);
    // the left panel is narrow, the row already carries a button and the
    // module name is what the sentence ends on: the one word the bar exists
    // to give is the first one the ellipsis takes
    gtk_widget_set_tooltip_text(d->creation_label, text);
    g_free(text);
    g_free(name);
  }
  else if(d->arm_library)
  {
    // the same sentence, the library where the module name goes: the bracket
    // is the one place of the row that says where the shape lands
    const gchar *text = _("draw the shape on the photograph (shape library)");
    gtk_label_set_text(GTK_LABEL(d->creation_label), text);
    gtk_widget_set_tooltip_text(d->creation_label, text);
  }

  // and the six toggles follow the very same answer: lit while this panel's
  // creation is in flight, and only the type actually being drawn. upstream
  // switched all six off by hand from two call sites (_lib_masks_inactivate_
  // icons); this is the one place that knows, so it is the only one that
  // touches them -- and it can also light the armed one, which upstream could
  // not, its buttons being left to flip themselves. no re-entry guard: a dtgtk
  // togglebutton wires "toggled" to gtk_widget_queue_draw and nothing else,
  // its click living in the gesture controller gui_init installs
  const dt_masks_form_t *drawn =
    (d->arm_module || d->arm_library) ? darktable.develop->form_visible : NULL;
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
// this panel: the selection change that moves rule 1, the refresh
// _target_set_cb() asks for when rule 2 moves, and the list rebuild an image
// change goes through, which is where a pick of the previous image is
// forgotten (_mask_picked_target). the focus is no longer part of the rule,
// so the one event this panel cannot hear -- a focus taken while nothing was
// focused -- no longer moves the answer either
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
    // no module was named: the shape goes to the library, on its own. the
    // same arrow, so the line reads the same way whichever answer it gives.
    // the tooltip says both what will happen and the two ways to a module,
    // since this is the state a fresh image opens in
    gtk_label_set_text(GTK_LABEL(d->target_label), _("→ shape library"));
    gtk_widget_set_tooltip_text
      (d->target_label,
       _("the next shape goes to the shape library, linked to no module\n"
         "pick a module here, or right-click the shape afterwards to add it"
         " to a module"));
  }
}

// the operator bar. it is up exactly where arming means something: a mask is
// pointed at -- a row of it is SELECTED, or a shape is being drawn into it
// from this panel. what varies inside the bar is what a mask holding nothing
// yet can take: the first shape lays the base and takes no operator, so two of
// the three buttons are insensitive there rather than the whole row gone.
//
// the selection and not _mask_default_target(): M2 draws this bar only on the
// panel where a mask is selected. a module picked in the target row's menu
// names where the NEXT shape goes, not a mask to combine into -- its group
// may not exist yet -- and the library has no operator to arm at all.
//
// refreshed from _update_all_properties(), so from all four refresh paths of
// this panel, the selection change included -- which is the one that matters
//
// rule 1: the mask a row was deliberately clicked on. rule 2: the creation
// this panel armed -- _start_creation() clears both selections on its way to
// the canvas (change_form_gui -> dt_dev_masks_selection_change), and rule 1
// alone therefore took the bar down at the exact moment its promise was about
// to be kept: the words were on screen everywhere except while the shape was
// being drawn. read after _creation_bar_update(), which is what keeps
// arm_module honest -- see _update_all_properties.
//
// its own function because _bt_arm_cb() has to answer the same question: the
// pair it records is compared against this answer on the very next refresh,
// and a button that resolved the target by rule 1 alone recorded NULL during a
// creation, was found to disagree, and put itself back down inside its own
// click
static dt_iop_module_t *_arm_target_now(dt_lib_masks_t *d)
{
  dt_iop_module_t *target = d->treeview ? _mask_selected_target(d) : NULL;
  if(!target && _mask_target_alive(d->arm_module)) target = d->arm_module;
  return target;
}

static void _arm_bar_update(dt_lib_masks_t *d)
{
  dt_iop_module_t *target = _arm_target_now(d);

  const dt_masks_form_t *grp = target
    ? dt_masks_get_from_id(darktable.develop, target->blend_params->mask_id)
    : NULL;
  // an empty mask takes no operator at all: dt_masks_gui_form_save_creation()
  // only reads the armament inside `if(grp->points)`, the first shape being
  // the base. F2-4 greys the two that cannot work and keeps the bar up; it
  // used to vanish, and a row that disappears teaches nothing
  const gboolean armable = target != NULL;
  const gboolean composable =
    grp && (grp->type & DT_MASKS_GROUP) && grp->points != NULL;

  // the armament belongs to the mask it was chosen for and to no other. it
  // survives the selection going EMPTY, which is not a change of mind but the
  // panel's own creation path: _start_creation() takes the darkroom focus, and
  // dt_masks_reset_form_gui() clears this list before the shape is drawn. it
  // does not survive another mask taking its place, nor the module it was
  // armed for leaving the pipe -- a freed module's address can be handed back
  // to the next one allocated, and the pointer is only ever compared
  if(d->arm_op != DT_MASKS_STATE_NONE
     && ((armable && target != d->arm_target)
         || !_mask_target_alive(d->arm_target)))
  {
    d->arm_op = DT_MASKS_STATE_NONE;
    d->arm_target = NULL;
    dt_masks_set_next_operator(DT_MASKS_STATE_NONE, NULL);
  }

  // sensitivity and not visibility: "add" is what the first shape does anyway,
  // the two others need something already in the mask to work on. the word on
  // the bar is fixed and set once in gui_init -- it names the moment, which
  // does not move, where the mask name it used to carry is already on the
  // highlighted row and on the target line above
  for(int i = 0; i < (int)G_N_ELEMENTS(_arm_operators); i++)
    gtk_widget_set_sensitive
      (d->bt_arm[i],
       composable || _arm_operators[i].state == DT_MASKS_STATE_UNION);

  d->arm_updating = TRUE;
  for(int i = 0; i < (int)G_N_ELEMENTS(_arm_operators); i++)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->bt_arm[i]),
                                 d->arm_op == _arm_operators[i].state);
  d->arm_updating = FALSE;

  gtk_widget_set_visible(d->arm_bar, armable);

  // and the hint row under it, raised here because whether an operator is
  // still armed was settled six lines above. three things it can say, in the
  // order the photographer meets them:
  //   a creation in flight -- the sentence _creation_bar_update() just wrote;
  //   an operator armed and nothing drawn yet -- F2-2 asks for a persistent
  //     chip with a cross rather than a pressed button, since a pressed button
  //     alone never says what to do next, which is the question the three
  //     words on the bar kept raising;
  //   a mask holding nothing -- F2-4 asks for the reason two of the three are
  //     greyed, and a tooltip cannot give it: gtk shows none on an insensitive
  //     widget.
  // the cross goes with the first two only: there is nothing to call off in
  // the third, and a cross beside a plain statement offers to undo the mask.
  // both of the last two are gated on `armable`, so the row never explains a
  // bar that is not up: an armament outlives a cancelled creation on purpose
  // -- see the reset above -- and would otherwise be pointing at three buttons
  // that went away with their target
  const gboolean armed = armable && d->arm_op != DT_MASKS_STATE_NONE;
  if(!d->arm_module && !d->arm_library)
  {
    const gchar *hint = NULL;
    if(armed)
      hint = _("now pick a shape above and draw it");
    else if(armable && !composable)
      hint = _("the first shape lays the base and takes no operator");

    if(hint)
    {
      gtk_label_set_text(GTK_LABEL(d->creation_label), hint);
      gtk_widget_set_tooltip_text(d->creation_label, hint);
    }
    gtk_widget_set_visible(d->bt_creation_cancel, armed);
    gtk_widget_set_visible(d->creation_bar, hint != NULL);
  }
  else
  {
    gtk_widget_set_visible(d->bt_creation_cancel, TRUE);
    gtk_widget_set_visible(d->creation_bar, TRUE);
  }
}

// the glyph on one of the three operator buttons of that bar, drawn and not
// rasterised. a GtkImage built from d->ic_op[] would be frozen in the colour
// gui_init found -- DT_GUI_COLOR_BUTTON_FG, 55 % of the foreground -- so the
// button would go on saying nothing while the word beside it lit, and a theme
// change would move one and not the other. this is dtgtk/button.c's own three
// lines: the widget's style context, the widget's state, the paint function.
// the drawing area inherits the CSS `color` of the button it sits in, so
// :checked reaches it exactly as it reaches the label
static gboolean _arm_glyph_draw(GtkWidget *area,
                                cairo_t *cr,
                                gpointer op_index)
{
  GdkRGBA fg;
  gtk_style_context_get_color(gtk_widget_get_style_context(area),
                              gtk_widget_get_state_flags(area), &fg);
  GtkAllocation alloc;
  gtk_widget_get_allocation(area, &alloc);
  gdk_cairo_set_source_rgba(cr, &fg);
  _masks_operators[GPOINTER_TO_INT(op_index)].paint
    (cr, 0, 0, alloc.width, alloc.height, 0, NULL);
  return TRUE;
}

// one of the three, or the same one again to put it down. M2 draws a cross on
// the lit button; a toggle disarming on a second click is that cross with one
// target fewer, and it is what every other toggle of this panel already does
static void _bt_arm_cb(GtkToggleButton *button, gpointer op)
{
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  dt_lib_masks_t *d = self ? self->data : NULL;
  if(!d || d->arm_updating) return;

  d->arm_op = gtk_toggle_button_get_active(button)
    ? (dt_masks_state_t)GPOINTER_TO_INT(op)
    : DT_MASKS_STATE_NONE;
  // the mask this is being armed for, recorded with it: the funnel that saves
  // a drawn shape is shared with every other way of drawing one, and only the
  // pair says which of them this answers for. the same two rules the bar
  // itself reads, and it has to be the same two -- see _arm_target_now()
  d->arm_target = d->arm_op ? _arm_target_now(d) : NULL;
  dt_masks_set_next_operator(d->arm_op, d->arm_target);
  // the other two go down: one operator is armed at a time, and the bar has to
  // show which
  _arm_bar_update(d);
}

// call it off without having to find the image first. these are the calls a
// right-click on the canvas makes; the disarming itself is left to
// _creation_bar_update(), which change_form_gui() reaches through the
// selection proxy, so the bar goes down in one place whatever took it down
static void _creation_bar_cancel(GtkButton *button, dt_lib_masks_t *d)
{
  dt_iop_module_t *module = d->arm_module;

  // one cross for whichever half of the row is up: an armed operator goes down
  // with it, and it is the only half there is when nothing is being drawn
  d->arm_op = DT_MASKS_STATE_NONE;
  d->arm_target = NULL;
  dt_masks_set_next_operator(DT_MASKS_STATE_NONE, NULL);

  _creation_end_continuous();
  if(_mask_target_alive(module))
  {
    dt_masks_set_edit_mode(module, DT_MASKS_EDIT_FULL);
    dt_masks_iop_update(module);
  }
  else if(module || d->arm_library)
    // a module gone from the pipe, or a shape armed for the library: there is
    // no module whose mask could come back on screen, so the form in flight
    // is simply dropped. the library claim goes down in _creation_bar_update,
    // reached through the selection proxy this call ends on
    dt_masks_change_form_gui(NULL);
  else
    // nothing was being drawn, so there is no form to drop -- and dropping one
    // here would take the shapes a selected row put on the photograph off it,
    // which no cross on this row ever offered to do. the refresh by hand,
    // because nothing else will come: no form changed
    _update_all_properties(d);

  dt_control_queue_redraw_center();
}

// the one creation path of this panel. the catalogue, the tree context menu,
// the six buttons and the shape shortcuts all land here, so all of them put
// a module in listening state the same way -- and all of them save a shape
// for no module the same way. `module` NULL is the shape library: nothing is
// enabled, nothing takes the focus, and the shape is saved on its own by
// dt_masks_gui_form_save_creation(dev, NULL, ...), the path upstream's mask
// manager has always taken. a module that is named but cannot take a drawn
// mask is refused, not demoted to the library: the photographer pointed at it
static gboolean _start_creation(dt_lib_module_t *self,
                                dt_iop_module_t *module,
                                const dt_masks_type_t type,
                                const gboolean continuous)
{
  if(!self) return FALSE;

  // creating a shape calls dt_masks_change_form_gui, which tears down a mask
  // computation in flight without a word -- refuse to start one until it is
  // done, whatever the shape asked for
  if(dt_masks_shapes_locked())
  {
    dt_control_log(_("mask still computing, try again in a moment"));
    return FALSE;
  }

#ifdef HAVE_AI
  if(type == DT_MASKS_OBJECT && !dt_masks_object_available())
  {
    dt_control_log(_("AI model is not available. Check preferences > AI"));
    return FALSE;
  }
#endif

  // the module may have gone away between the click that opened a menu and the
  // click that picked a shape, or be on a raster mask by now
  if(module && (!_mask_target_alive(module) || !_mask_target_ok(module)))
  {
    dt_control_log(_("this module cannot take a drawn shape"));
    return FALSE;
  }

  if(module)
  {
    // the target has to be listening. this is what a module's own shape
    // button does before creating anything, and it is not decoration:
    // nothing in develop/masks/*.c ever writes mask_mode, so a shape
    // attached to a module still on DEVELOP_MASK_DISABLED would be drawn
    // and do nothing. the call takes the focus, sets the mask indicator and
    // adds the history item that enables the module -- all of it inside
    // blend_gui.c, where it belongs
    dt_iop_gui_enable_drawn_mask(module);
    // enable_drawn_mask is a no-op when drawn masking is already on, and the
    // focus has to be taken in that case too
    dt_iop_request_focus(module);
  }
#ifdef HAVE_AI
  else if(type == DT_MASKS_OBJECT)
    // the one shape whose destination is not creation_module: the clicked
    // session closes into the module the canvas hands its handlers, which is
    // dev->gui_module at that moment (masks.c dt_masks_events_button_pressed,
    // object.c's closing gesture). a module left focused would therefore
    // receive a mask the photographer sent to the library. dropping the focus
    // is what object.c's own edit path does before opening a session for no
    // sink, and it is the one focus change this path makes -- the drawn
    // shapes read creation_module and need none of it
    dt_iop_request_focus(NULL);
#endif

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
  // just above, read back by _creation_bar_update(). one claim or the other,
  // never both: a library creation has no module to hold, so it holds a flag
  // the whole refresh and not that call alone: the hint row is raised by
  // _arm_bar_update(), which runs after it, and the change_form_gui() above
  // already went round the proxy while both claims were still unset
  dt_lib_masks_t *d = self->data;
  d->arm_module = module;
  d->arm_library = module == NULL;
  _update_all_properties(d);

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

#ifdef HAVE_AI
  // two behaviours live behind the object button -- the one-shot
  // detections and the clicked session (decision of the UX framing) --
  // and a press cannot say which one it means: the button opens a menu
  // instead, labels distinguishing the two
  if(GPOINTER_TO_INT(shape) == DT_MASKS_OBJECT)
  {
    _object_button_menu(self,
      gtk_event_controller_get_widget(GTK_EVENT_CONTROLLER(gesture)));
    return;
  }
#endif

  // taken or refused, the six buttons end up showing what is armed and
  // nothing else. only the refusal needs the call: _start_creation() refreshes
  // the panel itself on the other path, and has already said why in the same
  // words every other entry point uses
  if(!_start_creation(self, _mask_default_target(self),
                      GPOINTER_TO_INT(shape), continuous))
    _update_all_properties(self->data);
}

// menu-item adapter. the catalogue and the context menu write the module they
// mean on the item; an item without one -- which is also what an item FOR the
// library carries, a NULL being unset -- falls back to the default rule, and
// that rule answers the library when nothing was pointed at
static void _tree_add_shape(GtkWidget *widget, gpointer shape)
{
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  if(!self) return;

  dt_iop_module_t *module = g_object_get_data(G_OBJECT(widget), "target");
  if(!module) module = _mask_default_target(self);

  _start_creation(self, module, GPOINTER_TO_INT(shape), FALSE);
}

#ifdef HAVE_AI
// activate an automatic-selection entry: resolve the target the item
// carries (falling back like _tree_add_shape) and hand it to the one-shot
// detection job. the entry was sensitive, so the state allowed it -- but
// the registry and the pipe may have moved between the menu and the
// click: the guards of _start_creation, in its order and with its words,
// every one of them BEFORE the target is touched. the target is then put
// in listening state exactly as _start_creation puts it -- the apply
// idle rewrites mask_mode when the shape lands, but the job's result
// must land on a module that renders it, visibly armed for the wait.
// no target is the library, as everywhere in this panel: the job is
// launched for no module and the apply idle files the raster shape on
// its own, selected in the library so it shows on the photograph
static void _detect_activate(GtkWidget *widget, gpointer detector)
{
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  if(!self) return;

  // enable_drawn_mask below takes the focus, and a focus change tears
  // down a mask computation in flight without a word -- the very guard
  // _start_creation opens with
  if(dt_masks_shapes_locked())
  {
    dt_control_log(_("mask still computing, try again in a moment"));
    return;
  }

  // the launch re-derives this state and refuses on its own; asking
  // first keeps that refusal ahead of the module mutations below
  const dt_masks_object_detect_state_t state
    = dt_masks_object_detect_state(detector);
  if(state != DT_MASKS_OBJECT_DETECT_READY
     && state != DT_MASKS_OBJECT_DETECT_DOWNLOAD)
  {
    dt_control_log(_("AI model is not available. Check preferences > AI"));
    return;
  }

  dt_iop_module_t *module = g_object_get_data(G_OBJECT(widget), "target");
  if(!module) module = _mask_default_target(self);

  if(module && (!_mask_target_alive(module) || !_mask_target_ok(module)))
  {
    dt_control_log(_("this module cannot take a drawn shape"));
    return;
  }

  // before the launch, deliberately: the launch flushes the history and
  // captures the distort hash, so the item that enables the module must
  // exist by then -- reversed, arming a disabled distort module
  // (liquify) would change the geometry after the capture and the apply
  // would discard the finished mask as a geometry change. nothing of it
  // for the library: no module is enabled and none takes the focus
  if(module)
  {
    dt_iop_gui_enable_drawn_mask(module);
    dt_iop_request_focus(module);
  }

  dt_masks_object_detect_launch(detector, module);
}

// one entry per row of the detector table. what a click can do RIGHT NOW
// is in the label: gtk3 delivers no tooltip to an insensitive item, so
// the reason an entry cannot detect is part of it -- the convention of
// every refusal in this panel's menus. READY detects; DOWNLOAD stays
// sensitive too, because the click can act -- it downloads the model and
// the detection follows on the same job; everything else states why it
// is inert. `target` NULL is the library, and the entries are built for
// it all the same: the NULL rides on the item like a module would
static gboolean _detect_menu_items(GtkMenuShell *menu,
                                   dt_iop_module_t *target)
{
  gboolean any = FALSE;
  for(size_t i = 0; i < G_N_ELEMENTS(dt_detectors); i++)
  {
    const dt_detector_t *det = &dt_detectors[i];
    const char *reason = NULL;
    gboolean sensitive = FALSE;
    switch(dt_masks_object_detect_state(det))
    {
      case DT_MASKS_OBJECT_DETECT_READY:
        sensitive = TRUE;
        break;
      case DT_MASKS_OBJECT_DETECT_DOWNLOAD:
        reason = _("downloads the model");
        sensitive = TRUE;
        break;
      case DT_MASKS_OBJECT_DETECT_DOWNLOADING:
        reason = _("downloading the model...");
        break;
      case DT_MASKS_OBJECT_DETECT_AI_OFF:
        reason = _("AI disabled in preferences");
        break;
      default:
        reason = _("model not installed");
    }

    // the glyph of the detector's table row heads the label: plain
    // text, so it greys with an insensitive entry and follows the
    // theme -- these menus carry no image icons
    gchar *name = g_strdup_printf(_("select %s"), _(det->label));
    gchar *label = reason
      ? g_strdup_printf("%s %s (%s)", det->glyph, name, reason)
      : g_strdup_printf("%s %s", det->glyph, name);
    GtkWidget *item = gtk_menu_item_new_with_label(label);
    g_free(label);
    g_free(name);

    gtk_widget_set_sensitive(item, sensitive);
    g_object_set_data(G_OBJECT(item), "target", target);
    g_signal_connect(item, "activate", G_CALLBACK(_detect_activate),
                     (gpointer)det);
    gtk_menu_shell_append(menu, item);
    any = TRUE;
  }
  return any;
}
#endif

// one row per shape type, wired to the callback the context menu and the shape
// shortcuts already use. the module travels on the item, so the same function
// fills the top level of the catalogue, every per-module submenu, and the
// context menu
static void _new_mask_shape_items(GtkMenuShell *menu, dt_iop_module_t *target)
{
  // `target` NULL is the library: the same list, the drawn shape saved on
  // its own. what is written per entry below really is per entry -- one
  // model that is not installed

#ifdef HAVE_AI
  // the automatic selections head the catalogue (M2.1): a detection is
  // picked like a shape, minus the drawing, and it targets the very
  // module this list hangs from. the heading is an insensitive item --
  // these menus have no header widget, and a separator alone would not
  // name what follows
  GtkWidget *dhead = gtk_menu_item_new_with_label(_("automatic selection"));
  gtk_widget_set_sensitive(dhead, FALSE);
  gtk_menu_shell_append(menu, dhead);
  _detect_menu_items(menu, target);
  gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
#endif

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
    const gchar *glyph = "";

#ifdef HAVE_AI
    if(_new_mask_shapes[i].type == DT_MASKS_OBJECT)
    {
      // the ✦ of the automatic entries, on this one too: not a
      // detectors row, but the same AI family -- and at the menu
      // site, not in the table, whose label also names the panel
      // button and its tooltip
      glyph = "✦ ";
      if(!dt_masks_object_available())
        reason = _("AI model not available");
    }
#endif

    // gtk3 delivers no event, hence no tooltip, to an insensitive widget:
    // whatever an entry cannot do has to be readable in the entry itself
    gchar *label = reason
      ? g_strdup_printf("%s%s (%s)", glyph,
                        _(_new_mask_shapes[i].label), reason)
      : g_strdup_printf("%s%s", glyph, _(_new_mask_shapes[i].label));

    GtkWidget *item = gtk_menu_item_new_with_label(label);
    g_free(label);

    gtk_widget_set_sensitive(item, reason == NULL);
    g_object_set_data(G_OBJECT(item), "target", target);
    g_signal_connect(item, "activate", G_CALLBACK(_tree_add_shape),
                     GINT_TO_POINTER(_new_mask_shapes[i].type));
    gtk_menu_shell_append(menu, item);
  }

  // the raster masks, in the same catalogue: assigning one is choosing
  // a mask for the target, minus the drawing. only sources before the
  // target exist for the pipe, so only they appear -- and none, no
  // section, like the missing ctrl lines: absence over noise. no section
  // either for the library: a raster mask is wired INTO a module's
  // blending, there is nothing to wire it into here
  GtkWidget *rsub = gtk_menu_new();
  if(target && _raster_source_items(GTK_MENU_SHELL(rsub), target))
  {
    gtk_menu_shell_append(menu, gtk_separator_menu_item_new());
    GtkWidget *ritem = gtk_menu_item_new_with_label(_("use a raster mask"));
    gtk_menu_item_set_submenu(GTK_MENU_ITEM(ritem), rsub);
    gtk_menu_shell_append(menu, ritem);
  }
  else
  {
    g_object_ref_sink(rsub);
    g_object_unref(rsub);
  }
}

#ifdef HAVE_AI
// the AI menu: the automatic selections and the clicked session side by
// side -- the two behaviours the ✦ button stands for, distinguished by
// their labels since an icon cannot say which one a press means. `target`
// is where whatever is picked lands, NULL being the library; the entries
// carry it exactly as the catalogue's do, and the clicked entry rides
// _tree_add_shape, so every guard of _start_creation applies unchanged.
//
// the ✦ of a module's own blending panel opens this very menu through
// dev->proxy.masks.object_menu, with that module for target: one menu, one
// set of words, whichever of the two buttons was pressed
static void _object_menu_popup(dt_lib_module_t *self,
                               dt_iop_module_t *target,
                               GtkWidget *anchor)
{
  if(!self) return;

  GtkMenuShell *menu = GTK_MENU_SHELL(gtk_menu_new());
  _detect_menu_items(menu, target);
  gtk_menu_shell_append(menu, gtk_separator_menu_item_new());

  // the interactive session, under its own name -- the state convention
  // of the shape entries: the reason lives in the label. the ✦ of the
  // automatic entries is hardcoded here: not a detectors row, but the
  // same AI family, and the menu reads as one section
  const gboolean session_ok = dt_masks_object_available();
  gchar *label = session_ok
    ? g_strdup_printf("✦ %s", _("select by clicking"))
    : g_strdup_printf("✦ %s (%s)", _("select by clicking"),
                      _("AI model not available"));
  GtkWidget *item = gtk_menu_item_new_with_label(label);
  g_free(label);
  gtk_widget_set_sensitive(item, session_ok);
  g_object_set_data(G_OBJECT(item), "target", target);
  g_signal_connect(item, "activate", G_CALLBACK(_tree_add_shape),
                   GINT_TO_POINTER(DT_MASKS_OBJECT));
  gtk_menu_shell_append(menu, item);

  // dt_gui_menu_popup takes the floating ref and drops it on "deactivate"
  dt_gui_menu_popup(GTK_MENU(menu), anchor,
                    GDK_GRAVITY_SOUTH_WEST, GDK_GRAVITY_NORTH_WEST);
}

// the panel's own ✦: the menu above, for where the next shape goes
static void _object_button_menu(dt_lib_module_t *self, GtkWidget *button)
{
  _object_menu_popup(self, _mask_default_target(self), button);
}

// ... and a module's ✦, through the proxy: the same menu for that module.
// the module is checked the way every entry point of this panel checks the
// module it is handed -- blend_gui.c only calls with its own live module,
// but a proxy is a public door and the guard costs one list walk
static void _object_menu_proxy(dt_lib_module_t *self,
                               dt_iop_module_t *module,
                               GtkWidget *anchor)
{
  if(!_mask_target_alive(module) || !_mask_target_ok(module))
  {
    // a module on a raster mask, reached through its shortcut: the
    // blending panel hides the button, and _blendop_masks_modes_toggle()
    // would refuse the switch to drawn masking. said, not swallowed
    dt_control_log(_("this module cannot take a drawn shape"));
    return;
  }
  _object_menu_popup(self, module, anchor);
}
#endif

// picking an entry from the "..." of the target row: it says where the next
// shape goes, and stops there. the pick is written down, rule 2 of
// _mask_default_target(), and nothing else moves: no focus is taken -- the
// focus used to BE the answer, and a module focused for other reasons then
// became the target without a word -- and no form is dropped, since what a
// selected row put on the photograph is not this line's to take away.
//
// rule 1, the module of a selected mask row, still outranks the pick: with a
// row selected the line would keep answering for that row and the menu would
// look inert, so the selection is cleared through the two lists first --
// _tree_selection_change() then rebuilds the canvas from an empty selection,
// which is the one thing this does take off the photograph, and it is what
// the photographer asked for by naming another destination. `module` NULL is
// the library entry
static void _target_set_cb(GtkMenuItem *item, gpointer module)
{
  dt_lib_module_t *self = darktable.develop->proxy.masks.module;
  dt_lib_masks_t *d = self ? self->data : NULL;
  if(!d) return;

  // a menu outlives nothing, but the pipe under it can be rebuilt while it is
  // open -- same guard _start_creation() puts on the module it is handed
  if(module && !_mask_target_alive(module)) return;

  d->pick_target = module;
  d->pick_imgid = module ? darktable.develop->image_storage.id : NO_IMGID;

  gboolean cleared = FALSE;
  for(int v = 0; v < DT_MASKS_NVIEWS; v++)
  {
    GtkWidget *view = _masks_view(d, v);
    GtkTreeSelection *sel =
      view ? gtk_tree_view_get_selection(GTK_TREE_VIEW(view)) : NULL;
    if(sel && gtk_tree_selection_count_selected_rows(sel) > 0)
    {
      cleared = TRUE;
      gtk_tree_selection_unselect_all(sel);
    }
  }

  // the unselect above refreshed the panel on its way through
  // _tree_selection_change(); with nothing selected nothing did, and the
  // target row has to be told
  if(!cleared) _update_all_properties(d);
}

// send an existing shape to a module, wherever the gesture came from: the
// context menu of a library row now, the drop of a library drag later. the
// engine creates the module's mask when it has none, the module is switched
// to drawn masking first -- nothing in develop/masks/*.c ever writes
// mask_mode, see _start_creation -- and a refused add writes no history
static gboolean _shape_add_to_module(dt_iop_module_t *module,
                                     const dt_mask_id_t formid)
{
  // the pipe can be rebuilt between the gesture that opened the menu and the
  // click that lands here -- same guard as _start_creation
  if(!_mask_target_alive(module) || !_mask_target_ok(module)) return FALSE;

  dt_iop_gui_enable_drawn_mask(module);

  if(!dt_masks_iop_add_exist(module, formid)) return FALSE;

  // reselect the mask that took the shape once the lists are rebuilt, the
  // way _tree_add_exist follows its own add
  dt_dev_masks_selection_change(darktable.develop, NULL,
                                module->blend_params->mask_id);
  return TRUE;
}

// menu-item adapter: the module is the callback argument, the shape rides on
// the item as an id -- never a form pointer, the menu outlives the pipe
static void _add_to_module_cb(GtkMenuItem *item, gpointer module)
{
  _shape_add_to_module(module,
    GPOINTER_TO_INT(g_object_get_data(G_OBJECT(item), "formid")));
}

// every module that could take the shape. `flat` picks which of the two menus
// this is: in the catalogue each entry carries the same list of types one
// level down, so picking there picks the target AND the shape in one gesture;
// under the "..." of the target row there is no such list, because the six
// shapes are the row 20 px underneath and a menu repeating them would be a
// second, slower copy of it. `current` is the module to leave out: it is the
// one already named on the line this list hangs from
// ... and `add_formid`, when valid, turns the list into its third job:
// activating an entry adds THAT shape to the module -- the mirror, seen from
// a shape, of the group rows' "add existing shape"
static gboolean _new_mask_other_modules(GtkMenuShell *menu,
                                        const dt_iop_module_t *current,
                                        const gboolean flat,
                                        const dt_mask_id_t add_formid)
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

      // does this module's mask already hold the shape on offer? direct
      // membership only, never dt_masks_is_in_module(): a shape filed in a
      // sub-group is another row and a legitimate second add
      const gboolean has_it = dt_is_valid_maskid(add_formid)
        && _group_point_index(dt_masks_get_from_id(darktable.develop,
                                                   m->blend_params->mask_id),
                              add_formid) >= 0;

      gchar *label = _mask_target_label(m);
      if(has_it)
      {
        // gtk3 delivers no tooltip to an insensitive item: the reason lives
        // in the label, like every other refusal of this menu
        gchar *said = g_strdup_printf("%s (%s)", label,
                                      _("already has this shape"));
        g_free(label);
        label = said;
      }
      GtkWidget *item = gtk_menu_item_new_with_label(label);
      g_free(label);

      if(!_mask_target_ok(m) || has_it)
        // on a raster mask: _blendop_masks_modes_toggle() would refuse the
        // switch, so the entry stays and says so rather than disappearing.
        // a mask already holding the shape follows the same rule
        gtk_widget_set_sensitive(item, FALSE);
      else if(dt_is_valid_maskid(add_formid))
      {
        g_object_set_data(G_OBJECT(item), "formid",
                          GUINT_TO_POINTER(add_formid));
        g_signal_connect(item, "activate",
                         G_CALLBACK(_add_to_module_cb), m);
      }
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
  // the library is named the way the target row names it: the arrow and
  // the same two words, so the menu and the line above the buttons cannot
  // be read as two different places. the arrow already says "goes to", so
  // it takes no "apply to" in front
  gchar *header;
  if(target)
  {
    gchar *name = _mask_target_label(target);
    header = g_strdup_printf(_("apply to: %s"), name);
    g_free(name);
  }
  else
    header = g_strdup(_("→ shape library"));

  GtkWidget *item = gtk_menu_item_new_with_label(header);
  g_free(header);

  // the modules, each carrying the shapes one level down. the library needs
  // no entry of its own here: the shapes at the top level of this very menu
  // are already the library's when the header says so, and with a module
  // named, the "..." of the target row is where the library is picked
  GtkWidget *others = gtk_menu_new();
  if(_new_mask_other_modules(GTK_MENU_SHELL(others), target, FALSE,
                             INVALID_MASKID))
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
  // NULL is the library, and the menu opens on it like on any module: this
  // is the button of the empty state, an image just opened, and the library
  // is exactly where a shape drawn before any module was named belongs
  dt_iop_module_t *target = _mask_default_target(self);

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

// the "..." of the target row: send the next shape somewhere else. the
// library first -- the one destination that is not a module, and the way
// back from a picked module -- then the modules and only the modules, one
// level, no shapes: the row of six sits right under the line this hangs
// from, so an entry per shape here would be the same targets reached the
// slow way. picking an entry is the whole gesture, and the shape after it is
// one click on the row.
//
// the entry already on the line is greyed rather than left out -- the
// convention _new_mask_other_modules() applies to an entry that cannot act,
// the reason being readable in the menu itself. it leaves the modules out
// instead, being shared with the catalogue, where the current one heads the
// menu; here the header is the line above, so the list is complete.
//
// built at click for the same reason the catalogue is: the pipe moves, and
// the module list is read from it
static void _target_menu_clicked(GtkButton *button, dt_lib_module_t *self)
{
  const dt_iop_module_t *target = _mask_default_target(self);

  GtkMenuShell *menu = GTK_MENU_SHELL(gtk_menu_new());

  GtkWidget *item =
    gtk_menu_item_new_with_label(_("shape library (no module)"));
  if(target)
    g_signal_connect(item, "activate", G_CALLBACK(_target_set_cb), NULL);
  else
    gtk_widget_set_sensitive(item, FALSE);
  gtk_menu_shell_append(menu, item);
  gtk_menu_shell_append(menu, gtk_separator_menu_item_new());

  if(!_new_mask_other_modules(menu, target, TRUE, INVALID_MASKID) && !target)
  {
    // the library is on the line and not one module of the pipe can hold a
    // drawn mask: a menu with one greyed entry is nothing to open. sink the
    // floating reference first, as dt_gui_menu_popup would have
    g_object_ref_sink(menu);
    g_object_unref(menu);
    dt_control_log(_("no module of this image can take a shape"));
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

// which kind an icon should show for a row, -1 for none.
//
// a shape answers for itself. a group answers when no member disagrees -- a
// mask made of three circles IS a circle mask, and "no member disagrees" is
// the one count at which the icon cannot start lying. read recursively, so a
// module's mask holding one AI object resolves through that object's group
// down to its paths, which is the row M2 draws. read from grp->points and not
// from the row's children: on a first pass the row is being built and has none.
// the old rule was "exactly one member", which left a group of two -- an AI
// object and its hole -- with no icon at all.
//
// an AI object first, and by its DEFAULT NAME, which is the only mark it has.
// develop/masks/object.c destroys the object shape at finalisation and files
// plain paths, so nothing in the data says where they came from; the name is
// the mark object.c itself reads back to number the next one, so the panel and
// the producer are right together or wrong together. renamed by hand, or
// written in a locale other than the one running, the group falls back to the
// kind of its members -- a path, which is what it is, and not a lie. saying it
// durably needs DT_MASKS_OBJECT on the group's type, i.e. a new bit in
// main.masks_history: deliberately not done here.
// the depth cap is insurance, not a case: nothing builds a cycle, and this now
// runs on every row of every refresh
static int _row_type_index(const dt_masks_form_t *form, const int depth)
{
  if(!form || depth > 4) return -1;
  if(!(form->type & DT_MASKS_GROUP)) return _type_index(form->type);

#ifdef HAVE_AI
  const char *ai = _("ai object group");
  if(!strncmp(form->name, ai, strlen(ai)))
    return _type_index(DT_MASKS_OBJECT);
#endif

  int ty = -1;
  for(const GList *l = form->points; l; l = g_list_next(l))
  {
    const dt_masks_point_group_t *pt = l->data;
    const int t = _row_type_index
      (dt_masks_get_from_id(darktable.develop, pt->formid), depth + 1);
    if(t < 0 || (ty >= 0 && t != ty)) return -1;
    ty = t;
  }

  return ty;
}

static void _set_iter_name(dt_lib_masks_t *lm,
                           dt_masks_form_t *form,
                           const int state,
                           const float opacity,
                           GtkTreeModel *model,
                           GtkTreeIter *iter)
{
  if(!form)
  {
    // a raster row: no form anywhere, the consumer module carries
    // everything. one writer for the derived columns, this function, on
    // these rows too -- which is what lets _update_foreach refresh them
    // in place when the consumer or its source is switched
    dt_iop_module_t *rmod = NULL;
    dt_mask_id_t rid = INVALID_MASKID;
    _lib_masks_get_values(model, iter, &rmod, NULL, &rid);
    if(!_raster_row(rmod, rid) || !_mask_target_alive(rmod)
       || !_raster_consumer(rmod))
      return;
    dt_iop_module_t *src = rmod->raster_mask.sink.source;
    if(!_mask_target_alive(src)) return;

    // the name is exactly what a rename writes back: the source
    // instance's own name, or its module name while it has none -- the
    // prefill rule of the module header's rename entry
    gchar *rname = (*src->multi_name && strcmp(src->multi_name, "0"))
      ? dt_util_localize_segmented_name(src->multi_name, FALSE)
      : g_strdup(src->name());

    gchar *cname = dt_history_item_get_name(rmod);
    gchar *rtarget = g_strdup_printf("→ %s", cname);
    g_free(cname);

    // the row tooltip is derived from the source's current name, so it
    // is written here with the other derived columns: an in-place
    // refresh after a rename then updates it along with the name,
    // instead of leaving it stale until the next full rebuild
    gchar *sname = dt_history_item_get_name(src);
    gchar *rtip = g_strdup_printf
      (_("raster mask: pixels handed over by '%s'\n"
         "it has no shapes to edit on the photograph\n"
         "right-click to change or remove the link"), sname);
    g_free(sname);

    // the three states that silently break a raster wiring, said on the
    // row instead of in a toast that is gone: the pipe refuses a source
    // at or past its consumer and blends against zeros, a source
    // switched off writes no mask at all, and a source that stopped
    // advertising the wanted mask -- turned raster consumer itself, or
    // stripped of its blending -- has nothing to hand over either
    const char *rlink = "";
    if(src->iop_order >= rmod->iop_order)
      rlink = _("source later in pipe");
    else if(!src->enabled)
      rlink = _("source is off");
    else if(!g_hash_table_contains(src->raster_mask.source.masks,
                                   GINT_TO_POINTER(rmod->raster_mask.sink.id)))
      rlink = _("source provides no mask");

    const gboolean rmoff = !rmod->enabled;
    const gboolean rpower = rmod->off != NULL
      && !rmod->hide_enable_button;
    // develop/blend.c honours the show request of a focused raster
    // consumer (request_raster_display), so the cell works unchanged
    const gboolean rshow = rmod->blend_data != NULL;
    const gboolean rshow_on = rshow
      && rmod->enabled
      && dt_iop_has_focus(rmod)
      && (rmod->request_mask_display & DT_DEV_PIXELPIPE_DISPLAY_MASK);

    GdkPixbuf *ricinv = rmod->blend_params->raster_mask_invert
      ? lm->ic_inverse : NULL;

    gtk_tree_store_set(GTK_TREE_STORE(model), iter,
                       TREE_TEXT, rname,
                       TREE_OPACITY, "",
                       TREE_NUM, "",
                       TREE_LINK, rlink,
                       TREE_IC_OP, NULL,
                       TREE_IC_OP_VISIBLE, FALSE,
                       TREE_IC_INVERSE, ricinv,
                       TREE_IC_INVERSE_VISIBLE, (ricinv != NULL),
                       TREE_IC_TYPE, lm->ic_raster,
                       TREE_IC_TYPE_VISIBLE, TRUE,
                       TREE_TARGET, rtarget,
                       TREE_USED_TEXT, rtip,
                       TREE_MODULE_OFF, rmoff,
                       TREE_MODULE_ON, !rmoff,
                       TREE_POWER, rpower,
                       TREE_SHOW, rshow,
                       TREE_SHOW_ON, rshow_on,
                       -1);
    g_free(rtip);
    g_free(rtarget);
    g_free(rname);
    return;
  }

  // TREE_TEXT is exactly form->name, on every row and with nothing appended:
  // _tree_cell_edited copies the displayed string straight back into
  // form->name, so whatever else is shown here is what a rename would write
  // into the name. that used to hold by coincidence only -- the "%" suffix was
  // built right here, and TREE_EDITABLE being (grp_id == 0) simply never
  // coincided with a row carrying one. the rank, the kind and the opacity
  // live in columns of their own
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
  // the base is not tied to that: group.c forbids an operator on index 0 of
  // *every* group and the context menu greys the five "mode:" entries
  // accordingly, so the tooltip that says why has to answer wherever that
  // rule bites -- otherwise the menu is disabled without saying anything.
  // that tooltip is _tree_query_tooltip()'s DT_MASKS_OP_HIT_BASE case, which
  // reads the rank back out of the group and needs no column of its own.
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

  if(dt_is_valid_maskid(grid))
  {
    rank =
      _group_point_index(dt_masks_get_from_id(darktable.develop, grid), id);

    // the base carries no operator and cannot be given one. nothing is
    // written in its slot for it: the strip is still its zone, the hand
    // cursor still stays away from it, and the tooltip _tree_query_tooltip()
    // already puts there says why the five "mode:" entries are greyed
    if(rank >= 0 && module) snprintf(num, sizeof(num), "%d", rank + 1);
  }

  // a row hanging from no group. two kinds reach this column and they are told
  // apart below, because what they have to say is not the same thing.
  //
  // a library row first -- the only row in either store that is a shape with
  // no parent group, and the only place a shape exists as itself. two words
  // where the "used" badge was one mark: that badge was
  // dtgtk_cairo_paint_masks_used, a ring with a stem down from the top of it,
  // a few rows under dtgtk_cairo_paint_switch, which is a ring with a stem
  // down from the top of it. words also split the two negative states the
  // badge folded together and leave the ordinary one -- a module renders this
  // shape -- silent, as it should be. "unused" is exact on this row and still
  // forbidden on the caption under the list, which covers both states at once
  // (see _shape_scope).
  // the walk this costs is paid on library rows only, over a list a human drew
  const char *link = "";

  if(!dt_is_valid_maskid(grid))
  {
    if(form->type & DT_MASKS_GROUP)
    {
      // ... and a group at the root of the masks list that no module wears.
      // deliberately NOT the library's "no module", which says "some group
      // still holds this shape, but nothing renders it": read on this row it
      // would announce a mask whose module the panel had failed to name. this
      // row is not a mask at all, it is a group that belongs to nothing --
      // "group the forms" makes one on purpose, a module taken out of the pipe
      // leaves one behind -- and the file already has the word for that state,
      // in the TREE_MODULE_OFF rule below.
      // TREE_MODULE and not `live`: an empty pointer is gui_update saying no
      // module holds this group, a stale one is the pipe having moved under
      // the store, and the second must not be reported as the first
      if(!module) link = _("unattached");
    }
    else
    {
      const dt_masks_shape_scope_t scope = _shape_scope(form->formid, NULL, 0);
      if(scope == DT_MASKS_SCOPE_GROUP_ONLY) link = _("no module");
      else if(scope == DT_MASKS_SCOPE_ORPHAN) link = _("unused");
    }
  }

  // a raster shape whose backing file is gone renders as an absent group
  // member without a word on canvas -- the render skips it and, when the
  // recipe allows, repairs it in the background -- so the row is the one
  // surface that says it. overrides the library words above: a shape
  // that cannot render outranks a shape nothing renders. one stat per
  // row refresh, and the refresh-in-place path updates it like every
  // other derived column
  if(form->type & DT_MASKS_RASTER)
  {
    const dt_masks_point_raster_t *rpt = dt_masks_raster_point(form);
    gchar *rpath = rpt
      ? dt_masks_raster_resolve_path(rpt, &darktable.develop->image_storage)
      : NULL;
    if(!rpath || !g_file_test(rpath, G_FILE_TEST_EXISTS))
    {
#ifdef HAVE_AI
      // a valid recipe means the render path schedules the recompute on
      // its own -- but only when the replay's model gates would pass:
      // "recomputing" must not promise what the replay refuses, so the
      // row states the model-gap verdict instead. the diagnostic is the
      // exact mirror of those gates, promptless recipes included, and it
      // is never cached: installs and rebinds move it between refreshes
      if(rpt && dt_rf_recipe_valid(&rpt->recipe))
      {
        // the matting stage is a build capability, not a model: asked
        // before the model gap because that verdict would answer OK --
        // "file missing, recomputing" -- for a replay whose matting gate
        // refuses, and the anti-respawn table then pins the failure for
        // the session. the row would promise a recompute that never comes
        if(!dt_object_recipe_matting_reproducible(&rpt->recipe))
          link = _("file missing, matting stage not supported");
        else switch(dt_object_recipe_model_gap(&rpt->recipe, NULL))
        {
          case DT_OBJECT_RECIPE_MODELS_OK:
            // the render path schedules the repair only when a module
            // renders the shape: a shape sitting in the library, or in a
            // group no module wears, is never asked for and would say
            // "recomputing" forever. ask from here then -- the schedule
            // is idempotent (anti-respawn table), and it declines while
            // a finalisation legitimately holds the file absent. the row
            // keeps its word: a recompute really is on its way
            if(_shape_scope(form->formid, NULL, 0) != DT_MASKS_SCOPE_MODULE
               && !dt_object_mask_finalize_running())
              dt_object_recipe_schedule_recompute
                (&rpt->recipe, darktable.develop->image_storage.id);
            link = _("file missing, recomputing");
            break;
          case DT_OBJECT_RECIPE_MODELS_INSTALLABLE:
          case DT_OBJECT_RECIPE_MODELS_DRIFT_BEHIND:
            link = _("file missing, model download needed");
            break;
          case DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD:
            link = _("file missing, recorded model changed");
            break;
          case DT_OBJECT_RECIPE_MODELS_AI_OFF:
            link = _("file missing, AI disabled");
            break;
          default:
            link = _("file missing, model unknown");
        }
      }
      else
        link = _("file missing");
#else
      // without the AI subsystem no recipe can be replayed here
      link = _("file missing");
#endif
    }
    g_free(rpath);
  }

  // M2: "what is this group for?" answered on the row that raises the
  // question. the mask rows only -- which is a root row carrying a module:
  // grp_id is 0 there, and a library row carries no module at all.
  // dt_history_item_get_name and the same "→ %s" the target line at the top
  // of the panel is built from: one says where a mask lives, the other where
  // the next shape goes, and they have to read as the same statement
  gchar *target = NULL;

  // ... and not on a mask that has never been renamed. the name a module
  // writes into its own mask already IS the module's name -- `group
  // `exposure'' -- so the row stated it twice and cut the name short to fit
  // the repeat. asked of develop/masks.c, which composes that name, rather
  // than matched here against a copy of its format. rename the mask and the
  // arrow comes back, which is precisely the row M2 draws: "sky -> exposure"
  if(live && !dt_is_valid_maskid(grid)
     && !dt_masks_group_name_is_default(form, live))
  {
    gchar *mname = dt_history_item_get_name(live);
    target = g_strdup_printf("→ %s", mname);
    g_free(mname);
  }

  // M2 note 7: a mask whose module is switched off is struck through, and the
  // kind icon beside its name steps back with it. the whole mask and not just
  // its top row -- the shapes under it carry the same module, and nothing
  // under an off module renders. a row with no module at all (a stand-alone
  // group, every library row) is not off, it is unattached: full contrast
  const gboolean moff = (live != NULL) && !live->enabled;

  // ... and M2's power switch, on the mask rows only: a shape inside a mask
  // does not own the module, the mask does -- the same rule TREE_TARGET above
  // follows, and grp_id is what tells the two apart. a module whose switch is
  // hidden (hide_enable_button) gets no cell: dt_iop_gui_init() makes its own
  // button insensitive, and _enable_module_callback() refuses outright, so a
  // cell here would be the one control in darktable that pretends otherwise
  const gboolean power = (live != NULL)
    && !dt_is_valid_maskid(grid)
    && live->off != NULL
    && !live->hide_enable_button;

  // M2: see this module's mask filled over the photograph. the mask rows only,
  // like the switch beside it, and only where there is a drawn mask to fill.
  // NOT conditioned on the module being switched on: M2 note 7 steps the whole
  // row back when it is, it does not empty it, and darktable itself does the
  // same with the module's own indicator -- dt_iop_gui_set_enable_button()
  // makes it insensitive, never absent. spelled out rather than derived from
  // `power` above: the two cells sit side by side but a hidden enable button
  // says nothing about whether a mask can be shown. blend_data is what the
  // public call needs
  const gboolean show = (live != NULL)
    && !dt_is_valid_maskid(grid)
    && live->blend_data != NULL
    && (live->blend_params->mask_mode & DEVELOP_MASK_MASK);
  // ... and whether it is the one on screen. word for word the test
  // develop/blend.c makes before it honours the request (`valid_request`):
  // the focus included, so a request left standing on a module that no longer
  // has it cannot light a row. that is what makes the cell a radio -- at most
  // one row lit, because at most one module has the focus -- without this
  // panel having to enforce one
  const gboolean show_on = show
    && live->enabled
    && dt_iop_has_focus(live)
    && (live->request_mask_display & DT_DEV_PIXELPIPE_DISPLAY_MASK);

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

  // M2.2: every shape says what it is, and a group says it when its content
  // agrees on one answer -- the whole of the rule is _row_type_index() above
  const int ty = _row_type_index(form, 0);
  GdkPixbuf *ictype = (ty >= 0) ? lm->ic_type[ty] : NULL;
  // a raster shape is none of the drawable kinds (it deliberately has no
  // _new_mask_shapes entry -- that table builds the creation buttons):
  // its row shows the glyph the raster consumer rows already use
  if(!ictype && form->type & DT_MASKS_RASTER) ictype = lm->ic_raster;

  gtk_tree_store_set(GTK_TREE_STORE(model), iter,
                     TREE_TEXT, str,
                     TREE_OPACITY, opac,
                     TREE_NUM, num,
                     TREE_LINK, link,
                     TREE_IC_OP, icop,
                     TREE_IC_OP_VISIBLE, (icop != NULL),
                     TREE_IC_INVERSE, icinv,
                     TREE_IC_INVERSE_VISIBLE, (icinv != NULL),
                     TREE_IC_TYPE, ictype,
                     TREE_IC_TYPE_VISIBLE, (ictype != NULL),
                     TREE_TARGET, target,
                     TREE_MODULE_OFF, moff,
                     TREE_MODULE_ON, !moff,
                     TREE_POWER, power,
                     TREE_SHOW, show,
                     TREE_SHOW_ON, show_on,
                     -1);

  g_free(target);
}

// the one destructive action of this panel that names no target: it removes
// every shape no module and no history step refers to. asked for first,
// because afterwards there is nothing left on screen to point at -- and
// because dt_masks_cleanup_unused() opens on dt_masks_change_form_gui(NULL),
// which drops a shape being drawn. the button is insensitive while one is,
// see _update_all_properties; this is the second lock
static void _tree_cleanup(GtkButton *button, dt_lib_module_t *self)
{
  const dt_lib_masks_t *d = self->data;

  // the count says what the caption above the button says -- how many shapes
  // the library shows as linked to nothing -- and the sentence then says which
  // of them go, because the two are not the same set: a shape an undone
  // history step still refers to is counted here and kept by the cleanup
  const gboolean go = dt_gui_show_yes_no_dialog
    (_("delete unused shapes"), "",
     ngettext("%d shape in the library is linked to no module.\n"
              "the ones no history step refers to either are removed.",
              "%d shapes in the library are linked to no module.\n"
              "the ones no history step refers to either are removed.",
              d->unlinked), d->unlinked);
  if(!go) return;

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

// one step of the application order, shared by the context menu and by the
// drop of a drag: the handover at the base crossing and the move are one
// unit, and a path that called dt_masks_form_move() without the first half
// would recreate the uninitialised-buffer render group.c is patched against
static void _shape_move_step(dt_masks_form_t *grp,
                             const dt_mask_id_t id,
                             const gboolean later)
{
  const int rank = _group_point_index(grp, id);

  // crossing index 0 hands the base over to another shape. both ids are
  // resolved BEFORE the move, and _handover_base_state only touches state
  // bits, never positions
  if(later && rank == 0)
    _handover_base_state(grp, _group_point_id(grp, 1), id);
  else if(!later && rank == 1)
    _handover_base_state(grp, id, _group_point_id(grp, 0));

  dt_masks_form_move(grp, id, later);
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
      _shape_move_step(grp, id, later);
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
  int removed = 0;

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

      // a raster row names no form: removing it is unlinking, offered
      // by its own menu entry and never by this one -- a multiple
      // selection can still carry one here
      if(!dt_is_valid_maskid(id))
      {
        gtk_tree_iter_free(prev_iter);
        gtk_tree_iter_free(next_iter);
        continue;
      }

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
      removed++;
    }
  }
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);

  // a selection of raster rows alone deletes nothing: no history item
  // and no rebuild for a gesture that changed nothing
  if(removed)
  {
    dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
    _lib_masks_recreate_list(self);
  }
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

// the editor opened. nothing else to do with the news: the receipt is read on
// the next line of _tree_start_rename() and by no one else
static void _tree_editing_started(GtkCellRenderer *cell,
                                  GtkCellEditable *editable,
                                  gchar *path,
                                  dt_lib_masks_t *lm)
{
  lm->rename_started = TRUE;
}

// a row owns its name when it hangs from no group: a shape in the library, and
// a mask at the root of the masks list -- which is what TREE_EDITABLE says, on
// both stores. a shape nested in a mask is the same form under a second row
// and is renamed in the library, in one place.
// the cell is armed here and disarmed the moment the edit ends, so nothing
// else can start one -- see the note in _build_masks_view. that is what
// libs/map_locations.c does with its own name cell, at each of its three entry
// points, and for the same reason
static void _tree_start_rename(dt_lib_module_t *self,
                               GtkWidget *view,
                               GtkTreePath *path)
{
  dt_lib_masks_t *lm = self->data;
  const int v = _masks_view_index(lm, view);
  if(v < 0 || !path || !lm->name_col[v] || !lm->name_cell[v]) return;

  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  GtkTreeIter iter;
  if(!model || !gtk_tree_model_get_iter(model, &iter, path)) return;

  gboolean editable = FALSE;
  gtk_tree_model_get(model, &iter, TREE_EDITABLE, &editable, -1);
  if(!editable) return;

  lm->rename_started = FALSE;
  g_object_set(lm->name_cell[v], "editable", TRUE, NULL);
  gtk_tree_view_set_cursor_on_cell(GTK_TREE_VIEW(view), path,
                                   lm->name_col[v], lm->name_cell[v], TRUE);

  // that call returns void and starts the editor from inside itself, so
  // "editing-started" has already been through by the time we get here and the
  // answer is in. it can decline -- the view has to be realized and the column
  // has to take the focus -- and an arming nothing ever ends is the one state
  // this whole mechanism cannot afford: neither "edited" nor "editing-canceled"
  // would come, and the cell would stay editable for good
  if(!lm->rename_started)
    g_object_set(lm->name_cell[v], "editable", FALSE, NULL);
}

// the menu entry and the keyboard both land here: the row is the selected one,
// there being no pointer to read in the second case
static void _tree_rename(GtkButton *button, dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  GtkWidget *view = _masks_active_view(lm);

  GList *items = gtk_tree_selection_get_selected_rows
    (gtk_tree_view_get_selection(GTK_TREE_VIEW(view)), NULL);
  if(!items) return;

  _tree_start_rename(self, view, items->data);
  g_list_free_full(items, (GDestroyNotify)gtk_tree_path_free);
}

// escape, and nothing else: GtkCellRendererText emits "edited" on a commit
// only. a focus loss is not a third case -- dt_gui_commit_on_focus_loss() ends
// it through "editing-done", which comes out here as an ordinary "edited"
static void _tree_editing_canceled(GtkCellRenderer *cell,
                                   gpointer user_data)
{
  g_object_set(cell, "editable", FALSE, NULL);
}

static void _tree_cell_edited(GtkCellRendererText *cell,
                              gchar *path_string,
                              gchar *new_text,
                              GtkWidget *view)
{
  // the arming ends with the edit, before anything below can return early:
  // the cell has to be inert again by the time the next click reaches it
  g_object_set(cell, "editable", FALSE, NULL);

  // the renderer belongs to one view; resolving the path against any other
  // would rename whatever row happens to sit at the same index
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  GtkTreeIter iter;
  if(!gtk_tree_model_get_iter_from_string(model, &iter, path_string)) return;

  dt_mask_id_t id = INVALID_MASKID;
  dt_iop_module_t *module = NULL;
  _lib_masks_get_values(model, &iter, &module, NULL, &id);
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(!form)
  {
    // a raster row: the name on it is the source instance's, so a
    // rename renames that instance -- the very write of the module
    // header's own rename entry, seen everywhere the instance is named
    if(_raster_row(module, id) && _mask_target_alive(module)
       && _raster_consumer(module)
       && _mask_target_alive(module->raster_mask.sink.source))
    {
      dt_iop_module_t *src = module->raster_mask.sink.source;
      // write only on an actual change, as the header's rename entry
      // does (_rename_module_key_press compares first): the forced
      // write would stamp a history item and re-enable a switched-off
      // source just for validating the prefilled name unchanged
      gboolean changed;
      if(strlen(new_text) == 0)
        changed = *src->multi_name != '\0';
      else
      {
        gchar *shown = (*src->multi_name && strcmp(src->multi_name, "0"))
          ? dt_util_localize_segmented_name(src->multi_name, FALSE)
          : g_strdup(src->name());
        changed = g_strcmp0(shown, new_text) != 0;
        g_free(shown);
      }
      if(changed)
      {
        if(strlen(new_text) == 0)
          dt_iop_update_multi_name(src, "", FALSE, FALSE, TRUE);
        else
          dt_iop_update_multi_name(src, new_text, TRUE, TRUE, TRUE);
        dt_dev_masks_list_update(darktable.develop);
      }
    }
    return;
  }

  // we want to make sure that the new name is not an empty
  // string. else this would convert in the xmp file into "<rdf:li/>"
  // which produces problems. we use a single whitespace as the pure
  // minimum text.
  gchar *text = strlen(new_text) == 0 ? " " : new_text;

  // first, we need to update the mask name

  g_strlcpy(form->name, text, sizeof(form->name));
  dt_dev_add_masks_history_item(darktable.develop, NULL, FALSE);
}

// the body of the selection handler, apart from its guard: what a click on a
// row does to the canvas -- the selected rows become the group on screen.
// called by hand from gui_update() for a shape that landed on a row carrying
// no module -- the library, or a group nothing owns -- while nothing was
// being drawn: the row is selected there under the gui update guard, so the
// handler itself stays silent, and such a shape has no module whose edit
// mode would put it on the photograph instead
static void _tree_selection_apply(GtkTreeSelection *selection,
                                  dt_lib_masks_t *self)
{
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

static void _tree_selection_change(GtkTreeSelection *selection,
                                   dt_lib_masks_t *self)
{
  DT_GUARD_GUI_UPDATE();
  _tree_selection_apply(selection, self);
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

// where the selected row sits in a tree view's bin window, for a menu that has
// no pointer to hang from. x and width come back as 0 -- no column asked for
// -- which anchors on the left edge of the row, and a row that is scrolled out
// of view or not realised gives a zero height, which is the caller's cue to
// fall back
static gboolean _selected_row_rect(GtkWidget *view, GdkRectangle *rect)
{
  GtkTreeSelection *sel = gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
  GList *rows = gtk_tree_selection_get_selected_rows(sel, NULL);
  if(!rows) return FALSE;

  gtk_tree_view_get_cell_area(GTK_TREE_VIEW(view), rows->data, NULL, rect);
  g_list_free_full(rows, (GDestroyNotify)gtk_tree_path_free);

  return rect->height > 0;
}

// ask the recompute of a raster shape's mask file from its provenance
// recipe. the answer lands asynchronously: the recompute job reprocesses
// the darkroom when the file is back, and the row badge follows on the
// next refresh. the formid travels on the menu item, as the "add
// existing shape" entries do
static void _tree_raster_recompute(GtkMenuItem *item, gpointer user_data)
{
  (void)user_data;
  const dt_mask_id_t id =
    GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(item), "formid"));
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  const dt_masks_point_raster_t *rpt = dt_masks_raster_point(form);
  if(!rpt || !dt_rf_recipe_valid(&rpt->recipe)) return;
  if(dt_object_recipe_schedule_recompute(&rpt->recipe,
                                         darktable.develop->image_storage.id))
    dt_control_log(_("recomputing the mask file"));
  else
    // the stub of a build without AI declines, and so does a job system
    // gone away: either way nothing will be recomputed
    dt_control_log(_("recomputing the mask file needs the AI subsystem"));
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
//
// `at_row` says the menu was asked for without a pointer -- see the tail of
// this function, where it decides where the menu comes up
static void _tree_context_menu(dt_lib_module_t *self,
                               GtkWidget *view,
                               GtkTreeSelection *selection,
                               GtkTreeModel *model,
                               GtkTreePath *mouse_path,
                               const gboolean on_row,
                               const gboolean at_row,
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

  // a raster row gets a menu of its own: nothing of the shape
  // vocabulary applies to it. rename edits the source instance's name,
  // the submenu rewires the consumer, the last entry unlinks it --
  // unlinking, never deleting: there is no form anywhere to destroy
  const gboolean raster_sel = (nb == 1) && !from_group
    && _raster_row(sel_module, grpid)
    && _mask_target_alive(sel_module)
    && _raster_consumer(sel_module);
  if(raster_sel)
  {
    item = gtk_menu_item_new_with_label(_("rename"));
    g_signal_connect(item, "activate", G_CALLBACK(_tree_rename), self);
    gtk_menu_shell_append(menu, item);

    GtkWidget *rsub = gtk_menu_new();
    if(_raster_source_items(GTK_MENU_SHELL(rsub), sel_module))
    {
      item = gtk_menu_item_new_with_label(_("use another raster mask"));
      gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), rsub);
      gtk_menu_shell_append(menu, item);
    }
    else
    {
      // sink the floating reference, as _new_mask_target_header does
      g_object_ref_sink(rsub);
      g_object_unref(rsub);
    }

    item = gtk_menu_item_new_with_label(_("stop using this raster mask"));
    g_signal_connect(item, "activate",
                     G_CALLBACK(_raster_detach_cb), sel_module);
    gtk_menu_shell_append(menu, item);
  }

  // a raster-SHAPE row -- a real form, unlike the consumer rows above:
  // its pixels come from a mask file a valid provenance recipe can
  // regenerate. offer that regeneration; the render repairs a missing
  // file on its own, this entry is for asking again by hand (the mask
  // root moved back in, the models changed). everything else on the
  // menu -- operators, opacity, move, remove, delete -- is the shared
  // shape vocabulary and applies unchanged. grpid holds TREE_FORMID of
  // the selected row, see the note above
  if(nb == 1 && grp && (grp->type & DT_MASKS_RASTER))
  {
    const dt_masks_point_raster_t *rpt = dt_masks_raster_point(grp);
    if(rpt && dt_rf_recipe_valid(&rpt->recipe))
    {
      // GTK3 lesson, the panel's convention: when the replay's model
      // gates would refuse -- the model-gap verdict is their exact
      // mirror -- the reason is spelled in the label and the entry
      // disabled, rather than promising a recompute that changes
      // nothing. a build without AI keeps the plain entry: its click
      // already answers with the subsystem toast
      const char *reason = NULL;
#ifdef HAVE_AI
      // the entry schedules a replay of the RECORDED recipe (it rebinds
      // nothing), so the matting gate applies to it exactly as it applies
      // to the automatic recompute: asked first, and for the same reason
      // the row above asks it first -- a build capability no download can
      // move must not be dressed as a model problem, nor left enabled
      if(!dt_object_recipe_matting_reproducible(&rpt->recipe))
        reason = _("matting stage not supported by this build");
      else switch(dt_object_recipe_model_gap(&rpt->recipe, NULL))
      {
        case DT_OBJECT_RECIPE_MODELS_OK:
          break;
        case DT_OBJECT_RECIPE_MODELS_INSTALLABLE:
        case DT_OBJECT_RECIPE_MODELS_DRIFT_BEHIND:
          reason = _("model download needed");
          break;
        case DT_OBJECT_RECIPE_MODELS_DRIFT_AHEAD:
          reason = _("recorded model changed");
          break;
        case DT_OBJECT_RECIPE_MODELS_AI_OFF:
          reason = _("AI disabled in preferences");
          break;
        default:
          reason = _("model unknown");
      }
#endif
      gchar *rlabel = reason
        ? g_strdup_printf("%s (%s)", _("recompute mask file"), reason)
        : g_strdup(_("recompute mask file"));
      item = gtk_menu_item_new_with_label(rlabel);
      g_free(rlabel);
      gtk_widget_set_sensitive(item, reason == NULL);
      g_object_set_data(G_OBJECT(item), "formid",
                        GUINT_TO_POINTER(grp->formid));
      g_signal_connect(item, "activate",
                       G_CALLBACK(_tree_raster_recompute), self);
      gtk_menu_shell_append(menu, item);
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

  // a library row offers the mirror of the group rows' "add existing shape":
  // send THIS shape to a module. the same rank-ordered module list as the
  // target row's "...", each entry carrying the shape; a module whose mask
  // already holds it stays, greyed, and says why -- F4's "already linked".
  // grpid holds TREE_FORMID of the selected row, see the note above
  if(nb == 1 && !from_group && view == lm->library
     && dt_is_valid_maskid(grpid))
  {
    GtkWidget *others = gtk_menu_new();
    if(_new_mask_other_modules(GTK_MENU_SHELL(others), NULL, FALSE, grpid))
    {
      item = gtk_menu_item_new_with_label(_("add to module"));
      gtk_menu_item_set_submenu(GTK_MENU_ITEM(item), others);
      gtk_menu_shell_append(menu, item);
    }
    else
    {
      // nothing in this pipe can take a drawn shape: no entry at all. sink
      // the floating reference first, as _new_mask_target_header does
      g_object_ref_sink(others);
      g_object_unref(others);
    }
  }

  if(!raster_sel && !from_group && nb > 0)
  {
    // the row hangs from no group, which is the whole of the rule: it owns its
    // name, TREE_EDITABLE is true on it, and the double click renames it.
    // said once here for both lists, and no longer inside the branch below
    // that only shapes reach: a root row of the library is a shape, a root row
    // of the masks list is a mask, and both are named by hand. renaming a mask
    // is not a detail -- it is what puts "sky -> exposure" on the row instead
    // of the module's own name twice over
    if(nb == 1)
    {
      item = gtk_menu_item_new_with_label(_("rename"));
      g_signal_connect(item, "activate", G_CALLBACK(_tree_rename), self);
      gtk_menu_shell_append(menu, item);
    }

    dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, grpid);
    if(!(grp && (grp->type & DT_MASKS_GROUP)))
    {
      if(nb == 1)
      {
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
  else if(!raster_sel && nb > 0 && depth < 3)
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

  // grouping is something one does to SHAPES, so it is offered where the
  // shapes are. a root row of the masks list is a mask: grouping two of them
  // built a group of masks that no module wears, and both masks then appeared
  // twice, once at the root as their module's and once inside it.
  // where the result lands, since it is not under the rows it was made from:
  // _tree_group() registers the new group in dev->forms with no module and no
  // parent, so it appears at the root of the MASKS list, marked "unattached".
  // that is the one object the model has no level for -- it is not a shape,
  // and it is not the mask of a module -- and this entry is its deliberate
  // producer. it stays here rather than in the masks list because what it
  // takes has to be shapes; what it makes has nowhere better to be shown
  if(nb > 1 && !from_group && view == lm->library)
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

  gtk_widget_show_all(GTK_WIDGET(menu));

  GdkEvent *event = gtk_get_current_event();

  // a key press carries no pointer. gtk_menu_popup_at_pointer() would then
  // hand GTK an event with no coordinates and the menu would come up at the
  // corner of the list, or wherever the mouse was last left -- over the canvas
  // as easily as not. anchor it under the row it is about instead, which is
  // where the right button would have opened it anyway
  GdkRectangle rect;
  if(at_row && _selected_row_rect(view, &rect))
    gtk_menu_popup_at_rect(GTK_MENU(menu),
                           gtk_tree_view_get_bin_window(GTK_TREE_VIEW(view)),
                           &rect,
                           GDK_GRAVITY_SOUTH_WEST, GDK_GRAVITY_NORTH_WEST,
                           event);
  else
    gtk_menu_popup_at_pointer(GTK_MENU(menu), event);

  gdk_event_free(event);
}

// the one target of this panel's drag and drop. the payload never travels
// through it -- both ends of the drag live in dt_lib_masks_t -- but the NAME
// is what keeps a filmstrip image or a tag from being droppable here
static const GtkTargetEntry _masks_dnd_target =
  { "masks-shape-dnd", GTK_TARGET_SAME_APP, DND_TARGET_MASK_SHAPE };

// defined below with the other tree walkers; the drag icon needs it early
gboolean _find_mask_iter_by_values(GtkTreeModel *model,
                                   GtkTreeIter *iter,
                                   const dt_iop_module_t *module,
                                   const dt_mask_id_t formid,
                                   const int level);

// what a drop at (x,y) over the masks list would do. `ok` FALSE means the
// drop is refused and there is no indicator to draw; `row` is owned by the
// caller when set
typedef struct dt_masks_drop_t
{
  gboolean ok;
  gboolean reorder;         // TRUE: reorder inside a mask; FALSE: library add
  dt_masks_form_t *grp;     // the group acted on
  dt_iop_module_t *module;  // live owner of the mask row, or NULL
  int src, dst;             // reorder only: ranks in grp->points
  GtkTreePath *row;         // the row the indicator is drawn on
  GtkTreeViewDropPosition pos;
} dt_masks_drop_t;

// resolved from the model of the moment and from grp->points -- never from
// anything remembered at the press beyond the two ids: gui_update swaps the
// stores under an open drag without notice, so every motion and the drop
// itself re-ask from scratch. a group gone missing simply answers "refused"
static void _drag_dest_resolve(dt_lib_masks_t *lm,
                               GtkWidget *view,
                               const gint x,
                               const gint y,
                               dt_masks_drop_t *drop)
{
  *drop = (dt_masks_drop_t){ 0 };

  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));
  if(!model || !lm->drag_view || !dt_is_valid_maskid(lm->drag_formid))
    return;

  GtkTreePath *path = NULL;
  GtkTreeViewDropPosition pos;
  if(!gtk_tree_view_get_dest_row_at_pos(GTK_TREE_VIEW(view), x, y,
                                        &path, &pos))
    return;

  GtkTreeIter iter;
  if(!gtk_tree_model_get_iter(model, &iter, path))
  {
    gtk_tree_path_free(path);
    return;
  }

  if(lm->drag_view == view)
  {
    // reordering, inside one group only (F5): the row under the pointer
    // must be a member of the very group the drag left from. TREE_GROUPID
    // is the immediate parent, so a nested row reorders inside its own
    // sub-group and never leaves it -- the same reach the menu has
    dt_mask_id_t grid = INVALID_MASKID;
    dt_mask_id_t id = INVALID_MASKID;
    _lib_masks_get_values(model, &iter, NULL, &grid, &id);

    dt_masks_form_t *grp =
      dt_masks_get_from_id(darktable.develop, lm->drag_groupid);
    const int src = _group_point_index(grp, lm->drag_formid);
    const int over = _group_point_index(grp, id);

    if(gtk_tree_path_get_depth(path) < 2 || grid != lm->drag_groupid
       || src < 0 || over < 0)
    {
      gtk_tree_path_free(path);
      return;
    }

    // an insertion line between rows, never a drop "into" a shape
    if(pos == GTK_TREE_VIEW_DROP_INTO_OR_BEFORE)
      pos = GTK_TREE_VIEW_DROP_BEFORE;
    if(pos == GTK_TREE_VIEW_DROP_INTO_OR_AFTER)
      pos = GTK_TREE_VIEW_DROP_AFTER;

    // the rank the shape would hold once out of its own slot. the screen
    // order IS grp->points -- base on top -- so this is list arithmetic
    int dst = over;
    if(pos == GTK_TREE_VIEW_DROP_BEFORE && src < over) dst--;
    if(pos == GTK_TREE_VIEW_DROP_AFTER && src > over) dst++;

    if(dst == src)
    {
      // dropping where it already sits: nothing would happen, say so
      gtk_tree_path_free(path);
      return;
    }

    drop->ok = TRUE;
    drop->reorder = TRUE;
    drop->grp = grp;
    drop->src = src;
    drop->dst = dst;
    drop->row = path;
    drop->pos = pos;
    return;
  }

  // a shape pulled out of the library: the drop adds it to the MASK under
  // the pointer -- the whole depth-1 row, wherever inside an unfolded mask
  // the pointer sits. one gesture, one meaning: a nested group keeps its own
  // "add existing shape" in the menu
  while(gtk_tree_path_get_depth(path) > 1)
    gtk_tree_path_up(path);
  if(!gtk_tree_model_get_iter(model, &iter, path))
  {
    gtk_tree_path_free(path);
    return;
  }

  dt_mask_id_t grp_id = INVALID_MASKID;
  _lib_masks_get_values(model, &iter, NULL, NULL, &grp_id);

  dt_masks_form_t *grp = dt_masks_get_from_id(darktable.develop, grp_id);
  // the live owner, from dev->iop -- the row's module pointer can be one
  // refresh stale
  dt_iop_module_t *module = _mask_group_owner(grp_id);

  // refused: not a group any more, a module that cannot take a drawn shape,
  // or a mask that already holds this very shape -- the direct membership
  // dt_masks_group_add_form() does not check on this branch, so the drop
  // would otherwise duplicate the row silently. the library never lists
  // groups, so self-inclusion cannot happen here
  if(!grp || !(grp->type & DT_MASKS_GROUP)
     || (module && !_mask_target_ok(module))
     || _group_point_index(grp, lm->drag_formid) >= 0)
  {
    gtk_tree_path_free(path);
    return;
  }

  drop->ok = TRUE;
  drop->reorder = FALSE;
  drop->grp = grp;
  drop->module = module;
  drop->row = path;
  drop->pos = GTK_TREE_VIEW_DROP_INTO_OR_BEFORE;
}

static gboolean _tree_drag_motion_cb(GtkWidget *widget,
                                     GdkDragContext *context,
                                     const gint x,
                                     const gint y,
                                     const guint time,
                                     dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  // a flags-0 dest hears about EVERY drag that crosses it -- a filmstrip
  // image, a module header, a file from the desktop -- and the resolver
  // reads the payload from lm, never from the context. the one test tying
  // THIS drag to that payload is the source widget, the way the module
  // reorder of develop/imageop.c asks it; a foreign drag answers NULL and
  // is refused with the rest
  dt_masks_drop_t drop = { 0 };
  if(lm->drag_view
     && gtk_drag_get_source_widget(context) == lm->drag_view)
    _drag_dest_resolve(lm, widget, x, y, &drop);

  if(!drop.ok)
  {
    gtk_tree_view_set_drag_dest_row(GTK_TREE_VIEW(widget), NULL,
                                    GTK_TREE_VIEW_DROP_BEFORE);
    // 0 is the refusal: the forbidden cursor of F5
    gdk_drag_status(context, 0, time);
    return TRUE;
  }

  gtk_tree_view_set_drag_dest_row(GTK_TREE_VIEW(widget), drop.row, drop.pos);
  gdk_drag_status(context,
                  drop.reorder ? GDK_ACTION_MOVE : GDK_ACTION_LINK, time);
  gtk_tree_path_free(drop.row);
  return TRUE;
}

static void _tree_drag_leave_cb(GtkWidget *widget,
                                GdkDragContext *context,
                                const guint time,
                                dt_lib_module_t *self)
{
  gtk_tree_view_set_drag_dest_row(GTK_TREE_VIEW(widget), NULL,
                                  GTK_TREE_VIEW_DROP_BEFORE);
}

static gboolean _tree_drag_drop_cb(GtkWidget *widget,
                                   GdkDragContext *context,
                                   const gint x,
                                   const gint y,
                                   const guint time,
                                   dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  // the same gate as "drag-motion": only the drag this panel began may land
  dt_masks_drop_t drop = { 0 };
  if(lm->drag_view
     && gtk_drag_get_source_widget(context) == lm->drag_view)
    _drag_dest_resolve(lm, widget, x, y, &drop);

  gtk_tree_view_set_drag_dest_row(GTK_TREE_VIEW(widget), NULL,
                                  GTK_TREE_VIEW_DROP_BEFORE);
  // the payload, copied out BEFORE the finish: what the finish tells the
  // source ends in "drag-end", which clears the drag state -- through the
  // event queue today, but nothing below should rest on that ordering
  const dt_mask_id_t dragged = lm->drag_formid;
  // finish first, commit after, as develop/imageop.c does: the rebuild the
  // commit queues must find the drag already over
  gtk_drag_finish(context, drop.ok, FALSE, time);
  if(!drop.ok) return TRUE;
  gtk_tree_path_free(drop.row);

  if(drop.reorder)
  {
    // the same path as "apply earlier / later", one step at a time: each
    // crossing of the base is a handover, and a jump straight to the slot
    // would skip it. gui->group_edited is a positional index into the very
    // list about to move, so the canvas gui goes down first, as the menu does
    dt_masks_clear_form_gui(darktable.develop);

    const gboolean later = drop.dst > drop.src;
    int steps = later ? drop.dst - drop.src : drop.src - drop.dst;
    while(steps-- > 0)
      _shape_move_step(drop.grp, dragged, later);

    // one gesture, one history entry, one undo -- however many steps
    dt_dev_add_masks_history_item(darktable.develop, NULL, TRUE);
    _lib_masks_recreate_list(self);
  }
  else if(drop.module)
    // the very path of the context menu's "add to <module>": enable, add,
    // reselect -- one gesture, one behaviour. the engine appends at the END
    // of grp->points with UNION, so no shape ever ARRIVES at index 0 and no
    // handover is due on this branch
    _shape_add_to_module(drop.module, dragged);
  else
  {
    // a stand-alone group: no module to enable and no combo to refresh --
    // otherwise the same queue as _tree_add_exist
    dt_masks_form_t *form =
      dt_masks_get_from_id(darktable.develop, dragged);
    if(form && dt_masks_group_add_form(drop.grp, form))
    {
      dt_dev_add_masks_history_item(darktable.develop, NULL, FALSE);
      dt_dev_masks_selection_change(darktable.develop, NULL,
                                    drop.grp->formid);
    }
  }

  return TRUE;
}

static void _tree_drag_begin_cb(GtkWidget *widget,
                                GdkDragContext *context,
                                dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  if(lm->drag_view != widget) return;

  // the pointer is grabbed for the length of the drag and gtk3 does not
  // reliably deliver the leave event for a grab crossing -- same cleanup as
  // the operator menu makes before popping up
  GdkWindow *bin = gtk_tree_view_get_bin_window(GTK_TREE_VIEW(widget));
  if(bin) gdk_window_set_cursor(bin, NULL);

  // the row itself as the drag icon, as libs/tagging.c does. resolved by id
  // and not by a stored path: first row carrying it -- for a shape worn by
  // two masks that may be the twin row, which shows the same shape under
  // the same name
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(widget));
  GtkTreeIter iter;
  if(model && gtk_tree_model_get_iter_first(model, &iter)
     && _find_mask_iter_by_values(model, &iter, NULL, lm->drag_formid, 1))
  {
    GtkTreePath *path = gtk_tree_model_get_path(model, &iter);
    cairo_surface_t *row =
      gtk_tree_view_create_row_drag_icon(GTK_TREE_VIEW(widget), path);
    if(row)
    {
      gtk_drag_set_icon_surface(context, row);
      cairo_surface_destroy(row);
    }
    gtk_tree_path_free(path);
  }
}

static void _tree_drag_end_cb(GtkWidget *widget,
                              GdkDragContext *context,
                              dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  // this can land after gui_update swapped the stores: touch nothing but
  // the drag state and the indicator
  lm->drag_view = NULL;
  lm->drag_formid = INVALID_MASKID;
  lm->drag_groupid = INVALID_MASKID;
  if(lm->treeview)
    gtk_tree_view_set_drag_dest_row(GTK_TREE_VIEW(lm->treeview), NULL,
                                    GTK_TREE_VIEW_DROP_BEFORE);
}

// nothing ever asks for the data -- both ends of the drag live in
// dt_lib_masks_t -- but a source is expected to answer, so answer empty,
// exactly as libs/tagging.c does for its own widget-bound payload
static void _tree_drag_data_get_cb(GtkWidget *widget,
                                   GdkDragContext *context,
                                   GtkSelectionData *selection_data,
                                   const guint target_type,
                                   const guint time,
                                   dt_lib_module_t *self)
{
  if(target_type == DND_TARGET_MASK_SHAPE)
    gtk_selection_data_set(selection_data,
                           gtk_selection_data_get_target(selection_data),
                           _DWORD, NULL, 0);
}

// what one of the clickable columns has to say at a point, if anything
typedef enum dt_masks_op_hit_t
{
  DT_MASKS_OP_HIT_NONE = 0,  // not a clickable column, or nothing to say
  DT_MASKS_OP_HIT_BASE,      // the base: says why it has none, not clickable
  DT_MASKS_OP_HIT_OPERATOR,  // an operator a click may change
  DT_MASKS_OP_HIT_POWER,     // the module switch of a mask row
  DT_MASKS_OP_HIT_SHOW,      // the show-mask cell of a mask row
  DT_MASKS_OP_HIT_SHOW_OFF   // ... whose module is off: says so, not clickable
} dt_masks_op_hit_t;

// the module a row's cell hands over, or nothing. `live` is the model column
// that says whether the cell is there at all -- TREE_POWER for the switch,
// TREE_SHOW for the show-mask cell -- so what a click may reach is read off
// the very column that decided what is drawn
static dt_iop_module_t *_row_cell_module(GtkTreeModel *model,
                                         GtkTreePath *path,
                                         const dt_masks_tree_cols_t live)
{
  GtkTreeIter iter;
  if(!gtk_tree_model_get_iter(model, &iter, path)) return NULL;

  gboolean live_cell = FALSE;
  gtk_tree_model_get(model, &iter, live, &live_cell, -1);
  if(!live_cell) return NULL;

  // TREE_MODULE holds a raw pointer and the store outlives the pipe it was
  // built from by one refresh, so the same check _mask_default_target and
  // _set_iter_name make before reading through it
  dt_iop_module_t *m = NULL;
  _lib_masks_get_values(model, &iter, &m, NULL, NULL);
  return _mask_target_alive(m) ? m : NULL;
}

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
//
// the power column answers the same way and is decided elsewhere: the cell is
// live exactly where TREE_POWER is set, so _row_cell_module() above reads that
// column back rather than restating the rule. one writer for what a row shows
// AND for what a click may reach -- restated here, the two would drift and a
// hand would appear over an empty cell.
static dt_masks_op_hit_t _op_cell_at_bin(dt_lib_masks_t *lm,
                                         GtkWidget *view,
                                         const gint bx,
                                         const gint by,
                                         dt_masks_state_t *state_out,
                                         dt_iop_module_t **module_out)
{
  if(!lm->op_col && !lm->power_col && !lm->show_col)
    return DT_MASKS_OP_HIT_NONE;

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
      // and the same function, as the rank on screen. a row whose group no
      // longer holds its shape yields NULL and the click stays a plain
      // selection
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
  else if(model && path && column == lm->power_col)
  {
    dt_iop_module_t *m = _row_cell_module(model, path, TREE_POWER);
    if(m)
    {
      if(module_out) *module_out = m;
      hit = DT_MASKS_OP_HIT_POWER;
    }
  }
  else if(model && path && column == lm->show_col)
  {
    dt_iop_module_t *m = _row_cell_module(model, path, TREE_SHOW);
    if(m)
    {
      if(module_out) *module_out = m;
      // a module that is switched off renders no blend, so its mask cannot be
      // put on screen: the cell is there, stepped back, and says why rather
      // than setting a request nothing would honour
      hit = m->enabled ? DT_MASKS_OP_HIT_SHOW : DT_MASKS_OP_HIT_SHOW_OFF;
    }
  }

  if(path) gtk_tree_path_free(path);
  return hit;
}

// exactly the zone the click reacts to, and back to NULL as soon as we leave
// it -- a cursor set on the bin window and never reset stays a hand over the
// whole panel. the base gets no hand, and neither does a show-mask cell whose
// module is off: nothing there acts.
// three zones now, one cursor: the operator glyph, the power switch and the
// show-mask cell. they are the only cells in either list a click acts on
static void _tree_motion_cb(GtkEventControllerMotion *controller,
                            double x,
                            double y,
                            dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  GtkWidget *view = dt_gui_get_widget(controller);
  GdkWindow *bin = gtk_tree_view_get_bin_window(GTK_TREE_VIEW(view));
  if(!bin) return;

  // an armed press that travels the drag threshold with the button still
  // down becomes a drag, started by hand exactly as views/map.c starts its
  // own. gtk_drag_source_set() is deliberately not used: it would start a
  // drag from ANY press on the widget -- the power switch, the show-mask
  // cell, the operator strip -- and the only veto it leaves, cancelling from
  // inside "drag-begin", tears the source info down while gtk is still
  // setting it up
  if(lm->drag_view == view)
  {
    GdkModifierType state = 0;
    gtk_get_current_event_state(&state);
    if(!(state & GDK_BUTTON1_MASK))
      // released before the threshold: an ordinary click, stand down
      lm->drag_view = NULL;
    else if(gtk_drag_check_threshold(view, lm->drag_x, lm->drag_y,
                                     (gint)x, (gint)y))
    {
      GtkTargetList *targets = gtk_target_list_new(&_masks_dnd_target, 1);
      GdkEvent *event = gtk_get_current_event();
      // LINK from the library -- adding a shape links it, F4's whole point,
      // and the copy cursor's "+" badge would promise the independent copy
      // F4 argues against -- MOVE inside the masks: the row leaves its slot
      gtk_drag_begin_with_coordinates(view, targets,
                                      view == lm->library ? GDK_ACTION_LINK
                                                          : GDK_ACTION_MOVE,
                                      GDK_BUTTON_PRIMARY, event,
                                      lm->drag_x, lm->drag_y);
      if(event) gdk_event_free(event);
      gtk_target_list_unref(targets);
      return;
    }
  }

  gint bx, by;
  gtk_tree_view_convert_widget_to_bin_window_coords(GTK_TREE_VIEW(view),
                                                    (gint)x, (gint)y, &bx, &by);

  // this runs on every motion event over the list, so touch the window only
  // when the answer changes. nothing else puts a cursor on it, so its current
  // cursor is a faithful record of what we last decided
  const dt_masks_op_hit_t hit = _op_cell_at_bin(lm, view, bx, by, NULL, NULL);
  const gboolean over = (hit == DT_MASKS_OP_HIT_OPERATOR)
                     || (hit == DT_MASKS_OP_HIT_POWER)
                     || (hit == DT_MASKS_OP_HIT_SHOW);
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
  GtkTreeViewColumn *mouse_col = NULL;
  GtkTreeIter iter;
  dt_iop_module_t *module = NULL;
  gboolean on_row = FALSE;
  if(gtk_tree_view_get_path_at_pos(GTK_TREE_VIEW(treeview),
                                   bin_x, bin_y, &mouse_path, &mouse_col,
                                   NULL, NULL))
  {
    on_row = TRUE;
    // we retrieve the iter and module from path
    if(gtk_tree_model_get_iter(model, &iter, mouse_path))
    {
      _lib_masks_get_values(model, &iter, &module, NULL, NULL);
    }
  }

  const guint button = gtk_gesture_single_get_current_button(gesture);

  // a drag may leave this press: record what it would carry, with the same
  // hit information as every other gesture of the two lists. the name column
  // only -- the operator strip, the power switch and the show-mask cell act
  // on the press, and a held click on any of them must never become a drag;
  // bin_x against the cell area keeps the expander triangle out, gtk reports
  // it as part of the name column. one row, the pressed one, never the
  // selection: reordering is a per-shape operation (F5). any depth under a
  // mask: a row in a nested group is a member of its own group and moves
  // inside it, the same reach the menu has. NOT guarded by rename_started,
  // which is a one-shot receipt that stays TRUE long after an edit ended:
  // the cell's own "editable" is armed for exactly one rename
  lm->drag_view = NULL;
  const int dview = _masks_view_index(lm, treeview);
  gboolean editing = FALSE;
  if(dview >= 0)
    g_object_get(lm->name_cell[dview], "editable", &editing, NULL);

  if(button == GDK_BUTTON_PRIMARY && n_press == 1 && mouse_path && !editing
     && dview >= 0 && mouse_col == lm->name_col[dview]
     && dt_modifier_is(dt_gui_current_state(gesture), 0)
     && gtk_tree_path_get_depth(mouse_path) >= (dview == 1 ? 1 : 2))
  {
    GdkRectangle cell = { 0 };
    gtk_tree_view_get_cell_area(GTK_TREE_VIEW(treeview), mouse_path,
                                mouse_col, &cell);
    if(bin_x >= cell.x && gtk_tree_model_get_iter(model, &iter, mouse_path))
    {
      dt_mask_id_t grid = INVALID_MASKID;
      dt_mask_id_t id = INVALID_MASKID;
      _lib_masks_get_values(model, &iter, NULL, &grid, &id);
      if(dt_is_valid_maskid(id))
      {
        lm->drag_view = treeview;
        lm->drag_formid = id;
        lm->drag_groupid = grid;
        lm->drag_x = (gint)x;
        lm->drag_y = (gint)y;
      }
    }
  }

  /* single click with the right mouse button? */
  if(button == GDK_BUTTON_PRIMARY)
  {
    // dt_gui_current_state() and not gtk_get_current_event_state(): a press
    // emitted by a shortcut carries no current event, which leaves that call
    // writing nothing at all into its out parameter. the helper hands back the
    // effect's own modifiers there, and "no modifier" is the one answer we
    // must not get wrong -- it is what decides whether a menu opens
    const GdkModifierType mods = dt_gui_current_state(gesture);
    dt_masks_state_t op_state = DT_MASKS_STATE_NONE;

    // a plain left click on one of this list's click targets. a modified click
    // still belongs to the selection -- ctrl and shift build a multiple one,
    // and it would be unbuildable if these vertical strips swallowed those
    // clicks.
    // note what we do NOT do: dt_gui_claim(gesture). this handler has never
    // claimed the sequence and must not start: claiming cancels the
    // treeview's own gesture, and with it selection, the rename double click
    // and the expanders
    dt_iop_module_t *row_module = NULL;
    dt_masks_op_hit_t hit = (mouse_path && dt_modifier_is(mods, 0))
      ? _op_cell_at_bin(lm, treeview, bin_x, bin_y, &op_state, &row_module)
      : DT_MASKS_OP_HIT_NONE;

    // the menu is the one target that must not answer twice: "pressed" fires
    // once per press, so a double click on the glyph would pop it and then pop
    // a second one over the first. the cells beside it are switches, and two
    // quick clicks on a switch are two switches -- switch the module off, look
    // at the photograph, switch it back on IS a double click at one pixel, and
    // it is the whole gesture M2 asks the power column for. a guard written
    // once for the menu must not decide for them
    if(hit == DT_MASKS_OP_HIT_OPERATOR && n_press != 1)
      hit = DT_MASKS_OP_HIT_NONE;

    if(hit == DT_MASKS_OP_HIT_OPERATOR)
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
    // M2, the comparison gesture: switch off the module this mask serves and
    // the photograph goes back to what it was, without leaving the panel.
    //
    // the module's OWN switch and never module->enabled: the toggle carries
    // the history item, the pipe recompute, the fold rule
    // ("darkroom/ui/activate_expand"), the accelerators and the header class
    // with it -- _gui_off_callback does all of that and nothing here repeats
    // any of it. read the button's state and not the module's, as
    // _enable_module_callback (develop/imageop.c) and views/darkroom.c both
    // do; dt_iop_gui_set_enable_button keeps the two in step.
    //
    // the way back is already paid for: _gui_off_callback ends in
    // dt_dev_masks_list_update(), which rewrites every derived column of every
    // row in place -- TREE_POWER included. no loop: that path sets store cells
    // and touches no toggle, and dt_dev_add_history_item() rebuilds no list.
    //
    // the selection is deliberately left alone. the operator branch above
    // forces it because the menu acts on the selection; this acts on a module
    // the row hands over, and the tree view's own gesture will select the row
    // on release like any other click
    else if(hit == DT_MASKS_OP_HIT_POWER && row_module && row_module->off)
    {
      const gboolean on =
        gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(row_module->off));
      gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(row_module->off), !on);
    }
    // M2: see this mask filled over the photograph. one click, never a hover
    // -- each toggle costs a pipe recompute.
    // the known cost, and it is not a design choice: this takes the darkroom
    // focus, so the cell of every other row goes out. develop/blend.c only
    // honours a display request for the module that has the focus
    else if(hit == DT_MASKS_OP_HIT_SHOW && row_module)
    {
      dt_iop_set_mask_display
        (row_module,
         !(row_module->request_mask_display & DT_DEV_PIXELPIPE_DISPLAY_MASK));
    }
    // if click on a blank space, then deselect all
    else if(!on_row)
    {
      gtk_tree_selection_unselect_all(selection);
    }
  }
  else if(button == GDK_BUTTON_SECONDARY)
  {
    // the menu of a ROW, and only that: it is what M2 defines the right button
    // as, and since the six shapes and the target line became a row of their
    // own there is nothing left for a blank click to offer that is not already
    // 20 px above it. the row under the pointer and not the selection either
    // -- _tree_context_menu() selects what was pointed at before it counts, so
    // on_row is enough for the menu to have a subject, and a menu built on a
    // row 200 px away from the click answers a question nobody asked
    if(on_row)
      _tree_context_menu(self, treeview, selection, model,
                         mouse_path, on_row, FALSE, module);
  }

  // ours since gtk_tree_view_get_path_at_pos succeeded, on every button and
  // whatever the menu did with it -- the menu only borrows it
  if(mouse_path) gtk_tree_path_free(mouse_path);
}

// the Menu key and shift+F10, on the row that is selected. GTK raises
// "popup-menu" on a tree view for both, and the whole of this panel's per-row
// vocabulary -- rename, duplicate, delete, the five operators, apply
// earlier/later -- lived behind the right button alone until now. that is what
// replaces the per-row "..." of M2: a column of ellipsis buttons would have
// cost ~20 px on every row to double a gesture that already exists, where the
// right-click is the convention of every other list panel of darktable without
// one exception. it just had to be reachable from the keyboard.
//
// gui/gtk.h enforces the return type with a _Static_assert: this signal wants
// a gboolean, TRUE meaning handled. libs/collect.c is the precedent for the
// wiring; it is not one for the position, and that is the `at_row` argument
// below -- collect.c hands a key event to gtk_menu_popup_at_pointer() and the
// menu lands wherever the mouse was left, which for a gesture whose whole
// purpose is to work without one is the one thing to get right
static gboolean _tree_popup_menu_cb(GtkWidget *view, dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  // the menu is built now and its entries read the selection back when they
  // fire: point them at this list, exactly as the right button does before
  // anything can pop up
  lm->active_view = view;

  GtkTreeSelection *selection =
    gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
  GtkTreeModel *model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));

  // nothing selected is nothing to open a menu about, this gesture being the
  // row's. FALSE, so GTK does what it does when a panel has no such menu
  if(gtk_tree_selection_count_selected_rows(selection) == 0) return FALSE;

  // the module of the row the menu is about. the right button reads it off the
  // row under the pointer; there is no pointer here, so the selection is the
  // row. "add existing shape" is the only entry that uses it, and only for a
  // single selection -- which is also the only case where it means anything
  dt_iop_module_t *module = NULL;
  GList *sel = gtk_tree_selection_get_selected_rows(selection, NULL);

  if(sel && !g_list_next(sel))
  {
    GtkTreeIter iter;
    if(gtk_tree_model_get_iter(model, &iter, sel->data))
      _lib_masks_get_values(model, &iter, &module, NULL, NULL);
  }
  g_list_free_full(sel, (GDestroyNotify)gtk_tree_path_free);

  // no row under a pointer, so no path to hand over and nothing to reselect:
  // the menu reads the selection back itself, which is the whole reason it can
  // be opened from the keyboard at all
  _tree_context_menu(self, view, selection, model, NULL, FALSE, TRUE, module);
  return TRUE;
}

// M3: double click to rename, and a single click that does nothing but select.
// GtkTreeView emits "row-activated" on the second press of a double click --
// and only now that no cell of this column is editable at rest, since the
// editing branch of its gesture returns before the signal is reached.
// the gesture is free: the widget attaches no default handler to that signal,
// which is why gui/preferences.c expands and collapses its own rows from its
// own "row-activated" callback rather than letting the view do it, and nothing
// else in this file listens for it. a group is still folded and unfolded by
// its triangle, the way it always was, and by nothing else.
// the COLUMN is checked and not just the row: the switch and the show-mask
// cell are single-click targets, and two quick clicks on a switch are two
// switches -- the comparison gesture M2 asks the power column for. opening an
// editor on top of it would fight the very thing it is there for. the operator
// glyph is the same story one column over. what the check does NOT separate is
// the cells packed inside the name column itself -- the kind icon, the
// opacity, the target, the word for a row that reaches no module -- so a
// double click anywhere along that column renames. none of them answers a
// click of its own, so there is nothing there to fight
static void _tree_row_activated_cb(GtkTreeView *view,
                                   GtkTreePath *path,
                                   GtkTreeViewColumn *column,
                                   dt_lib_module_t *self)
{
  dt_lib_masks_t *lm = self->data;
  GtkWidget *widget = GTK_WIDGET(view);
  const int v = _masks_view_index(lm, widget);
  if(v < 0 || column != lm->name_col[v]) return;

  // the same pinning every other entry point does before acting on a list
  lm->active_view = widget;
  _tree_start_rename(self, widget, path);
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
  const gchar *hint = NULL;
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
    dt_iop_module_t *row_module = NULL;
    const dt_masks_op_hit_t hit =
      _op_cell_at_bin(lm, widget, x, y, &op_state, &row_module);

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
    else if(hit == DT_MASKS_OP_HIT_POWER && row_module)
    {
      // the first line is the very sentence the module's own switch puts in
      // its tooltip (develop/imageop.c), from the same two msgids: one switch,
      // two places to reach it, one wording. the second says what THIS click
      // does, which is not the same sentence in both directions
      gchar *name = dt_history_item_get_name(row_module);
      gchar *line = g_strdup_printf(row_module->enabled
                                      ? _("'%s' is switched on")
                                      : _("'%s' is switched off"),
                                    name);
      gchar *text =
        g_strdup_printf("%s\n%s", line,
                        row_module->enabled
                          ? _("click to see the photograph without it")
                          : _("click to switch it back on"));
      gtk_tooltip_set_text(tooltip, text);
      g_free(text);
      g_free(line);
      g_free(name);
    }
    else if(hit == DT_MASKS_OP_HIT_SHOW && row_module)
    {
      gtk_tooltip_set_text
        (tooltip,
         (row_module->request_mask_display & DT_DEV_PIXELPIPE_DISPLAY_MASK)
           ? _("this mask is on screen\nclick to hide it")
           : _("see this mask filled over the photograph\n"
               "it takes the focus, so only one is shown at a time"));
    }
    else if(hit == DT_MASKS_OP_HIT_SHOW_OFF)
      gtk_tooltip_set_text(tooltip,
                           _("switch the module on to see its mask"));

    if(hit != DT_MASKS_OP_HIT_NONE)
    {
      // the hit names the column it came from, so there is nothing to look up
      GtkTreeViewColumn *hit_col = lm->op_col;
      if(hit == DT_MASKS_OP_HIT_POWER) hit_col = lm->power_col;
      else if(hit == DT_MASKS_OP_HIT_SHOW
              || hit == DT_MASKS_OP_HIT_SHOW_OFF) hit_col = lm->show_col;
      gtk_tree_view_set_tooltip_cell(tree_view, tooltip, path,
                                     hit_col, NULL);
      gtk_tree_path_free(path);
      return TRUE;
    }

    // the name cell is the one affordance of this panel with no glyph and
    // no pointer change to announce it: it can be dragged. every other
    // click target of these lists names itself in a tooltip, see the hits
    // above, so this one does too -- on the row, next to the "used by"
    // text when the row carries one. the same conditions as the arming in
    // _tree_button_pressed_cb, or the sentence promises a drag that could
    // not start
    GtkTreeViewColumn *col = NULL;
    if(gtk_tree_view_get_path_at_pos(tree_view, x, y, NULL, &col,
                                     NULL, NULL))
    {
      const int dview = _masks_view_index(lm, widget);
      dt_mask_id_t id = INVALID_MASKID;
      _lib_masks_get_values(model, &iter, NULL, NULL, &id);
      if(dview >= 0 && col == lm->name_col[dview] && dt_is_valid_maskid(id))
      {
        const int depth = gtk_tree_path_get_depth(path);
        if(dview == 0 && depth >= 2)
          hint = _("drag to change when it applies");
        else if(dview == 1 && depth == 1)
          hint = _("drag onto a mask to add it");
      }
    }
  }

  gtk_tree_model_get(model, &iter, TREE_USED_TEXT, &tmp, -1);
  // it used to be tied to the "used" badge being visible, so the one row that
  // most needs a word -- a shape nothing references -- was the one row that
  // stayed silent
  show = (tmp && *tmp) || hint != NULL;
  if(show)
  {
    // plain text, not markup: this string carries group names typed by the
    // photographer, and a single "&" in one of them blanked the whole tooltip
    gchar *text = (tmp && *tmp && hint)
      ? g_strdup_printf("%s\n%s", tmp, hint)
      : g_strdup(tmp && *tmp ? tmp : hint);
    gtk_tooltip_set_text(tooltip, text);
    g_free(text);
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
// cleanup takes it away all the same. a ROW may say "unused", and only an
// ORPHAN one does -- that row is in no group and in no module, which is the
// whole of the word.
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

// the module whose mask THIS group IS -- the one pointing straight at it, and
// never one that merely holds it somewhere below. that distinction is the
// whole of what tells a mask from a group filed inside one, and it is why
// _shape_scope() cannot answer here: dt_masks_is_in_module() is recursive on
// purpose and calls both of them MODULE
static dt_iop_module_t *_mask_group_owner(const dt_mask_id_t formid)
{
  if(!dt_is_valid_maskid(formid)) return NULL;

  for(const GList *iops = darktable.develop->iop;
      iops;
      iops = g_list_next(iops))
  {
    dt_iop_module_t *iop = iops->data;
    if((iop->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
       && !(iop->flags() & IOP_FLAGS_NO_MASKS)
       && iop->blend_params->mask_id == formid)
      return iop;
  }
  return NULL;
}

// ... and whether some other group holds it. _is_form_used() walks every group
// of dev->forms, stand-alone ones included, which is exactly this question.
// only "> 0" is read here, so its long-standing habit of counting a nested
// group twice cannot reach the screen through this call
static gboolean _group_is_filed(const dt_mask_id_t formid)
{
  char str[1000] = "";
  int nb = 0;
  _is_form_used(formid, NULL, str, sizeof(str), &nb);
  return nb > 0;
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
  char str2[1000] = "";
  // the badge this count used to raise is gone and TREE_LINK now says more
  // than a number could, so nothing displays it any more. it survives as a
  // yes-or-no: is this form filed in any group at all, which is what decides
  // whether the tooltip below has a list to introduce
  int nbuse = 0;

  if(grp_id == 0)
  {
    // the names of the groups holding this form, which is the tooltip's
    // historical content, quirks included. kept apart from str2 now: it is a
    // bare list, and a bare list is an ingredient and not a sentence
    char groups[1000] = "";
    _is_form_used(form->formid, NULL, groups, sizeof(groups), &nbuse);
    g_strlcpy(str2, groups, sizeof(str2));

    if(!(form->type & DT_MASKS_GROUP))
    {
      // a library row. the word itself goes in TREE_LINK, written by
      // _set_iter_name like every other derived column; what is settled here is
      // the sentence behind it, which needs the names of the groups holding the
      // shape and so cannot be had from the scope alone. short on the row,
      // spelled out under the list and in this tooltip: a library where every
      // row carries a sentence is a library nobody reads.
      // NULL for the names: `groups` above already holds them, and asking
      // _shape_scope() for them a second time would clear it on the very case
      // that needs it, a shape a module does render
      const dt_masks_shape_scope_t scope = _shape_scope(form->formid, NULL, 0);

      if(scope == DT_MASKS_SCOPE_GROUP_ONLY)
        // "unused" would be a lie in one direction and a trap in the other:
        // no module renders it, yet the group holding it is real
        snprintf(str2, sizeof(str2),
                 _("no module uses this shape\n"
                   "it is only filed in:\n%s"), groups);
      else if(scope == DT_MASKS_SCOPE_ORPHAN)
        g_strlcpy(str2, _("no module and no group uses this shape"),
                  sizeof(str2));
      else if(nbuse > 0)
        // a module does render it, and the list of groups was all this
        // tooltip ever said on that row. it follows a sentence now, so it
        // needs one of its own rather than standing there as a bare name
        snprintf(str2, sizeof(str2), _("filed in:\n%s"), groups);
    }
  }

  // what a click DOES, on every row of both lists and above whatever else the
  // row has to say. a mask row hands its WHOLE content over:
  // _tree_selection_change() flattens the group through
  // dt_masks_group_ungroup(), so every shape under it -- the two paths of an
  // AI object included -- is drawn at once, and the one under the pointer is
  // the one being edited (develop/masks/group.c reassigns group_edited on
  // every move, so no click is needed to change shape). a shape row hands over
  // that shape alone. said here because nothing else in the panel says it, and
  // a row that only ever answered with a rename box read as a row that could
  // do nothing else
  gchar *state = g_strdup(str2);
  // a raster shape would lie with the "edit on the photograph" promise:
  // it has no anchors, a click can only select it. its sentence says
  // what it is instead, badge behaviour included -- the one place the
  // missing-file state is explained rather than just named
  const char *click_line = (form->type & DT_MASKS_GROUP)
    ? _("click to edit these shapes on the photograph")
    : (form->type & DT_MASKS_RASTER)
      ? _("AI mask: this shape renders a mask file computed from the image\n"
          "it has no anchors to edit on the photograph; it combines,\n"
          "inverts and takes opacity like any other shape.\n"
          "if its file is missing the row says so; right-click to\n"
          "recompute it")
      : _("click to edit this shape on the photograph");
  snprintf(str2, sizeof(str2), "%s%s%s",
           click_line, *state ? "\n\n" : "", state);
  g_free(state);

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
                       TREE_USED_TEXT, str2,
                       -1);
    _set_iter_name(lm, form, gstate, opacity, GTK_TREE_MODEL(treestore), &child);
  }
  else
  {
    // we first check if it's a "module" group or not. the same walk gui_update
    // runs to decide whether this group is a root row at all, so what a row
    // states about its module and whether the row exists are one decision
    if(grp_id == 0 && !module)
      module = _mask_group_owner(form->formid);

    // we add the group node to the tree
    GtkTreeIter child;
    if(toplevel)
      // a group nested in another group is a member of its parent's
      // application order like any shape: same rule as above
      gtk_tree_store_append(treestore, &child, toplevel);
    else
      // at the root there is no application order to show. keep the historical
      // stacking so root rows -- and _tree_group, which builds a new group in
      // the visual order of the selection and is offered on the root rows of
      // the library -- are untouched
      gtk_tree_store_prepend(treestore, &child, NULL);
    gtk_tree_store_set(treestore, &child,
                       TREE_TEXT, str,
                       TREE_MODULE, module,
                       TREE_GROUPID, grp_id,
                       TREE_FORMID, form->formid,
                       TREE_EDITABLE, (grp_id == 0),
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
    // raster rows all share the invalid formid: only the module pointer
    // tells them apart, or a restored selection lands on the first one
    if(found && !dt_is_valid_maskid(formid))
      found = (mod == module);
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
    if(m->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
    {
      _MIX(m->blend_params->mask_id);
      // the raster rows exist per wired consumer and are named after
      // the source: both are structure, or the early-out would keep
      // serving a list without the row a wiring just created
      if(_raster_consumer(m))
      {
        const size_t rsrc = (size_t)m->raster_mask.sink.source;
        _MIX(1u);
        _MIX(rsrc);
        _MIX(rsrc >> 32);
      }
    }
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
    if(!(form->type & DT_MASKS_GROUP)) continue;

    // a root row of this list is a MASK: the group a module points at, or a
    // group nothing else holds. a group filed inside another one is shown
    // where it is filed and only there.
    // dev->forms is a flat sack of ids and membership lives in the groups'
    // points, so being registered there and being a member are not exclusive:
    // an AI object registers its own group AND hands it to the module's mask,
    // which put the same composite on two rows -- one nested and complete, one
    // at the root among the masks with no module, no switch, no target, no
    // rank and no kind, and a "delete group (shapes are kept)" that emptied
    // the mask above it. what is skipped here is a duplicate, never the only
    // row a form has
    if(!_mask_group_owner(form->formid) && _group_is_filed(form->formid))
      continue;

    _lib_masks_list_recurs(store[0], NULL, form, 0, NULL, 0, 1.0, lm);
  }

  // the raster consumers, after the masks: a module taking its mask from
  // another module's raster output had no row anywhere until now. root
  // rows carrying a module and no form -- TREE_FORMID stays the invalid
  // sentinel every formid reader already answers NULL for -- one per
  // wired consumer, in pipe order
  for(const GList *l = darktable.develop->iop; l; l = g_list_next(l))
  {
    dt_iop_module_t *m = l->data;
    if(!(m->flags() & IOP_FLAGS_SUPPORTS_BLENDING)
       || !_raster_consumer(m)
       || !_mask_target_alive(m->raster_mask.sink.source))
      continue;

    GtkTreeIter riter;
    gtk_tree_store_append(store[0], &riter, NULL);
    gtk_tree_store_set(store[0], &riter,
                       TREE_TEXT, "",
                       TREE_MODULE, m,
                       TREE_GROUPID, 0,
                       TREE_FORMID, INVALID_MASKID,
                       TREE_EDITABLE, TRUE,
                       -1);
    _set_iter_name(lm, NULL, 0, 1.0f, GTK_TREE_MODEL(store[0]), &riter);
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

        // a shape that landed in the library has no module to show it: a
        // module's shape comes back on screen through dt_masks_set_edit_mode
        // on that module, a library shape only through its row. the select
        // above ran under this update's guard, so the handler said nothing;
        // this is the click on the row it stands for, provided the canvas is
        // free -- a creation in flight is somebody's, and a run between two
        // shapes (creation_continuous, see _creation_bar_update) is about to
        // put its next form up and must not find it dropped from under it.
        // the raster shape of an automatic selection is the case this exists
        // for: nothing else ever puts it on the photograph. the same for a
        // group nothing owns -- the paths of an automatic selection sent to
        // no module -- which is a root row of the masks list carrying no
        // module: TREE_MODULE says which rows have one, a module's own
        // shapes carry it down to the leaves. before the id below on
        // purpose: rebuilding the canvas clears the form gui, and the
        // selected id with it
        GList *rows = gtk_tree_selection_get_selected_rows(selection, NULL);
        dt_iop_module_t *owner = NULL;
        GtkTreeIter sel_iter;
        if(rows && gtk_tree_model_get_iter(model, &sel_iter, rows->data))
          _lib_masks_get_values(model, &sel_iter, &owner, NULL, NULL);

        const dt_masks_form_gui_t *fg = darktable.develop->form_gui;
        const gboolean busy = fg && (fg->creation || fg->creation_module
                                     || fg->creation_continuous);
        if(!owner && !busy)
          _tree_selection_apply(selection, lm);

        // make the just-created shape the active one so the properties reflect
        // it, rather than the previously selected shape: mask_form_selected_id
        // is otherwise only set once a shape is clicked on the canvas.
        darktable.develop->mask_form_selected_id = lm->pending_selectid;
        _update_all_properties(lm);

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
  lm->unlinked = unlinked;
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
  dt_iop_module_t *module = NULL;
  _lib_masks_get_values(model, iter, &module, &grid, &id);

  // we retrieve the forms
  dt_masks_form_t *form = dt_masks_get_from_id(darktable.develop, id);
  if(!form)
  {
    // a raster row refreshes in place like any other; _set_iter_name
    // owns the whole rule
    if(_raster_row(module, id))
      _set_iter_name(data, NULL, 0, 1.0f, model, iter);
    return 0;
  }
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

// a cell renderer has no CSS node, so nothing in the theme can reach it on a
// state: what the row says about itself has to be handed over by hand, once
// per row, right before it is drawn. GtkCellLayout applies the model
// attributes first and this second, so "visible" and "sensitive" above keep
// working untouched -- same pairing as libs/map_locations.c, which binds
// "text" and sets a data func on one renderer.
// two functions and not one taking the column in its user data: the two model
// columns are not interchangeable, and the wrong one here would light the
// wrong glyph on every row of the list at once
static void _power_cell_data(GtkTreeViewColumn *col,
                             GtkCellRenderer *cell,
                             GtkTreeModel *model,
                             GtkTreeIter *iter,
                             gpointer user_data)
{
  gboolean on = FALSE;
  gtk_tree_model_get(model, iter, TREE_MODULE_ON, &on, -1);
  dtgtk_paint_cell_set_active(DTGTK_PAINT_CELL(cell), on);
}

static void _show_cell_data(GtkTreeViewColumn *col,
                            GtkCellRenderer *cell,
                            GtkTreeModel *model,
                            GtkTreeIter *iter,
                            gpointer user_data)
{
  gboolean on = FALSE;
  gtk_tree_model_get(model, iter, TREE_SHOW_ON, &on, -1);
  dtgtk_paint_cell_set_active(DTGTK_PAINT_CELL(cell), on);
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
    // rank and operator: everything that says where the shape sits in the
    // application order, in one strip that does not move with depth
    d->op_col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(d->op_col, "operator");
    gtk_tree_view_append_column(GTK_TREE_VIEW(view), d->op_col);

    // the application rank, first thing on the row: read the column top-down
    // and you read the order the shapes are applied in. right-aligned, small
    // and insensitive: the theme greys it for us -- dark theme and light theme
    // alike -- and no colour is hardcoded.
    // no reserved width any more. two characters were held so a group reaching
    // ten shapes would not shift the names sideways; the bill was 18 px of
    // gutter on every row of every list against the 9 px a digit takes, and M2
    // gives the rank 10. a mask that reaches ten shapes widens the strip once,
    // by one digit, like every other content-driven cell in this view
    renderer = gtk_cell_renderer_text_new();
    g_object_set(renderer,
                 "xalign", 1.0,
                 "xpad", (guint)DT_PIXEL_APPLY_DPI(1),
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

    // no cell for the base marker. M2 draws it as a chip INSIDE the operator
    // slot -- the two are mutually exclusive, rank 0 never has an operator --
    // and a GtkCellAreaBox cannot share one slot between two renderers: it
    // reserves the maximum of each cell over every row of the view, so a
    // second cell cost 30 px on every row of both lists to speak on one row
    // per mask. the strip is still the base's own zone: _op_cell_at_bin()
    // answers DT_MASKS_OP_HIT_BASE over it, and the tooltip there says in a
    // sentence what a four-letter word only hinted at
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
  // the one cell M2 note 7 steps back beside the struck-through name, and the
  // only one of them that reaches a mask row at all. the operator glyph and
  // the inverse marker are left alone on purpose: they exist on shape rows
  // only, the operator is still what a click changes there whatever the module
  // does, and note 7 does not name them
  gtk_tree_view_column_add_attribute(col, renderer,
                                     "sensitive", TREE_MODULE_ON);

  renderer = gtk_cell_renderer_text_new();
  // by the END and not by the middle: what a name loses at the end is a
  // suffix, what it loses in the middle is the word itself. `group
  // `exposure'' ellipsized in the middle came out "gro...n'", which names
  // nothing at all -- M2 draws no cut name, and END is the closest thing to
  // that when one has to be cut
  g_object_set(renderer, "ellipsize", PANGO_ELLIPSIZE_END, NULL);
  gtk_tree_view_column_pack_start(col, renderer, TRUE);
  gtk_tree_view_column_add_attribute(col, renderer, "text", TREE_TEXT);
  // "editable" is deliberately NOT bound to TREE_EDITABLE. a cell editable at
  // rest is a cell GTK starts editing on a plain click:
  // gtk_tree_view_multipress_gesture_pressed() decides whether to edit BEFORE
  // it looks at the click count, on one test -- is the clicked row already the
  // anchor -- and a selection made in code sets that anchor
  // (_gtk_tree_selection_internal_select_node). so the first click on a row
  // _restore_selection() had just put back opened the editor, ten seconds
  // after the selection or ten minutes, and the second press of a real double
  // click was swallowed by the same branch: "row-activated" never fired on a
  // row that had a name. the cell is armed for the length of one rename
  // instead, by _tree_start_rename(), exactly as libs/map_locations.c arms its
  // own. TREE_EDITABLE stays the policy -- which rows own their name -- and is
  // read by that function and by the context menu
  // struck through, never greyed: an insensitive cell is not editable, and
  // greying the name is the one change that would make a mask unrenamable.
  // both properties bound, not just the first: "strikethrough" is ignored
  // while "strikethrough-set" is false, and one renderer draws the whole
  // column -- set once on one row it would stay set on every row below
  gtk_tree_view_column_add_attribute(col, renderer,
                                     "strikethrough", TREE_MODULE_OFF);
  gtk_tree_view_column_add_attribute(col, renderer,
                                     "strikethrough-set", TREE_MODULE_OFF);
  g_signal_connect(renderer, "edited", G_CALLBACK(_tree_cell_edited), view);
  // the other way an edit ends. without it an escaped rename would leave the
  // cell armed, and the next plain click on that row would open the editor --
  // the very behaviour the unbinding above exists to remove
  g_signal_connect(renderer, "editing-canceled",
                   G_CALLBACK(_tree_editing_canceled), NULL);
  // ... and the receipt that the editor opened at all, which is what tells an
  // arming that will end from one that never would
  g_signal_connect(renderer, "editing-started",
                   G_CALLBACK(_tree_editing_started), d);
  dt_gui_commit_on_focus_loss(renderer, NULL);
  // both lists: a mask at the root of the top list owns its name as much as a
  // shape in the library does -- renaming it is what puts "sky -> exposure" on
  // the row, see the TREE_TARGET note in _set_iter_name
  d->name_col[library ? 1 : 0] = col;
  d->name_cell[library ? 1 : 0] = renderer;

  if(!library)
  {
    // the opacity of this shape in this mask, at the right edge of the row.
    // small and insensitive, like the rank on the other side -- the theme
    // greys it, dark and light alike, and no colour is written here.
    // no reserved gutter any more: four characters were held for "100%", the
    // one string this cell is written never to show, and the 32 px it cost
    // were paid on every row of every panel whose shapes are all at 100 % --
    // which is most panels. the names move by that much the first time a
    // shape leaves 100 %, once, and stop moving after that
    renderer = gtk_cell_renderer_text_new();
    g_object_set(renderer,
                 "xalign", 1.0,
                 "xpad", (guint)DT_PIXEL_APPLY_DPI(2),
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

  if(!library)
  {
    // M2's power switch, at the right edge and in a column of its own -- a
    // click target beside the operator strip, and the last thing the row
    // reads. appended after the name column, which is the one marked
    // expanding, so the leftover width still goes to the names and this strip
    // keeps its own.
    //
    // no fixed width. sixteen raw points against a cell that asks for the
    // row's line height gave a 14 px cell, and _paint_cell_render then took a
    // further fifth off each side: 10 px of glyph, where a .dt_module_btn
    // holding the very same switch is near 16. it was also the one
    // GTK_TREE_VIEW_COLUMN_FIXED left in darktable, and the precedent this
    // column claimed -- gui/preferences_ai.c and its info column -- sets no
    // sizing and no width at all. left natural, the column is 25 px and the
    // glyph 19: the size of the same switch in the module header, which is
    // the one place a reader compares it to
    d->power_col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(d->power_col, "power");
    gtk_tree_view_append_column(GTK_TREE_VIEW(view), d->power_col);

    // ONE renderer, one drawing: dtgtk_cairo_paint_switch, the glyph the
    // module's own enable button draws (dt_iop_gui_set_enable_button_icon), so
    // the two switches of one module are the same switch. that button says on
    // or off by its COLOUR and nothing else -- button:checked, one CSS rule --
    // and this cell now says it the same way, which is also what M2 asks of it
    // (".pw" against ".pw.on", one glyph, two colours).
    // NOT dtgtk_cairo_paint_switch_on for the lit state, tempting as the
    // filled disc is: that drawing already means "this module cannot be
    // switched off" in the header and in the history stack, and a third
    // meaning for it in a third list is how an icon set stops being one.
    // "visible" is TREE_POWER -- a mask row with a reachable switch --
    // "sensitive" is TREE_MODULE_ON, already written for the struck-through
    // name, and the lit colour is set per row by the data func below. three
    // levels: lit, resting, stepped back, which is M2 note 7
    renderer = dtgtk_paint_cell_new(dtgtk_cairo_paint_switch, 0, NULL);
    g_object_set(renderer, "xpad", (guint)DT_PIXEL_APPLY_DPI(1), NULL);
    dtgtk_paint_cell_set_active_color(DTGTK_PAINT_CELL(renderer),
                                      "masks_row_on_fg");
    gtk_tree_view_column_pack_start(d->power_col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(d->power_col, renderer,
                                       "visible", TREE_POWER);
    gtk_tree_view_column_add_attribute(d->power_col, renderer,
                                       "sensitive", TREE_MODULE_ON);
    gtk_tree_view_column_set_cell_data_func(d->power_col, renderer,
                                            _power_cell_data, NULL, NULL);

    // M2 reads the row left to right and ends on the switch, with the
    // show-mask cell just before it. INSERTED and not appended, at the index
    // the switch built ten lines above now occupies: appending would put this
    // to the right of it. counted rather than written down, so a column added
    // before this block does not silently move it
    d->show_col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(d->show_col, "show mask");
    gtk_tree_view_insert_column
      (GTK_TREE_VIEW(view), d->show_col,
       gtk_tree_view_get_n_columns(GTK_TREE_VIEW(view)) - 1);

    // dtgtk_cairo_paint_showmask is the glyph the module's own header
    // indicator draws (dt_iop_add_remove_mask_indicator), so the two ways to
    // reach the same request are the same drawing -- and, like the switch
    // beside it, the same drawing lit and unlit, told apart by colour alone.
    // the very colour, in fact: one name for both cells, because a reader
    // scanning the right edge of the list is reading one question, "what is
    // this row doing right now", and two accents would read as two.
    // this is the one place M2 and the blending panel disagree. the blending
    // panel's own display-mask button lights in @field_active_fg, a grey;
    // M2 puts its accent on the cell, and so does this, because the mask is
    // drawn in yellow OVER THE PHOTOGRAPH and this cell is what puts it there.
    // named in darktable.css, so a theme that wants the grey back is one line
    renderer = dtgtk_paint_cell_new(dtgtk_cairo_paint_showmask, 0, NULL);
    g_object_set(renderer, "xpad", (guint)DT_PIXEL_APPLY_DPI(1), NULL);
    dtgtk_paint_cell_set_active_color(DTGTK_PAINT_CELL(renderer),
                                      "masks_row_on_fg");
    gtk_tree_view_column_pack_start(d->show_col, renderer, FALSE);
    gtk_tree_view_column_add_attribute(d->show_col, renderer,
                                       "visible", TREE_SHOW);
    // "sensitive" is TREE_MODULE_ON and no longer TREE_SHOW_ON: stepped back
    // here means "this module renders no blend, so there is nothing of its
    // mask to put on screen" -- the same thing the struck-through name says
    // one column over, and the same thing M2 note 7 does to the whole row.
    // and it is not silence either: _op_cell_at_bin() still answers over the
    // cell with DT_MASKS_OP_HIT_SHOW_OFF, which is the tooltip that says why.
    // what TREE_SHOW_ON says, "this one is the mask on screen", is now said
    // by the lit colour, where it belongs
    gtk_tree_view_column_add_attribute(d->show_col, renderer,
                                       "sensitive", TREE_MODULE_ON);
    gtk_tree_view_column_set_cell_data_func(d->show_col, renderer,
                                            _show_cell_data, NULL, NULL);
  }

  // a row that reaches no module says so in words, on BOTH lists and for a
  // different row of each: in the library, what the badge could never say --
  // this shape reaches no module, and whether it at least sits in a group; in
  // the masks list, the one root row that is not a module's mask. blank on
  // every other row, the shapes a module renders included, which is the
  // ordinary case and needs no word. same treatment as the rank column --
  // small, insensitive, greyed by the theme, no colour in the C. no
  // ellipsizing: the name cell expands and gives way, a truncated statement
  // would read as a different statement. written in _set_iter_name like every
  // other derived column, and packed at the end after the opacity and the
  // target, which are both empty on that root row
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
  if(library)
    gtk_tree_view_set_show_expanders(GTK_TREE_VIEW(view), FALSE);

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
  // both lists now: the library has no operator column -- the hit test can
  // never match there, the handler returns at once -- but the drag of one of
  // its rows starts from this very handler
  dt_gui_connect_motion(view, _tree_motion_cb, NULL, _tree_leave_cb, self);
  dt_gui_connect_click_all(view, _tree_button_pressed_cb, NULL, self);
  // the double click, on both lists. it is emitted by the tree view's own
  // gesture, which our handler above never claims -- see the note there
  g_signal_connect(view, "row-activated",
                   G_CALLBACK(_tree_row_activated_cb), self);
  // both lists: the same menu, the same entries, and no reason for one of them
  // to answer the keyboard and the other not
  g_signal_connect(view, "popup-menu", G_CALLBACK(_tree_popup_menu_cb), self);
  // the drag leaves either list -- started by hand from _tree_motion_cb once
  // a press armed above travels the threshold -- and only the masks list
  // receives: the library is where shapes rest, not a place a drag
  // rearranges. no GTK_DEST_DEFAULT_*: the veto lives in "drag-motion",
  // which answers per row, where DEFAULT_HIGHLIGHT frames the whole view
  g_signal_connect(view, "drag-begin", G_CALLBACK(_tree_drag_begin_cb), self);
  g_signal_connect(view, "drag-end", G_CALLBACK(_tree_drag_end_cb), self);
  g_signal_connect(view, "drag-data-get",
                   G_CALLBACK(_tree_drag_data_get_cb), self);
  if(!library)
  {
    gtk_drag_dest_set(view, 0, &_masks_dnd_target, 1,
                      GDK_ACTION_MOVE | GDK_ACTION_LINK);
    g_signal_connect(view, "drag-motion",
                     G_CALLBACK(_tree_drag_motion_cb), self);
    g_signal_connect(view, "drag-leave",
                     G_CALLBACK(_tree_drag_leave_cb), self);
    g_signal_connect(view, "drag-drop",
                     G_CALLBACK(_tree_drag_drop_cb), self);
  }
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
  // the operator glyphs, from the one table that also builds the menu entries
  // and decides which one a row shows. wider than they are high because they
  // are two overlapping circles: dtgtk_cairo_paint_masks_union() and its four
  // siblings draw 3.4 radii across and 2 down, the radius capped under both
  // w / 3.4 and h / 2. those two caps meet at w = 1.7 * h, so that is the
  // narrowest pixbuf still giving the glyph its full height. twice the height
  // was past that point and paid for it in transparency -- two and a half
  // points down each side of every shape row, in the one strip with nothing
  // to spare
  const int opw = (bs2 * 17) / 10;
  for(int i = 0; i < (int)G_N_ELEMENTS(_masks_operators); i++)
    d->ic_op[i] = _get_pixbuf_from_cairo(_masks_operators[i].paint,
                                         opw, bs2);

  // the kind glyphs, from the table that also draws the six buttons above the
  // list: a row and the button that made it show the same drawing. square,
  // like the inverse marker -- the operators are the only wide ones
  for(int i = 0; i < (int)G_N_ELEMENTS(_new_mask_shapes); i++)
    d->ic_type[i] = _get_pixbuf_from_cairo(_new_mask_shapes[i].paint,
                                           bs2, bs2);
  d->ic_raster = _get_pixbuf_from_cairo(dtgtk_cairo_paint_masks_raster,
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
  // M2 puts 5 to 6 px between the objects of a row; dt_gui_hbox() opens at 0
  // and only the 0.07em margin of a button separated anything. two points on
  // top of that margin lands on M2 without touching the theme
  gtk_box_set_spacing(GTK_BOX(d->target_row), DT_PIXEL_APPLY_DPI(2));

  // R2: the six shapes, one click each. six buttons ask for about 148 px of
  // their own (1.15em min-width + 0.07em margin + 1px padding + 1px border,
  // six times over) plus the five 2 px gaps below, 180 px inside the panel's
  // 0.65em side padding -- against 320 px for the same six with the module
  // name on the same line, twice the 150 px min_panel_width. hence two rows
  d->shape_row = dt_gui_hbox();
  // the same gap the row above takes, for the same reason
  gtk_box_set_spacing(GTK_BOX(d->shape_row), DT_PIXEL_APPLY_DPI(2));
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

  // the hint row. it belongs to the panel and not to either resize wrapper:
  // a row added inside one of them would change what
  // "plugins/darkroom/masks/heightview" measures, and the list would come back
  // one row shorter at the next start
  d->creation_label = dt_ui_label_new("");
  GtkWidget *cancel = dtgtk_button_new(dtgtk_cairo_paint_cancel, 0, NULL);
  d->bt_creation_cancel = cancel;
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("cancel creation"),
                   cancel, &dt_action_def_button);
  // it calls off whichever of the two the row is showing -- a shape about to
  // be drawn, or an operator waiting for one -- and the row only shows the
  // cross when there is one of them to call off, so a word for both
  gtk_widget_set_tooltip_text(cancel, _("call it off"));
  g_signal_connect(G_OBJECT(cancel), "clicked",
                   G_CALLBACK(_creation_bar_cancel), d);
  // the operator bar, above the creation bar and below both lists: one says
  // how the next shape combines, the other where it goes.
  // the glyph AND the word, which is what M2 draws -- the earlier reading of
  // that sketch, "words against three unlabelled symbols", was a choice
  // between two things it never offered. the glyph is the drawing the shape's
  // own row will show once it is in, so the button and its result are one
  // drawing; the word says what a glyph cannot before anything has been drawn.
  // and the row is measurable now: minimum 172 px against 214 before, under
  // the 228 px the six shape buttons of the row above already ask for, so it
  // can no longer be the row GTK cuts off at the right edge
  // the two words the bar never said. the question it kept raising was not
  // which operator, it was WHEN: the answer is "the next one you draw", and it
  // is written rather than left to a tooltip nobody hovers
  d->arm_label = dt_ui_label_new(_("next shape:"));
  // NOT expanding: shoulder to shoulder with three hexpanding buttons an
  // expanding label claimed a quarter of the bar. and NOT capped either any
  // more -- the cap was ten characters, set when the label repeated a mask
  // name the highlighted row already carried and could give way first. it now
  // carries the only statement of what the three buttons are about, and the
  // cap would have cut it in every language whose translation runs past it:
  // "prochaine forme :" is seventeen characters. what still lets the bar be
  // narrower than its sentence is the ellipsize dt_ui_label_new() sets, on a
  // natural width that is now the sentence itself.
  // dimmed, which is the level M2 gives it
  gtk_widget_set_tooltip_text
    (d->arm_label, _("how the next shape you draw enters this mask"));
  gtk_label_set_ellipsize(GTK_LABEL(d->arm_label), PANGO_ELLIPSIZE_END);
  dt_gui_add_class(d->arm_label, "dt_dimmed");
  d->arm_bar = dt_gui_hbox(d->arm_label);
  gtk_box_set_spacing(GTK_BOX(d->arm_bar), DT_PIXEL_APPLY_DPI(2));

  for(int i = 0; i < (int)G_N_ELEMENTS(_arm_operators); i++)
  {
    const int op = _op_index(_arm_operators[i].state);
    GtkWidget *w = gtk_toggle_button_new();
    GtkWidget *word = gtk_label_new(_(_arm_operators[i].label));
    // ellipsized, so the minimum of the button is the glyph and a letter and
    // the bar can always be narrower than the panel; named, so the theme sets
    // it one step under the body text as M2 does, and no size is written here
    gtk_label_set_ellipsize(GTK_LABEL(word), PANGO_ELLIPSIZE_END);
    gtk_widget_set_name(word, "masks-arm-word");
    // the same two sizes the row's own operator pixbuf is rasterised at, so
    // the button and the row it will produce show one glyph at one size
    GtkWidget *glyph = gtk_drawing_area_new();
    gtk_widget_set_size_request(glyph, opw, bs2);
    gtk_widget_set_valign(glyph, GTK_ALIGN_CENTER);
    g_signal_connect(G_OBJECT(glyph), "draw", G_CALLBACK(_arm_glyph_draw),
                     GINT_TO_POINTER(op));
    GtkWidget *content = dt_gui_hbox(glyph, word);
    gtk_box_set_spacing(GTK_BOX(content), DT_PIXEL_APPLY_DPI(2));
    gtk_container_add(GTK_CONTAINER(w), content);
    // the operator's own name, from the msgid the row's context menu already
    // uses: the word on the button and the word in that menu are two names for
    // one thing, and this is where they are put side by side. the third line
    // is M2's cross on the lit button, said rather than drawn -- a toggle that
    // disarms on a second click is that cross with one target fewer
    gchar *tip = g_strdup_printf(_("%s\nthe next shape you draw enters the "
                                   "mask this way\nit stays armed until you "
                                   "click it again"),
                                 _(_masks_operators[op].name));
    gtk_widget_set_tooltip_text(w, tip);
    g_free(tip);
    g_signal_connect(G_OBJECT(w), "toggled", G_CALLBACK(_bt_arm_cb),
                     GINT_TO_POINTER(_arm_operators[i].state));
    d->bt_arm[i] = w;
    dt_gui_box_add(d->arm_bar, dt_gui_expand(w));
  }

  d->creation_bar = dt_gui_hbox(dt_gui_expand(d->creation_label), cancel);
  gtk_box_set_spacing(GTK_BOX(d->creation_bar), DT_PIXEL_APPLY_DPI(2));

  // the masks on top, the shapes they are drawn from below. a row in the top
  // list belongs to a mask -- deleting it detaches it. a row in the library IS
  // the shape. that is the whole point of the split.
  //
  // both lists go away together, and only when both are empty -- and the
  // library's heading goes with them, the cleanup button on it included. that
  // costs nothing it could reach: the library lists every shape dev->forms
  // holds bar the ones retouch and spot removal own, so an empty library is a
  // library with nothing unlinked in it, and what the cleanup would still find
  // beyond that -- a shape only an undone history step refers to -- was never
  // reachable in that state either, both trees being hidden and the blank
  // click that used to open the menu with them.
  //
  // the second argument is a FLOOR and not a starting height: dt_ui_resize_wrap
  // stashes it as the scrolled window's negative min-content-height, and
  // _resize_wrap_draw() raises the content height to it before it clamps.
  // 200 and 120 meant a panel holding one mask and one shape -- about 75 px of
  // rows -- reserved 320 px and drew 245 of them empty. four rows here, two
  // below: enough for the box to read as a list rather than as a slot, and the
  // box still grows with its content up to the two conf values (300 / 150)
  d->masks_box = dt_ui_resize_wrap(d->treeview, 96,
                                   "plugins/darkroom/masks/heightview");
  // xalign 0: dt_ui_section_label_new centres its text, and centred over a
  // full-width rule is how darktable draws a separator -- which is why this one
  // read as a footer under the list above instead of a heading for the list
  // below. the class stays, so colour, weight and rule still come from the
  // theme. same fix on "properties", built by the same helper
  d->lib_label = dt_ui_section_label_new(C_("section", "shape library"));
  gtk_label_set_xalign(GTK_LABEL(d->lib_label), 0.0f);
  d->bt_cleanup = dtgtk_button_new(dtgtk_cairo_paint_remove, 0, NULL);
  gtk_widget_set_tooltip_text
    (d->bt_cleanup,
     _("delete the shapes no module and no history step uses"));
  dt_action_define(DT_ACTION(self), N_("shapes"), N_("delete unused shapes"),
                   d->bt_cleanup, &dt_action_def_button);
  g_signal_connect(G_OBJECT(d->bt_cleanup), "clicked",
                   G_CALLBACK(_tree_cleanup), self);
  // on the heading of the list it empties, and as far from the six creation
  // buttons as this panel goes. it was in a menu that only opened where there
  // was blank space left to click -- so it went out of reach on a full list
  // and out of existence in the empty state
  d->lib_row = dt_gui_hbox(dt_gui_expand(d->lib_label), d->bt_cleanup);
  gtk_box_set_spacing(GTK_BOX(d->lib_row), DT_PIXEL_APPLY_DPI(2));
  d->lib_box = dt_ui_resize_wrap(d->library, 48,
                                 "plugins/darkroom/masks/heightlibrary");

  self->widget = dt_gui_vbox
    (d->empty_title,
     d->bt_new,
     d->empty_hint,
     d->target_row,
     d->shape_row,
     d->masks_box,
     d->lib_row,
     d->lib_box,
     d->lib_unlinked,
     d->arm_bar,
     d->creation_bar);

  // gui_update decides whether the caption is up. show_all first, then
  // no_show_all and an explicit hide, or a later panel show_all brings it back
  // -- same pattern as the shrink/grow slider further down
  gtk_widget_show_all(d->lib_unlinked);
  gtk_widget_set_no_show_all(d->lib_unlinked, TRUE);
  gtk_widget_hide(d->lib_unlinked);

  // same treatment for the operator bar, and for the same reason: raise the
  // label and the three toggles once, then no_show_all, or the panel-wide
  // gtk_widget_show_all() of libs/lib.c decides this instead of
  // _arm_bar_update. hidden, a box child takes no height at all
  gtk_widget_show_all(d->arm_bar);
  gtk_widget_set_no_show_all(d->arm_bar, TRUE);
  gtk_widget_hide(d->arm_bar);

  // same for the hint row: show_all reaches the label and the cross once, then
  // no_show_all keeps a later panel-wide show_all from putting the row back up
  // -- and from putting the cross back with it, which _arm_bar_update() takes
  // down on the one state of the row that has nothing to call off
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
  // the row and not the label alone: show_all has to reach the cleanup button
  // packed beside it, and it is the row the empty state now hides
  gtk_widget_show_all(d->lib_row);
  gtk_widget_show_all(d->lib_box);
  gtk_widget_set_no_show_all(d->masks_box, TRUE);
  gtk_widget_set_no_show_all(d->lib_row, TRUE);
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
                                       0, _masks_properties[i].defval, 2);
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
    if(_masks_properties[i].tooltip)
      gtk_widget_set_tooltip_text(w, _(_masks_properties[i].tooltip));
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
  // the AI menu, for the ✦ of every blending panel: one menu for both
  // buttons, built here where the detectors and the session entry live
#ifdef HAVE_AI
  darktable.develop->proxy.masks.object_menu = _object_menu_proxy;
#else
  darktable.develop->proxy.masks.object_menu = NULL;
#endif
}

void gui_cleanup(dt_lib_module_t *self)
{
  dt_lib_masks_t *d = self->data;
  if(d && d->resize_timer)
    g_source_remove(d->resize_timer);
  // the armament outlives the form on purpose; it must not outlive the panel
  // that is the only way to see it and the only way to put it down
  dt_masks_set_next_operator(DT_MASKS_STATE_NONE, NULL);
  g_free(self->data);
  self->data = NULL;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
