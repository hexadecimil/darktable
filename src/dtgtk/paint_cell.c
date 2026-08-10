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

#include "dtgtk/paint_cell.h"
#include "common/darktable.h"
#include "gui/gtk.h"

G_DEFINE_TYPE(GtkDarktablePaintCell, dtgtk_paint_cell, GTK_TYPE_CELL_RENDERER)

// icon edge length, derived from the widget's line height
static int _paint_cell_compute_size(GtkWidget *widget)
{
  int s = DT_PIXEL_APPLY_DPI(12);
  if(widget)
  {
    PangoContext *pctx = gtk_widget_get_pango_context(widget);
    const PangoFontDescription *fd = pango_context_get_font_description(pctx);
    PangoFontMetrics *m = pango_context_get_metrics(pctx, fd, NULL);
    const int line_h = (pango_font_metrics_get_ascent(m)
                        + pango_font_metrics_get_descent(m)) / PANGO_SCALE;
    pango_font_metrics_unref(m);
    if(line_h > 0) s = line_h;
  }
  return s;
}

// xpad and ypad were ignored in both directions -- neither asked for here nor
// taken off the drawing area below. that breaks the GtkCellRenderer contract
// every stock renderer honours, and it left a caller no way to buy air around
// the glyph other than shrinking the glyph
static void _paint_cell_get_preferred_width(GtkCellRenderer *r,
                                            GtkWidget *widget,
                                            gint *minimum_size,
                                            gint *natural_size)
{
  gint xpad = 0, ypad = 0;
  gtk_cell_renderer_get_padding(r, &xpad, &ypad);
  const int s = _paint_cell_compute_size(widget) + 2 * xpad;
  if(minimum_size) *minimum_size = s;
  if(natural_size) *natural_size = s;
}

static void _paint_cell_get_preferred_height(GtkCellRenderer *r,
                                             GtkWidget *widget,
                                             gint *minimum_size,
                                             gint *natural_size)
{
  gint xpad = 0, ypad = 0;
  gtk_cell_renderer_get_padding(r, &xpad, &ypad);
  const int s = _paint_cell_compute_size(widget) + 2 * ypad;
  if(minimum_size) *minimum_size = s;
  if(natural_size) *natural_size = s;
}

static void _paint_cell_render(GtkCellRenderer *r,
                               cairo_t *cr,
                               GtkWidget *widget,
                               const GdkRectangle *bg_area,
                               const GdkRectangle *cell_area,
                               GtkCellRendererState flags)
{
  (void)bg_area;
  GtkDarktablePaintCell *self = DTGTK_PAINT_CELL(r);
  if(!self->paint) return;

  // the state of the CELL, not of the widget. a widget is never selected, so
  // reading its state alone drew every icon of a list in the plain foreground
  // colour -- including the one on the selected row, next to a label that had
  // just switched -- and gave the prelight of the whole view to every icon in
  // it at once. gtk_cell_renderer_get_state() is the one call that folds the
  // widget's state, this renderer's own "sensitive" property and the per-cell
  // flags into one, and it is what every stock renderer of GTK uses
  const GtkStateFlags state = gtk_cell_renderer_get_state(r, widget, flags);

  // and the lit colour by name, because no :checked rule of the theme can
  // reach a cell: there is no CSS node to hang one on. looked up exactly as
  // dt_gui_apply_theme() and bauhaus look up every colour they then draw in
  // cairo. insensitive still wins -- "you cannot reach this" outranks "this
  // one is on", and the two columns using it never combine the two anyway.
  // a caller that named no colour keeps the plain path, byte for byte
  GdkRGBA fg;
  GtkStyleContext *ctx = gtk_widget_get_style_context(widget);
  if(!(self->active
       && self->active_color
       && !(state & GTK_STATE_FLAG_INSENSITIVE)
       && gtk_style_context_lookup_color(ctx, self->active_color, &fg)))
    gtk_style_context_get_color(ctx, state, &fg);

  gint xpad = 0, ypad = 0;
  gtk_cell_renderer_get_padding(r, &xpad, &ypad);
  const int w = cell_area->width  - 2 * xpad;
  const int h = cell_area->height - 2 * ypad;
  if(w <= 0 || h <= 0) return;

  // 5 % a side, which is the breathing room #button-canvas gives every icon
  // button of darktable (data/themes/darktable.css). it was 20 %, and a glyph
  // in a row came out at a third of the area of the same glyph on the button
  // that reaches the same setting -- 10 px against 16.
  // never zero: 5 % of a cell the height of a line of text rounds down to
  // nothing, and a paint function is entitled to the whole box it is given --
  // dtgtk_cairo_paint_showmask fills it corner to corner, and the round cap
  // dtgtk_cairo_paint_switch puts on the top of its stem reaches half a
  // stroke past it. one point keeps either off the cell next door
  const int mx = MAX(1, w / 20);
  const int my = MAX(1, h / 20);

  // CPF_ACTIVE with the lit state, the way dtgtk/togglebutton.c raises it.
  // it draws nothing by itself for the two functions in a cell today --
  // dtgtk_cairo_paint_switch reads only CPF_FOCUS, dtgtk_cairo_paint_showmask
  // reads no flag at all -- it is the contract the rest of paint.c is written
  // against, and the next function dropped into a cell will find it already
  // true
  cairo_save(cr);
  gdk_cairo_set_source_rgba(cr, &fg);
  self->paint(cr,
              cell_area->x + xpad + mx, cell_area->y + ypad + my,
              w - 2 * mx, h - 2 * my,
              self->paint_flags | (self->active ? CPF_ACTIVE : 0),
              self->paint_data);
  cairo_restore(cr);
}

static void dtgtk_paint_cell_class_init(GtkDarktablePaintCellClass *klass)
{
  GtkCellRendererClass *cr_class = GTK_CELL_RENDERER_CLASS(klass);
  cr_class->get_preferred_width  = _paint_cell_get_preferred_width;
  cr_class->get_preferred_height = _paint_cell_get_preferred_height;
  cr_class->render               = _paint_cell_render;
}

static void dtgtk_paint_cell_init(GtkDarktablePaintCell *self)
{
  self->paint = NULL;
  self->paint_flags = 0;
  self->paint_data = NULL;
  self->active_color = NULL;
  self->active = FALSE;
}

GtkCellRenderer *dtgtk_paint_cell_new(DTGTKCairoPaintIconFunc paint,
                                      gint paint_flags,
                                      void *paint_data)
{
  GtkDarktablePaintCell *cell = g_object_new(dtgtk_paint_cell_get_type(), NULL);
  cell->paint = paint;
  cell->paint_flags = paint_flags;
  cell->paint_data = paint_data;
  return GTK_CELL_RENDERER(cell);
}

void dtgtk_paint_cell_set_active_color(GtkDarktablePaintCell *cell,
                                       const char *css_color)
{
  g_return_if_fail(cell != NULL);
  // kept by reference and never copied: the caller hands over a literal, the
  // same way the table of dt_gui_apply_theme() holds its colour names
  cell->active_color = css_color;
}

void dtgtk_paint_cell_set_active(GtkDarktablePaintCell *cell,
                                 const gboolean active)
{
  g_return_if_fail(cell != NULL);
  cell->active = active;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
