"""
Display functions for DSVector objects.

Multiple output formats are available:

    repr_ansi(m)   — colored terminal output with ANSI escape codes
    repr_plain(m)  — plain text, no colors (for logging, files)
    repr_html(m)   — HTML table (for Jupyter notebooks)
    repr_latex(m)  — LaTeX tabular (for academic papers)

DSVector integration
--------------------
- DSVector.__repr__      calls repr_ansi  (terminal default)
- DSVector._repr_html_   calls repr_html  (Jupyter auto-display)
- DSVector.display(fmt)  calls the requested format explicitly

Usage
-----
    from evtools.display import repr_plain, repr_latex

    m = DSVector.from_focal(["a", "h", "r"], {"a": 0.5, "r": 0.5})

    print(repr_plain(m))   # plain text table
    print(repr_latex(m))   # LaTeX tabular environment
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dsvector import DSVector, Kind

from .constants import DISPLAY_TOL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subset_label(subset: frozenset, frame: list[str]) -> str:
    """Return a human-readable label for a subset."""
    if not subset:
        return "∅"
    return "{" + ", ".join(sorted(subset, key=frame.index)) + "}"


def _kind_symbol(m: "DSVector") -> str:
    """Return the mathematical symbol for the kind."""
    from .dsvector import Kind
    symbols = {
        Kind.M:   "m",
        Kind.BEL: "bel",
        Kind.PL:  "pl",
        Kind.B:   "b",
        Kind.Q:   "q",
        Kind.V:   "v",
        Kind.W:   "w",
    }
    return symbols.get(m.kind, m.kind.value)


def _sorted_items(m: "DSVector") -> list[tuple[frozenset, float]]:
    """Return focal elements sorted by binary index."""
    from .dsvector import _subset_index
    return sorted(m.sparse.items(), key=lambda kv: _subset_index(kv[0], m.frame))


def _kind_label(m: "DSVector") -> str:
    """Return the full name of the kind."""
    from .dsvector import Kind
    labels = {
        Kind.M:   "Basic Belief Assignment",
        Kind.BEL: "Belief function",
        Kind.PL:  "Plausibility function",
        Kind.B:   "Commonality function",
        Kind.Q:   "Implicability function",
        Kind.V:   "Disjunctive weights",
        Kind.W:   "Conjunctive weights",
    }
    return labels.get(m.kind, m.kind.value)


# ---------------------------------------------------------------------------
# ANSI — colored terminal display
# ---------------------------------------------------------------------------

# ANSI codes
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_BLUE    = "\033[34m"
_GREEN   = "\033[32m"
_ORANGE  = "\033[33m"
_BAR_FULL  = "█"
_BAR_HALF  = "▌"
_BAR_EMPTY = "░"

_KIND_COLOR = {
    "m":   "\033[36m",
    "bel": "\033[34m",
    "pl":  "\033[32m",
    "b":   "\033[33m",
    "q":   "\033[35m",
    "v":   "\033[31m",
    "w":   "\033[91m",
}


def _bar_ansi(value: float, max_val: float, width: int = 16) -> str:
    """Return a colored progress bar."""
    ratio = max(0.0, min(1.0, abs(value) / max_val)) if max_val else 0.0
    filled = int(ratio * width * 2)
    full, half, empty = filled // 2, filled % 2, width - filled // 2 - filled % 2
    filled_part = f"{_GREEN}{_BAR_FULL * full}{_BAR_HALF * half}{_RESET}"
    empty_part  = f"{_DIM}{_BAR_EMPTY * empty}{_RESET}"
    return filled_part + empty_part


def repr_ansi(m: "DSVector") -> str:
    """
    Render a DSVector as a colored terminal string using ANSI escape codes.

    Includes a progress bar for each focal element, colored by value,
    and a colored total line for mass functions.

    Parameters
    ----------
    m : DSVector
        The belief function to display.

    Returns
    -------
    str
        ANSI-colored string suitable for terminal output.
    """
    from .dsvector import Kind
    B, R, D = _BOLD, _RESET, _DIM
    kind_color = _KIND_COLOR.get(m.kind.value, "")
    kind_label = _kind_label(m)
    frame_str  = "{" + ", ".join(m.frame) + "}"
    n_focal    = len(m.sparse)

    lines = [
        f"{B}DSVector{R}  "
        f"kind={kind_color}{B}{m.kind.value}{R}  "
        f"{D}({kind_label}){R}  "
        f"frame={B}{frame_str}{R}  "
        f"{D}{n_focal} focal element{'s' if n_focal != 1 else ''}{R}"
    ]

    if not m.sparse:
        lines.append(f"  {D}(empty){R}")
        return "\n".join(lines)

    items = _sorted_items(m)
    col_w = max(len(_subset_label(s, m.frame)) for s, _ in items)
    col_w = max(col_w, 6)
    sep = f"  {D}{'─' * (col_w + 2)}{'─' * 10}{'─' * 19}{R}"

    sym = _kind_symbol(m)
    lines += ["", f"  {B}{'Subset':<{col_w}}  {sym:>8}  {'':19}{R}", sep]

    max_val = max(abs(v) for _, v in items) if items else 1.0
    for subset, value in items:
        label = _subset_label(subset, m.frame)
        bar   = _bar_ansi(value, max_val)
        lines.append(f"  {_BLUE}{label:<{col_w}}{R}  {B}{value:>8.4f}{R}  {bar}")

    lines.append(sep)

    if m.kind == Kind.M:
        total = sum(m.sparse.values())
        ok = abs(total - 1.0) < DISPLAY_TOL
        total_color = _GREEN if ok else _ORANGE
        lines.append(f"  {D}{'Total':<{col_w}}  {total_color}{B}{total:>8.4f}{R}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plain text — no colors
# ---------------------------------------------------------------------------

def repr_plain(m: "DSVector") -> str:
    """
    Render a DSVector as plain text without ANSI escape codes.

    Suitable for logging, file output, or terminals that do not support
    ANSI colors.

    Parameters
    ----------
    m : DSVector
        The belief function to display.

    Returns
    -------
    str
        Plain text string.
    """
    from .dsvector import Kind
    kind_label = _kind_label(m)
    frame_str  = "{" + ", ".join(m.frame) + "}"
    n_focal    = len(m.sparse)

    lines = [
        f"DSVector  kind={m.kind.value}  ({kind_label})  "
        f"frame={frame_str}  "
        f"{n_focal} focal element{'s' if n_focal != 1 else ''}"
    ]

    if not m.sparse:
        lines.append("  (empty)")
        return "\n".join(lines)

    items = _sorted_items(m)
    col_w = max(len(_subset_label(s, m.frame)) for s, _ in items)
    col_w = max(col_w, 6)
    sep = "  " + "-" * (col_w + 14)

    sym = _kind_symbol(m)
    lines += ["", f"  {'Subset':<{col_w}}  {sym:>8}", sep]
    for subset, value in items:
        label = _subset_label(subset, m.frame)
        lines.append(f"  {label:<{col_w}}  {value:>8.4f}")

    lines.append(sep)

    if m.kind == Kind.M:
        total = sum(m.sparse.values())
        lines.append(f"  {'Total':<{col_w}}  {total:>8.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML — Jupyter notebook display
# ---------------------------------------------------------------------------

def repr_html(m: "DSVector") -> str:
    """
    Render a DSVector as an HTML table for display in Jupyter notebooks.

    Jupyter calls DSVector._repr_html_() automatically, which delegates
    to this function. The table includes a color-coded value bar column.

    Parameters
    ----------
    m : DSVector
        The belief function to display.

    Returns
    -------
    str
        HTML string.
    """
    from .dsvector import Kind

    kind_label = _kind_label(m)
    frame_str  = "{" + ", ".join(m.frame) + "}"
    n_focal    = len(m.sparse)
    items      = _sorted_items(m)
    max_val    = max(abs(v) for _, v in items) if items else 1.0

    kind_colors = {
        "m": "#17a2b8", "bel": "#007bff", "pl": "#28a745",
        "b": "#ffc107", "q": "#6f42c1",  "v": "#dc3545", "w": "#fd7e14",
    }
    color = kind_colors.get(m.kind.value, "#6c757d")

    html = [
        f'<div style="font-family:monospace; margin:8px 0;">',
        f'  <span style="font-weight:bold;">DSVector</span>',
        f'  <span style="color:{color}; font-weight:bold;">{m.kind.value}</span>',
        f'  <span style="color:#6c757d;">({kind_label})</span>',
        f'  &nbsp; frame=<b>{frame_str}</b>',
        f'  &nbsp; <span style="color:#6c757d;">{n_focal} focal element{"s" if n_focal != 1 else ""}</span>',
        f'</div>',
        f'<table style="border-collapse:collapse; font-family:monospace; font-size:0.9em;">',
        f'  <thead>',
        f'    <tr style="border-bottom:2px solid #dee2e6;">',
        f'      <th style="text-align:left; padding:4px 12px;">Subset</th>',
        f'      <th style="text-align:right; padding:4px 12px;">${_kind_symbol(m)}$</th>',
        f'      <th style="text-align:left; padding:4px 12px; min-width:160px;"></th>',
        f'    </tr>',
        f'  </thead>',
        f'  <tbody>',
    ]

    for subset, value in items:
        label   = _subset_label(subset, m.frame)
        ratio   = abs(value) / max_val if max_val else 0.0
        bar_pct = int(ratio * 100)
        html += [
            f'    <tr style="border-bottom:1px solid #f0f0f0;">',
            f'      <td style="padding:3px 12px; color:#0056b3;">{label}</td>',
            f'      <td style="padding:3px 12px; text-align:right; font-weight:bold;">{value:.4f}</td>',
            f'      <td style="padding:3px 12px;">',
            f'        <div style="background:#e9ecef; border-radius:3px; height:12px; width:150px;">',
            f'          <div style="background:{color}; border-radius:3px; height:12px; width:{bar_pct}%;"></div>',
            f'        </div>',
            f'      </td>',
            f'    </tr>',
        ]

    html.append('  </tbody>')

    if m.kind == Kind.M:
        total = sum(m.sparse.values())
        ok = abs(total - 1.0) < DISPLAY_TOL
        total_color = "#28a745" if ok else "#fd7e14"
        html += [
            f'  <tfoot>',
            f'    <tr style="border-top:2px solid #dee2e6;">',
            f'      <td style="padding:3px 12px; color:#6c757d;">Total</td>',
            f'      <td style="padding:3px 12px; text-align:right; font-weight:bold; color:{total_color};">{total:.4f}</td>',
            f'      <td></td>',
            f'    </tr>',
            f'  </tfoot>',
        ]

    html.append('</table>')
    return "\n".join(html)


# ---------------------------------------------------------------------------
# LaTeX — academic paper display
# ---------------------------------------------------------------------------

def repr_latex(m: "DSVector") -> str:
    """
    Render a DSVector as a LaTeX tabular environment.

    Produces a ready-to-use LaTeX table that can be embedded in a paper.
    Requires no special packages beyond standard LaTeX.

    Parameters
    ----------
    m : DSVector
        The belief function to display.

    Returns
    -------
    str
        LaTeX string containing a tabular environment.

    Example output
    --------------
    \\begin{tabular}{lr}
    \\hline
    Subset & $m$ \\\\
    \\hline
    $\\{a\\}$       & 0.5000 \\\\
    $\\{r\\}$       & 0.5000 \\\\
    \\hline
    Total  & 1.0000 \\\\
    \\hline
    \\end{tabular}
    """
    from .dsvector import Kind

    kind_symbol = {
        Kind.M:   "m",
        Kind.BEL: "\\mathrm{bel}",
        Kind.PL:  "\\mathrm{pl}",
        Kind.B:   "b",
        Kind.Q:   "q",
        Kind.V:   "v",
        Kind.W:   "w",
    }.get(m.kind, m.kind.value)

    def latex_label(subset: frozenset) -> str:
        if not subset:
            return "$\\emptyset$"
        atoms = ", ".join(sorted(subset, key=m.frame.index))
        return f"$\\{{{atoms}\\}}$"

    items = _sorted_items(m)
    lines = [
        "\\begin{tabular}{lr}",
        "\\hline",
        f"Subset & ${kind_symbol}$ \\\\",
        "\\hline",
    ]

    for subset, value in items:
        label = latex_label(subset)
        lines.append(f"{label} & {value:.4f} \\\\")

    lines.append("\\hline")

    if m.kind == Kind.M:
        total = sum(m.sparse.values())
        lines.append(f"Total & {total:.4f} \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    return "\n".join(lines)
