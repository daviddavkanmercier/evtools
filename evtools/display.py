"""
Display functions for DSVector objects.

Four mutually exclusive output formats are available, each as both a plain
function and a DSVector method:

    to_string(m)  / m.to_string()    — plain text, no colors (logs, files)
    to_ansi(m)    / m.to_ansi()      — colored terminal output (ANSI codes)
    to_html(m)    / m.to_html()      — HTML table for Jupyter notebooks
    to_latex(m)   / m.to_latex()     — LaTeX tabular environment for papers

All four accept ``all_kinds=False``. When ``True`` (and ``m.kind == Kind.M``),
they render every standard representation (m, bel, pl, b, q, plus v if the
BBA is subnormal and w if it is non-dogmatic) in a single table.

DSVector integration
--------------------
- DSVector.__repr__      calls to_ansi   (terminal default)
- DSVector._repr_html_   calls to_html   (Jupyter auto-display)

Usage
-----
    from evtools.display import to_string, to_latex

    m = DSVector.from_focal(["a", "h", "r"], {"a": 0.5, "r": 0.5})

    print(m)                             # ANSI (via __repr__)
    print(m.to_string())                 # plain text
    print(m.to_latex())                  # LaTeX for a paper
    print(m.to_string(all_kinds=True))   # m + bel + pl + b + q + (v) + (w)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dsvector import DSVector

from .constants import DISPLAY_TOL, ZERO_MASS


# ---------------------------------------------------------------------------
# Internal helpers — shared by all four formats
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


def _all_kinds_data(m: "DSVector"):
    """
    Build the data needed by the ``all_kinds=True`` view.

    Returns
    -------
    col_labels : list of kind symbols ("m", "bel", "pl", "b", "q", optionally "v", "w")
    subset_labels : list of human-readable subset labels (one per row)
    rows : list of (subset, [value_for_each_kind]) pairs (filtered to non-zero rows)
    """
    from .dsvector import Kind, _subset_index
    from .conversions import mtobel, mtopl, mtob, mtoq, mtov, mtow

    if m.kind != Kind.M:
        raise ValueError(
            f"all_kinds=True requires a BBA (kind='m'), got kind='{m.kind.value}'."
        )

    omega = frozenset(m.frame)
    is_subnormal   = m[frozenset()] > ZERO_MASS
    is_nondogmatic = m[omega] > ZERO_MASS
    kinds = [
        ("m",   m.dense),
        ("bel", mtobel(m.dense)),
        ("pl",  mtopl(m.dense)),
        ("b",   mtob(m.dense)),
        ("q",   mtoq(m.dense)),
    ]
    if is_subnormal:
        try:
            kinds.append(("v", mtov(m.dense)))
        except Exception:
            pass
    if is_nondogmatic:
        try:
            kinds.append(("w", mtow(m.dense)))
        except Exception:
            pass

    n_atoms = len(m.frame)
    n = 2 ** n_atoms
    all_subsets = sorted(
        [frozenset(atom for k, atom in enumerate(m.frame) if i >> k & 1)
         for i in range(n)],
        key=lambda s: _subset_index(s, m.frame),
    )

    def _row_values(subset: frozenset) -> list[float]:
        idx = _subset_index(subset, m.frame)
        return [arr[idx] for _, arr in kinds]

    rows = [(s, _row_values(s)) for s in all_subsets
            if any(abs(v) > ZERO_MASS for v in _row_values(s))]

    col_labels = [k for k, _ in kinds]
    subset_labels = [_subset_label(s, m.frame) for s, _ in rows]
    return col_labels, subset_labels, rows


# ---------------------------------------------------------------------------
# ANSI — colored terminal output
# ---------------------------------------------------------------------------

_RESET     = "\033[0m"
_BOLD      = "\033[1m"
_DIM       = "\033[2m"
_BLUE      = "\033[34m"
_GREEN     = "\033[32m"
_ORANGE    = "\033[33m"
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


def _to_ansi_single(m: "DSVector") -> str:
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


def _to_ansi_all_kinds(m: "DSVector") -> str:
    B, R, D = _BOLD, _RESET, _DIM
    col_labels, subset_labels, rows = _all_kinds_data(m)
    val_w = 9
    col_w = max(max((len(s) for s in subset_labels), default=6), 6)
    frame_str = "{" + ", ".join(m.frame) + "}"
    is_sub = m[frozenset()] > ZERO_MASS

    lines = [
        f"{B}DSVector{R}  kind={B}m{R}  {D}(Basic Belief Assignment){R}  "
        f"frame={B}{frame_str}{R}"
        + (f"  {D}(subnormal){R}" if is_sub else ""),
        "",
    ]
    header = f"  {B}{'Subset':<{col_w}}{R}"
    for k in col_labels:
        color = _KIND_COLOR.get(k, "")
        header += f"  {color}{B}{k:>{val_w}}{R}"
    sep = f"  {D}{'─' * (col_w + (val_w + 2) * len(col_labels))}{R}"
    lines += [header, sep]
    for label, (_, vals) in zip(subset_labels, rows):
        line = f"  {_BLUE}{label:<{col_w}}{R}"
        for v in vals:
            line += f"  {B}{v:>{val_w}.4f}{R}"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def to_ansi(m: "DSVector", all_kinds: bool = False) -> str:
    """
    Render *m* as an ANSI-colored terminal string.

    Parameters
    ----------
    m : DSVector
        The belief function to display.
    all_kinds : bool, default False
        If ``True``, render every standard representation (m, bel, pl, b, q,
        and v/w when applicable) in a single table. Requires ``m.kind == M``.

    Returns
    -------
    str
        ANSI-colored string suitable for terminal output.
    """
    return _to_ansi_all_kinds(m) if all_kinds else _to_ansi_single(m)


# ---------------------------------------------------------------------------
# Plain text — no colors
# ---------------------------------------------------------------------------

def _to_string_single(m: "DSVector") -> str:
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


def _to_string_all_kinds(m: "DSVector") -> str:
    col_labels, subset_labels, rows = _all_kinds_data(m)
    val_w = 9
    col_w = max(max((len(s) for s in subset_labels), default=6), 6)
    header = f"  {'Subset':<{col_w}}" + "".join(f"  {k:>{val_w}}" for k in col_labels)
    sep = "  " + "-" * (col_w + (val_w + 2) * len(col_labels))
    lines = ["", header, sep]
    for label, (_, vals) in zip(subset_labels, rows):
        line = f"  {label:<{col_w}}" + "".join(f"  {v:>{val_w}.4f}" for v in vals)
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def to_string(m: "DSVector", all_kinds: bool = False) -> str:
    """
    Render *m* as plain text (no ANSI codes).

    Parameters
    ----------
    m : DSVector
        The belief function to display.
    all_kinds : bool, default False
        If ``True``, render every standard representation (m, bel, pl, b, q,
        and v/w when applicable) in a single table. Requires ``m.kind == M``.

    Returns
    -------
    str
        Plain text string.
    """
    return _to_string_all_kinds(m) if all_kinds else _to_string_single(m)


# ---------------------------------------------------------------------------
# HTML — Jupyter notebook display
# ---------------------------------------------------------------------------

_KIND_HTML_COLOR = {
    "m":   "#17a2b8",
    "bel": "#007bff",
    "pl":  "#28a745",
    "b":   "#ffc107",
    "q":   "#6f42c1",
    "v":   "#dc3545",
    "w":   "#fd7e14",
}


def _to_html_single(m: "DSVector") -> str:
    from .dsvector import Kind

    kind_label = _kind_label(m)
    frame_str  = "{" + ", ".join(m.frame) + "}"
    n_focal    = len(m.sparse)
    items      = _sorted_items(m)
    max_val    = max(abs(v) for _, v in items) if items else 1.0
    color      = _KIND_HTML_COLOR.get(m.kind.value, "#6c757d")

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


def _to_html_all_kinds(m: "DSVector") -> str:
    col_labels, subset_labels, rows = _all_kinds_data(m)
    frame_str = "{" + ", ".join(m.frame) + "}"

    html = [
        f'<div style="font-family:monospace; margin:8px 0;">',
        f'  <span style="font-weight:bold;">DSVector</span>',
        f'  <span style="color:#17a2b8; font-weight:bold;">m</span>',
        f'  <span style="color:#6c757d;">(Basic Belief Assignment)</span>',
        f'  &nbsp; frame=<b>{frame_str}</b>',
        f'  &nbsp; <span style="color:#6c757d;">all representations</span>',
        f'</div>',
        f'<table style="border-collapse:collapse; font-family:monospace; font-size:0.9em;">',
        f'  <thead><tr style="border-bottom:2px solid #dee2e6;">',
        f'    <th style="text-align:left; padding:4px 12px;">Subset</th>',
    ]
    for k in col_labels:
        color = _KIND_HTML_COLOR.get(k, "#6c757d")
        html.append(f'    <th style="text-align:right; padding:4px 12px; color:{color};">${k}$</th>')
    html += ['  </tr></thead>', '  <tbody>']

    for label, (_, vals) in zip(subset_labels, rows):
        html.append(f'    <tr style="border-bottom:1px solid #f0f0f0;">')
        html.append(f'      <td style="padding:3px 12px; color:#0056b3;">{label}</td>')
        for v in vals:
            html.append(f'      <td style="padding:3px 12px; text-align:right;">{v:.4f}</td>')
        html.append(f'    </tr>')

    html += ['  </tbody>', '</table>']
    return "\n".join(html)


def to_html(m: "DSVector", all_kinds: bool = False) -> str:
    """
    Render *m* as an HTML table for display in Jupyter notebooks.

    Jupyter calls ``DSVector._repr_html_()`` automatically, which delegates
    to this function with ``all_kinds=False``.

    Parameters
    ----------
    m : DSVector
        The belief function to display.
    all_kinds : bool, default False
        If ``True``, render every standard representation in a single table.
        Requires ``m.kind == M``.

    Returns
    -------
    str
        HTML string.
    """
    return _to_html_all_kinds(m) if all_kinds else _to_html_single(m)


# ---------------------------------------------------------------------------
# LaTeX — academic paper display
# ---------------------------------------------------------------------------

def _latex_kind_symbol(m: "DSVector") -> str:
    from .dsvector import Kind
    return {
        Kind.M:   "m",
        Kind.BEL: "\\mathrm{bel}",
        Kind.PL:  "\\mathrm{pl}",
        Kind.B:   "b",
        Kind.Q:   "q",
        Kind.V:   "v",
        Kind.W:   "w",
    }.get(m.kind, m.kind.value)


def _latex_subset(subset: frozenset, frame: list[str]) -> str:
    if not subset:
        return "$\\emptyset$"
    atoms = ", ".join(sorted(subset, key=frame.index))
    return f"$\\{{{atoms}\\}}$"


def _to_latex_single(m: "DSVector") -> str:
    from .dsvector import Kind

    kind_symbol = _latex_kind_symbol(m)
    items = _sorted_items(m)
    lines = [
        "\\begin{tabular}{lr}",
        "\\hline",
        f"Subset & ${kind_symbol}$ \\\\",
        "\\hline",
    ]
    for subset, value in items:
        lines.append(f"{_latex_subset(subset, m.frame)} & {value:.4f} \\\\")
    lines.append("\\hline")

    if m.kind == Kind.M:
        total = sum(m.sparse.values())
        lines.append(f"Total & {total:.4f} \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _to_latex_all_kinds(m: "DSVector") -> str:
    col_labels, subset_labels, rows = _all_kinds_data(m)
    col_spec = "l" + "r" * len(col_labels)
    header = "Subset & " + " & ".join(f"${k}$" for k in col_labels) + " \\\\"
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header,
        "\\hline",
    ]
    for (subset, vals), label in zip(rows, subset_labels):
        # Use the LaTeX-aware label
        latex_label = _latex_subset(subset, m.frame)
        row = latex_label + " & " + " & ".join(f"{v:.4f}" for v in vals) + " \\\\"
        lines.append(row)
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)


def to_latex(m: "DSVector", all_kinds: bool = False) -> str:
    """
    Render *m* as a LaTeX tabular environment.

    Produces a ready-to-use LaTeX table that can be embedded in a paper.
    Requires no special packages beyond standard LaTeX.

    Parameters
    ----------
    m : DSVector
        The belief function to display.
    all_kinds : bool, default False
        If ``True``, render every standard representation in a single table.
        Requires ``m.kind == M``.

    Returns
    -------
    str
        LaTeX string containing a tabular environment.
    """
    return _to_latex_all_kinds(m) if all_kinds else _to_latex_single(m)
