"""
Tests for evtools.display — to_ansi, to_string, to_html, to_latex.
"""

import numpy as np
import pytest
from evtools.dsvector import DSVector, Kind
from evtools.display import to_ansi, to_string, to_html, to_latex

FRAME_AHR = ["a", "h", "r"]

# Normal BBA: m({a})=0.5, m({r})=0.5
M = DSVector.from_focal(FRAME_AHR, {"a": 0.5, "r": 0.5})

# Subnormal BBA
M_SUB = DSVector.from_focal(FRAME_AHR, {"": 0.1, "a": 0.4, "a,h,r": 0.5}, complete=False)

# All representations
M_BEL = M.to_bel()
M_PL  = M.to_pl()
M_B   = M.to_b()
M_Q   = M.to_q()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_all_kinds(fn, check):
    """Apply check(output, kind_symbol) for all standard kinds."""
    for v, sym in [(M, "m"), (M_BEL, "bel"), (M_PL, "pl"), (M_B, "b"), (M_Q, "q")]:
        result = fn(v)
        check(result, sym, v)


# ---------------------------------------------------------------------------
# to_string
# ---------------------------------------------------------------------------

class TestReprPlain:

    def test_header_contains_dsvector(self):
        assert "DSVector" in to_string(M)

    def test_header_contains_kind_label(self):
        assert "Basic Belief Assignment" in to_string(M)
        assert "Belief function"         in to_string(M_BEL)
        assert "Plausibility function"   in to_string(M_PL)
        assert "Implicability function"  in to_string(M_B)
        assert "Commonality function"    in to_string(M_Q)

    def test_column_header_shows_kind_symbol(self):
        for v, sym in [(M,"m"), (M_BEL,"bel"), (M_PL,"pl"), (M_B,"b"), (M_Q,"q")]:
            assert sym in to_string(v), f"Symbol '{sym}' not found for kind {v.kind}"

    def test_subset_labels_present(self):
        p = to_string(M)
        assert "{a}" in p
        assert "{r}" in p

    def test_empty_set_label(self):
        assert "∅" in to_string(M_Q)  # commonality q(∅) = 1 always

    def test_total_only_for_m(self):
        assert "Total" in to_string(M)
        assert "Total" not in to_string(M_BEL)
        assert "Total" not in to_string(M_PL)
        assert "Total" not in to_string(M_B)
        assert "Total" not in to_string(M_Q)

    def test_total_value_correct(self):
        assert "1.0000" in to_string(M)

    def test_no_ansi_codes(self):
        assert "\033[" not in to_string(M)
        assert "\033[" not in to_string(M_BEL)

    def test_subnormal_bba(self):
        p = to_string(M_SUB)
        assert "∅" in p
        assert "Total" in p


# ---------------------------------------------------------------------------
# to_ansi
# ---------------------------------------------------------------------------

class TestReprAnsi:

    def test_header_contains_dsvector(self):
        # Strip ANSI codes for content check
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', to_ansi(M))
        assert "DSVector" in clean

    def test_column_header_shows_kind_symbol(self):
        import re
        for v, sym in [(M,"m"), (M_BEL,"bel"), (M_PL,"pl"), (M_B,"b"), (M_Q,"q")]:
            clean = re.sub(r'\033\[[0-9;]*m', '', to_ansi(v))
            assert sym in clean, f"Symbol '{sym}' not found for kind {v.kind}"

    def test_contains_ansi_codes(self):
        assert "\033[" in to_ansi(M)

    def test_total_only_for_m(self):
        import re
        assert "Total" in re.sub(r'\033\[[0-9;]*m', '', to_ansi(M))
        assert "Total" not in re.sub(r'\033\[[0-9;]*m', '', to_ansi(M_BEL))

    def test_focal_count_singular(self):
        m1 = DSVector.from_focal(FRAME_AHR, {"a": 1.0})
        assert "1 focal element" in to_ansi(m1)
        assert "1 focal elements" not in to_ansi(m1)

    def test_focal_count_plural(self):
        assert "2 focal elements" in to_ansi(M)

    def test_bar_characters_present(self):
        assert "█" in to_ansi(M)


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------

class TestReprHtml:

    def test_is_valid_html(self):
        h = to_html(M)
        assert "<table" in h
        assert "</table>" in h

    def test_column_header_shows_kind_symbol(self):
        for v, sym in [(M,"m"), (M_BEL,"bel"), (M_PL,"pl"), (M_B,"b"), (M_Q,"q")]:
            assert sym in to_html(v), f"Symbol '{sym}' not found for kind {v.kind}"

    def test_subset_labels_present(self):
        h = to_html(M)
        assert "{a}" in h
        assert "{r}" in h

    def test_total_only_for_m(self):
        assert "Total" in to_html(M)
        assert "Total" not in to_html(M_BEL)
        assert "Total" not in to_html(M_PL)

    def test_repr_html_method(self):
        """Jupyter calls _repr_html_ automatically."""
        assert to_html(M) == M._repr_html_()

    def test_bar_div_present(self):
        assert "<div" in to_html(M)

    def test_kind_color_in_header(self):
        # Each kind has a distinct color
        h_m   = to_html(M)
        h_bel = to_html(M_BEL)
        # Both should have color styles but potentially different colors
        assert "color:" in h_m
        assert "color:" in h_bel


# ---------------------------------------------------------------------------
# to_latex
# ---------------------------------------------------------------------------

class TestReprLatex:

    def test_tabular_environment(self):
        l = to_latex(M)
        assert "\\begin{tabular}" in l
        assert "\\end{tabular}" in l

    def test_hline_present(self):
        assert "\\hline" in to_latex(M)

    def test_column_header_shows_kind_symbol(self):
        for v, sym in [(M,"m"), (M_BEL,"bel"), (M_PL,"pl"), (M_B,"b"), (M_Q,"q")]:
            assert sym in to_latex(v), f"Symbol '{sym}' not found for kind {v.kind}"

    def test_emptyset_label(self):
        assert "\\emptyset" in to_latex(M_Q)

    def test_subset_braces(self):
        l = to_latex(M)
        assert "\\{" in l
        assert "\\}" in l

    def test_total_only_for_m(self):
        assert "Total" in to_latex(M)
        assert "Total" not in to_latex(M_BEL)
        assert "Total" not in to_latex(M_PL)

    def test_value_format(self):
        l = to_latex(M)
        assert "0.5000" in l


# ---------------------------------------------------------------------------
# DSVector methods (m.to_string / m.to_ansi / m.to_html / m.to_latex)
# ---------------------------------------------------------------------------

class TestDSVectorMethods:

    def test_method_to_string_matches_function(self):
        assert M.to_string() == to_string(M)

    def test_method_to_ansi_matches_function(self):
        assert M.to_ansi() == to_ansi(M)

    def test_method_to_html_matches_function(self):
        assert M.to_html() == to_html(M)

    def test_method_to_latex_matches_function(self):
        assert M.to_latex() == to_latex(M)

    def test_repr_uses_to_ansi(self):
        assert repr(M) == M.to_ansi()

    def test_repr_html_uses_to_html(self):
        assert M._repr_html_() == M.to_html()

    def test_method_all_kinds_is_forwarded(self):
        assert M.to_string(all_kinds=True) == to_string(M, all_kinds=True)
        assert M.to_html(all_kinds=True)   == to_html(M, all_kinds=True)
        assert M.to_latex(all_kinds=True)  == to_latex(M, all_kinds=True)

    def test_all_kinds_methods_run_for_all_kinds(self):
        for v in [M, M_BEL, M_PL, M_B, M_Q]:
            for method in ("to_string", "to_ansi", "to_html", "to_latex"):
                result = getattr(v, method)()
                assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# all_kinds=True — multi-representation table
# ---------------------------------------------------------------------------

# Non-dogmatic, normal BBA: m(Ω)>0, m(∅)=0  → w only
M_ND  = DSVector.from_focal(FRAME_AHR, {"a": 0.3, "r": 0.3, "a,h,r": 0.4})

# Subnormal, non-dogmatic: m(∅)>0, m(Ω)>0   → v and w
M_BOTH = DSVector.from_focal(FRAME_AHR, {"": 0.1, "a": 0.3, "r": 0.4, "a,h,r": 0.2}, complete=False)

# Subnormal, dogmatic: m(∅)>0, m(Ω)=0        → v only
M_DOG_SUB = DSVector.from_focal(FRAME_AHR, {"": 0.1, "a": 0.5, "r": 0.4}, complete=False)

# Normal, dogmatic: m(∅)=0, m(Ω)=0           → neither v nor w
M_DOG = DSVector.from_focal(FRAME_AHR, {"a": 0.6, "r": 0.4}, complete=False)


def _has_col(m, col, fn=to_string):
    """Check whether column `col` appears in the header line of the all_kinds view."""
    for line in fn(m, all_kinds=True).split("\n"):
        if "Subset" in line:
            return col in line.split()
    return False


class TestAllKindsTable:

    # --- column selection rules ---

    def test_v_shown_only_when_subnormal(self):
        """v appears iff m(∅) > 0."""
        assert not _has_col(M_ND,     "v")
        assert     _has_col(M_BOTH,    "v")
        assert     _has_col(M_DOG_SUB, "v")
        assert not _has_col(M_DOG,     "v")

    def test_w_shown_only_when_nondogmatic(self):
        """w appears iff m(Ω) > 0."""
        assert     _has_col(M_ND,     "w")
        assert     _has_col(M_BOTH,    "w")
        assert not _has_col(M_DOG_SUB, "w")
        assert not _has_col(M_DOG,     "w")

    def test_always_shows_m_bel_pl_b_q(self):
        for m in [M, M_ND, M_BOTH, M_DOG_SUB, M_DOG]:
            for col in ["m", "bel", "pl", "b", "q"]:
                assert _has_col(m, col), f"Column '{col}' missing for {m.kind}"

    # --- formats ---

    def test_to_string_no_ansi(self):
        assert "\033[" not in to_string(M_ND, all_kinds=True)

    def test_to_ansi_has_codes(self):
        assert "\033[" in to_ansi(M_ND, all_kinds=True)

    def test_to_html_table(self):
        h = to_html(M_ND, all_kinds=True)
        assert "<table" in h and "</table>" in h

    def test_to_html_shows_w_col(self):
        assert "$w$" in to_html(M_ND, all_kinds=True)

    def test_to_html_no_v_when_normal(self):
        assert "$v$" not in to_html(M_ND, all_kinds=True)

    def test_to_latex_tabular(self):
        l = to_latex(M_ND, all_kinds=True)
        assert "\\begin{tabular}" in l and "\\end{tabular}" in l

    def test_to_latex_shows_w_col(self):
        assert "$w$" in to_latex(M_ND, all_kinds=True)

    def test_to_latex_shows_v_and_w_when_both(self):
        l = to_latex(M_BOTH, all_kinds=True)
        assert "$v$" in l and "$w$" in l

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            to_string(M.to_bel(), all_kinds=True)
        with pytest.raises(ValueError, match="kind"):
            M.to_bel().to_html(all_kinds=True)

    # --- content checks ---

    def test_all_subsets_with_nonzero_values_present(self):
        p = to_string(M, all_kinds=True)
        assert "{a}" in p
        assert "{r}" in p

    def test_values_correct(self):
        """m({a})=0.5 and bel({a})=0.5 appear in the table."""
        p = to_string(M, all_kinds=True)
        assert "0.5000" in p

    def test_all_formats_run_for_all_cases(self):
        for m in [M, M_ND, M_BOTH, M_DOG_SUB, M_DOG]:
            for fn in (to_string, to_ansi, to_html, to_latex):
                result = fn(m, all_kinds=True)
                assert isinstance(result, str) and len(result) > 0
