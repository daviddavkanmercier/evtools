"""
Tests for evtools.display — repr_ansi, repr_plain, repr_html, repr_latex.
"""

import numpy as np
import pytest
from evtools.dsvector import DSVector, Kind
from evtools.display import repr_ansi, repr_plain, repr_html, repr_latex

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
# repr_plain
# ---------------------------------------------------------------------------

class TestReprPlain:

    def test_header_contains_dsvector(self):
        assert "DSVector" in repr_plain(M)

    def test_header_contains_kind_label(self):
        assert "Basic Belief Assignment" in repr_plain(M)
        assert "Belief function"         in repr_plain(M_BEL)
        assert "Plausibility function"   in repr_plain(M_PL)
        assert "Commonality function"    in repr_plain(M_B)
        assert "Implicability function"  in repr_plain(M_Q)

    def test_column_header_shows_kind_symbol(self):
        for v, sym in [(M,"m"), (M_BEL,"bel"), (M_PL,"pl"), (M_B,"b"), (M_Q,"q")]:
            assert sym in repr_plain(v), f"Symbol '{sym}' not found for kind {v.kind}"

    def test_subset_labels_present(self):
        p = repr_plain(M)
        assert "{a}" in p
        assert "{r}" in p

    def test_empty_set_label(self):
        assert "∅" in repr_plain(M_Q)  # q has ∅ as focal element

    def test_total_only_for_m(self):
        assert "Total" in repr_plain(M)
        assert "Total" not in repr_plain(M_BEL)
        assert "Total" not in repr_plain(M_PL)
        assert "Total" not in repr_plain(M_B)
        assert "Total" not in repr_plain(M_Q)

    def test_total_value_correct(self):
        assert "1.0000" in repr_plain(M)

    def test_no_ansi_codes(self):
        assert "\033[" not in repr_plain(M)
        assert "\033[" not in repr_plain(M_BEL)

    def test_subnormal_bba(self):
        p = repr_plain(M_SUB)
        assert "∅" in p
        assert "Total" in p


# ---------------------------------------------------------------------------
# repr_ansi
# ---------------------------------------------------------------------------

class TestReprAnsi:

    def test_header_contains_dsvector(self):
        # Strip ANSI codes for content check
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', repr_ansi(M))
        assert "DSVector" in clean

    def test_column_header_shows_kind_symbol(self):
        import re
        for v, sym in [(M,"m"), (M_BEL,"bel"), (M_PL,"pl"), (M_B,"b"), (M_Q,"q")]:
            clean = re.sub(r'\033\[[0-9;]*m', '', repr_ansi(v))
            assert sym in clean, f"Symbol '{sym}' not found for kind {v.kind}"

    def test_contains_ansi_codes(self):
        assert "\033[" in repr_ansi(M)

    def test_total_only_for_m(self):
        import re
        assert "Total" in re.sub(r'\033\[[0-9;]*m', '', repr_ansi(M))
        assert "Total" not in re.sub(r'\033\[[0-9;]*m', '', repr_ansi(M_BEL))

    def test_focal_count_singular(self):
        m1 = DSVector.from_focal(FRAME_AHR, {"a": 1.0})
        assert "1 focal element" in repr_ansi(m1)
        assert "1 focal elements" not in repr_ansi(m1)

    def test_focal_count_plural(self):
        assert "2 focal elements" in repr_ansi(M)

    def test_bar_characters_present(self):
        assert "█" in repr_ansi(M)


# ---------------------------------------------------------------------------
# repr_html
# ---------------------------------------------------------------------------

class TestReprHtml:

    def test_is_valid_html(self):
        h = repr_html(M)
        assert "<table" in h
        assert "</table>" in h

    def test_column_header_shows_kind_symbol(self):
        for v, sym in [(M,"m"), (M_BEL,"bel"), (M_PL,"pl"), (M_B,"b"), (M_Q,"q")]:
            assert sym in repr_html(v), f"Symbol '{sym}' not found for kind {v.kind}"

    def test_subset_labels_present(self):
        h = repr_html(M)
        assert "{a}" in h
        assert "{r}" in h

    def test_total_only_for_m(self):
        assert "Total" in repr_html(M)
        assert "Total" not in repr_html(M_BEL)
        assert "Total" not in repr_html(M_PL)

    def test_repr_html_method(self):
        """Jupyter calls _repr_html_ automatically."""
        assert repr_html(M) == M._repr_html_()

    def test_bar_div_present(self):
        assert "<div" in repr_html(M)

    def test_kind_color_in_header(self):
        # Each kind has a distinct color
        h_m   = repr_html(M)
        h_bel = repr_html(M_BEL)
        # Both should have color styles but potentially different colors
        assert "color:" in h_m
        assert "color:" in h_bel


# ---------------------------------------------------------------------------
# repr_latex
# ---------------------------------------------------------------------------

class TestReprLatex:

    def test_tabular_environment(self):
        l = repr_latex(M)
        assert "\\begin{tabular}" in l
        assert "\\end{tabular}" in l

    def test_hline_present(self):
        assert "\\hline" in repr_latex(M)

    def test_column_header_shows_kind_symbol(self):
        for v, sym in [(M,"m"), (M_BEL,"bel"), (M_PL,"pl"), (M_B,"b"), (M_Q,"q")]:
            assert sym in repr_latex(v), f"Symbol '{sym}' not found for kind {v.kind}"

    def test_emptyset_label(self):
        assert "\\emptyset" in repr_latex(M_Q)

    def test_subset_braces(self):
        l = repr_latex(M)
        assert "\\{" in l
        assert "\\}" in l

    def test_total_only_for_m(self):
        assert "Total" in repr_latex(M)
        assert "Total" not in repr_latex(M_BEL)
        assert "Total" not in repr_latex(M_PL)

    def test_value_format(self):
        l = repr_latex(M)
        assert "0.5000" in l


# ---------------------------------------------------------------------------
# display() method
# ---------------------------------------------------------------------------

class TestDisplayMethod:

    def test_display_ansi(self):
        assert "\033[" in M.display("ansi")

    def test_display_plain(self):
        assert "\033[" not in M.display("plain")
        assert "DSVector" in M.display("plain")

    def test_display_html(self):
        assert "<table" in M.display("html")

    def test_display_latex(self):
        assert "\\begin{tabular}" in M.display("latex")

    def test_display_default_is_ansi(self):
        assert M.display() == M.display("ansi")

    def test_display_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            M.display("pdf")

    def test_display_all_kinds_all_formats(self):
        for v in [M, M_BEL, M_PL, M_B, M_Q]:
            for fmt in ["ansi", "plain", "html", "latex"]:
                result = v.display(fmt)
                assert isinstance(result, str)
                assert len(result) > 0


# ---------------------------------------------------------------------------
# display_all
# ---------------------------------------------------------------------------

from evtools.display import display_all

# Non-dogmatic, normal BBA: m(Ω)>0, m(∅)=0  → w only
M_ND  = DSVector.from_focal(FRAME_AHR, {"a": 0.3, "r": 0.3, "a,h,r": 0.4})

# Subnormal, non-dogmatic: m(∅)>0, m(Ω)>0   → v and w
M_BOTH = DSVector.from_focal(FRAME_AHR, {"": 0.1, "a": 0.3, "r": 0.4, "a,h,r": 0.2}, complete=False)

# Subnormal, dogmatic: m(∅)>0, m(Ω)=0        → v only
M_DOG_SUB = DSVector.from_focal(FRAME_AHR, {"": 0.1, "a": 0.5, "r": 0.4}, complete=False)

# Normal, dogmatic: m(∅)=0, m(Ω)=0           → neither v nor w
M_DOG = DSVector.from_focal(FRAME_AHR, {"a": 0.6, "r": 0.4}, complete=False)


def _has_col(m, col, fmt="plain"):
    """Check whether column `col` appears in the header line of display_all."""
    for line in display_all(m, fmt).split("\n"):
        if "Subset" in line:
            return col in line.split()
    return False


class TestDisplayAll:

    # --- column selection rules ---

    def test_v_shown_only_when_subnormal(self):
        """v appears iff m(∅) > 0."""
        assert not _has_col(M_ND,  "v"), "v should be absent for normal BBA"
        assert     _has_col(M_BOTH, "v"), "v should be present for subnormal BBA"
        assert     _has_col(M_DOG_SUB, "v"), "v should be present for subnormal dogmatic BBA"
        assert not _has_col(M_DOG,  "v"), "v should be absent for normal dogmatic BBA"

    def test_w_shown_only_when_nondogmatic(self):
        """w appears iff m(Ω) > 0."""
        assert     _has_col(M_ND,  "w"), "w should be present for non-dogmatic BBA"
        assert     _has_col(M_BOTH, "w"), "w should be present for non-dogmatic subnormal BBA"
        assert not _has_col(M_DOG_SUB, "w"), "w should be absent for dogmatic BBA"
        assert not _has_col(M_DOG,  "w"), "w should be absent for dogmatic BBA"

    def test_always_shows_m_bel_pl_b_q(self):
        for m in [M, M_ND, M_BOTH, M_DOG_SUB, M_DOG]:
            for col in ["m", "bel", "pl", "b", "q"]:
                assert _has_col(m, col), f"Column '{col}' missing for {m.kind}"

    # --- formats ---

    def test_plain_no_ansi(self):
        assert "\033[" not in display_all(M_ND, "plain")

    def test_ansi_has_codes(self):
        assert "\033[" in display_all(M_ND, "ansi")

    def test_html_table(self):
        h = display_all(M_ND, "html")
        assert "<table" in h and "</table>" in h

    def test_html_shows_w_col(self):
        assert "$w$" in display_all(M_ND, "html")

    def test_html_no_v_when_normal(self):
        assert "$v$" not in display_all(M_ND, "html")

    def test_latex_tabular(self):
        l = display_all(M_ND, "latex")
        assert "\\begin{tabular}" in l and "\\end{tabular}" in l

    def test_latex_shows_w_col(self):
        assert "$w$" in display_all(M_ND, "latex")

    def test_latex_shows_v_and_w_when_both(self):
        l = display_all(M_BOTH, "latex")
        assert "$v$" in l and "$w$" in l

    def test_default_format_is_ansi(self):
        assert display_all(M) == display_all(M, "ansi")

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="unknown format"):
            display_all(M, "pdf")

    def test_wrong_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            display_all(M.to_bel())

    # --- method on DSVector ---

    def test_display_all_method(self):
        assert M_ND.display_all("plain") == display_all(M_ND, "plain")

    def test_display_all_method_default(self):
        assert M_ND.display_all() == display_all(M_ND, "ansi")

    # --- content checks ---

    def test_all_subsets_with_nonzero_values_present(self):
        p = display_all(M, "plain")
        assert "{a}" in p
        assert "{r}" in p
        assert "∅" in p or "\\emptyset" in p or "∅" in p

    def test_values_correct(self):
        """m({a})=0.5 and bel({a})=0.5 appear in the table."""
        p = display_all(M, "plain")
        assert "0.5000" in p

    def test_all_formats_run_for_all_cases(self):
        for m in [M, M_ND, M_BOTH, M_DOG_SUB, M_DOG]:
            for fmt in ["ansi", "plain", "html", "latex"]:
                result = display_all(m, fmt)
                assert isinstance(result, str) and len(result) > 0
