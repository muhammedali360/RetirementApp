import numpy as np

import report


# -------------------------
# delta_pts
# -------------------------
def test_delta_pts_rounds_to_points():
    assert report.delta_pts(0.123) == 12
    assert report.delta_pts(0.125) == 12  # banker's rounding via round()
    assert report.delta_pts(-0.04) == -4


def test_delta_pts_none():
    assert report.delta_pts(None) is None


# -------------------------
# pick_best_levers
# -------------------------
def _row(change, delta):
    return {"Change": change, "delta": delta}


def test_pick_best_levers_filters_non_positive():
    rows = [_row("a", 0.05), _row("b", 0.0), _row("c", -0.02), _row("d", None)]
    best = report.pick_best_levers(rows)
    assert [r["Change"] for r in best] == ["a"]


def test_pick_best_levers_sorts_descending_and_limits():
    rows = [_row("a", 0.01), _row("b", 0.08), _row("c", 0.05)]
    best = report.pick_best_levers(rows, limit=2)
    assert [r["Change"] for r in best] == ["b", "c"]


def test_pick_best_levers_drops_sub_point_deltas():
    # delta rounds to 0 points -> excluded.
    rows = [_row("tiny", 0.004)]
    assert report.pick_best_levers(rows) == []


# -------------------------
# format_lever_recommendation
# -------------------------
def test_format_lever_recommendation_empty_not_at_target():
    msg = report.format_lever_recommendation([], at_target=False)
    assert "No tested adjustment" in msg


def test_format_lever_recommendation_empty_at_target():
    assert report.format_lever_recommendation([], at_target=True) is None


def test_format_lever_recommendation_at_target_optional_margin():
    levers = [_row("Save more", 0.07)]
    msg = report.format_lever_recommendation(levers, at_target=True)
    assert msg == "Optional margin: Save more (+7 pts)."


def test_format_lever_recommendation_best_and_second():
    levers = [_row("Save more", 0.07), _row("Retire later", 0.03)]
    msg = report.format_lever_recommendation(levers, at_target=False)
    assert "Best lever: Save more (+7 pts)." in msg
    assert "Also consider Retire later (+3 pts)." in msg


def test_format_lever_recommendation_html():
    levers = [_row("Save more", 0.07)]
    msg = report.format_lever_recommendation(levers, at_target=False, html=True)
    assert "<strong>Save more</strong>" in msg


# -------------------------
# build_plan_narrative
# -------------------------
def test_build_plan_narrative_clears_threshold():
    text = report.build_plan_narrative(
        target_prob=0.97, target_age=60, on_track=True,
        withdrawal_rate=0.03, ranges=[(58, 60)], years_until=0,
    )
    assert "clears the 90% bar at age 60" in text
    assert "Earliest age with ≥90% success: 58." in text
    assert "within a typical safe range" in text


def test_build_plan_narrative_below_threshold_warns():
    text = report.build_plan_narrative(
        target_prob=0.70, target_age=55, on_track=False,
        withdrawal_rate=0.06, ranges=[], years_until=5,
    )
    assert "consider saving more" in text
    assert "above the common 4–5% guideline" in text
    assert "more working years" in text


def test_build_plan_narrative_handles_no_target_prob():
    text = report.build_plan_narrative(
        target_prob=None, target_age=60, on_track=True,
        withdrawal_rate=None, ranges=[], years_until=0,
    )
    # No crash, no probability sentence.
    assert "90% bar" not in text


# -------------------------
# safe_retirement_ranges
# -------------------------
def test_safe_retirement_ranges_empty_when_none_safe():
    ages = np.arange(50, 60)
    probs = np.full(10, 0.5)
    assert report.safe_retirement_ranges(ages, probs) == []


def test_safe_retirement_ranges_single_contiguous():
    ages = np.arange(50, 60)
    probs = np.array([0.1, 0.1, 0.96, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99])
    assert report.safe_retirement_ranges(ages, probs) == [(52, 59)]


def test_safe_retirement_ranges_with_gap():
    ages = np.array([50, 51, 52, 53, 54, 55])
    probs = np.array([0.96, 0.97, 0.4, 0.4, 0.96, 0.97])
    assert report.safe_retirement_ranges(ages, probs) == [(50, 51), (54, 55)]


def test_safe_retirement_ranges_respects_threshold():
    ages = np.array([50, 51, 52])
    probs = np.array([0.80, 0.85, 0.90])
    assert report.safe_retirement_ranges(ages, probs, threshold=0.85) == [(51, 52)]


# -------------------------
# format_safe_ranges_plain
# -------------------------
def test_format_safe_ranges_plain_none():
    assert report.format_safe_ranges_plain([], max_eval_age=85) is None


def test_format_safe_ranges_plain_single_age():
    assert report.format_safe_ranges_plain([(60, 60)], max_eval_age=85) == (
        "Only age 60 meets the >=90% threshold"
    )


def test_format_safe_ranges_plain_all_later_safe():
    msg = report.format_safe_ranges_plain([(58, 85)], max_eval_age=85)
    assert "Earliest retirement age with >=90% success: 58" in msg
    assert "all later ages through 85 are also safe" in msg


def test_format_safe_ranges_plain_bounded_range():
    assert report.format_safe_ranges_plain([(55, 60)], max_eval_age=85) == (
        "Ages 55-60 meet the >=90% threshold"
    )


def test_format_safe_ranges_plain_multiple_ranges():
    msg = report.format_safe_ranges_plain([(50, 51), (54, 54)], max_eval_age=85)
    assert msg == "Safe retirement ages: 50-51, 54"


# -------------------------
# summaries
# -------------------------
def test_spending_summary_with_reduction():
    cfg = {"annual_spending": 100_000, "spending_reduction_after_75": 0.2}
    assert report._spending_summary(cfg) == "$100,000/yr ( −20% after 75)"


def test_spending_summary_no_reduction():
    cfg = {"annual_spending": 80_000, "spending_reduction_after_75": 0.0}
    assert report._spending_summary(cfg) == "$80,000/yr"


def test_ss_summary_included_and_excluded():
    assert report._ss_summary({"include_social_security": True, "social_security_claim_age": 70}) == (
        "Claim at age 70"
    )
    assert report._ss_summary({"include_social_security": False}) == "Not included"


# -------------------------
# PDF generation (smoke)
# -------------------------
def _cfg():
    return {
        "current_age": 40,
        "max_age": 95,
        "starting_amount": 1_500_000,
        "annual_contribution": 55_000,
        "annual_spending": 80_000,
        "spending_reduction_after_75": 0.1,
        "inflation_rate": 0.025,
        "trials": 1000,
        "include_social_security": True,
        "social_security_claim_age": 67,
    }


def test_generate_report_bytes_produces_pdf():
    ages = np.arange(50, 66)
    probs = np.linspace(0.5, 0.99, len(ages))
    data = report.generate_report_bytes(
        _cfg(), ages, probs,
        executive_summary="Test summary.",
        optimizer_result=["Need 20 career years."],
    )
    assert data.startswith(b"%PDF")
    assert len(data) > 500


def test_generate_report_writes_file(tmp_path):
    ages = np.arange(50, 66)
    probs = np.full(len(ages), 0.4)  # no safe age -> exercises the empty branch
    out = tmp_path / "report.pdf"
    report.generate_report(str(out), _cfg(), ages, probs)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF")
