import numpy as np
import pytest

import model


# -------------------------
# SPENDING
# -------------------------
def test_annual_spending_prefers_annual_spending_key(cfg):
    cfg["annual_spending"] = 123_456
    assert model._annual_spending(cfg) == 123_456


def test_annual_spending_falls_back_to_spending_under_75():
    assert model._annual_spending({"spending_under_75": 90_000}) == 90_000


def test_annual_spending_default():
    assert model._annual_spending({}) == 200_000


def test_build_spending_array_no_reduction(cfg):
    cfg["annual_spending"] = 100_000
    cfg["spending_reduction_after_75"] = 0.0
    arr = model.build_spending_array(cfg)
    assert arr.shape == (150,)
    assert arr[40] == 100_000
    assert arr[74] == 100_000
    assert arr[75] == 100_000


def test_build_spending_array_with_reduction(cfg):
    cfg["annual_spending"] = 100_000
    cfg["spending_reduction_after_75"] = 0.20
    arr = model.build_spending_array(cfg)
    assert arr[74] == 100_000
    assert arr[75] == pytest.approx(80_000)
    assert arr[120] == pytest.approx(80_000)


def test_get_spending_matches_array(cfg):
    cfg["annual_spending"] = 100_000
    cfg["spending_reduction_after_75"] = 0.10
    assert model.get_spending(70, cfg) == 100_000
    assert model.get_spending(80, cfg) == pytest.approx(90_000)


# -------------------------
# SOCIAL SECURITY
# -------------------------
def test_compute_social_security_full_career_at_67():
    # 35+ years worked, claim at 67 => exactly max benefit
    assert model.compute_social_security(35, 67, max_benefit=24_000) == pytest.approx(24_000)
    assert model.compute_social_security(40, 67, max_benefit=24_000) == pytest.approx(24_000)


def test_compute_social_security_partial_career_scales_linearly():
    # half the 35-year cap
    assert model.compute_social_security(17.5, 67, max_benefit=24_000) == pytest.approx(12_000)


def test_compute_social_security_claim_age_adjustment():
    # SSA-style schedule: +8%/yr delayed credits past 67 (capped at 70),
    # -6.67%/yr for the first 3 years of early claiming.
    later = model.compute_social_security(35, 70, max_benefit=24_000)
    earlier = model.compute_social_security(35, 64, max_benefit=24_000)
    assert later == pytest.approx(24_000 * (1 + 0.08 * 3))
    assert earlier == pytest.approx(24_000 * (1 - 0.0667 * 3))


def test_ss_claim_adjustment_full_retirement_age_is_unity():
    assert model.ss_claim_adjustment(67) == pytest.approx(1.0)


def test_ss_claim_adjustment_early_beyond_three_years():
    # 5 years early: 3 years at -6.67% + 2 years at -5%.
    expected = 1.0 - 0.0667 * 3 - 0.05 * 2
    assert model.ss_claim_adjustment(62) == pytest.approx(expected)


def test_ss_claim_adjustment_clamps_outside_62_to_70():
    assert model.ss_claim_adjustment(75) == model.ss_claim_adjustment(70)
    assert model.ss_claim_adjustment(55) == model.ss_claim_adjustment(62)


def test_social_security_batch_matches_scalar():
    years = [0, 10, 35, 40]
    batch = model.compute_social_security_batch(years, 70, max_benefit=24_000)
    scalar = [model.compute_social_security(y, 70, max_benefit=24_000) for y in years]
    np.testing.assert_allclose(batch, scalar)


def test_social_security_income_disabled(cfg):
    cfg["include_social_security"] = False
    assert model.social_security_income(cfg, 35) == 0.0


def test_social_security_income_batch_disabled_returns_zeros(cfg):
    cfg["include_social_security"] = False
    out = model.social_security_income_batch(cfg, [10, 20, 30])
    np.testing.assert_array_equal(out, np.zeros(3))


def test_social_security_start_age(cfg):
    assert model.social_security_start_age(cfg) == 67
    cfg["include_social_security"] = False
    assert model.social_security_start_age(cfg) is None


def test_social_security_claim_age_fallback():
    cfg = {"include_social_security": True, "social_security_start_age": 70}
    assert model.social_security_start_age(cfg) == 70
    assert model.social_security_income(cfg, 35) == pytest.approx(24_000 * (1 + 0.08 * 3))


# -------------------------
# RETURNS
# -------------------------
def test_generate_returns_shape():
    r = model.generate_returns(0.06, 0.12, trials=100, n_years=30, seed=1)
    assert r.shape == (100, 30)


def test_generate_returns_seed_reproducible():
    a = model.generate_returns(0.06, 0.12, 50, 20, seed=42)
    b = model.generate_returns(0.06, 0.12, 50, 20, seed=42)
    np.testing.assert_array_equal(a, b)


def test_generate_returns_different_seeds_differ():
    a = model.generate_returns(0.06, 0.12, 50, 20, seed=1)
    b = model.generate_returns(0.06, 0.12, 50, 20, seed=2)
    assert not np.array_equal(a, b)


def test_generate_returns_mean_approximates_arithmetic_mean():
    # Log-normal calibrated so arithmetic mean of simple returns ~= mean_return.
    r = model.generate_returns(0.06, 0.12, trials=200_000, n_years=1, seed=7)
    assert r.mean() == pytest.approx(0.06, abs=0.01)


def test_generate_returns_zero_volatility_is_constant():
    r = model.generate_returns(0.05, 0.0, trials=10, n_years=5, seed=0)
    np.testing.assert_allclose(r, 0.05)


# -------------------------
# MONTE CARLO
# -------------------------
def test_simulate_batch_returns_probability(cfg):
    p = model.simulate_batch(cfg, 60, 0.06, 0.12, ss_income=20_000, trials=cfg["trials"], seed=1)
    assert 0.0 <= p <= 1.0


def test_simulate_batch_seed_reproducible(cfg):
    kw = dict(ss_income=20_000, trials=cfg["trials"], seed=3)
    p1 = model.simulate_batch(cfg, 60, 0.06, 0.12, **kw)
    p2 = model.simulate_batch(cfg, 60, 0.06, 0.12, **kw)
    assert p1 == p2


def test_simulate_batch_more_starting_money_helps(cfg):
    poor = dict(cfg, starting_amount=200_000)
    rich = dict(cfg, starting_amount=5_000_000)
    p_poor = model.simulate_batch(poor, 55, 0.06, 0.12, ss_income=20_000, trials=cfg["trials"], seed=5)
    p_rich = model.simulate_batch(rich, 55, 0.06, 0.12, ss_income=20_000, trials=cfg["trials"], seed=5)
    assert p_rich > p_poor


def test_simulate_batch_later_retirement_helps(cfg):
    early = model.simulate_batch(cfg, 50, 0.06, 0.12, ss_income=20_000, trials=cfg["trials"], seed=9)
    late = model.simulate_batch(cfg, 65, 0.06, 0.12, ss_income=20_000, trials=cfg["trials"], seed=9)
    assert late >= early


# -------------------------
# CURVE
# -------------------------
def test_compute_curve_shapes(cfg):
    ages, probs = model.compute_curve(cfg, range(50, 66), 0.06, 0.12, seed=11)
    assert len(ages) == len(probs) == 16
    assert np.all((probs >= 0) & (probs <= 1))


def test_compute_curve_monotonic_in_retirement_age(cfg):
    # Retiring later should never reduce success probability.
    ages, probs = model.compute_curve(cfg, range(45, 71), 0.06, 0.12, seed=13)
    assert np.all(np.diff(probs) >= -1e-9)


def test_compute_curve_matches_simulate_batch_single_age(cfg):
    # Both paths build the same returns from the same seed, so a single-age
    # curve point must equal the corresponding simulate_batch call.
    age = 60
    years_worked = cfg["years_already_worked"] + (age - cfg["current_age"])
    ss_income = model.social_security_income(cfg, years_worked)

    ages, probs = model.compute_curve(cfg, [age], 0.06, 0.12, seed=21)
    batch = model.simulate_batch(
        cfg, age, 0.06, 0.12, ss_income=ss_income, trials=cfg["trials"], seed=21,
    )
    assert probs[0] == pytest.approx(batch)


# -------------------------
# SMART SPENDING (GUARDRAILS)
# -------------------------
def test_withdrawal_strategy_defaults_to_fixed(cfg):
    assert model.withdrawal_strategy(cfg) == "fixed"
    assert model.withdrawal_strategy(dict(cfg, withdrawal_strategy="guardrails")) == "guardrails"


def test_guardrail_income_band(cfg):
    cfg["annual_spending"] = 100_000
    floor, base, ceiling = model.guardrail_income_band(cfg)
    assert base == 100_000
    assert floor == pytest.approx(100_000 * model.SPEND_FLOOR_MULT)
    assert ceiling == pytest.approx(100_000 * model.SPEND_CEILING_MULT)
    assert floor < base < ceiling


def test_guardrails_curve_shapes_and_bounds(cfg):
    smart = dict(cfg, withdrawal_strategy="guardrails")
    ages, probs = model.compute_curve(smart, range(50, 66), 0.06, 0.12, seed=11)
    assert len(ages) == len(probs) == 16
    assert np.all((probs >= 0) & (probs <= 1))


def test_guardrails_curve_matches_simulate_batch_single_age(cfg):
    # The vectorized curve and the single-age path must agree under guardrails,
    # just as they do for the fixed strategy.
    smart = dict(cfg, withdrawal_strategy="guardrails")
    age = 60
    years_worked = smart["years_already_worked"] + (age - smart["current_age"])
    ss_income = model.social_security_income(smart, years_worked)

    _, probs = model.compute_curve(smart, [age], 0.06, 0.12, seed=21)
    batch = model.simulate_batch(
        smart, age, 0.06, 0.12, ss_income=ss_income, trials=smart["trials"], seed=21,
    )
    assert probs[0] == pytest.approx(batch)


def test_guardrails_helps_a_marginal_plan(cfg):
    # On a plan that is not a sure thing, flexing spending down after bad markets
    # should not lower — and in practice raises — the odds of lasting.
    marginal = dict(
        cfg, starting_amount=900_000, annual_contribution=0,
        annual_spending=70_000, include_social_security=False,
    )
    fixed = model.compute_curve(marginal, [60], 0.06, 0.14, seed=7)[1][0]
    smart = model.compute_curve(
        dict(marginal, withdrawal_strategy="guardrails"), [60], 0.06, 0.14, seed=7,
    )[1][0]
    assert 0.0 < fixed < 1.0  # genuinely marginal, so there is room to improve
    assert smart >= fixed


def test_guardrails_spending_stays_within_band(cfg):
    # No trial path should ever withdraw more than the ceiling lifestyle implies:
    # the realized spend multiplier is clamped to [floor, ceiling].
    smart = dict(cfg, withdrawal_strategy="guardrails", include_social_security=False)
    returns = model.generate_returns(0.06, 0.18, smart["trials"], 56, seed=3)
    _, _, paths = model._simulate_guardrails(
        smart, [55], returns, collect_paths=True,
    )
    assert paths.shape == (smart["trials"], 1, 56)
    # Surviving balances are finite and the tensor is well-formed.
    assert np.all(np.isfinite(paths))
    assert np.all(paths >= 0)


# -------------------------
# PERCENTILE PATHS
# -------------------------
def test_simulate_percentile_paths_ordering(cfg):
    ages, pct = model.simulate_percentile_paths(cfg, 60, 0.06, 0.12, seed=17)
    assert len(ages) == cfg["max_age"] - cfg["current_age"] + 1
    # 10th <= 50th <= 90th percentile at every age
    assert np.all(pct[10] <= pct[50] + 1e-6)
    assert np.all(pct[50] <= pct[90] + 1e-6)


# -------------------------
# OPTIMIZER
# -------------------------
def test_find_min_years_worked_returns_int_or_none(cfg):
    result = model.find_min_years_worked(cfg, 60, 0.06, 0.12, target=0.95, seed=23)
    assert result is None or isinstance(result, int)


def test_find_min_years_worked_meets_target(cfg):
    target = 0.90
    result = model.find_min_years_worked(cfg, 62, 0.06, 0.12, target=target, seed=29)
    assert result is not None
    # At the returned career length the success probability meets the target,
    # and one fewer year does not (it is the *minimum*).
    p_at = model._success_at_career_years(cfg, 62, result, 0.06, 0.12, seed=29)
    assert p_at >= target
    if result > cfg["years_already_worked"]:
        p_below = model._success_at_career_years(cfg, 62, result - 1, 0.06, 0.12, seed=29)
        assert p_below < target


def test_find_min_years_worked_impossible_target(cfg):
    broke = dict(cfg, starting_amount=0, annual_contribution=0, annual_spending=500_000)
    result = model.find_min_years_worked(broke, 45, 0.06, 0.12, target=0.95, seed=2)
    assert result is None


# -------------------------
# SUSTAINABLE SPENDING SOLVER
# -------------------------
def _success_at_spend(cfg, retirement_age, spend, mean_return, volatility, seed):
    """Recompute success for a fixed budget using the same seed as the solver,
    so the returns draw (and therefore the probability) matches exactly."""
    years_worked = cfg["years_already_worked"] + max(retirement_age - cfg["current_age"], 0)
    ss_income = model.social_security_income(cfg, years_worked)
    trial_cfg = dict(cfg, annual_spending=float(spend))
    return model.simulate_batch(
        trial_cfg, retirement_age, mean_return, volatility,
        ss_income=ss_income, trials=cfg["trials"], seed=seed,
    )


def test_find_max_sustainable_spending_returns_int_or_none(cfg):
    result = model.find_max_sustainable_spending(cfg, 65, 0.06, 0.12, seed=23)
    assert result is None or isinstance(result, int)


def test_find_max_sustainable_spending_lands_on_boundary(cfg):
    # The returned budget clears the target, and one grid step more does not —
    # i.e. it is the maximum sustainable spend, not merely a sustainable one.
    target, seed, age, step = 0.90, 41, 65, 1000
    safe = model.find_max_sustainable_spending(
        cfg, age, 0.06, 0.12, target=target, seed=seed, step=step,
    )
    assert safe is not None and safe > 0
    assert _success_at_spend(cfg, age, safe, 0.06, 0.12, seed) >= target
    assert _success_at_spend(cfg, age, safe + step, 0.06, 0.12, seed) < target


def test_find_max_sustainable_spending_is_multiple_of_step(cfg):
    safe = model.find_max_sustainable_spending(cfg, 65, 0.06, 0.12, seed=5, step=2500)
    assert safe % 2500 == 0


def test_find_max_sustainable_spending_more_money_allows_more_spend(cfg):
    poor = dict(cfg, starting_amount=500_000)
    rich = dict(cfg, starting_amount=3_000_000)
    s_poor = model.find_max_sustainable_spending(poor, 65, 0.06, 0.12, seed=7)
    s_rich = model.find_max_sustainable_spending(rich, 65, 0.06, 0.12, seed=7)
    assert s_rich > s_poor


def test_find_max_sustainable_spending_higher_target_is_stricter(cfg):
    # Demanding more confidence can only lower (never raise) the safe budget.
    relaxed = model.find_max_sustainable_spending(cfg, 65, 0.06, 0.12, target=0.80, seed=9)
    strict = model.find_max_sustainable_spending(cfg, 65, 0.06, 0.12, target=0.99, seed=9)
    assert strict <= relaxed


def test_find_max_sustainable_spending_degenerate_returns_none(cfg):
    # Retire today with no money and no savings: not even $0 spending survives.
    broke = dict(cfg, current_age=65, starting_amount=0, annual_contribution=0)
    assert model.find_max_sustainable_spending(broke, 65, 0.06, 0.12, seed=2) is None


# -------------------------
# COAST FIRE NUMBER
# -------------------------
def _coast_success(cfg, retirement_age, start, mean_return, volatility, seed):
    """Success at a fixed starting balance with contributions zeroed, sharing the
    solver's seed so the returns draw (and probability) match exactly."""
    years_worked = cfg["years_already_worked"] + max(retirement_age - cfg["current_age"], 0)
    ss_income = model.social_security_income(cfg, years_worked)
    trial_cfg = dict(cfg, starting_amount=float(start), annual_contribution=0)
    return model.simulate_batch(
        trial_cfg, retirement_age, mean_return, volatility,
        ss_income=ss_income, trials=cfg["trials"], seed=seed,
    )


def test_find_coast_number_returns_int_or_none(cfg):
    result = model.find_coast_number(cfg, 65, 0.06, 0.12, seed=23)
    assert result is None or isinstance(result, int)


def test_find_coast_number_seed_reproducible(cfg):
    a = model.find_coast_number(cfg, 65, 0.06, 0.12, seed=3)
    b = model.find_coast_number(cfg, 65, 0.06, 0.12, seed=3)
    assert a == b


def test_find_coast_number_is_multiple_of_step(cfg):
    coast = model.find_coast_number(cfg, 65, 0.06, 0.12, seed=5, step=25_000)
    assert coast % 25_000 == 0


def test_find_coast_number_lands_on_boundary(cfg):
    # The returned balance clears the target with no further contributions, and
    # one grid step less does not — it is the *minimum* coast portfolio.
    target, seed, age, step = 0.90, 41, 65, 10_000
    coast = model.find_coast_number(
        cfg, age, 0.06, 0.12, target=target, seed=seed, step=step,
    )
    assert coast is not None and coast > 0
    assert _coast_success(cfg, age, coast, 0.06, 0.12, seed) >= target
    if coast - step >= 0:
        assert _coast_success(cfg, age, coast - step, 0.06, 0.12, seed) < target


def test_find_coast_number_ignores_contributions(cfg):
    # Coasting zeroes savings internally, so the current contribution rate must
    # not change the answer.
    saver = dict(cfg, annual_contribution=10_000)
    super_saver = dict(cfg, annual_contribution=90_000)
    assert (
        model.find_coast_number(saver, 65, 0.06, 0.12, seed=7)
        == model.find_coast_number(super_saver, 65, 0.06, 0.12, seed=7)
    )


def test_find_coast_number_more_spending_needs_bigger_coast(cfg):
    lean = dict(cfg, annual_spending=60_000)
    lavish = dict(cfg, annual_spending=120_000)
    c_lean = model.find_coast_number(lean, 65, 0.06, 0.12, seed=13)
    c_lavish = model.find_coast_number(lavish, 65, 0.06, 0.12, seed=13)
    assert c_lavish > c_lean


def test_find_coast_number_later_retirement_lowers_coast(cfg):
    # More years for the existing balance to compound before drawdown means a
    # smaller portfolio is needed today to coast.
    early = model.find_coast_number(cfg, 55, 0.06, 0.12, seed=17)
    late = model.find_coast_number(cfg, 70, 0.06, 0.12, seed=17)
    assert late < early


# -------------------------
# DETERMINISTIC NET WORTH
# -------------------------
def test_simulate_net_worth_success(cfg):
    age, years, nominal, real = model.simulate_net_worth(cfg, 60, 0.06)
    assert age == 60
    assert years[0] == cfg["current_age"]
    assert years[-1] == cfg["max_age"]
    assert len(years) == len(nominal) == len(real)
    # Real values are nominal discounted by inflation, so smaller for later years.
    assert real[-1] < nominal[-1]


def test_simulate_net_worth_depletion_returns_path_to_depletion(cfg):
    broke = dict(cfg, starting_amount=100_000, annual_contribution=0, annual_spending=400_000)
    age, years, nominal, real = model.simulate_net_worth(broke, 41, 0.0)
    # No fully-funded age, but the path runs up to the depletion point so the
    # chart can still show where it ran out.
    assert age is None
    assert years and len(years) == len(nominal) == len(real)
    assert nominal[-1] <= 0
    # Stops at depletion rather than continuing to max_age.
    assert years[-1] < broke["max_age"]


def test_simulate_net_worth_grows_before_retirement(cfg):
    no_spend = dict(cfg, current_age=40, max_age=60)
    _, years, nominal, _ = model.simulate_net_worth(no_spend, 60, 0.06)
    # Pure accumulation phase: nominal net worth strictly increases each year.
    assert all(b > a for a, b in zip(nominal, nominal[1:]))


# -------------------------
# MC NET WORTH FAN
# -------------------------
def test_compute_mc_net_worth_fan_real_below_nominal(cfg):
    ages, pct, real = model.compute_mc_net_worth_fan(cfg, 60, 0.06, 0.12, seed=31)
    assert set(pct.keys()) == {10, 50, 90}
    assert set(real.keys()) == {10, 50, 90}
    # At the first year inflation discount factor is 1, so real == nominal there.
    assert real[50][0] == pytest.approx(pct[50][0])
    # Later, real is discounted below nominal (where there is wealth left).
    assert real[90][-1] <= pct[90][-1] + 1e-6
