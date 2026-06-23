import numpy as np

DEFAULT_SS_MAX_BENEFIT = 24_000
SS_FULL_RETIREMENT_AGE = 67

# Probability of portfolio survival we treat as "safe" for retirement.
SUCCESS_THRESHOLD = 0.90
SUCCESS_THRESHOLD_PCT = round(SUCCESS_THRESHOLD * 100)

# -------------------------
# DYNAMIC SPENDING ("smart spending" / guardrails)
# -------------------------
# Smart spending flexes the annual withdrawal with the market instead of holding
# it flat in real terms. We follow a Guyton-Klinger-style guardrail: each
# retirement year we compare the current withdrawal rate (planned withdrawal /
# current portfolio) to the rate locked in at the retirement date. If it drifts
# more than GUARDRAIL_BAND above that anchor — markets fell and the rate spiked —
# we cut spending by GUARDRAIL_ADJUST; if it drifts the same distance below, we
# raise it. Spending is clamped to a real-terms band so a long bull or bear run
# can't push the lifestyle to an unrealistic extreme (and so a deep cut can't
# inflate the success rate by quietly starving the plan).
GUARDRAIL_BAND = 0.20
GUARDRAIL_ADJUST = 0.10
SPEND_FLOOR_MULT = 0.75
SPEND_CEILING_MULT = 1.25

# Config fields that actually affect the simulation / report output. The Monte
# Carlo cache key is derived from only these so that toggling UI-only state
# (e.g. show_real_values, advanced_mode, active_profile) does not force a
# re-run — the fan already returns both nominal and real series.
MODEL_CFG_FIELDS = (
    "annual_contribution",
    "annual_spending",
    "current_age",
    "include_social_security",
    "inflation_rate",
    "max_age",
    "social_security_claim_age",
    "social_security_start_age",
    "spending_reduction_after_75",
    "spending_under_75",
    "ss_max_benefit",
    "starting_amount",
    "trials",
    "withdrawal_strategy",
    "years_already_worked",
)


# -------------------------
# SPENDING
# -------------------------
def _annual_spending(cfg):
    return cfg.get("annual_spending", cfg.get("spending_under_75", 200_000))


def build_spending_array(cfg):
    base = _annual_spending(cfg)
    reduction = float(cfg.get("spending_reduction_after_75", 0.0))
    reduced = base * (1.0 - reduction)
    return np.array([
        base if age < 75 else reduced
        for age in range(150)
    ], dtype=np.float64)


def get_spending(age, cfg):
    return float(build_spending_array(cfg)[age])


def withdrawal_strategy(cfg):
    """Active spending strategy: 'fixed' (inflation-only) or 'guardrails'."""
    return cfg.get("withdrawal_strategy", "fixed")


def guardrail_income_band(cfg, base=None):
    """Real-terms (floor, base, ceiling) annual spending under smart spending.

    The guardrail multiplier is clamped to [SPEND_FLOOR_MULT, SPEND_CEILING_MULT],
    so these are the lowest and highest lifestyle the strategy will ever settle
    on, expressed in today's dollars around a base budget. `base` defaults to the
    configured plan spend, but callers pass the sustainable base so the band
    straddles the same number the Sustainable-spending card headlines.
    """
    if base is None:
        base = _annual_spending(cfg)
    return base * SPEND_FLOOR_MULT, base, base * SPEND_CEILING_MULT


# -------------------------
# SOCIAL SECURITY
# -------------------------
def ss_claim_adjustment(claim_age):
    """Benefit multiplier relative to the full-retirement-age (67) benefit.

    Approximates the SSA schedule rather than a flat +/-3%/yr:
      - Delayed-retirement credits of +8%/yr from 67, capped at age 70 (+24%).
      - Early-claiming reductions of ~6.67%/yr for the first 3 years before 67
        and ~5%/yr earlier, bottoming out near -30% at age 62.
    Claim ages outside 62-70 are clamped to that range.
    """
    claim_age = min(max(claim_age, 62), 70)
    if claim_age >= SS_FULL_RETIREMENT_AGE:
        return 1.0 + 0.08 * (claim_age - SS_FULL_RETIREMENT_AGE)
    early_years = SS_FULL_RETIREMENT_AGE - claim_age
    first_three = min(early_years, 3)
    beyond_three = max(early_years - 3, 0)
    return 1.0 - 0.0667 * first_three - 0.05 * beyond_three


def compute_social_security(years_worked, claim_age, max_benefit=DEFAULT_SS_MAX_BENEFIT):
    base = max_benefit * min(years_worked, 35) / 35
    return base * ss_claim_adjustment(claim_age)


def compute_social_security_batch(years_worked, claim_age, max_benefit=DEFAULT_SS_MAX_BENEFIT):
    years_worked = np.asarray(years_worked, dtype=np.float64)
    base = max_benefit * np.minimum(years_worked, 35) / 35
    return base * ss_claim_adjustment(claim_age)


def social_security_income(cfg, years_worked):
    if not cfg.get("include_social_security", True):
        return 0.0
    claim_age = cfg.get("social_security_claim_age", cfg.get("social_security_start_age", 67))
    max_benefit = cfg.get("ss_max_benefit", DEFAULT_SS_MAX_BENEFIT)
    return compute_social_security(years_worked, claim_age, max_benefit)


def social_security_income_batch(cfg, years_worked):
    if not cfg.get("include_social_security", True):
        return np.zeros_like(np.asarray(years_worked, dtype=np.float64), dtype=np.float64)
    claim_age = cfg.get("social_security_claim_age", cfg.get("social_security_start_age", 67))
    max_benefit = cfg.get("ss_max_benefit", DEFAULT_SS_MAX_BENEFIT)
    return compute_social_security_batch(years_worked, claim_age, max_benefit)


def social_security_start_age(cfg):
    if not cfg.get("include_social_security", True):
        return None
    return cfg.get("social_security_claim_age", cfg.get("social_security_start_age", 67))


# -------------------------
# RETURN MODEL (log-normal)
# -------------------------
def generate_returns(mean_return, volatility, trials, n_years, seed=None):
    rng = np.random.default_rng(seed)
    variance = volatility ** 2
    sigma_ln = np.sqrt(np.log(1 + variance / (1 + mean_return) ** 2))
    mu_ln = np.log(1 + mean_return) - sigma_ln ** 2 / 2
    return np.expm1(rng.normal(mu_ln, sigma_ln, (trials, n_years)))


# -------------------------
# VECTORIZED MONTE CARLO
# -------------------------
def _spending_matrix(retirement_ages, current_age, max_age, inflation_rate, ss_start_age, ss_incomes, spending_by_age):
    calendar_ages = np.arange(current_age, max_age + 1)
    ages = np.asarray(retirement_ages, dtype=np.int64)

    yrs_since = np.maximum(calendar_ages[None, :] - ages[:, None], 0)
    retired = calendar_ages[None, :] >= ages[:, None]
    spending_base = spending_by_age[calendar_ages]
    spending = spending_base * ((1 + inflation_rate) ** yrs_since)
    spending = np.where(retired, spending, 0.0)

    if ss_start_age is not None:
        # Social Security has a cost-of-living adjustment, so the benefit
        # inflates from the retirement year just like spending does. This keeps
        # the real value of the SS offset constant instead of letting a fixed
        # nominal amount erode over decades of retirement.
        ss_active = calendar_ages[None, :] >= ss_start_age
        ss_offset = ss_incomes[:, None] * ((1 + inflation_rate) ** yrs_since)
        spending = np.maximum(spending - np.where(ss_active, ss_offset, 0.0), 0.0)
    return spending


def _guardrail_inputs(cfg, retirement_ages, ss_incomes=None):
    """Per-(retirement age, calendar year) matrices for the guardrail simulator.

    Unlike `_spending_matrix`, the Social Security offset is kept *separate* from
    spending: the guardrail multiplier scales lifestyle spending, while the SS
    benefit is a fixed real floor that does not flex with the market.
    """
    current_age = cfg["current_age"]
    max_age = cfg["max_age"]
    calendar_ages = np.arange(current_age, max_age + 1)
    ages = np.asarray(retirement_ages, dtype=np.int64)
    inflation = cfg["inflation_rate"]
    spending_by_age = build_spending_array(cfg)

    yrs_since = np.maximum(calendar_ages[None, :] - ages[:, None], 0)
    retired = calendar_ages[None, :] >= ages[:, None]
    gross = spending_by_age[calendar_ages][None, :] * ((1 + inflation) ** yrs_since)
    gross = np.where(retired, gross, 0.0)

    if ss_incomes is None:
        years_worked = cfg["years_already_worked"] + (ages - current_age)
        ss_incomes = social_security_income_batch(cfg, years_worked)
    else:
        ss_incomes = np.asarray(ss_incomes, dtype=np.float64)

    ss_start = social_security_start_age(cfg)
    if ss_start is not None:
        ss_active = calendar_ages[None, :] >= ss_start
        ss_off = ss_incomes[:, None] * ((1 + inflation) ** yrs_since)
        ss_off = np.where(ss_active, ss_off, 0.0)
    else:
        ss_off = np.zeros_like(gross)

    contribute = np.where(
        calendar_ages[None, :] < ages[:, None], cfg["annual_contribution"], 0.0,
    )
    is_retire_year = calendar_ages[None, :] == ages[:, None]
    return calendar_ages, gross, ss_off, contribute, retired, is_retire_year


def _simulate_guardrails(cfg, retirement_ages, returns, ss_incomes=None, collect_paths=False):
    """Monte Carlo with Guyton-Klinger spending guardrails, vectorized across
    trials and retirement ages at once.

    Returns (calendar_ages, alive, paths) where `alive` is the final
    per-(trial, age) survival mask and `paths` is the full net-worth tensor of
    shape (trials, ages, years) — or None when not requested, to avoid the
    allocation on the success-curve path that only needs survival.
    """
    calendar_ages, gross, ss_off, contribute, retired, is_retire_year = _guardrail_inputs(
        cfg, retirement_ages, ss_incomes,
    )
    n_trials = returns.shape[0]
    n_ages = gross.shape[0]
    n_years = len(calendar_ages)

    net = np.full((n_trials, n_ages), cfg["starting_amount"], dtype=np.float64)
    alive = np.ones((n_trials, n_ages), dtype=bool)
    spend_mult = np.ones((n_trials, n_ages), dtype=np.float64)
    anchor_wr = np.zeros((n_trials, n_ages), dtype=np.float64)
    anchored = np.zeros((n_trials, n_ages), dtype=bool)

    paths = (
        np.zeros((n_trials, n_ages, n_years), dtype=np.float64)
        if collect_paths else None
    )

    for i in range(n_years):
        net = np.where(alive, net + contribute[:, i], net)
        net = np.where(alive, net * (1.0 + returns[:, i:i + 1]), net)

        retired_i = retired[:, i]
        if retired_i.any():
            gross_i = gross[:, i][None, :]
            ss_i = ss_off[:, i][None, :]
            retire_year_i = is_retire_year[:, i][None, :]

            # Withdrawal rate at the current spend level vs. the current balance.
            withdrawal = np.maximum(gross_i * spend_mult - ss_i, 0.0)
            wr = withdrawal / np.maximum(net, 1.0)

            # Lock the anchor rate the first retired year, then flex against it.
            lock = retire_year_i & alive & ~anchored
            anchor_wr = np.where(lock, wr, anchor_wr)
            anchored |= lock

            flex = retired_i[None, :] & alive & anchored & ~lock
            upper = anchor_wr * (1.0 + GUARDRAIL_BAND)
            lower = anchor_wr * (1.0 - GUARDRAIL_BAND)
            spend_mult = np.where(
                flex & (wr > upper), spend_mult * (1.0 - GUARDRAIL_ADJUST), spend_mult,
            )
            spend_mult = np.where(
                flex & (wr < lower), spend_mult * (1.0 + GUARDRAIL_ADJUST), spend_mult,
            )
            spend_mult = np.clip(spend_mult, SPEND_FLOOR_MULT, SPEND_CEILING_MULT)

            # Re-derive the withdrawal after any adjustment, then deduct it.
            withdrawal = np.maximum(gross_i * spend_mult - ss_i, 0.0)
            spend_active = retired_i[None, :] & alive
            net = np.where(spend_active, net - withdrawal, net)
            alive &= ~(spend_active & (net <= 0))

        if collect_paths:
            paths[:, :, i] = np.where(alive, net, 0.0)

    return calendar_ages, alive, paths


def _simulate_trial_paths(cfg, retirement_age, returns, ss_income=None):
    """Run trials and return net-worth path for each trial (0 after depletion)."""
    if withdrawal_strategy(cfg) == "guardrails":
        ss_incomes = None if ss_income is None else [ss_income]
        calendar_ages, _, paths = _simulate_guardrails(
            cfg, [retirement_age], returns, ss_incomes=ss_incomes, collect_paths=True,
        )
        return calendar_ages, paths[:, 0, :]

    current_age = cfg["current_age"]
    max_age = cfg["max_age"]
    n_years = max_age - current_age + 1
    trials = returns.shape[0]
    spending_by_age = build_spending_array(cfg)

    if ss_income is None:
        years_worked = cfg["years_already_worked"] + (retirement_age - current_age)
        ss_income = social_security_income(cfg, years_worked)

    spending = _spending_matrix(
        np.array([retirement_age], dtype=np.int64),
        current_age,
        max_age,
        cfg["inflation_rate"],
        social_security_start_age(cfg),
        np.array([ss_income], dtype=np.float64),
        spending_by_age,
    )[0]

    calendar_ages = np.arange(current_age, max_age + 1)
    contribute = (calendar_ages < retirement_age).astype(np.float64) * cfg["annual_contribution"]

    paths = np.zeros((trials, n_years), dtype=np.float64)
    net = np.full(trials, cfg["starting_amount"], dtype=np.float64)
    alive = np.ones(trials, dtype=bool)

    for i in range(n_years):
        net = np.where(alive, net + contribute[i], net)
        net = np.where(alive, net * (1.0 + returns[:, i]), net)
        if spending[i] > 0:
            net = np.where(alive, net - spending[i], net)
            alive &= net > 0
        paths[:, i] = np.where(alive, net, 0.0)

    return calendar_ages, paths


def simulate_batch(
    cfg,
    retirement_age,
    mean_return,
    volatility,
    ss_income,
    trials=5000,
    returns=None,
    seed=None,
):
    current_age = cfg["current_age"]
    max_age = cfg["max_age"]
    n_years = max_age - current_age + 1

    if returns is None:
        returns = generate_returns(mean_return, volatility, trials, n_years, seed)

    _, paths = _simulate_trial_paths(cfg, retirement_age, returns, ss_income)
    return (paths[:, -1] > 0).mean()


def _simulate_all_retirement_ages(cfg, retirement_ages, returns):
    """Vectorized Monte Carlo across all retirement ages at once."""
    if withdrawal_strategy(cfg) == "guardrails":
        _, alive, _ = _simulate_guardrails(cfg, retirement_ages, returns)
        return alive.mean(axis=0)

    current_age = cfg["current_age"]
    max_age = cfg["max_age"]
    n_years = max_age - current_age + 1
    ages = np.asarray(retirement_ages, dtype=np.int64)

    calendar_ages = np.arange(current_age, max_age + 1)
    inflation = cfg["inflation_rate"]
    ss_start = social_security_start_age(cfg)
    contribution = cfg["annual_contribution"]
    spending_by_age = build_spending_array(cfg)

    years_worked = cfg["years_already_worked"] + (ages - current_age)
    ss_incomes = social_security_income_batch(cfg, years_worked)

    spending = _spending_matrix(
        ages, current_age, max_age, inflation, ss_start, ss_incomes, spending_by_age,
    )

    contribute = np.where(
        calendar_ages[None, :] < ages[:, None], contribution, 0.0,
    )

    net = np.full((returns.shape[0], len(ages)), cfg["starting_amount"], dtype=np.float64)
    alive = np.ones((returns.shape[0], len(ages)), dtype=bool)

    for i in range(n_years):
        net = np.where(alive, net + contribute[:, i], net)
        net = np.where(alive, net * (1.0 + returns[:, i:i + 1]), net)
        if spending[:, i].any():
            net = np.where(alive, net - spending[:, i], net)
            alive &= net > 0

    return alive.mean(axis=0)


# -------------------------
# CURVE
# -------------------------
def compute_curve(cfg, retirement_age_range, mean_return, volatility, seed=None):
    ages = np.array(list(retirement_age_range), dtype=np.int64)
    n_years = cfg["max_age"] - cfg["current_age"] + 1
    returns = generate_returns(
        mean_return, volatility, cfg["trials"], n_years, seed,
    )
    probs = _simulate_all_retirement_ages(cfg, ages, returns)
    return ages, probs


# -------------------------
# SEQUENCE-OF-RETURNS / PERCENTILE PATHS
# -------------------------
def simulate_percentile_paths(
    cfg,
    retirement_age,
    mean_return,
    volatility,
    seed=None,
    percentiles=(10, 50, 90),
):
    n_years = cfg["max_age"] - cfg["current_age"] + 1
    returns = generate_returns(
        mean_return, volatility, cfg["trials"], n_years, seed,
    )
    calendar_ages, paths = _simulate_trial_paths(cfg, retirement_age, returns)
    pct_values = np.percentile(paths, percentiles, axis=0)
    return calendar_ages, dict(zip(percentiles, pct_values))


# -------------------------
# OPTIMIZER
# -------------------------
def _success_at_career_years(
    cfg, retirement_age, years_worked, mean_return, volatility,
    returns=None, seed=None,
):
    ss_income = social_security_income(cfg, years_worked)
    return simulate_batch(
        cfg, retirement_age, mean_return, volatility, ss_income, cfg["trials"],
        returns=returns, seed=seed,
    )


def find_min_years_worked(
    cfg, retirement_age, mean_return, volatility, target=SUCCESS_THRESHOLD, max_years=50,
    seed=None,
):
    """Minimum total career years by retirement for target success."""
    lo = cfg["years_already_worked"]
    hi = max_years
    result = None

    n_years = cfg["max_age"] - cfg["current_age"] + 1
    returns = generate_returns(
        mean_return, volatility, cfg["trials"], n_years, seed,
    )

    while lo <= hi:
        mid = (lo + hi) // 2
        if _success_at_career_years(
            cfg, retirement_age, mid, mean_return, volatility,
            returns=returns, seed=seed,
        ) >= target:
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return result


def find_max_sustainable_spending(
    cfg, retirement_age, mean_return, volatility,
    target=SUCCESS_THRESHOLD, seed=None, step=1000,
):
    """Largest annual spending whose success rate at `retirement_age` still
    clears `target`.

    The inverse of the success curve: instead of scoring a fixed budget, it
    solves for the budget the plan can actually sustain. Spending and success
    move in opposite directions monotonically, so a binary search on a `step`
    grid lands exactly on the boundary. A single returns draw is shared across
    every probe so the comparison is apples-to-apples and fast.

    Returns dollars rounded down to the nearest `step`, or None if the plan
    misses the target even at zero spending (a degenerate, broke config).
    """
    n_years = cfg["max_age"] - cfg["current_age"] + 1
    returns = generate_returns(mean_return, volatility, cfg["trials"], n_years, seed)
    years_worked = cfg["years_already_worked"] + max(retirement_age - cfg["current_age"], 0)
    ss_income = social_security_income(cfg, years_worked)

    def success(spend):
        trial_cfg = dict(cfg)
        trial_cfg["annual_spending"] = float(spend)
        return simulate_batch(
            trial_cfg, retirement_age, mean_return, volatility,
            ss_income, cfg["trials"], returns=returns,
        )

    # Zero spending never depletes a non-empty portfolio; if even that misses
    # the target there is nothing to solve for.
    if success(0) < target:
        return None

    # Expand an upper bound until spending is clearly unsustainable, seeding
    # from the current plan so the common case converges in a few probes.
    base = max(step, int(np.ceil(cfg.get("annual_spending", step) / step)) * step)
    hi = base
    for _ in range(40):
        if success(hi) < target:
            break
        hi *= 2
    else:
        return int(hi)  # extraordinarily well-funded — return the cap reached

    # Binary search the grid for the largest sustainable budget.
    lo_k, hi_k, best = 0, hi // step, 0
    while lo_k <= hi_k:
        mid_k = (lo_k + hi_k) // 2
        if success(mid_k * step) >= target:
            best = mid_k * step
            lo_k = mid_k + 1
        else:
            hi_k = mid_k - 1
    return int(best)


# -------------------------
# NET WORTH PROJECTION (deterministic overlay)
# -------------------------
def simulate_net_worth(cfg, retirement_age, return_rate):
    current_age = cfg["current_age"]
    max_age = cfg["max_age"]

    years_worked = cfg["years_already_worked"] + (retirement_age - current_age)
    ss_income = social_security_income(cfg, years_worked)
    ss_start = social_security_start_age(cfg)

    spending_array = build_spending_array(cfg)

    age = current_age
    net_worth = cfg["starting_amount"]
    years = []
    nominal = []
    real = []
    depleted_at = None

    while age <= max_age:
        if age < retirement_age:
            net_worth += cfg["annual_contribution"]

        net_worth *= (1 + return_rate)

        if age >= retirement_age:
            base = spending_array[age]
            yrs = age - retirement_age
            spending = base * ((1 + cfg["inflation_rate"]) ** yrs)

            if ss_start is not None and age >= ss_start:
                # SS benefit inflates with a COLA, matching the spending path.
                spending -= ss_income * ((1 + cfg["inflation_rate"]) ** yrs)
                spending = max(0, spending)

            net_worth -= spending

        real_nw = net_worth / ((1 + cfg["inflation_rate"]) ** (age - current_age))
        years.append(age)
        nominal.append(net_worth)
        real.append(real_nw)

        if age >= retirement_age and net_worth <= 0:
            # Funds depleted: record the depletion point and stop so the
            # chart can still show where the path ran out.
            depleted_at = age
            break

        age += 1

    fully_funded_age = retirement_age if depleted_at is None else None
    return fully_funded_age, years, nominal, real


def compute_mc_net_worth_fan(cfg, retirement_age, mean_return, volatility, seed=None):
    """Monte Carlo percentile fan at a fixed retirement age."""
    calendar_ages, pct = simulate_percentile_paths(
        cfg, retirement_age, mean_return, volatility, seed,
    )
    inflation = cfg["inflation_rate"]
    current_age = cfg["current_age"]
    yrs = calendar_ages - current_age
    real = {
        p: pct[p] / ((1 + inflation) ** yrs)
        for p in pct
    }
    return calendar_ages, pct, real
