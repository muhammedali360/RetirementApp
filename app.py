import base64
import io
import json
import os
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.ticker import FuncFormatter

try:
    from streamlit_local_storage import LocalStorage
except ImportError:  # optional dependency; falls back to the on-disk file
    LocalStorage = None

from model import (
    MODEL_CFG_FIELDS,
    SUCCESS_THRESHOLD,
    SUCCESS_THRESHOLD_PCT,
    coast_growth_paths,
    coast_success,
    compute_curve,
    compute_mc_net_worth_fan,
    find_coast_number,
    find_max_sustainable_spending,
    find_min_years_worked,
    guardrail_income_band,
    simulate_net_worth,
    social_security_claim_comparison,
    social_security_start_age,
)
from report import (
    build_executive_summary,
    delta_pts,
    generate_report_bytes,
    pick_best_levers,
    safe_retirement_ranges,
)

SCENARIO_FILE = Path("scenarios.json")
LEGACY_SAVED_FILE = Path("saved_scenarios.json")
CUSTOM_CSS = Path(__file__).parent / ".streamlit" / "static" / "custom.css"

APP_NAME = "Retirement Runway"
SS_CLAIM_AGES = [62, 67, 70]
# Grid resolution for the sustainable-spending solver; also the threshold below
# which "headroom vs. plan" rounds to break-even. Keep in sync with the
# `step` default of model.find_max_sustainable_spending.
SAFE_SPEND_STEP = 1000

RISK_PRESETS = {
    "Conservative": {"mean_return": 0.05, "volatility": 0.09},
    "Balanced": {"mean_return": 0.06, "volatility": 0.12},
    "Aggressive": {"mean_return": 0.07, "volatility": 0.14},
}

RISK_PRESET_LABELS = {
    "Conservative": "Conservative — 5% return, 9% volatility",
    "Balanced": "Balanced — 6% return, 12% volatility",
    "Aggressive": "Aggressive — 7% return, 14% volatility",
}


def match_risk_preset(mean_return, volatility):
    best = "Balanced"
    best_dist = float("inf")
    for name, preset in RISK_PRESETS.items():
        dist = abs(preset["mean_return"] - mean_return) + abs(preset["volatility"] - volatility)
        if dist < best_dist:
            best_dist = dist
            best = name
    return best


THEME = {
    "primary": "#ec4899",
    "secondary": "#fafafa",
    "accent": "#f472b6",
    "success": "#34d399",
    "danger": "#f87171",
    "muted": "#737373",
    "threshold": "#a3a3a3",
    "grid": "#2a2a2a",
    "face": "#141414",
    "text": "#fafafa",
}

CHART_PALETTE = ["#ec4899", "#fafafa", "#737373", "#a3a3a3", "#f472b6", "#525252"]

FAN_COLORS = {
    10: "#60a5fa",
    50: "#ec4899",
    90: "#34d399",
}

WHAT_IF_VARIANTS = [
    ("save_10k", "Save +$10K/yr", lambda cfg, _ta: {
        "annual_contribution": cfg["annual_contribution"] + 10_000,
    }),
    ("save_25k", "Save +$25K/yr", lambda cfg, _ta: {
        "annual_contribution": cfg["annual_contribution"] + 25_000,
    }),
    ("spend_down", "Spend −10%", lambda cfg, _ta: {
        "annual_spending": int(cfg["annual_spending"] * 0.9),
    }),
    ("retire_later", "Retire later", lambda cfg, ta: {
        "target_retirement_age": min(ta + 2, 85),
    }),
    ("retire_earlier", "Retire earlier", lambda cfg, ta: {
        "target_retirement_age": max(cfg["current_age"] + 1, ta - 2),
    }),
]


def apply_chart_style():
    plt.rcParams.update(
        {
            "figure.facecolor": THEME["face"],
            "axes.facecolor": THEME["face"],
            "axes.edgecolor": THEME["grid"],
            "axes.labelcolor": THEME["text"],
            "text.color": THEME["text"],
            "xtick.color": THEME["muted"],
            "ytick.color": THEME["muted"],
            "grid.color": THEME["grid"],
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "legend.framealpha": 0.92,
            "legend.edgecolor": THEME["grid"],
        }
    )


def format_currency(value):
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"


def format_money(value):
    """Full-precision dollars (e.g. $1,700) for human-scale figures — a monthly
    check or annual income — where rounding to $1K/$2K hides the detail that
    matters (and can make two different amounts look identical). Use
    format_currency for big-picture portfolio / lifetime sums."""
    return f"${value:,.0f}"


def prob_at_age(ages, probs, target_age):
    ages = np.asarray(ages)
    probs = np.asarray(probs)
    idx = np.where(ages == target_age)[0]
    if len(idx) == 0:
        return None
    return float(probs[idx[0]])


def chart_age_range(current_age, target_age):
    start_age = int(current_age)
    end_age = int(min(85, max(target_age + 5, 75)))
    return start_age, end_age


PROFILE_ALIASES = {
    "Early career": "Early",
    "Mid-career": "Mid",
    "High net worth": "Mid",
    "Wealthy": "Mid",
    "Near retirement": "Pre-retire",
    "Conservative": "Pre-retire",
}

PRESET_PROFILES = {
    "Early": {
        "tagline": "Building your foundation",
        "current_age": 28,
        "starting_amount": 45_000,
        "annual_contribution": 15_000,
        "target_retirement_age": 65,
        "annual_spending": 72_000,
        "spending_reduction_after_75": 0.17,
        "years_already_worked": 4,
        "mean_return": 0.07,
        "volatility": 0.14,
        "inflation_rate": 0.03,
        "include_social_security": True,
        "social_security_claim_age": 67,
    },
    "Mid": {
        "tagline": "Balanced growth and spending",
        "current_age": 42,
        "starting_amount": 350_000,
        "annual_contribution": 55_000,
        "target_retirement_age": 65,
        "annual_spending": 90_000,
        "spending_reduction_after_75": 0.17,
        "years_already_worked": 18,
        "mean_return": 0.06,
        "volatility": 0.12,
        "inflation_rate": 0.03,
        "include_social_security": True,
        "social_security_claim_age": 67,
    },
    "Pre-retire": {
        "tagline": "Fine-tuning your exit",
        "current_age": 58,
        "starting_amount": 1_500_000,
        "annual_contribution": 65_000,
        "target_retirement_age": 65,
        "annual_spending": 100_000,
        "spending_reduction_after_75": 0.15,
        "years_already_worked": 34,
        "mean_return": 0.05,
        "volatility": 0.10,
        "inflation_rate": 0.03,
        "include_social_security": True,
        "social_security_claim_age": 67,
    },
}

# App-level defaults: seeded from the mid-career preset (so a fresh session
# opens on a balanced, generic profile) plus simulation/UI settings that aren't
# part of any preset.
DEFAULT_PROFILE = "Mid"

DEFAULT_CFG = {
    **{k: v for k, v in PRESET_PROFILES[DEFAULT_PROFILE].items() if k != "tagline"},
    "max_age": 90,
    "trials": 2000,
    "advanced_mode": False,
    "active_profile": DEFAULT_PROFILE,
    "show_real_values": False,
    "withdrawal_strategy": "fixed",
    # Part-time "bridge" income during early retirement — off by default; only
    # surfaced as a suggested lever when a plan is at risk (see the bridge card).
    "bridge_income": 0,
    "bridge_end_age": None,
}

# A plan is treated as "near retirement" when the person is already in their late
# 50s+ or only a handful of years from their target age. For this group the
# accumulation framing (coast number, earliest-safe-age) is the wrong lead — they
# care about their guaranteed income floor and how much they can safely spend.
NEAR_RETIREMENT_AGE = 55
NEAR_RETIREMENT_WINDOW = 7


def is_near_retirement(cfg, target_age):
    years_until = target_age - cfg["current_age"]
    return cfg["current_age"] >= NEAR_RETIREMENT_AGE or years_until <= NEAR_RETIREMENT_WINDOW

GLOSSARY = {
    "Success rate": "Share of Monte Carlo trials where your portfolio stays positive through your planning horizon.",
    "Earliest safe age": f"The youngest retirement age (searched up to 85) whose success rate reaches ≥{SUCCESS_THRESHOLD_PCT}% — where success means the portfolio lasts through your planning horizon.",
    "Withdrawal rate": "Annual spending divided by portfolio at retirement — the 4% rule targets ≤4%.",
    "Sequence-of-returns risk": "Bad market returns early in retirement can permanently reduce portfolio durability.",
    "Planning horizon": "The age through which the model checks whether your money lasts (default: 100).",
    "Return assumptions": "Returns are modeled on a broad stock-market index like the S&P 500 — annual returns are drawn from a log-normal distribution whose average and volatility you set via the preset (or the Market sliders in advanced mode).",
    "Guaranteed income": "Inflation-adjusted income you receive for life regardless of markets — here, your Social Security benefit at your chosen claim age. It's the floor your plan is built on top of.",
    "Safe spend": f"The largest total annual budget that still clears ≥{SUCCESS_THRESHOLD_PCT}% success at your target age — what your savings can sustain on top of guaranteed income.",
    "Bridge income": "Part-time earnings during early retirement that cover some spending so you can delay drawing your portfolio (and delay claiming Social Security for a bigger benefit).",
}


def normalize_profile_name(name):
    if not name:
        return name
    return PROFILE_ALIASES.get(name, name)


def apply_profile(cfg, profile_name):
    profile_name = normalize_profile_name(profile_name)
    profile = PRESET_PROFILES[profile_name]
    for key, value in profile.items():
        if key != "tagline":
            cfg[key] = value
    cfg["active_profile"] = profile_name
    return cfg


def apply_saved_scenario(cfg, entry):
    merged = merge_cfg(entry.get("cfg", {}))
    cfg.clear()
    cfg.update(merged)
    return cfg


def portfolio_at_retirement(cfg, retirement_age, mean_return):
    if retirement_age <= cfg["current_age"]:
        return cfg["starting_amount"]
    _, years, nominal, _ = simulate_net_worth(cfg, retirement_age, mean_return)
    if not years or retirement_age not in years:
        return None
    idx = years.index(retirement_age)
    if idx == 0:
        return cfg["starting_amount"]
    return nominal[idx - 1]


def compute_withdrawal_rate(cfg, retirement_age, mean_return):
    spending = cfg.get("annual_spending", 0)
    if spending <= 0:
        return None
    portfolio = portfolio_at_retirement(cfg, retirement_age, mean_return)
    if portfolio is None or portfolio <= 0:
        return None
    return spending / portfolio


def inject_custom_css():
    if CUSTOM_CSS.exists():
        st.markdown(f"<style>{CUSTOM_CSS.read_text()}</style>", unsafe_allow_html=True)


def insight_status(target_prob, on_track):
    if on_track and target_prob is not None and target_prob >= SUCCESS_THRESHOLD:
        return "good"
    if target_prob is not None and target_prob < 0.80:
        return "warn"
    return "neutral"


def hero_status(target_prob, on_track):
    if target_prob is not None and target_prob >= SUCCESS_THRESHOLD and on_track:
        return "On track", "good"
    if target_prob is not None and target_prob >= 0.80:
        return "Close", "neutral"
    return "At risk", "warn"


def prob_semantic_class(target_prob):
    if target_prob is None:
        return "neutral"
    if target_prob >= SUCCESS_THRESHOLD:
        return "good"
    if target_prob >= 0.80:
        return "neutral"
    return "warn"


def withdrawal_semantic_class(withdrawal_rate):
    if withdrawal_rate is None:
        return "neutral"
    if withdrawal_rate <= 0.04:
        return "good"
    if withdrawal_rate <= 0.05:
        return "neutral"
    return "warn"


def format_delta_arrow(delta):
    pts = delta_pts(delta)
    if pts is None or pts == 0:
        return "—", "neutral"
    if pts > 0:
        return f"▲ {pts:+d} pts", "up"
    return f"▼ {pts:+d} pts", "down"


def build_verdict(target_prob, target_age, max_age, ranges):
    """One plain-English sentence — the lead, before any chart."""
    if target_prob is None:
        return "Not enough data to estimate your plan’s success."
    headline = (
        f"Retiring at {target_age} has a {target_prob * 100:.0f}% chance "
        f"of lasting to {max_age}."
    )
    if ranges:
        headline += f" Earliest safe age: {ranges[0][0]}."
    return headline


def build_verdict_subline(target_prob, best_levers):
    if not best_levers:
        return None
    top = best_levers[0]
    pts = delta_pts(top.get("delta"))
    if pts in (None, 0):
        return None
    at_target = target_prob is not None and target_prob >= SUCCESS_THRESHOLD
    prefix = "Optional margin" if at_target else "Best lever"
    return f"{prefix}: {top['Change']} ({pts:+d} pts)."


def build_verdict_near_retiree(target_age, ss_floor, ss_claim_age, safe_spend):
    """Lead with the floor + safe budget for someone close to (or in) retirement.

    'Earliest safe age' is the wrong headline this late — it's usually irrelevant
    or None for the under-saved. What matters is the guaranteed income they'll
    have for life and how much, realistically, they can spend on top of it.
    """
    parts = []
    if ss_floor and ss_claim_age:
        parts.append(
            f"Social Security gives you a guaranteed {format_money(ss_floor)}/yr "
            f"for life from age {ss_claim_age}."
        )
    if safe_spend:
        lead = "On top of that, your" if parts else "Your"
        parts.append(
            f"{lead} savings can safely support about {format_money(safe_spend)}/yr "
            f"of total spending retiring at {target_age}."
        )
    else:
        parts.append(
            f"At age {target_age} your savings add little on their own — part-time work "
            "or claiming Social Security later would do the heavy lifting."
        )
    return " ".join(parts)


def render_verdict(headline, subline, status="neutral"):
    sub_html = f'<p class="verdict-sub">{subline}</p>' if subline else ""
    st.markdown(
        f'<div class="verdict-card">'
        f'<div class="signal-header">'
        f'<span class="signal-dot {status}"></span>'
        f'<span class="signal-label">Verdict</span>'
        f"</div>"
        f'<p class="verdict-headline">{headline}</p>'
        f"{sub_html}</div>",
        unsafe_allow_html=True,
    )


def render_hero(
    *,
    current_age,
    target_age,
    portfolio,
    active_profile,
    target_prob=None,
    on_track=False,
):
    status_label, status_class = hero_status(target_prob, on_track)
    chips = []
    if active_profile and active_profile in PRESET_PROFILES:
        chips.append(f'<span class="hero-chip">{active_profile}</span>')
    chips.append(f'<span class="hero-chip hero-chip-mono">Age {current_age} → {target_age}</span>')
    chips.append(
        f'<span class="hero-chip hero-chip-mono">{format_currency(portfolio)} portfolio</span>'
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-header">
                <h1 class="hero-title">{APP_NAME}</h1>
                <span class="hero-status {status_class}">{status_label}</span>
                <span class="hero-chips">{"".join(chips)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label, value, value_cls, tooltip):
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label" title="{tooltip}">{label}<span class="kpi-info">ⓘ</span></div>'
        f'<div class="kpi-value {value_cls}">{value}</div>'
        f"</div>"
    )


def render_kpi_row(
    *,
    target_age,
    target_prob,
    ranges,
    withdrawal_rate,
    near_retiree=False,
    ss_floor=None,
    ss_claim_age=None,
    safe_spend=None,
):
    prob_val = f"{target_prob * 100:.0f}%" if target_prob is not None else "—"
    prob_cls = prob_semantic_class(target_prob)
    cards = [_kpi_card(
        f"Success rate · age {target_age}", prob_val, prob_cls, GLOSSARY["Success rate"],
    )]

    if near_retiree:
        # Lead with the two questions a near-retiree actually asks: what income is
        # guaranteed for life, and how much can I safely spend — not "earliest
        # safe age," which is usually irrelevant (or None) this close to the exit.
        if ss_floor:
            floor_val = f"{format_money(ss_floor)}<span class='kpi-unit'>/yr</span>"
            floor_cls = "good"
            floor_label = (
                f"Guaranteed income · from {ss_claim_age}" if ss_claim_age
                else "Guaranteed income"
            )
        else:
            floor_val, floor_cls, floor_label = "None", "warn", "Guaranteed income"
        cards.append(_kpi_card(floor_label, floor_val, floor_cls, GLOSSARY["Guaranteed income"]))

        if safe_spend:
            spend_val = f"{format_money(safe_spend)}<span class='kpi-unit'>/yr</span>"
            spend_cls = "neutral"
        else:
            spend_val, spend_cls = "—", "warn"
        cards.append(_kpi_card(
            f"Safe spend · age {target_age}", spend_val, spend_cls, GLOSSARY["Safe spend"],
        ))
    else:
        earliest = str(ranges[0][0]) if ranges else "None"
        if ranges and ranges[0][0] <= target_age:
            earliest_cls = "good"
        elif ranges:
            earliest_cls = "neutral"
        else:
            earliest_cls = "warn"
        cards.append(_kpi_card(
            "Earliest safe age", earliest, earliest_cls, GLOSSARY["Earliest safe age"],
        ))
        wr_val = f"{withdrawal_rate * 100:.1f}%" if withdrawal_rate is not None else "—"
        wr_cls = withdrawal_semantic_class(withdrawal_rate)
        cards.append(_kpi_card(
            f"Withdrawal rate · age {target_age}", wr_val, wr_cls, GLOSSARY["Withdrawal rate"],
        ))

    st.markdown(
        f'<div class="kpi-sticky"><div class="kpi-grid">{"".join(cards)}</div></div>',
        unsafe_allow_html=True,
    )


def safe_spend_state(safe_spend, current_spend, target_age):
    """Compare the sustainable budget to the user's plan → (status, message)."""
    if safe_spend is None or safe_spend < SAFE_SPEND_STEP:
        return "warn", (
            f"No budget reaches {SUCCESS_THRESHOLD_PCT}% at age {target_age} — "
            "retire later, save more, or claim Social Security."
        )
    headroom = safe_spend - current_spend
    if headroom >= SAFE_SPEND_STEP:
        return "good", (
            f"▲ {format_currency(headroom)}/yr of headroom above your "
            f"{format_currency(current_spend)} plan — room to spend more."
        )
    if headroom <= -SAFE_SPEND_STEP:
        return "warn", (
            f"▼ {format_currency(-headroom)}/yr over your safe max — "
            f"trim toward {format_currency(safe_spend)} or retire later."
        )
    return "neutral", (
        f"Right at your safe max — your {format_currency(current_spend)} "
        "plan leaves little margin for a rough market."
    )


def render_safe_spend(safe_spend, current_spend, target_age, withdrawal_rate, band=None):
    """Headline 'how much can I actually spend?' card — the inverse insight.

    Under smart spending the sustainable answer is a base *plus* a flex band, so
    `band` (floor, _base, ceiling) is folded in as a sub-line that straddles the
    headline number instead of competing with it in a separate card.
    """
    if safe_spend is None:
        return
    status, message = safe_spend_state(safe_spend, current_spend, target_age)
    wr_txt = (
        f' <span class="safespend-sep">·</span> {withdrawal_rate * 100:.1f}% withdrawal rate'
        if withdrawal_rate is not None
        else ""
    )
    conf = f"{SUCCESS_THRESHOLD_PCT}% confidence"
    band_html = ""
    if band is not None:
        floor, _base, ceiling = band
        conf += " · smart"
        band_html = (
            f'<div class="safespend-band">Flexes '
            f"{format_currency(floor)}–{format_currency(ceiling)}/yr "
            "as markets move</div>"
        )
    st.markdown(
        f'<div class="safespend-card {status}">'
        f'<div class="signal-header">'
        f'<span class="signal-dot {status}"></span>'
        f'<span class="signal-label">Sustainable spending</span>'
        f'<span class="safespend-conf">{conf}</span>'
        f"</div>"
        f'<div class="safespend-value">{format_currency(safe_spend)}'
        f'<span class="safespend-unit">/yr</span></div>'
        f'<div class="safespend-sub">retiring at age {target_age}{wr_txt}</div>'
        f"{band_html}"
        f'<div class="safespend-delta {status}">{message}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_ss_timing(comparison, ss_success, plan_claim_age, target_age):
    """Side-by-side 'when to claim Social Security' comparison.

    For each candidate claim age it shows the monthly check, the lifetime total
    in today's dollars, and the plan's success rate if you claim then — so the
    single biggest lever for an under-saved retiree is a real, clickable choice.
    Returns the claim age the user chose to switch to, or None.
    """
    if not comparison:
        return None
    lo, hi = comparison[0], comparison[-1]
    cols = st.columns(len(comparison))
    apply_choice = None
    for col, row in zip(cols, comparison):
        claim = row["claim_age"]
        is_plan = claim == plan_claim_age
        succ = ss_success.get(claim)
        succ_txt = f"{succ * 100:.0f}%" if succ is not None else "—"
        succ_cls = prob_semantic_class(succ)
        badge = ""
        if is_plan:
            badge = '<span class="ss-badge plan">Your plan</span>'
        elif claim == hi["claim_age"]:
            badge = '<span class="ss-badge best">Biggest check</span>'
        with col:
            st.markdown(
                f'<div class="ss-card {"plan" if is_plan else ""}">'
                f'<div class="ss-claim">Claim at {claim}{badge}</div>'
                f'<div class="ss-monthly">{format_money(row["monthly"])}'
                f'<span class="ss-unit">/mo</span></div>'
                f'<div class="ss-detail">{format_money(row["annual"])}/yr · '
                f'lifetime {format_currency(row["lifetime"])}</div>'
                f'<div class="ss-detail">Plan success: '
                f'<span class="ss-succ {succ_cls}">{succ_txt}</span></div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            if is_plan:
                st.button("Current plan", key=f"ss_apply_{claim}", disabled=True, width="stretch")
            elif st.button(f"Switch to {claim}", key=f"ss_apply_{claim}", width="stretch"):
                apply_choice = claim

    if lo["monthly"] > 0:
        pct = (hi["monthly"] / lo["monthly"] - 1) * 100
        msg = (
            f"Delaying from {lo['claim_age']} to {hi['claim_age']} raises your monthly "
            f"check about {pct:.0f}% — guaranteed and inflation-adjusted for life."
        )
        s_lo, s_hi = ss_success.get(lo["claim_age"]), ss_success.get(hi["claim_age"])
        if s_lo is not None and s_hi is not None and abs(s_hi - s_lo) >= 0.01:
            direction = "lifts" if s_hi >= s_lo else "lowers"
            msg += (
                f" For this plan it {direction} success at {target_age} from "
                f"{s_lo * 100:.0f}% to {s_hi * 100:.0f}% (claiming later means more "
                "years drawing the portfolio before benefits start)."
            )
        st.caption(msg)
    return apply_choice


def render_bridge_card(cfg, target_age, mean_return, volatility, max_eval_age):
    """Conditional 'could part-time work bridge the gap?' lever.

    Only shown when a plan is at risk (or a bridge is already set), never as a
    default input. Part-time income during early retirement covers some spending
    so the portfolio keeps compounding and Social Security can be claimed later.
    Returns overrides to apply (or None).
    """
    bridge_active = float(cfg.get("bridge_income") or 0) > 0
    ss_claim = social_security_start_age(cfg)
    horizon_cap = min(75, int(cfg["max_age"]))
    end_hi = max(target_age + 2, horizon_cap)
    default_end = ss_claim if (ss_claim and target_age < ss_claim <= end_hi) else min(target_age + 5, end_hi)
    default_end = int(min(max(int(cfg.get("bridge_end_age") or default_end), target_age + 1), end_hi))
    # Keep the default on the slider's 5K grid and within range.
    raw_income = int(cfg.get("bridge_income") or 0) or 20_000
    default_income = int(min(60_000, max(0, round(raw_income / 5_000) * 5_000)))

    st.markdown(
        '<div class="bridge-card">'
        '<div class="signal-header">'
        '<span class="signal-dot neutral"></span>'
        '<span class="signal-label">Could part-time work bridge the gap?</span>'
        "</div>"
        '<p class="bridge-intro">Working part-time for a few years into retirement lets '
        'your savings keep growing instead of being drawn down — and lets you delay '
        'Social Security for a bigger check. Try it without committing:</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    income = c1.slider(
        "Part-time income ($/yr)", 0, 60_000, value=default_income, step=5_000,
        key="bridge_income_slider",
        help=GLOSSARY["Bridge income"],
    )
    end_age = c2.slider(
        "Work part-time until age", target_age + 1, end_hi, value=default_end,
        key="bridge_end_slider",
        help="Bridge income runs from your retirement age until this age.",
    )

    base_variant = dict(cfg, bridge_income=0, bridge_end_age=None)
    base_ages, base_probs = cached_model(
        "curve", cfg_cache_key(base_variant),
        curve_args(cfg, mean_return, volatility, max_eval_age),
    )
    base_prob = prob_at_age(base_ages, base_probs, target_age)

    if income > 0:
        variant = dict(cfg, bridge_income=income, bridge_end_age=end_age)
        v_ages, v_probs = cached_model(
            "curve", cfg_cache_key(variant),
            curve_args(cfg, mean_return, volatility, max_eval_age),
        )
        new_prob = prob_at_age(v_ages, v_probs, target_age)
        if new_prob is not None and base_prob is not None:
            pts = round((new_prob - base_prob) * 100)
            delta_txt = (
                f'<span class="up">+{pts} pts</span>' if pts > 0
                else f'<span class="flat">{pts:+d} pts</span>'
            )
            st.markdown(
                f'<p class="bridge-preview">{format_money(income)}/yr until age '
                f"{end_age}: success at {target_age} goes "
                f'<b>{base_prob * 100:.0f}% → {new_prob * 100:.0f}%</b> {delta_txt}.</p>',
                unsafe_allow_html=True,
            )

    overrides = None
    b1, b2 = st.columns(2)
    if b1.button("Apply to plan", key="bridge_apply", width="stretch", disabled=income == 0):
        overrides = {"bridge_income": int(income), "bridge_end_age": int(end_age)}
    if bridge_active and b2.button("Remove bridge income", key="bridge_remove", width="stretch"):
        overrides = {"bridge_income": 0, "bridge_end_age": None}
    return overrides


def coast_state(coast, current_portfolio, target_age):
    """Compare today's portfolio to the coast number → (status, message)."""
    if coast is None:
        return "warn", (
            f"No nest egg coasts to age {target_age} at {SUCCESS_THRESHOLD_PCT}% "
            "without more savings — spending outpaces any starting balance, so "
            "trim spending or retire later."
        )
    gap = coast - current_portfolio
    if gap <= 0:
        return "good", (
            f"▲ You're coasting — {format_currency(-gap)} past the line. You could stop "
            f"saving today and let it grow untouched — no contributions, no withdrawals — "
            f"until you retire at {target_age}."
        )
    return "neutral", (
        f"▼ {format_currency(gap)} to go. Keep contributing until your portfolio reaches "
        f"this number; from there it can grow untouched to retirement at {target_age}."
    )


def render_coast(coast, current_portfolio, target_age):
    """Headline 'have I saved enough to stop?' card — the Coast FIRE number.

    Shows the smallest portfolio that, with no further contributions, still
    clears the success threshold at the target age, and compares it to today's
    balance so the user knows whether they can downshift now or how far they
    have left.
    """
    status, message = coast_state(coast, current_portfolio, target_age)
    if coast is None:
        conf = f"{SUCCESS_THRESHOLD_PCT}% confidence"
        body = ""
    else:
        conf = f"{SUCCESS_THRESHOLD_PCT}% confidence · grows untouched"
        body = (
            f'<div class="safespend-value">{format_currency(coast)}</div>'
            f'<div class="safespend-sub">balance that, left untouched (no contributions, '
            f"no withdrawals), still funds retirement at age {target_age} "
            f'<span class="safespend-sep">·</span> '
            f"you have {format_currency(current_portfolio)} today</div>"
        )
    st.markdown(
        f'<div class="safespend-card {status}">'
        f'<div class="signal-header">'
        f'<span class="signal-dot {status}"></span>'
        f'<span class="signal-label">Coast number</span>'
        f'<span class="safespend-conf">{conf}</span>'
        f"</div>"
        f"{body}"
        f'<div class="safespend-delta {status}">{message}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_coast_progress(coast, current_portfolio):
    """Horizontal progress bar: today's balance against the coast number.

    A one-glance read of how close the portfolio is to the point where saving
    becomes optional. Once today's balance clears the line the bar fills and
    flips green; otherwise it shows the fraction banked so far.
    """
    if not coast or coast <= 0:
        return
    pct = max(0.0, min(current_portfolio / coast, 1.0))
    reached = current_portfolio >= coast
    status = "good" if reached else "neutral"
    pct_label = "100%" if reached else f"{pct * 100:.0f}%"
    headline = "Coast number reached" if reached else f"{pct_label} of the way there"
    st.markdown(
        f'<div class="coast-progress {status}">'
        f'<div class="coast-progress-top">'
        f'<span class="coast-progress-label">{headline}</span>'
        f'<span class="coast-progress-pct">{pct_label}</span>'
        f"</div>"
        f'<div class="coast-progress-track">'
        f'<div class="coast-progress-fill {status}" style="width:{pct * 100:.1f}%"></div>'
        f"</div>"
        f'<div class="coast-progress-scale">'
        f'<span>{format_currency(current_portfolio)} today</span>'
        f'<span>{format_currency(coast)} to coast</span>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_coast_kpis(coast_age, current_age, target_age, stop_now_odds):
    """Three compact stats translating the coast number into a decision.

    Covers three distinct axes: *when* saving becomes optional (coast age),
    *how long* until then (years left), and *how risky it is to quit today*
    (the stop-now odds — success at today's balance with no contributions,
    which the coast number is the threshold-clearing version of).
    """
    if coast_age is None:
        # Portfolio never catches the line by retirement — saving never becomes
        # optional on the current plan.
        age_val, age_cls = f"After {target_age}", "neutral"
        years_val, years_cls = f"{max(target_age - current_age, 0)}+ yrs", "neutral"
    else:
        saving_years = max(coast_age - current_age, 0)
        age_val = "Now" if coast_age <= current_age else str(coast_age)
        age_cls = "good"
        years_val = f"{saving_years} yr" if saving_years == 1 else f"{saving_years} yrs"
        years_cls = "good" if saving_years == 0 else "neutral"
    # Classify on the rounded percentage actually shown so the colour never
    # disagrees with the number (e.g. 0.799 displays "80%" but would otherwise
    # colour as if below the 80% band).
    odds_pct = round(stop_now_odds * 100)
    odds_val = f"{odds_pct}%"
    odds_cls = prob_semantic_class(odds_pct / 100)
    st.markdown(
        f"""
        <div class="kpi-grid coast-kpis">
            <div class="kpi-card">
                <div class="kpi-label">Coast age · {SUCCESS_THRESHOLD_PCT}% confidence</div>
                <div class="kpi-value {age_cls}">{age_val}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Saving years left</div>
                <div class="kpi-value {years_cls}">{years_val}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Odds if you stop now</div>
                <div class="kpi-value {odds_cls}">{odds_val}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_coast_compounding(coast, nest_egg, target_age):
    """Stacked bar splitting the future nest egg into your money vs. growth.

    The coast number is what you hold today; the rest of the retirement nest
    egg is pure compounding it earns untouched. Showing the split makes the
    Coast FIRE idea concrete — most of the finish line is growth you haven't
    earned yet.
    """
    if not coast or coast <= 0 or nest_egg <= coast:
        return
    growth = nest_egg - coast
    growth_pct = growth / nest_egg
    principal_pct = coast / nest_egg
    st.markdown(
        f'<div class="coast-split">'
        f'<div class="coast-split-head">'
        f'<span class="coast-split-pct">{growth_pct * 100:.0f}%</span> of your '
        f'{format_currency(nest_egg)} nest egg at age {target_age} is growth you '
        f"haven't earned yet"
        f"</div>"
        f'<div class="coast-split-track">'
        f'<div class="coast-split-seg principal" style="width:{principal_pct * 100:.1f}%"></div>'
        f'<div class="coast-split-seg growth" style="width:{growth_pct * 100:.1f}%"></div>'
        f"</div>"
        f'<div class="coast-split-legend">'
        f'<span class="coast-split-key"><span class="coast-split-dot principal"></span>'
        f"{format_currency(coast)} you hold today</span>"
        f'<span class="coast-split-key"><span class="coast-split-dot growth"></span>'
        f"{format_currency(growth)} from compounding</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_skeleton():
    st.markdown(
        """
        <div class="skeleton-wrap">
            <div class="skeleton skeleton-hero"></div>
            <div class="skeleton skeleton-insight"></div>
            <div class="skeleton-kpi-row">
                <div class="skeleton skeleton-kpi"></div>
                <div class="skeleton skeleton-kpi"></div>
                <div class="skeleton skeleton-kpi"></div>
            </div>
            <div class="skeleton skeleton-chart"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_what_if_scenarios(base_cfg, base_prob, target_age, mean_return, volatility):
    rows = []
    for variant_id, label, build_overrides in WHAT_IF_VARIANTS:
        overrides = build_overrides(base_cfg, target_age)
        display_label = label
        if variant_id == "retire_later":
            display_label = f"Retire at {overrides['target_retirement_age']}"
        elif variant_id == "retire_earlier":
            display_label = f"Retire at {overrides['target_retirement_age']}"

        variant_cfg = base_cfg.copy()
        variant_cfg.update(overrides)
        variant_key = cfg_cache_key(variant_cfg)
        ages, probs = cached_model(
            "curve",
            variant_key,
            curve_args(variant_cfg, mean_return, volatility, min(85, variant_cfg["max_age"])),
        )
        variant_age = variant_cfg["target_retirement_age"]
        prob = prob_at_age(ages, probs, variant_age)
        delta = (prob - base_prob) if prob is not None and base_prob is not None else None
        rows.append({
            "id": variant_id,
            "Change": display_label,
            "Success @ target": f"{prob * 100:.0f}%" if prob is not None else "—",
            "Δ vs plan": (
                f"{delta_pts(delta):+d} pts"
                if delta_pts(delta) not in (None, 0)
                else "—"
            ),
            "delta": delta,
            "overrides": overrides,
        })
    return rows


def render_what_if_table(rows, best_levers):
    best_labels = {row["Change"] for row in best_levers}
    col_weights = [3.5, 2, 2, 1.5]
    header = st.columns(col_weights, vertical_alignment="center")
    header[0].markdown("**Scenario**")
    header[1].markdown("**Success rate @ target**")
    header[2].markdown("**Change vs plan**")
    header[3].markdown("")

    for row in rows:
        delta_text, delta_dir = format_delta_arrow(row.get("delta"))
        row_class = "what-if-row-label best" if row["Change"] in best_labels else "what-if-row-label"
        cols = st.columns(col_weights, vertical_alignment="center")
        cols[0].markdown(
            f'<span class="{row_class}">{row["Change"]}</span>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(f'<span class="mono">{row["Success @ target"]}</span>', unsafe_allow_html=True)
        cols[2].markdown(
            f'<span class="delta-{delta_dir}">{delta_text}</span>',
            unsafe_allow_html=True,
        )
        if cols[3].button("Apply", key=f"what_if_apply_{row['id']}"):
            return row["overrides"]
    return None


def render_comparison_legend(comparison_runs):
    swatches = []
    for i, (name, _, _) in enumerate(comparison_runs):
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        swatches.append(
            f'<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{color};"></span>'
            f"{name}</span>"
        )
    st.markdown(
        f'<div class="chart-legend">{"".join(swatches)}</div>',
        unsafe_allow_html=True,
    )


def show_figure(fig):
    st.pyplot(fig, clear_figure=True)


def figure_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return buf.getvalue()


def render_chart_with_download(fig, *, file_name, label):
    png = figure_to_png_bytes(fig)
    plt.close(fig)
    b64 = base64.b64encode(png).decode()
    src = f"data:image/png;base64,{b64}"
    fs_id = f"chart-fs-{file_name.replace('.', '-')}"
    st.markdown(
        f"""
        <div class="chart-frame">
            <input type="checkbox" id="{fs_id}" class="chart-fs-toggle" aria-hidden="true"/>
            <img src="{src}" alt="{label}"/>
            <div class="chart-actions">
                <label for="{fs_id}" class="chart-action chart-fullscreen"
                       title="View full screen" aria-label="View full screen">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="15 3 21 3 21 9"/>
                        <polyline points="9 21 3 21 3 15"/>
                        <line x1="21" y1="3" x2="14" y2="10"/>
                        <line x1="3" y1="21" x2="10" y2="14"/>
                    </svg>
                </label>
                <a class="chart-action chart-download" href="{src}"
                   download="{file_name}" title="{label}" aria-label="{label}">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                </a>
            </div>
            <div class="chart-fs-overlay" role="dialog" aria-modal="true">
                <label for="{fs_id}" class="chart-fs-backdrop" aria-label="Close full screen"></label>
                <img src="{src}" alt="{label}"/>
                <label for="{fs_id}" class="chart-fs-close" aria-label="Close full screen">×</label>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_legacy_saved():
    try:
        with LEGACY_SAVED_FILE.open("r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _nearest_ss_claim_age(age):
    return min(SS_CLAIM_AGES, key=lambda x: abs(x - age))


def _migrate_spending_fields(cfg, saved):
    if saved and "annual_spending" not in saved:
        under = saved.get("spending_under_75", DEFAULT_CFG["annual_spending"])
        cfg["annual_spending"] = under
        over = saved.get("spending_75_89", under)
        if under > 0 and over < under:
            cfg["spending_reduction_after_75"] = round(1 - over / under, 2)
    if saved and saved.get("same_spending_all_ages", True):
        cfg["spending_reduction_after_75"] = 0.0


def _empty_store():
    return {"last": None, "saved": []}


def _load_store_file():
    """Read the scenario store from the on-disk JSON file (local dev / fallback)."""
    try:
        with SCENARIO_FILE.open("r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"last": None, "saved": _load_legacy_saved()}
    except json.JSONDecodeError:
        st.sidebar.warning("Could not parse scenarios.json — using defaults.")
        return _empty_store()

    if isinstance(data, dict) and "last" in data:
        return data

    saved = _load_legacy_saved()
    return {"last": data, "saved": saved}


def _save_store_file(store):
    with SCENARIO_FILE.open("w") as f:
        json.dump(store, f, indent=2)


# --- Per-user persistence -------------------------------------------------
# When deployed (e.g. Streamlit Community Cloud) the server filesystem is shared
# across every visitor and is wiped on each restart, so storing scenarios in a
# server-side file would mean all users see and overwrite each other's data and
# lose it on reboot. Instead each visitor's scenarios live in their own browser
# via localStorage, keeping them private and durable. Running locally, the
# on-disk JSON file is used (and imported into localStorage on first load) so an
# existing single user keeps their saved scenarios.
_LS_COMPONENT_KEY = "rr_local_storage"      # Streamlit widget key for the component
_LS_ITEM_KEY = "retirement_runway_store"    # localStorage key holding the JSON store
# Streamlit Community Cloud checks the repo out under /mount/src; treat anything
# else as a local run where the on-disk file is the user's own data. Set RR_LOCAL
# (1/0) to override the auto-detection if needed.
_RR_LOCAL_ENV = os.environ.get("RR_LOCAL")
if _RR_LOCAL_ENV is not None:
    IS_LOCAL = _RR_LOCAL_ENV.strip().lower() not in ("", "0", "false", "no")
else:
    IS_LOCAL = not str(Path(__file__).resolve()).startswith("/mount/")


def _ls_handle():
    """Return (LocalStorage handle, ready).

    ``ready`` is True once the browser's localStorage has actually been read: the
    first script run of a session returns empty before the browser responds and
    then triggers an automatic rerun. Returns (None, True) when the component is
    unavailable so callers fall back to the on-disk file.
    """
    if LocalStorage is None:
        return None, True
    ready = _LS_COMPONENT_KEY in st.session_state
    return LocalStorage(key=_LS_COMPONENT_KEY), ready


def load_store():
    ls, ready = _ls_handle()
    if ls is None:
        return _load_store_file()

    if st.session_state.get("_store_synced"):
        return st.session_state["_store"]

    if not ready:
        # Browser not read yet: serve transient defaults for this one render
        # without caching them, so real values load on the automatic rerun.
        return _load_store_file() if IS_LOCAL else _empty_store()

    raw = ls.getItem(_LS_ITEM_KEY)
    if raw:
        try:
            store = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            store = _empty_store()
    elif IS_LOCAL:
        # First visit in this browser: import the existing local file once.
        store = _load_store_file()
    else:
        store = _empty_store()

    st.session_state["_store"] = store
    st.session_state["_store_synced"] = True
    return store


def save_store(last=None, saved=None):
    store = load_store()
    if last is not None:
        store["last"] = last
    if saved is not None:
        store["saved"] = saved

    ls, ready = _ls_handle()
    if ls is None:
        _save_store_file(store)
        return
    if not ready:
        # Don't persist before the browser has been read, or we'd clobber the
        # user's stored scenarios with transient defaults.
        return

    st.session_state["_store"] = store
    st.session_state["_store_synced"] = True
    payload = json.dumps(store)
    if st.session_state.get("_store_written") == payload:
        return
    # Each setItem needs a unique widget key when more than one write happens in
    # the same run (e.g. saving a scenario and the end-of-run config autosave).
    n = st.session_state.get("_ls_write_count", 0) + 1
    st.session_state["_ls_write_count"] = n
    ls.setItem(_LS_ITEM_KEY, payload, key=f"rr_ls_set_{n}")
    st.session_state["_store_written"] = payload
    if IS_LOCAL:
        _save_store_file(store)


def save_cfg(cfg):
    save_store(last=cfg)


def load_cfg():
    store = load_store()
    return merge_cfg(store.get("last"))


def load_saved_scenarios():
    return load_store().get("saved", [])


def save_saved_scenarios(scenarios):
    save_store(saved=scenarios)


def merge_cfg(saved):
    cfg = DEFAULT_CFG.copy()
    if saved:
        cfg.update(saved)
    if saved and "target_retirement_age" not in saved:
        cfg["target_retirement_age"] = saved.get(
            "optimizer_age",
            saved.get("fan_chart_age", DEFAULT_CFG["target_retirement_age"]),
        )
    _migrate_spending_fields(cfg, saved)
    if saved and "social_security_claim_age" not in saved:
        legacy_age = saved.get("social_security_start_age", 67)
        cfg["social_security_claim_age"] = _nearest_ss_claim_age(legacy_age)
    elif saved:
        cfg["social_security_claim_age"] = _nearest_ss_claim_age(cfg["social_security_claim_age"])
    if saved and "include_social_security" not in saved:
        cfg["include_social_security"] = True
    if cfg.get("active_profile"):
        cfg["active_profile"] = normalize_profile_name(cfg["active_profile"])
    return cfg


def validate_inputs(cfg):
    warnings = []
    annual_spend = cfg.get("annual_spending", 0)
    target_age = cfg.get("target_retirement_age")
    mean_return = cfg.get("mean_return")
    if annual_spend > 0 and target_age is not None and mean_return is not None:
        withdrawal_rate = compute_withdrawal_rate(cfg, target_age, mean_return)
        if withdrawal_rate is not None and withdrawal_rate > 0.05:
            warnings.append(
                f"Withdrawal rate at age {target_age} would be "
                f"{withdrawal_rate * 100:.1f}% — above the common 4–5% guideline; "
                "success rates may be low."
            )
    if cfg["target_retirement_age"] <= cfg["current_age"]:
        warnings.append("Target retirement age should be after your current age.")
    if cfg["annual_contribution"] == 0 and cfg["starting_amount"] < annual_spend * 5:
        warnings.append(
            "Low savings with no contributions — consider adding expected savings."
        )
    return warnings


def load_comparison_runs(saved_scenarios, compare_names, current_run=None):
    runs = []
    if current_run is not None:
        runs.append(current_run)
    for entry in saved_scenarios:
        if entry["name"] not in compare_names:
            continue
        scfg = merge_cfg(entry.get("cfg", {}))
        s_key = cfg_cache_key(scfg)
        mean_return = entry.get("mean_return", scfg["mean_return"])
        volatility = entry.get("volatility", scfg["volatility"])
        s_ages, s_probs = cached_model(
            "curve",
            s_key,
            curve_args(scfg, mean_return, volatility, min(85, scfg["max_age"])),
        )
        runs.append((entry["name"], s_ages, s_probs))
    return runs


def load_preset_comparison_runs(preset_names):
    runs = []
    for name in preset_names:
        pcfg = DEFAULT_CFG.copy()
        apply_profile(pcfg, name)
        p_key = cfg_cache_key(pcfg)
        s_ages, s_probs = cached_model(
            "curve",
            p_key,
            curve_args(pcfg, pcfg["mean_return"], pcfg["volatility"], min(85, pcfg["max_age"])),
        )
        runs.append((f"Preset: {name}", s_ages, s_probs))
    return runs


def show_section(title, description):
    st.subheader(title)
    st.caption(description)


def cfg_cache_key(cfg):
    # Key on only the model-relevant fields so UI-only toggles (e.g.
    # show_real_values, advanced_mode, active_profile) don't bust the cache.
    model_cfg = {k: cfg[k] for k in MODEL_CFG_FIELDS if k in cfg}
    return json.dumps(model_cfg, sort_keys=True)


def curve_args(cfg, mean_return, volatility, age_end, seed=None):
    """Serialized args for a ``cached_model("curve", ...)` call.

    The success curve always starts at the plan's current age; callers pass the
    upper age bound (the eval horizon, or a capped display max).
    """
    return json.dumps({
        "age_start": cfg["current_age"],
        "age_end": age_end,
        "mean_return": mean_return,
        "volatility": volatility,
        "seed": seed,
    })


@st.cache_data(show_spinner=False)
def cached_report_bytes(
    cfg_key, ages_key, probs_key, mean_return, volatility, optimizer_key, executive_summary,
):
    return generate_report_bytes(
        json.loads(cfg_key),
        json.loads(ages_key),
        json.loads(probs_key),
        mean_return,
        volatility,
        json.loads(optimizer_key),
        executive_summary=executive_summary,
    )


@st.cache_data(show_spinner=False)
def cached_model(fn_name, cfg_key, args_key):
    cfg = json.loads(cfg_key)
    args = json.loads(args_key)

    if fn_name == "curve":
        ages, probs = compute_curve(
            cfg,
            range(args["age_start"], args["age_end"] + 1),
            args["mean_return"],
            args["volatility"],
            args.get("seed"),
        )
        return ages.tolist(), probs.tolist()

    if fn_name == "min_years":
        return find_min_years_worked(
            cfg,
            args["retirement_age"],
            args["mean_return"],
            args["volatility"],
            seed=args.get("seed"),
        )

    if fn_name == "max_spend":
        return find_max_sustainable_spending(
            cfg,
            args["retirement_age"],
            args["mean_return"],
            args["volatility"],
            target=args.get("target", SUCCESS_THRESHOLD),
            seed=args.get("seed"),
        )

    if fn_name == "coast":
        return find_coast_number(
            cfg,
            args["retirement_age"],
            args["mean_return"],
            args["volatility"],
            target=args.get("target", SUCCESS_THRESHOLD),
            seed=args.get("seed"),
        )

    if fn_name == "coast_success":
        return coast_success(
            cfg,
            args["retirement_age"],
            args["mean_return"],
            args["volatility"],
            seed=args.get("seed"),
        )

    if fn_name == "fan":
        calendar_ages, nominal, real = compute_mc_net_worth_fan(
            cfg,
            args["retirement_age"],
            args["mean_return"],
            args["volatility"],
            args.get("seed"),
        )
        return (
            calendar_ages.tolist(),
            {k: v.tolist() for k, v in nominal.items()},
            {k: v.tolist() for k, v in real.items()},
        )

    raise ValueError(f"Unknown cached model operation: {fn_name}")


def dollar_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))


def _glow_line(ax, x, y, color, linewidth=2.5, label=None, zorder=3):
    line, = ax.plot(x, y, color=color, linewidth=linewidth, label=label, zorder=zorder)
    line.set_path_effects([
        pe.withStroke(linewidth=linewidth + 4, foreground=color),
        pe.Normal(),
    ])
    return line


def plot_success_curve(
    ages,
    probs,
    start_age,
    end_age,
    current_age,
    highlight_age=None,
    earliest_safe_age=None,
    overlay=None,
    primary_label="Success probability",
):
    apply_chart_style()
    ages = np.asarray(ages)
    probs = np.asarray(probs)
    fig, ax = plt.subplots(figsize=(10, 5))
    chart_mask = (ages >= start_age) & (ages <= end_age)
    chart_ages = ages[chart_mask]
    chart_probs = probs[chart_mask]

    ax.fill_between(
        chart_ages, 0, chart_probs,
        color=THEME["primary"], alpha=0.08, interpolate=True,
    )
    ax.fill_between(
        chart_ages, chart_probs, SUCCESS_THRESHOLD, where=chart_probs >= SUCCESS_THRESHOLD,
        color=THEME["success"], alpha=0.14, interpolate=True,
    )
    if overlay is not None:
        ov_ages, ov_probs, ov_label = overlay
        ov_ages = np.asarray(ov_ages)
        ov_probs = np.asarray(ov_probs)
        ov_mask = (ov_ages >= start_age) & (ov_ages <= end_age)
        ax.plot(
            ov_ages[ov_mask], ov_probs[ov_mask], linestyle="--",
            color=THEME["muted"], linewidth=1.6, alpha=0.9, label=ov_label,
        )
    _glow_line(ax, chart_ages, chart_probs, THEME["primary"], label=primary_label)
    ax.axhline(SUCCESS_THRESHOLD, linestyle="--", color=THEME["threshold"], linewidth=1.0, label=f"{SUCCESS_THRESHOLD_PCT}% threshold")
    ax.axvline(current_age, linestyle=":", color=THEME["secondary"], alpha=0.7, linewidth=1.2, label="Current age")
    if highlight_age is not None:
        prob_at_target = prob_at_age(ages, probs, highlight_age)
        ax.axvline(
            highlight_age, linestyle=":", color=THEME["accent"], alpha=0.85,
            linewidth=1.2, label="Target age",
        )
        if prob_at_target is not None:
            ax.plot(
                highlight_age, prob_at_target, "o",
                color=THEME["accent"], markersize=7, zorder=6,
            )
            ax.annotate(
                f"{prob_at_target * 100:.0f}% @ {highlight_age}",
                xy=(highlight_age, prob_at_target),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=9,
                color=THEME["accent"],
                fontweight="600",
            )
    if earliest_safe_age is not None and earliest_safe_age != highlight_age:
        prob_at_earliest = prob_at_age(ages, probs, earliest_safe_age)
        ax.axvline(
            earliest_safe_age, linestyle=":", color=THEME["success"], alpha=0.75,
            linewidth=1.2, label="Earliest safe age",
        )
        if prob_at_earliest is not None:
            ax.plot(
                earliest_safe_age, prob_at_earliest, "o",
                color=THEME["success"], markersize=6, zorder=6,
            )
            ax.annotate(
                f"Safe @ {earliest_safe_age}",
                xy=(earliest_safe_age, prob_at_earliest),
                xytext=(8, -14),
                textcoords="offset points",
                fontsize=8,
                color=THEME["success"],
            )
    ax.set_xlabel("Retirement Age")
    ax.set_ylabel("Success Probability")
    ax.set_title("Will your money last if you retire at this age?")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def plot_fan_chart(calendar_ages, pct_values, title, *, real_values=False, det_overlay=None):
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    calendar_ages = np.asarray(calendar_ages)
    p10 = np.asarray(pct_values[10])
    p50 = np.asarray(pct_values[50])
    p90 = np.asarray(pct_values[90])

    ax.fill_between(calendar_ages, p10, p90, alpha=0.14, color=FAN_COLORS[50])
    ax.plot(
        calendar_ages, p10, linestyle="--", color=FAN_COLORS[10],
        alpha=0.85, linewidth=1.2, label="10th percentile",
    )
    _glow_line(ax, calendar_ages, p50, FAN_COLORS[50], label="Median (50th)")
    ax.plot(
        calendar_ages, p90, linestyle="--", color=FAN_COLORS[90],
        alpha=0.85, linewidth=1.2, label="90th percentile",
    )
    ax.axhline(0, color=THEME["danger"], linestyle="-", linewidth=0.8, alpha=0.45, label="Depleted")

    depleted = np.where(p50 <= 0)[0]
    if len(depleted):
        depletion_age = int(calendar_ages[depleted[0]])
        ax.annotate(
            f"Median depleted @ {depletion_age}",
            xy=(depletion_age, 0),
            xytext=(10, 18),
            textcoords="offset points",
            fontsize=8,
            color=THEME["danger"],
            arrowprops={"arrowstyle": "->", "color": THEME["danger"], "lw": 0.8},
        )

    if det_overlay is not None:
        det_ages, det_values, det_label = det_overlay
        ax.plot(
            det_ages,
            det_values,
            color=THEME["success"],
            linewidth=2.2,
            linestyle="-",
            alpha=0.95,
            label=det_label,
            zorder=5,
        )
    ax.set_xlabel("Age")
    ylabel = "Real Portfolio Value ($)" if real_values else "Portfolio Value ($)"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    dollar_axis(ax)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def plot_scenario_comparison(comparison_runs):
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, ages, probs) in enumerate(comparison_runs):
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        ax.plot(np.asarray(ages), np.asarray(probs), linewidth=2.2, label=name, color=color)
    ax.axhline(SUCCESS_THRESHOLD, linestyle="--", color=THEME["threshold"], linewidth=1.2)
    ax.set_xlabel("Retirement Age")
    ax.set_ylabel("Success Probability")
    ax.set_title("Scenario Comparison · Monte Carlo")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def plot_coast_growth(ages, coast_line, portfolio_line, coast_age, current_age, target_age):
    """Coast FIRE crossover chart.

    The coast line is the balance needed at each age to stop saving and still
    retire on time; your savings (with contributions) climb to meet it. Where
    they cross is the age saving becomes optional — shaded green beyond it.
    """
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ages = np.asarray(ages)
    coast_line = np.asarray(coast_line)
    portfolio_line = np.asarray(portfolio_line)

    # Shade the cushion: where your savings sit above the coast line you've
    # cleared it and could downshift.
    ax.fill_between(
        ages, coast_line, portfolio_line, where=portfolio_line >= coast_line,
        color=THEME["success"], alpha=0.13, interpolate=True,
    )
    ax.plot(
        ages, coast_line, linestyle="--", color=THEME["threshold"],
        linewidth=1.6, alpha=0.95, label="Coast number (left untouched)",
    )
    _glow_line(ax, ages, portfolio_line, THEME["primary"], label="Your savings (with contributions)")

    ax.axvline(
        target_age, linestyle=":", color=THEME["accent"], alpha=0.85,
        linewidth=1.2, label=f"Retire @ {target_age}",
    )
    if coast_age is not None:
        idx = int(np.where(ages == coast_age)[0][0])
        ax.plot(coast_age, portfolio_line[idx], "o", color=THEME["success"], markersize=8, zorder=6)
        label = "Already coasting" if coast_age <= current_age else f"Coast age {coast_age}"
        ax.annotate(
            label,
            xy=(coast_age, portfolio_line[idx]),
            xytext=(10, -16 if coast_age <= current_age else 12),
            textcoords="offset points",
            fontsize=9,
            color=THEME["success"],
            fontweight="600",
        )

    ax.set_xlabel("Age")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("When does saving for retirement become optional?")
    ax.set_ylim(bottom=0)
    dollar_axis(ax)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def planned_career_years(cfg, target_age):
    years_until = target_age - cfg["current_age"]
    total = cfg["years_already_worked"] + years_until
    return years_until, total


def render_career_years_section(
    *,
    cfg,
    target_age,
    baseline_career_years,
    min_years,
    on_track,
):
    """Only surfaced when SS work history is a real constraint — not sidebar arithmetic."""
    if min_years is None:
        st.error(f"No career length up to 50 years reaches {SUCCESS_THRESHOLD_PCT}% success at age {target_age}.")
        return

    worked = cfg["years_already_worked"]
    if on_track and min_years <= worked:
        return

    if on_track:
        st.info(
            f"Social Security needs at least **{min_years}** total work years by age {target_age} "
            f"for {SUCCESS_THRESHOLD_PCT}% success — your plan reaches **{baseline_career_years}**."
        )
    else:
        shortfall = min_years - baseline_career_years
        st.warning(
            f"Social Security needs at least **{min_years}** total work years by age {target_age} "
            f"for {SUCCESS_THRESHOLD_PCT}% success — your plan only reaches **{baseline_career_years}**. "
            f"Work **{shortfall}** more year(s), retire later, or adjust savings/spending."
        )


def build_optimizer_lines(target_age, min_years, baseline_career_years, years_already_worked, years_until):
    if min_years is None:
        return [f"No career length up to 50 years reaches {SUCCESS_THRESHOLD_PCT}% at age {target_age}."]
    status = "On track" if baseline_career_years >= min_years else (
        f"Short by {min_years - baseline_career_years} year(s)"
    )
    return [
        f"Retire at age {target_age}",
        f"Minimum total career years at retirement: {min_years}",
        (
            f"Your plan: {baseline_career_years} total "
            f"({years_already_worked} worked + {years_until} more years working)"
        ),
        status,
    ]


# -------------------------
# UI
# -------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

cfg = load_cfg()
saved_scenarios = load_saved_scenarios()

with st.sidebar:
    st.markdown("### Parameters")
    st.caption("Select a profile preset, then fine-tune inputs.")

    profile_names = list(PRESET_PROFILES.keys())
    active = cfg.get("active_profile")
    profile_choice = st.selectbox(
        "Profile preset",
        options=profile_names,
        index=profile_names.index(active) if active in PRESET_PROFILES else None,
        placeholder="Choose a profile…",
    )
    if profile_choice:
        st.caption(PRESET_PROFILES[profile_choice]["tagline"])
    if profile_choice and profile_choice != active:
        apply_profile(cfg, profile_choice)
        save_cfg(cfg)
        st.rerun()

    if active in PRESET_PROFILES:
        if st.button(f"↺ Reset to {active} preset", width="stretch"):
            apply_profile(cfg, active)
            save_cfg(cfg)
            st.rerun()

    st.divider()
    advanced_mode = st.toggle(
        "Advanced parameters",
        value=bool(cfg.get("advanced_mode", False)),
        help="Inflation, spending after 75, market details, and planning horizon.",
    )
    cfg["advanced_mode"] = advanced_mode

    smart_spending = st.toggle(
        "Smart spending",
        value=(cfg.get("withdrawal_strategy", "fixed") == "guardrails"),
        help=(
            "Flex spending with the market — trim withdrawals after a downturn, "
            "spend more after a strong run (Guyton-Klinger guardrails). Improves "
            "the odds your money lasts versus a fixed inflation-only budget."
        ),
    )
    cfg["withdrawal_strategy"] = "guardrails" if smart_spending else "fixed"

    with st.expander("Essentials", expanded=True):
        st.caption("Changes apply instantly.")
        cfg["current_age"] = st.number_input(
            "Current age", 18, 80, int(cfg["current_age"]),
        )
        cfg["starting_amount"] = st.number_input(
            "Portfolio today ($)", 0, 10_000_000, int(cfg["starting_amount"]),
        )
        cfg["annual_contribution"] = st.number_input(
            "Annual savings ($)", 0, 500_000, int(cfg["annual_contribution"]),
        )
        cfg["target_retirement_age"] = st.number_input(
            "Target retirement age",
            cfg["current_age"] + 1, 85,
            min(max(int(cfg.get("target_retirement_age", 65)), cfg["current_age"] + 1), 85),
            help="Used for success rate, risk paths, and career planning.",
        )
        cfg["annual_spending"] = st.number_input(
            "Annual spending ($)",
            0, 500_000,
            int(cfg.get("annual_spending", DEFAULT_CFG["annual_spending"])),
            help="Pre-inflation spending from retirement onward.",
        )
        if advanced_mode:
            cfg["inflation_rate"] = st.slider(
                "Inflation", 0.0, 0.1, float(cfg["inflation_rate"]),
            )
            reduction_pct = int(round(float(cfg.get("spending_reduction_after_75", 0.0)) * 100))
            cfg["spending_reduction_after_75"] = st.slider(
                "Lower spending after 75 (%)",
                0, 50,
                reduction_pct,
                step=5,
                help="Reduce annual spending by this percentage from age 75 onward.",
            ) / 100.0

    target_retirement_age = int(cfg.get("target_retirement_age", 65))

    if advanced_mode:
        with st.expander("Market", expanded=False):
            st.caption(
                "Defaults are modeled on a broad stock-market index like the "
                "S&P 500; annual returns are drawn from a log-normal distribution."
            )
            mean_return = st.slider("Expected return", 0.0, 0.12, float(cfg["mean_return"]))
            volatility = st.slider("Volatility", 0.0, 0.25, float(cfg["volatility"]))
            cfg["trials"] = st.slider("Simulation trials", 500, 10_000, int(cfg["trials"]))
    else:
        preset_names = list(RISK_PRESETS.keys())
        current_preset = match_risk_preset(float(cfg["mean_return"]), float(cfg["volatility"]))
        risk_choice = st.selectbox(
            "Volatility preset",
            options=preset_names,
            index=preset_names.index(current_preset),
            format_func=lambda k: RISK_PRESET_LABELS[k],
        )
        preset = RISK_PRESETS[risk_choice]
        mean_return = preset["mean_return"]
        volatility = preset["volatility"]
        cfg["trials"] = 2000
        st.caption(
            "Return and volatility presets are modeled on a broad stock-market "
            "index like the S&P 500."
        )

    with st.expander("Social Security", expanded=False):
        cfg["include_social_security"] = st.toggle(
            "Include Social Security",
            value=bool(cfg.get("include_social_security", True)),
            help="Offsets spending from your claim age onward.",
        )
        if cfg["include_social_security"]:
            claim_index = SS_CLAIM_AGES.index(
                _nearest_ss_claim_age(int(cfg.get("social_security_claim_age", 67)))
            )
            cfg["social_security_claim_age"] = st.selectbox(
                "Claim age",
                options=SS_CLAIM_AGES,
                index=claim_index,
                format_func=lambda a: f"Age {a}",
            )
        cfg["years_already_worked"] = st.number_input(
            "Years worked so far",
            0, 50,
            int(cfg.get("years_already_worked", 6)),
            help=(
                "Career years you've already completed. Added to years until retirement "
                "to get your total career years at retirement."
            ),
        )

    if advanced_mode:
        with st.expander("More options", expanded=False):
            cfg["max_age"] = st.number_input(
                "Planning horizon (max age)", 70, 110, int(cfg["max_age"]),
                help=GLOSSARY["Planning horizon"],
            )

    st.divider()
    st.markdown("**Save scenario**")
    scenario_name = st.text_input(
        "Name this scenario",
        placeholder="e.g. Retire at 60",
    )
    if st.button("Save scenario", width="stretch"):
        if not scenario_name.strip():
            st.error("Enter a name first.")
        else:
            snapshot = cfg.copy()
            snapshot.update({
                "mean_return": mean_return,
                "volatility": volatility,
                "target_retirement_age": target_retirement_age,
            })
            entry = {
                "name": scenario_name.strip(),
                "cfg": snapshot,
                "mean_return": mean_return,
                "volatility": volatility,
            }
            updated = [s for s in saved_scenarios if s["name"] != entry["name"]]
            updated.append(entry)
            save_saved_scenarios(updated)
            saved_scenarios = updated
            st.success(f"Saved “{entry['name']}”.")

    if saved_scenarios:
        load_choice = st.selectbox(
            "Load saved scenario",
            options=[s["name"] for s in saved_scenarios],
            key="load_scenario_select",
        )
        if st.button("Load scenario", width="stretch"):
            entry = next(s for s in saved_scenarios if s["name"] == load_choice)
            apply_saved_scenario(cfg, entry)
            save_cfg(cfg)
            st.rerun()

cfg["mean_return"] = mean_return
cfg["volatility"] = volatility
cfg["target_retirement_age"] = target_retirement_age

input_warnings = validate_inputs(cfg)
cfg_key = cfg_cache_key(cfg)
max_eval_age = min(85, cfg["max_age"])
target_age = target_retirement_age
start_age, default_end_age = chart_age_range(cfg["current_age"], target_age)

model_args = {
    "mean_return": mean_return,
    "volatility": volatility,
    "seed": None,
}

main_body = st.empty()
with main_body.container():
    render_skeleton()

trial_count = int(cfg.get("trials", 2000))
with st.spinner(f"Running {trial_count:,} Monte Carlo trials…"):
    ages, probs = cached_model(
        "curve",
        cfg_key,
        curve_args(cfg, mean_return, volatility, max_eval_age),
    )
    min_years = cached_model(
        "min_years",
        cfg_key,
        json.dumps({"retirement_age": target_age, **model_args}),
    )
    fan_ages, fan_nominal, fan_real = cached_model(
        "fan",
        cfg_key,
        json.dumps({"retirement_age": target_age, **model_args}),
    )
    safe_spend = cached_model(
        "max_spend",
        cfg_key,
        json.dumps({"retirement_age": target_age, **model_args}),
    )
    coast_number = cached_model(
        "coast",
        cfg_key,
        json.dumps({"retirement_age": target_age, **model_args}),
    )
    coast_stop_now_odds = cached_model(
        "coast_success",
        cfg_key,
        json.dumps({"retirement_age": target_age, **model_args}),
    )

smart_spending_on = cfg.get("withdrawal_strategy") == "guardrails"
fixed_ages = fixed_probs = None
if smart_spending_on:
    # Same plan with a fixed inflation-only budget, for the success-curve overlay
    # so the lift smart spending buys is visible side by side.
    fixed_cfg = dict(cfg, withdrawal_strategy="fixed")
    fixed_ages, fixed_probs = cached_model(
        "curve",
        cfg_cache_key(fixed_cfg),
        curve_args(cfg, mean_return, volatility, max_eval_age),
    )

ranges = safe_retirement_ranges(ages, probs)
target_prob = prob_at_age(ages, probs, target_age)
years_until_retirement, baseline_career_years = planned_career_years(cfg, target_age)
on_track = min_years is not None and baseline_career_years >= min_years

withdrawal_rate = compute_withdrawal_rate(cfg, target_age, mean_return)
safe_spend_portfolio = portfolio_at_retirement(cfg, target_age, mean_return)
safe_spend_wr = (
    safe_spend / safe_spend_portfolio
    if safe_spend and safe_spend_portfolio and safe_spend_portfolio > 0
    else None
)
_, det_years, det_nominal, det_real = simulate_net_worth(cfg, target_age, mean_return)
det_label = f"Fixed {mean_return * 100:.1f}% return"
what_if_rows = run_what_if_scenarios(
    cfg, target_prob, target_age, mean_return, volatility,
)
best_levers = pick_best_levers(what_if_rows)
executive_summary = build_executive_summary(
    target_prob, target_age, on_track, withdrawal_rate, ranges, years_until_retirement,
    levers=best_levers,
)

optimizer_lines = build_optimizer_lines(
    target_age, min_years, baseline_career_years,
    cfg["years_already_worked"], years_until_retirement,
)

# Persona + Social Security framing. Near-retirees lead with their guaranteed
# income floor and safe budget rather than accumulation metrics; the SS claim
# comparison powers the timing section and that floor KPI.
near_retiree = is_near_retirement(cfg, target_age)
ss_on = cfg.get("include_social_security", True)
ss_claim_age = social_security_start_age(cfg)
ss_comparison = social_security_claim_comparison(cfg, target_age) if ss_on else []
ss_floor = next(
    (r["annual"] for r in ss_comparison if r["claim_age"] == ss_claim_age), 0.0
)
ss_success = {}
if ss_on:
    for row in ss_comparison:
        variant_cfg = dict(cfg, social_security_claim_age=row["claim_age"])
        v_ages, v_probs = cached_model(
            "curve",
            cfg_cache_key(variant_cfg),
            curve_args(cfg, mean_return, volatility, max_eval_age),
        )
        ss_success[row["claim_age"]] = prob_at_age(v_ages, v_probs, target_age)

bridge_active = float(cfg.get("bridge_income") or 0) > 0
plan_at_risk = target_prob is None or target_prob < SUCCESS_THRESHOLD

with main_body.container():
    render_hero(
        current_age=cfg["current_age"],
        target_age=target_age,
        portfolio=cfg["starting_amount"],
        active_profile=cfg.get("active_profile"),
        target_prob=target_prob,
        on_track=on_track,
    )

    for warning in input_warnings:
        st.warning(warning)

    if near_retiree:
        verdict_headline = build_verdict_near_retiree(
            target_age, ss_floor, ss_claim_age, safe_spend,
        )
    else:
        verdict_headline = build_verdict(target_prob, target_age, cfg["max_age"], ranges)
    render_verdict(
        verdict_headline,
        build_verdict_subline(target_prob, best_levers),
        status=insight_status(target_prob, on_track),
    )

    if bridge_active:
        st.caption(
            f"This plan includes {format_money(cfg['bridge_income'])}/yr of part-time "
            f"income until age {cfg['bridge_end_age']} — adjust it on the Plan tab."
        )

    top_lever = best_levers[0] if best_levers else None
    if (
        top_lever
        and delta_pts(top_lever.get("delta")) not in (None, 0)
        and top_lever.get("overrides")
    ):
        lever_pts = delta_pts(top_lever["delta"])
        at_target = target_prob is not None and target_prob >= SUCCESS_THRESHOLD
        lever_verb = "Add margin" if at_target else "Apply best lever"
        if st.button(
            f"{lever_verb}: {top_lever['Change']} ({lever_pts:+d} pts)",
            key="apply_best_lever",
            width="stretch",
        ):
            cfg.update(top_lever["overrides"])
            save_cfg(cfg)
            st.rerun()

    render_kpi_row(
        target_age=target_age,
        target_prob=target_prob,
        ranges=ranges,
        withdrawal_rate=withdrawal_rate,
        near_retiree=near_retiree,
        ss_floor=ss_floor if ss_on else None,
        ss_claim_age=ss_claim_age,
        safe_spend=safe_spend,
    )

    tab_plan, tab_whatif, tab_coast, tab_compare, tab_export = st.tabs(
        ["Plan", "What-if", "Coast", "Compare", "Export"]
    )

    with tab_plan:
        spend_band = (
            guardrail_income_band(cfg, base=safe_spend)
            if smart_spending_on and safe_spend
            else None
        )
        render_safe_spend(
            safe_spend,
            cfg.get("annual_spending", 0),
            target_age,
            safe_spend_wr,
            band=spend_band,
        )

        # Social Security timing — the single biggest lever for the under-saved.
        # Prominent (open) for near-retirees; tucked in an expander otherwise.
        if ss_on and ss_comparison:
            def _ss_timing_body():
                choice = render_ss_timing(ss_comparison, ss_success, ss_claim_age, target_age)
                if choice is not None:
                    cfg["social_security_claim_age"] = choice
                    save_cfg(cfg)
                    st.rerun()

            if near_retiree:
                show_section(
                    "Social Security timing — when to claim",
                    "Your claim age sets a guaranteed, inflation-adjusted income for life. "
                    "Claiming later means a bigger monthly check — often the highest-impact "
                    "choice when savings are thin.",
                )
                _ss_timing_body()
            else:
                with st.expander("Social Security timing — when to claim", expanded=False):
                    _ss_timing_body()

        # Part-time bridge income — surfaced only when the plan is at risk (or a
        # bridge is already set), never as a default sidebar input.
        if plan_at_risk or bridge_active:
            bridge_overrides = render_bridge_card(cfg, target_age, mean_return, volatility, max_eval_age)
            if bridge_overrides is not None:
                cfg.update(bridge_overrides)
                save_cfg(cfg)
                st.rerun()

        show_section(
            "Success probability curve",
            "Each point is the chance your portfolio stays funded through your planning "
            f"horizon if you retire at that age. The shaded band marks ≥{SUCCESS_THRESHOLD_PCT}% success.",
        )
        chart_end_age = st.slider(
            "Chart ends at age",
            min_value=start_age,
            max_value=max_eval_age,
            value=default_end_age,
            help="Zoom the success curve — left edge is always your current age.",
        )
        earliest_safe = ranges[0][0] if ranges else None
        curve_overlay = (
            (fixed_ages, fixed_probs, "Fixed spending")
            if smart_spending_on and fixed_probs is not None
            else None
        )
        success_fig = plot_success_curve(
            ages, probs, start_age, chart_end_age, cfg["current_age"], target_age,
            earliest_safe_age=earliest_safe,
            overlay=curve_overlay,
            primary_label="Smart spending" if smart_spending_on else "Success probability",
        )
        render_chart_with_download(
            success_fig,
            file_name="success_curve.png",
            label="Download chart PNG",
        )

        if curve_overlay is not None:
            st.caption(
                "The dashed line is the same plan with fixed inflation-only spending. "
                "Smart spending lifts the odds by trimming withdrawals after market "
                "downturns — at the cost of a leaner budget in those years."
            )

        if ranges:
            ss_age = social_security_start_age(cfg)
            earliest = ranges[0][0]
            if ss_age is not None and earliest > cfg["current_age"] and earliest == ss_age:
                st.caption(
                    f"Success jumps at age {earliest} because Social Security offsets "
                    "spending from that age onward."
                )
        else:
            st.error(f"No retirement age reaches ≥{SUCCESS_THRESHOLD_PCT}% success with these inputs.")

        with st.expander("Sequence-of-returns risk", expanded=True):
            show_section(
                "Portfolio paths at your target age",
                f"Monte Carlo paths if you retire at age {target_age}. "
                "Early adverse markets can leave you with far less than the median path. "
                f"The green line shows a deterministic path at your expected return "
                f"({mean_return * 100:.1f}%) with no volatility.",
            )
            show_real = st.toggle(
                "Adjust for inflation",
                value=bool(cfg.get("show_real_values", False)),
                help="Show portfolio values in today's dollars instead of nominal dollars.",
            )
            cfg["show_real_values"] = show_real
            fan_data = fan_real if show_real else fan_nominal
            det_values = det_real if show_real else det_nominal
            det_overlay = (det_years, det_values, det_label) if det_years else None
            fan_fig = plot_fan_chart(
                fan_ages,
                fan_data,
                f"What might your portfolio be worth if you retire at {target_age}?",
                real_values=show_real,
                det_overlay=det_overlay,
            )
            render_chart_with_download(
                fan_fig,
                file_name="portfolio_paths.png",
                label="Download chart PNG",
            )
            if det_overlay is None:
                st.caption(
                    f"The fixed {mean_return * 100:.1f}% return path depletes before your "
                    "planning horizon — no overlay shown."
                )

        render_career_years_section(
            cfg=cfg,
            target_age=target_age,
            baseline_career_years=baseline_career_years,
            min_years=min_years,
            on_track=on_track,
        )

        st.caption(
            f"Based on {trial_count:,} Monte Carlo trials per retirement age. "
            "Results are cached — rerunning with the same inputs is fast."
        )

    with tab_whatif:
        show_section(
            "What-if scenarios",
            "See how small changes to savings, spending, or retirement age affect your "
            "success rate at your target — without editing the sidebar.",
        )
        applied = render_what_if_table(what_if_rows, best_levers)
        if applied is not None:
            cfg.update(applied)
            save_cfg(cfg)
            st.rerun()
        st.caption(
            "Each row reruns Monte Carlo with one parameter change. "
            "Click Apply to load a scenario into the sidebar."
        )

    with tab_coast:
        show_section(
            "Coast number",
            "The smallest portfolio you'd need today that, left untouched — no further "
            "contributions and no withdrawals — still grows enough to retire at age "
            f"{target_age} with ≥{SUCCESS_THRESHOLD_PCT}% success. Reach it and saving "
            "for retirement becomes optional.",
        )
        render_coast(coast_number, cfg["starting_amount"], target_age)
        if coast_number:
            render_coast_progress(coast_number, cfg["starting_amount"])
            coast_ages, coast_line, portfolio_line, coast_age = coast_growth_paths(
                cfg, coast_number, target_age, cfg["mean_return"],
            )
            render_coast_kpis(
                coast_age, cfg["current_age"], target_age, coast_stop_now_odds,
            )
            coast_fig = plot_coast_growth(
                coast_ages, coast_line, portfolio_line, coast_age,
                cfg["current_age"], target_age,
            )
            render_chart_with_download(
                coast_fig, file_name="coast_growth.png", label="Download coast chart",
            )
            st.caption(
                "Both lines grow at your expected return. The dashed line is the coast "
                "number compounding untouched to the retirement nest egg; the solid line "
                "is today's balance plus your contributions. Where they meet is the age "
                "you could stop saving and let the rest ride."
            )
            render_coast_compounding(coast_number, coast_line[-1], target_age)
        st.caption(
            "Coasting assumes you keep working and cover your own expenses until "
            "retirement, so the balance is neither added to nor drawn from in the "
            "meantime — and career years still accrue toward Social Security."
        )

    with tab_compare:
        show_section(
            "Scenario comparison",
            "Overlay saved scenarios or profile presets on your current plan's success curve.",
        )
        saved_names = [s["name"] for s in saved_scenarios]
        preset_names = list(PRESET_PROFILES.keys())
        compare_col1, compare_col2 = st.columns(2)
        with compare_col1:
            compare_saved = st.multiselect(
                "Compare saved",
                options=saved_names,
                default=[saved_names[-1]] if saved_names else [],
                help="Select saved scenarios to overlay on the chart below.",
            )
        with compare_col2:
            compare_presets = st.multiselect(
                "Compare presets",
                options=preset_names,
                help="Overlay profile presets without saving them first.",
            )
        if compare_saved or compare_presets:
            comparison_runs = load_comparison_runs(
                saved_scenarios, compare_saved, current_run=("Current", ages, probs),
            )
            comparison_runs.extend(load_preset_comparison_runs(compare_presets))
            compare_fig = plot_scenario_comparison(comparison_runs)
            render_chart_with_download(
                compare_fig,
                file_name="scenario_comparison.png",
                label="Download chart PNG",
            )
            render_comparison_legend(comparison_runs)
        elif not saved_names:
            st.markdown(
                '<div class="empty-state"><h3>No saved scenarios</h3>'
                "<p>Save a plan in the sidebar, or compare profile presets above.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Select a saved scenario or preset above to compare.")

    with tab_export:
        show_section(
            "Export your plan",
            "Download your current inputs as JSON, or a full PDF report with charts and summary.",
        )
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download scenario JSON",
                json.dumps(cfg, indent=2),
                file_name="scenario.json",
                mime="application/json",
            )
        with col_dl2:
            pdf_bytes = cached_report_bytes(
                cfg_key,
                json.dumps(ages),
                json.dumps(probs),
                mean_return,
                volatility,
                json.dumps(optimizer_lines),
                executive_summary,
            )
            st.download_button(
                "Download PDF report",
                pdf_bytes,
                file_name="retirement_report.pdf",
                mime="application/pdf",
            )

    save_cfg(cfg)
