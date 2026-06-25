import base64
import io
import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.ticker import FuncFormatter

from model import (
    MODEL_CFG_FIELDS,
    SUCCESS_THRESHOLD,
    SUCCESS_THRESHOLD_PCT,
    compute_curve,
    compute_mc_net_worth_fan,
    find_coast_number,
    find_max_sustainable_spending,
    find_min_years_worked,
    guardrail_income_band,
    simulate_net_worth,
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


DEFAULT_CFG = {
    "current_age": 28,
    "starting_amount": 550_000,
    "annual_contribution": 50_000,
    "max_age": 90,
    "inflation_rate": 0.03,
    "include_social_security": True,
    "social_security_claim_age": 67,
    "years_already_worked": 6,
    "trials": 2000,
    "annual_spending": 120_000,
    "spending_reduction_after_75": 0.0,
    "mean_return": 0.06,
    "volatility": 0.12,
    "target_retirement_age": 65,
    "advanced_mode": False,
    "active_profile": None,
    "show_real_values": False,
    "withdrawal_strategy": "fixed",
}

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

GLOSSARY = {
    "Success rate": "Share of Monte Carlo trials where your portfolio stays positive through your planning horizon.",
    "Earliest safe age": f"The youngest retirement age (searched up to 85) whose success rate reaches ≥{SUCCESS_THRESHOLD_PCT}% — where success means the portfolio lasts through your planning horizon.",
    "Withdrawal rate": "Annual spending divided by portfolio at retirement — the 4% rule targets ≤4%.",
    "Sequence-of-returns risk": "Bad market returns early in retirement can permanently reduce portfolio durability.",
    "Planning horizon": "The age through which the model checks whether your money lasts (default: 100).",
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


def render_kpi_row(
    *,
    target_age,
    target_prob,
    ranges,
    withdrawal_rate,
):
    prob_val = f"{target_prob * 100:.0f}%" if target_prob is not None else "—"
    prob_cls = prob_semantic_class(target_prob)
    earliest = str(ranges[0][0]) if ranges else "None"
    if ranges and ranges[0][0] <= target_age:
        earliest_cls = "good"
    elif ranges:
        earliest_cls = "neutral"
    else:
        earliest_cls = "warn"
    wr_val = f"{withdrawal_rate * 100:.1f}%" if withdrawal_rate is not None else "—"
    wr_cls = withdrawal_semantic_class(withdrawal_rate)

    st.markdown(
        f"""
        <div class="kpi-sticky">
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Success rate · age {target_age}</div>
                    <div class="kpi-value {prob_cls}">{prob_val}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Earliest safe age</div>
                    <div class="kpi-value {earliest_cls}">{earliest}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Withdrawal rate · age {target_age}</div>
                    <div class="kpi-value {wr_cls}">{wr_val}</div>
                </div>
            </div>
        </div>
        """,
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
    if coast is None:
        status, message = coast_state(coast, current_portfolio, target_age)
        st.markdown(
            f'<div class="safespend-card {status}">'
            f'<div class="signal-header">'
            f'<span class="signal-dot {status}"></span>'
            f'<span class="signal-label">Coast number</span>'
            f'<span class="safespend-conf">{SUCCESS_THRESHOLD_PCT}% confidence</span>'
            f"</div>"
            f'<div class="safespend-delta {status}">{message}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        return
    status, message = coast_state(coast, current_portfolio, target_age)
    st.markdown(
        f'<div class="safespend-card {status}">'
        f'<div class="signal-header">'
        f'<span class="signal-dot {status}"></span>'
        f'<span class="signal-label">Coast number</span>'
        f'<span class="safespend-conf">{SUCCESS_THRESHOLD_PCT}% confidence · grows untouched</span>'
        f"</div>"
        f'<div class="safespend-value">{format_currency(coast)}</div>'
        f'<div class="safespend-sub">balance that, left untouched (no contributions, '
        f"no withdrawals), still funds retirement at age {target_age} "
        f'<span class="safespend-sep">·</span> '
        f"you have {format_currency(current_portfolio)} today</div>"
        f'<div class="safespend-delta {status}">{message}</div>'
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
            json.dumps({
                "age_start": variant_cfg["current_age"],
                "age_end": min(85, variant_cfg["max_age"]),
                "mean_return": mean_return,
                "volatility": volatility,
                "seed": None,
            }),
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


def load_store():
    try:
        with SCENARIO_FILE.open("r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"last": None, "saved": _load_legacy_saved()}
    except json.JSONDecodeError:
        st.sidebar.warning("Could not parse scenarios.json — using defaults.")
        return {"last": None, "saved": []}

    if isinstance(data, dict) and "last" in data:
        return data

    saved = _load_legacy_saved()
    return {"last": data, "saved": saved}


def save_store(last=None, saved=None):
    store = load_store()
    if last is not None:
        store["last"] = last
    if saved is not None:
        store["saved"] = saved
    with SCENARIO_FILE.open("w") as f:
        json.dump(store, f, indent=2)


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
            json.dumps({
                "age_start": scfg["current_age"],
                "age_end": min(85, scfg["max_age"]),
                "mean_return": mean_return,
                "volatility": volatility,
                "seed": None,
            }),
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
            json.dumps({
                "age_start": pcfg["current_age"],
                "age_end": min(85, pcfg["max_age"]),
                "mean_return": pcfg["mean_return"],
                "volatility": pcfg["volatility"],
                "seed": None,
            }),
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

    with st.expander("Definitions", expanded=False):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** — {definition}")

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
        with st.form("essentials_form", border=False):
            essentials_age = st.number_input(
                "Current age", 18, 80, int(cfg["current_age"]),
            )
            essentials_portfolio = st.number_input(
                "Portfolio today ($)", 0, 10_000_000, int(cfg["starting_amount"]),
            )
            essentials_savings = st.number_input(
                "Annual savings ($)", 0, 500_000, int(cfg["annual_contribution"]),
            )
            essentials_target = st.number_input(
                "Target retirement age",
                essentials_age + 1, 85,
                int(cfg.get("target_retirement_age", 65)),
                help="Used for success rate, risk paths, and career planning.",
            )
            essentials_spending = st.number_input(
                "Annual spending ($)",
                0, 500_000,
                int(cfg.get("annual_spending", DEFAULT_CFG["annual_spending"])),
                help="Pre-inflation spending from retirement onward.",
            )
            essentials_inflation = cfg["inflation_rate"]
            essentials_reduction = cfg.get("spending_reduction_after_75", 0.0)
            if advanced_mode:
                essentials_inflation = st.slider(
                    "Inflation", 0.0, 0.1, float(cfg["inflation_rate"]),
                )
                reduction_pct = int(round(float(cfg.get("spending_reduction_after_75", 0.0)) * 100))
                essentials_reduction = st.slider(
                    "Lower spending after 75 (%)",
                    0, 50,
                    reduction_pct,
                    step=5,
                    help="Reduce annual spending by this percentage from age 75 onward.",
                ) / 100.0
            if st.form_submit_button("Apply changes", width="stretch"):
                cfg["current_age"] = essentials_age
                cfg["starting_amount"] = essentials_portfolio
                cfg["annual_contribution"] = essentials_savings
                cfg["target_retirement_age"] = essentials_target
                cfg["annual_spending"] = essentials_spending
                if advanced_mode:
                    cfg["inflation_rate"] = essentials_inflation
                    cfg["spending_reduction_after_75"] = essentials_reduction
                save_cfg(cfg)
                st.rerun()

    target_retirement_age = int(cfg.get("target_retirement_age", 65))

    if advanced_mode:
        with st.expander("Market", expanded=False):
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
            cfg["max_age"] = st.number_input("Planning horizon (max age)", 70, 110, int(cfg["max_age"]))

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
        json.dumps({
            "age_start": cfg["current_age"],
            "age_end": max_eval_age,
            **model_args,
        }),
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

smart_spending_on = cfg.get("withdrawal_strategy") == "guardrails"
fixed_ages = fixed_probs = None
if smart_spending_on:
    # Same plan with a fixed inflation-only budget, for the success-curve overlay
    # so the lift smart spending buys is visible side by side.
    fixed_cfg = dict(cfg, withdrawal_strategy="fixed")
    fixed_ages, fixed_probs = cached_model(
        "curve",
        cfg_cache_key(fixed_cfg),
        json.dumps({
            "age_start": cfg["current_age"],
            "age_end": max_eval_age,
            **model_args,
        }),
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

    render_verdict(
        build_verdict(target_prob, target_age, cfg["max_age"], ranges),
        build_verdict_subline(target_prob, best_levers),
        status=insight_status(target_prob, on_track),
    )

    render_kpi_row(
        target_age=target_age,
        target_prob=target_prob,
        ranges=ranges,
        withdrawal_rate=withdrawal_rate,
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
