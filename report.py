import io

import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from model import SUCCESS_THRESHOLD, SUCCESS_THRESHOLD_PCT


def delta_pts(delta):
    return round(delta * 100) if delta is not None else None


def pick_best_levers(rows, limit=2):
    eligible = [
        row for row in rows
        if delta_pts(row.get("delta")) is not None and delta_pts(row["delta"]) > 0
    ]
    eligible.sort(key=lambda row: row["delta"], reverse=True)
    return eligible[:limit]


def format_lever_recommendation(levers, *, at_target, html=False):
    if not levers:
        if at_target:
            return None
        return (
            "No tested adjustment improves success — try larger changes to "
            "savings, spending, or retirement age."
        )

    def fmt_label(label, delta):
        pts = delta_pts(delta)
        if html:
            return f"<strong>{label}</strong> ({pts:+d} pts)"
        return f"{label} ({pts:+d} pts)"

    best = levers[0]
    best_str = fmt_label(best["Change"], best["delta"])
    if at_target:
        return f"Optional margin: {best_str}."
    text = f"Best lever: {best_str}."
    if len(levers) > 1:
        second = levers[1]
        second_str = fmt_label(second["Change"], second["delta"])
        text += f" Also consider {second_str}."
    return text


def build_plan_narrative(
    target_prob,
    target_age,
    on_track,
    withdrawal_rate,
    ranges,
    years_until,
    levers=None,
    html=False,
):
    bold = (lambda value: f"<strong>{value}</strong>") if html else str

    parts = []
    if target_prob is not None:
        if target_prob >= SUCCESS_THRESHOLD:
            parts.append(
                f"Your plan clears the {SUCCESS_THRESHOLD_PCT}% bar at age {bold(target_age)}."
            )
        elif target_prob >= 0.80:
            parts.append(
                f"You have an {bold(f'{target_prob * 100:.0f}')} chance of success at age "
                f"{bold(target_age)} — close, but not at the {SUCCESS_THRESHOLD_PCT}% threshold."
            )
        else:
            parts.append(
                f"At age {bold(target_age)}, success is {bold(f'{target_prob * 100:.0f}')} "
                "— consider saving more, spending less, or retiring later."
            )
    if ranges:
        parts.append(
            f"Earliest age with ≥{SUCCESS_THRESHOLD_PCT}% success: {bold(ranges[0][0])}."
        )
    if on_track:
        parts.append("Your planned career length supports your target.")
    elif years_until > 0:
        parts.append(
            f"You may need more working years to hit {SUCCESS_THRESHOLD_PCT}% at your target age."
        )
    if withdrawal_rate is not None and withdrawal_rate > 0.05:
        parts.append(
            f"Withdrawal rate at age {bold(target_age)} is "
            f"{bold(f'{withdrawal_rate * 100:.1f}')} — above the common 4–5% guideline."
        )
    elif withdrawal_rate is not None and withdrawal_rate > 0:
        parts.append(
            f"Withdrawal rate at age {bold(target_age)} is "
            f"{bold(f'{withdrawal_rate * 100:.1f}')} — within a typical safe range."
        )

    lever_text = format_lever_recommendation(
        levers or [],
        at_target=target_prob is not None and target_prob >= SUCCESS_THRESHOLD,
        html=html,
    )
    if lever_text:
        parts.append(lever_text)

    return " ".join(parts)


def build_executive_summary(
    target_prob,
    target_age,
    on_track,
    withdrawal_rate,
    ranges,
    years_until,
    levers=None,
):
    return build_plan_narrative(
        target_prob,
        target_age,
        on_track,
        withdrawal_rate,
        ranges,
        years_until,
        levers=levers,
        html=False,
    )


def _spending_summary(cfg):
    base = cfg.get("annual_spending", cfg.get("spending_under_75", 200_000))
    reduction = cfg.get("spending_reduction_after_75", 0.0)
    if reduction > 0:
        return f"${base:,.0f}/yr ( −{reduction * 100:.0f}% after 75)"
    return f"${base:,.0f}/yr"


def _ss_summary(cfg):
    if not cfg.get("include_social_security", True):
        return "Not included"
    claim_age = cfg.get("social_security_claim_age", cfg.get("social_security_start_age", 67))
    return f"Claim at age {claim_age}"


def safe_retirement_ranges(ages, probs, threshold=SUCCESS_THRESHOLD):
    ages = np.asarray(ages)
    probs = np.asarray(probs)
    safe = ages[probs >= threshold]
    if len(safe) == 0:
        return []

    ranges = []
    start = int(safe[0])
    prev = int(safe[0])
    for age in safe[1:]:
        age = int(age)
        if age - prev > 1:
            ranges.append((start, prev))
            start = age
        prev = age
    ranges.append((start, prev))
    return ranges


def format_safe_ranges_plain(ranges, max_eval_age):
    if not ranges:
        return None

    if len(ranges) == 1:
        start, end = ranges[0]
        if start == end:
            return f"Only age {start} meets the >={SUCCESS_THRESHOLD_PCT}% threshold"
        if end >= max_eval_age - 1:
            return (
                f"Earliest retirement age with >={SUCCESS_THRESHOLD_PCT}% success: {start} "
                f"(all later ages through {max_eval_age} are also safe)"
            )
        return f"Ages {start}-{end} meet the >={SUCCESS_THRESHOLD_PCT}% threshold"

    parts = []
    for start, end in ranges:
        parts.append(str(start) if start == end else f"{start}-{end}")
    return f"Safe retirement ages: {', '.join(parts)}"


def _draw_wrapped_text(c, text, x, y, max_width, font="Helvetica", size=11, leading=15):
    c.setFont(font, size)
    for line in simpleSplit(text, font, size, max_width):
        c.drawString(x, y, line)
        y -= leading
    return y


def _draw_report(
    c,
    cfg,
    ages,
    probs,
    mean_return,
    volatility,
    optimizer_result=None,
    executive_summary=None,
):
    ages = np.asarray(ages)
    probs = np.asarray(probs)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Retirement Simulation Report")
    y -= 30

    if executive_summary:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Executive Summary")
        y -= 18
        y = _draw_wrapped_text(c, executive_summary, 50, y, width - 100)
        y -= 12

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Assumptions")
    y -= 20

    c.setFont("Helvetica", 11)
    lines = [
        f"Current Age: {cfg['current_age']}",
        f"Portfolio: ${cfg['starting_amount']:,.0f}",
        f"Annual Contribution: ${cfg['annual_contribution']:,.0f}",
        f"Inflation: {cfg['inflation_rate'] * 100:.1f}%",
        f"Mean Return: {mean_return * 100:.1f}%  |  Volatility: {volatility * 100:.1f}%",
        f"Monte Carlo Trials: {cfg['trials']:,}",
        f"Planning Horizon: age {cfg['max_age']}",
        f"Social Security: {_ss_summary(cfg)}",
        f"Spending: {_spending_summary(cfg)}",
    ]
    for line in lines:
        c.drawString(50, y, line)
        y -= 18

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Success Probability Summary")
    y -= 20

    c.setFont("Helvetica", 11)
    max_eval_age = min(85, cfg.get("max_age", 100))
    safe_summary = format_safe_ranges_plain(
        safe_retirement_ranges(ages, probs), max_eval_age,
    )
    if safe_summary:
        y = _draw_wrapped_text(c, safe_summary, 50, y, width - 100)
    else:
        c.drawString(50, y, f"No retirement age reaches >={SUCCESS_THRESHOLD_PCT}% success")
        y -= 18

    if len(ages) > 0:
        best_idx = int(probs.argmax())
        c.drawString(
            50, y,
            f"Highest success: {int(ages[best_idx])} at {probs[best_idx] * 100:.1f}%",
        )
        y -= 18

    if optimizer_result:
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Career Years")
        y -= 20
        c.setFont("Helvetica", 11)
        for line in optimizer_result:
            c.drawString(50, y, line)
            y -= 18

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Educational model only — not financial advice.")


def generate_report(
    filename,
    cfg,
    ages,
    probs,
    mean_return=0.06,
    volatility=0.12,
    optimizer_result=None,
    executive_summary=None,
):
    c = canvas.Canvas(filename, pagesize=letter)
    _draw_report(
        c, cfg, ages, probs, mean_return, volatility, optimizer_result, executive_summary,
    )
    c.save()


def generate_report_bytes(
    cfg,
    ages,
    probs,
    mean_return=0.06,
    volatility=0.12,
    optimizer_result=None,
    executive_summary=None,
):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    _draw_report(
        c, cfg, ages, probs, mean_return, volatility, optimizer_result, executive_summary,
    )
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
