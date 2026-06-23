# Retirement Runway — Retirement Probability Simulator

A Monte Carlo retirement planning tool with a Streamlit dashboard. It estimates the probability that your portfolio lasts through retirement at different retirement ages, finds the minimum career length needed for 90% success, and visualizes sequence-of-returns risk.

## Requirements

- Python 3.9+

## Install

From the project root:

```bash
python3 -m pip install -r requirements.txt
```

Or activate the virtual environment and install there:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: Streamlit, NumPy, Matplotlib, and ReportLab (see `requirements.txt` for version constraints).

## Run

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (usually `http://localhost:8501`).

## How to Use

### 1. Set inputs (sidebar)

Pick a **profile preset** (Early, Mid, or Pre-retire) to load realistic defaults — applying a preset takes effect immediately. Then adjust values in **Essentials** and click **Apply changes** to update the simulation.

| Input | Description |
|-------|-------------|
| **Current age** | Your age today |
| **Portfolio today** | Current retirement savings |
| **Annual savings** | Amount added each year until retirement |
| **Target retirement age** | Age used for success rate, fan chart, and career planning |
| **Annual spending** | Pre-inflation spending from retirement onward |

Most other sidebar controls (presets, Social Security, market settings) rerun the simulation as soon as you change them. Only **Essentials** requires **Apply changes**.

Open **Definitions** for a glossary of terms (success rate, earliest safe age, withdrawal rate, sequence-of-returns risk, planning horizon).

Toggle **Advanced parameters** to expose inflation, lower spending after 75, custom market assumptions, and planning horizon.

#### Simple mode (Advanced parameters off)

**Volatility preset** sets expected return and volatility together:

| Preset | Expected return | Volatility |
|--------|-----------------|------------|
| Conservative | 5% | 9% |
| Balanced | 6% | 12% |
| Aggressive | 7% | 14% |

Simple mode runs **2,000** Monte Carlo trials per retirement age.

#### Advanced mode (Advanced parameters on)

With advanced mode on, **Essentials** also shows inflation and spending-after-75 sliders. Additional expanders:

**Market**

| Input | Description |
|-------|-------------|
| **Expected return** | Mean annual portfolio return |
| **Volatility** | Annual return standard deviation |
| **Simulation trials** | Monte Carlo paths per retirement age (500–10,000; higher = more accurate, slower) |

Returns are drawn from a **log-normal** distribution.

**More options**

| Input | Description |
|-------|-------------|
| **Planning horizon (max age)** | Age through which success is evaluated (default 100) |

#### Social Security

| Input | Description |
|-------|-------------|
| **Include Social Security** | Whether SS offsets spending from your claim age onward (the offset rises each year with a COLA) |
| **Claim age** | 62, 67, or 70 |
| **Years worked so far** | Work history before today; used for SS and career-years planning |

Benefit scales with years worked (capped at 35) and claim age. The claim-age adjustment approximates the SSA schedule: **+8% per year** of delayed credits from 67 up to 70 (about +24% at 70), and early-claiming reductions of roughly **6.67% per year** for the first three years before 67 and **5% per year** earlier (about −30% at 62). It is still a simplification — actual benefits depend on your full earnings record and birth-year full-retirement age. The model default max benefit is **$24,000**/yr at full work history and age-67 claim.

Once you claim, the SS benefit **offsets spending each year** and is inflated by a cost-of-living adjustment (COLA) equal to the inflation rate, so its real value stays constant rather than eroding over retirement.

Spending grows with inflation from your retirement year onward. **Annual savings (contributions) are flat in nominal terms** — they do not grow year to year — so in real terms your contributions decline slightly before retirement. With advanced mode on, you can also set a **lower spending after 75 (%)** reduction (0% = flat spending).

#### Save and load scenarios

Under **Save scenario**, name your plan and click **Save scenario** to store it for comparison. Use **Load saved scenario** to restore a previously saved plan into the sidebar.

Saved scenarios and your last-run config are written to `scenarios.json` (gitignored — fresh clones start with defaults).

### 2. Read results (main page)

After inputs change, the app reruns Monte Carlo simulations. Results are cached — rerunning with the same inputs is fast. A loading skeleton and spinner appear while trials run.

The main page shows, in order:

1. **Hero** — app title, status badge (On track / Close / At risk), and chips for active profile, age range, and portfolio size
2. **Input warnings** — e.g. withdrawal rate above 5%, target age not after current age, or very low savings with no contributions
3. **Signal** — plain-language summary of your plan, including best what-if levers when relevant
4. **KPIs** — success probability at target age, earliest safe age, and withdrawal rate at target age
5. **Sustainable spending** — the largest annual budget that still clears the success threshold at your target age, with its implied withdrawal rate and a status-aware delta vs. your plan (headroom when you can spend more, trim-to-target when you are over)
6. **Success probability curve** — Monte Carlo success vs. retirement age, with a **Chart ends at age** slider to zoom the x-axis and a **Download chart PNG** button
7. **Safe-age summary** — message below the curve when any age reaches ≥90% success (with an SS note if the earliest safe age aligns with your claim age)
8. **Sequence-of-returns risk** (expander) — Monte Carlo fan chart at your target retirement age, with **Adjust for inflation** toggle and **Download chart PNG**
9. **Career years** — shown only when Social Security work history is a constraint (see section 5)
10. **What if** (expander) — quick scenario changes without editing the sidebar
11. **Compare scenarios** (expander) — overlay saved plans or profile presets
12. **Export** — JSON config and PDF report

### 3. Success curve

The success curve uses Monte Carlo simulation: random annual returns are sampled from a log-normal distribution. **Success** means the portfolio stays positive through your planning horizon.

Safe retirement ages (≥90% success) are evaluated from your current age through `min(85, planning horizon)`. The chart defaults from your current age to roughly your target age + 5; use **Chart ends at age** to zoom in or out.

### 4. Sequence-of-returns risk

The fan chart shows portfolio paths if you retire at your **target retirement age**:

- **10th, 50th, and 90th percentiles** across all Monte Carlo trials
- Shaded band between 10th and 90th percentiles
- Green overlay: deterministic path at your expected return (no volatility), when the portfolio lasts through the planning horizon
- **Adjust for inflation** toggle switches between nominal and real (today's-dollar) values

This highlights how early-market luck affects outcomes.

### 5. Career years

For your target retirement age, the app finds the minimum total work history needed for ≥90% success and compares it to your plan (years worked so far + years until retirement). Longer careers mainly raise **Social Security** income in this model.

On the main page, this section appears **only when SS work history is a constraint** — i.e. when you are not on track, or when the required minimum exceeds years you have already worked. When your plan already satisfies the requirement, the section is hidden (results still appear in the PDF report).

When shown, you get an on-track info message or a shortfall warning with the gap in years.

### 6. What-if scenarios

The **What if** expander reruns Monte Carlo with one change at a time:

| Scenario | Change |
|----------|--------|
| Save +$10K/yr | Increases annual savings by $10,000 |
| Save +$25K/yr | Increases annual savings by $25,000 |
| Spend −10% | Reduces annual spending by 10% |
| Retire later | Target retirement age +2 (capped at 85) |
| Retire earlier | Target retirement age −2 (minimum current age + 1) |

Each row shows success probability at your target age and the delta vs. your current plan. Click **Apply** on a row to load that scenario into the sidebar. The Signal narrative may recommend the best levers from this table.

### 7. Compare scenarios

Overlay success curves on one chart:

1. **Compare saved** — select one or more saved scenarios (the most recent is selected by default when any exist)
2. **Compare presets** — overlay Early, Mid, or Pre-retire profile presets without saving them first

The current plan is always included. Use **Download chart PNG** to export the comparison chart.

To save a plan for comparison: configure inputs, name it under **Save scenario** in the sidebar, and click **Save scenario**.

### 8. Save and export

- All inputs are **auto-saved** to `scenarios.json` after each run
- **Download scenario JSON** — export your full config
- **Download PDF report** — executive summary, assumptions, safe ages, and career-years results
- On next launch, saved values load automatically from `scenarios.json`

## PDF Report

### From the UI

Click **Download PDF report** at the bottom of the main page.

### From Python

```python
from model import compute_curve
from report import generate_report, generate_report_bytes

cfg = {
    "current_age": 28,
    "starting_amount": 550000,
    "annual_contribution": 50000,
    "max_age": 100,
    "inflation_rate": 0.03,
    "include_social_security": True,
    "social_security_claim_age": 67,
    "years_already_worked": 6,
    "trials": 2000,
    "annual_spending": 120000,
    "spending_reduction_after_75": 0.15,
}

ages, probs = compute_curve(cfg, range(55, 76), mean_return=0.06, volatility=0.12)

generate_report("report.pdf", cfg, ages, probs, mean_return=0.06, volatility=0.12)

pdf_bytes = generate_report_bytes(cfg, ages, probs, mean_return=0.06, volatility=0.12)
```

## Project Structure

```
├── app.py                      # Streamlit UI
├── model.py                    # Monte Carlo simulation engine
├── report.py                   # PDF report generator
├── requirements.txt            # Python dependencies
├── scenarios.json              # Last-run config and saved scenarios (auto-written, gitignored)
├── .streamlit/
│   ├── config.toml             # Theme and server settings
│   └── static/custom.css       # Custom UI styling
└── .venv/                      # Python virtual environment (optional)
```

## Model Notes

- **Spending** uses one annual amount with optional reduction after age 75; grows with inflation from retirement onward
- **Contributions** (annual savings) are flat nominal until retirement — they do not grow with inflation, so their real value declines modestly over the accumulation phase
- **Social Security** can be toggled off; when on, benefit scales with years worked (capped at 35) and claim age (62 / 67 / 70). The claim-age adjustment approximates SSA rules (+8%/yr delayed credits to 70; ≈−30% at 62) rather than a flat ±3%/yr, and the benefit receives an inflation COLA once claimed so its real offset stays constant. Default max benefit is $24,000/yr
- **Success** = portfolio stays positive through your planning horizon (default age 100)
- **Safe retirement range** is evaluated from current age through `min(85, planning horizon)`
- **Returns** use a log-normal distribution (matches typical equity return modeling)
- **Simple mode** uses 2,000 trials; **advanced mode** allows 500–10,000
- Model runs are cached (`@st.cache_data`) — identical inputs reuse prior results
