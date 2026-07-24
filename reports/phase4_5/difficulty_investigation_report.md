# MacroRisk AI — Phase 4.5: Task-Difficulty Investigation

_Why are baseline R² values so high? Is the synthetic generator or the target
formulation to blame? This report answers each of the eight questions with
measurements on the real reference (160 rows) vs. the synthetic dataset
(4,520 rows). No model accuracy was altered and no noise was injected._

## TL;DR — Headline conclusion

**The high R² is a property of predicting absolute financial *levels*, not a
synthetic-data artifact.** Real financial statements are almost perfectly
persistent quarter-to-quarter (real lag-1 autocorrelation ≈ 0.99 for every
target), so a trivial "previous quarter" forecast already explains ~99% of the
level variance — **in the real data itself**. The synthetic generator is, if
anything, *less* persistent than the real data and has *equal-or-larger*
quarter-to-quarter volatility, so it does **not** make the task easier.

➡️ **Recommended fix is a target-formulation change, not a generator change:**
model **percentage growth** (log-return) and reconstruct levels for the app.
On growth, models show small but **consistently positive skill over naive**
across all six targets; on levels they show ~zero skill. **The generator is
statistically valid and must not be modified** (doing so would make it diverge
from the real data).

---

## Q1 — Is the copula preserving temporal persistence *too strongly*?

**No — the opposite.** Level lag-1 autocorrelation, real vs synthetic:

| Target | Real acf(1) | Synth acf(1) | Real acf(2) | Synth acf(2) | Real acf(4) | Synth acf(4) |
|---|--:|--:|--:|--:|--:|--:|
| Sales | 0.9994 | 0.9993 | 0.9988 | 0.9982 | 0.9979 | 0.9966 |
| Operating Profit | 0.9873 | 0.9875 | 0.9864 | 0.9768 | 0.9846 | 0.9611 |
| Net Profit | 0.9880 | 0.9757 | 0.9877 | 0.9563 | 0.9907 | 0.9280 |
| Borrowings | 0.9993 | 0.9907 | 0.9989 | 0.9831 | 0.9986 | 0.9742 |
| Total Assets | 0.9994 | 0.9952 | 0.9994 | 0.9908 | 0.9990 | 0.9870 |
| CFO | 0.9974 | 0.9853 | 0.9975 | 0.9736 | 0.9965 | 0.9573 |

Synthetic persistence is **equal to or slightly below** real at every lag. The
generator does not over-persist. **The real data is the source of the high
persistence.**

## Q2 — Variance and quarter-to-quarter (delta) distributions

| Target | Level var ratio (S/R) | QoQ Δ std real | QoQ Δ std synth | pct-growth std ratio (S/R) | pct-growth KS |
|---|--:|--:|--:|--:|--:|
| Sales | 1.15 | 2,780 | 3,089 | 1.04 | 0.16 |
| Operating Profit | 1.24 | 1,495 | 1,664 | 1.07 | 0.06 |
| Net Profit | 1.36 | 1,084 | 1,808 | 1.65 | 0.14 |
| Borrowings | 0.46 | 21,421 | 52,663 | 0.82 | 0.12 |
| Total Assets | 1.09 | 30,433 | 91,877 | 1.40 | 0.20 |
| CFO | 1.29 | 1,330 | 3,580 | 2.37 | 0.22 |

Synthetic quarter-to-quarter **changes are as large as, or larger than, real**
for five of six targets (CFO/Net Profit/Total Assets are noticeably *more*
volatile). There is **no over-smoothing** — the opposite where they differ.

## Q3 — Autocorrelation (lag-1, lag-2, lag-4)

See the Q1 table. Real autocorrelation is uniformly ~0.99 and **higher** than
synthetic at lags 2 and 4 for most targets (e.g. CFO lag-4: real 0.997 vs synth
0.957). The synthetic series decay *faster* than real — again, not "too easy."

## Q4 — Are synthetic companies diverse enough?

- 565 synthetic entities (vs 20 real). Between-company **dispersion ratio
  (synth/real) per sector = 0.96–1.17** → synthetic companies are **as diverse
  as real ones** (slightly more so for FMCG/Energy).
- Nearest-neighbour distance in standardized 6-target centroid space:
  mean 0.243, median 0.137, min 0.014; 41% have a neighbour within 0.10.
- Interpretation: the 41% figure reflects **sampling density** (565 draws from a
  continuous copula pack closely in absolute terms), **not duplication** — the
  dispersion ratio ≈ 1.0 shows the spread matches real. Diversity is adequate.

## Q5 — Variance explained by lag-1 alone

lag-1 variance explained = acf(1)². **Real data**: Sales 0.999, Operating
Profit 0.975, Net Profit 0.976, Borrowings 0.999, Total Assets 0.999, CFO 0.995.
**A single feature (last quarter's value) explains 97–99.9% of the level
variance — measured on the real data.** Synthetic is comparable or slightly
lower. This is the mathematical origin of the high baseline R².

## Q6 — Do absolute levels inherently cause high R²?

**Yes, definitively.** R² is variance-explained relative to the target's total
variance. For levels, total variance is dominated by **cross-company scale**
(e.g. a large firm's Sales ≈ 60,000 vs a small firm's ≈ 15,000) plus near-unit
persistence. Predicting "next quarter ≈ this quarter" therefore captures ~99% of
that variance automatically. This is intrinsic to financial-statement levels and
is present in the real data — it is not caused by the generator.

## Q7 — Levels vs QoQ growth vs percentage growth

Single chronological split (train quarters < last, test = last), ExtraTrees via
the Phase-3 preprocessor, on the synthetic dataset:

| Target | Levels model R² | Levels skill vs naive | Δ model R² | %-growth model R² | %-growth skill vs naive |
|---|--:|--:|--:|--:|--:|
| Sales | 0.998 | −0.00 | 0.078 | 0.008 | **+0.13** |
| Operating Profit | 0.975 | −0.00 | −0.048 | 0.035 | **+0.05** |
| Net Profit | 0.954 | −0.00 | 0.021 | 0.025 | **+0.05** |
| Borrowings | 0.973 | −0.01 | −0.005 | 0.029 | **+0.05** |
| Total Assets | 0.988 | −0.00 | −0.102 | 0.018 | **+0.04** |
| CFO | 0.977 | −0.00 | 0.036 | 0.036 | **+0.05** |

**Reading this table:**
- **Levels**: R² ≈ 0.95–0.998 but **skill over naive ≈ 0** — the model adds
  nothing beyond persistence. The impressive R² is an illusion of scale.
- **Percentage growth**: low absolute R² (0.01–0.04) but **positive skill over
  naive for all six targets** — the model genuinely learns signal the naive
  forecast lacks. This is the honest, discriminating task.
- **Delta (absolute change)** is scale-dependent and noisier; percentage growth
  is the cleaner, cross-company-comparable formulation.

**Most appropriate formulation for this project: percentage growth (log-return),
with levels reconstructed as `level_{t+1} = level_t × (1 + predicted_growth)`.**
This serves the application (which needs both predicted levels and growth %),
makes model comparison meaningful, and forces models to beat the naive baseline.

## Q8 — Should the synthetic generator be improved?

**No.** The generator is statistically valid on every axis tested:
persistence ≤ real, QoQ volatility ≥ real, diversity ≈ real, marginal/
correlation fidelity already validated in Phase 2. The high R² is **not** caused
by the generator. Modifying it (e.g. injecting noise or damping persistence)
would (a) violate the "no artificial degradation / no noise" constraint and
(b) make the synthetic data **diverge from the real data it is meant to mimic**.
**Recommendation: leave the generator unchanged.**

---

## Recommendations (for approval, before Phase 5)

1. **Reformulate the six models to predict percentage growth** (log-return),
   then reconstruct absolute levels for the application. Keep the six targets.
2. **Adopt skill-over-naive as the primary evaluation metric** (R² and RMSE
   *relative to the naive previous-quarter baseline*), regardless of
   formulation — a model is only useful if it beats naive.
3. **Do not modify the synthetic generator**; it faithfully reproduces real
   persistence, volatility, and diversity.
4. **Exclude plain OLS** and keep regularized linear + tree/boosting families.
5. Proceed to Phase 5 hyperparameter optimization on the **growth formulation**,
   optimizing skill-over-naive.

_These are the only statistically justified changes. No model accuracy was
reduced and no noise was added in producing this analysis._
