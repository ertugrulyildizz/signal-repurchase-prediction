# Pre-registration (`tanımlar`)

This is a cleaned, English version of the project's definitions document. In the
original project it was written **before any model result was seen**, dated, and
never edited in place: every change was added as a new version below the old
text. Real business figures have been removed; the rules are unchanged.

---

## v1.0 / v1.1 — definitions (locked before modelling)

### Business question
Which online customers will place at least one order in the next 90 days?
And what does the last 30 days of web and app behaviour add to that forecast?

### Scope
- Online store only (marketplaces excluded); one brand.
- Orders that were invoiced (pre-invoice cancellations never enter the data).
- Excluded: rows without a customer key; accounts with more than 50 orders in
  any calendar year (bulk buyers, staff, test accounts).

### Cutoffs (monthly snapshots)
| Set | Cutoffs | Note |
|---|---|---|
| Train | 2024-10-01 → 2025-07-01 (10) | Label windows close before the test cutoff |
| Buffer | 2025-08-01, 2025-09-01 | Label window overlaps the test cutoff → never used |
| **Test** | 2025-10-01 | The one-shot test |
| Test 2 | 2026-01-01 | Robustness in a different season; model retrained on 2024-10 → 2025-10 |

Model selection uses **rolling-origin validation** only: validation cutoffs
2025-04 / 05 / 06 / 07, each trained only on cutoffs whose label window closed
before it (`cutoff + 90 days ≤ validation cutoff`).

### Population and label
- Population at a cutoff: customers with ≥ 1 valid order in the previous 365 days.
- `y_etiket` = 1 if ≥ 1 valid order in `[cutoff, cutoff + 90 days)`. Returns are
  not available, so the label measures gross purchase intent (limitation).

### Feature windows
- Order counts and recency: previous 365 days (sub-windows 30/90/180/270).
- Monetary: previous 270 days, VAT and shipping excluded, divided by the
  cutoff's median unit price (inflation index). Raw currency never enters a model.
- Category shares by units.
- Behaviour (model B): previous 7 and 30 days, web + app. **Time rule:** a
  device→customer link is used only if its evidence (login or purchase)
  occurred before the cutoff.
- Not used: lifetime counts (history is left-truncated) and "current state"
  fields (active flag, opt-ins, current age).

### Experiment arms
- **A** — order and customer features (24).
- **A + flag** — A plus "any time-ruled behaviour in the last 30 days" (25).
- **B** — A plus 14 behaviour features (38). Behaviour counts enter as
  within-cutoff percentile ranks (see v1.2).

### Baselines
Recency rule · RFM score · BG/NBD (`lifetimes`).

### Success criteria (fixed before results)
1. Primary metric: PR-AUC on the test cutoff.
2. Business metric: share of all buyers captured in the top-scored 10 %
   (capture@10) and precision in that slice.
3. The best model must beat the best baseline on both PR-AUC and capture@10.
4. **"Material" B − A difference** — all of:
   - capture@10 difference ≥ 1 percentage point on the test cutoff;
   - customer-level paired bootstrap (1,000 resamples) 95 % CI excludes 0;
   - same sign in the rolling validation folds (clarified in v1.2).
5. Calibration: Brier score and reliability curve, before and after.
6. Decision-focused view: all metrics also for customers whose last order is
   more than 30 days old.
7. Coverage-matched view: B vs A also only among customers with behaviour data.
8. Robustness: B − A on Test 2 must have the same sign as on the test.
9. Pre-listed subgroups: single-order vs repeat customers; with / without
   behaviour; last order > 180 days ago.
10. **Secondary hypothesis:** B − A is larger for single-order customers than
    for repeat customers (their order history carries little signal).

### Alternatives considered and not chosen
| Dimension | Alternative | Why not |
|---|---|---|
| Target | Uplift (who responds to contact) | Best answer to the business question, but needs historical campaigns with random control groups. The project proposes that control group. |
| Target | 90-day revenue | Doubles the work; next step. |
| Target | Time to next order (survival) | "When?" question; BG/NBD baseline is from this family. |
| Horizon | 30 / 180 days | 30 days: too few positives. 180 days: fewer usable training cutoffs (gap rule) and much slower feedback. |
| Population | Members who never ordered | Different problem ("first purchase"). |
| Metric | Relative lift or a money threshold | A fixed percentage-point threshold is easier to pre-register and to translate into "extra buyers per list". |

### Known limitations
No returns · marketplace purchases invisible · left-truncated identity
evidence in the earliest cutoffs · event duplication not cleaned · model
predicts *who will buy*, not *whom contact will change*.

---

## v1.2 — before any model result (after code review)

1. **Behaviour counts → percentile among the active only.** In v1.1 the
   percentile was taken over the whole cutoff, so the many zeros shifted the
   value of the same relative interest whenever behaviour coverage changed.
   New rule: percentile rank within the cutoff **only among customers with
   behaviour** (values in (0, 1]); customers without behaviour get 0. The flag
   already carries active/inactive. Four options (this, v1.1 percentile,
   z-score, raw) are compared in rolling validation; if differences are below
   0.3 points the default stays. Reversible via a function argument.
2. **Web day share** enters as the same within-active percentile (NaN for the
   inactive), because its level drifted with channel mix.
3. **Sign consistency, made explicit:** "same sign in at least 2 of 3 rolling
   windows" means **at least 3 of the 4 folds**, compared with the sign of the
   test difference (`ISARET_MIN_KAT = 3`).
4. Added limitations: left-truncated identity evidence (conservative for B);
   bootstrap covers customer sampling, not model seed variance.

**The test was opened once, after v1.2.** Later versions only add new
evaluations; they never change the test criteria.

---

## v1.3 — forward test and live forecast (before any 2026 result)

- **New cutoffs:** extra training (2025-11, 12, 2026-02, 03), forward test
  (2026-04 → 07), live (2026-10-01, labels empty).
- **Forward test (one shot, lock file):** the frozen Test 2 models, no
  retraining. "Supports" = B − A has the test's sign in ≥ 3 of 4 cutoffs **and**
  B beats the best baseline in every cutoff.
- **Live forecast:** same settings, retrained on every cutoff whose label
  closed before 2026-10-01; level calibration only; scores written to a
  write-once table and sealed with SHA-256 before any outcome is known.

## v1.4 — behaviour source switch (before live scoring)

The identity-matched behaviour table ended before the live cutoff; the last
days were rebuilt from raw GA4 events with the same time rule. Validation
conditions: historical rows unchanged; identity accuracy ≈ 1; no break at the
seam. The seam recognised noticeably more logged-in activity than the old
source; scoring continued and this was declared a limitation of the live run.

## v1.5 — read-out criteria (before the interim read)

- **Interim read (first month of the live window, informational):** capture@10
  among customers whose first order fell in that month, compared with the same
  measurement one year earlier. Criterion 1: live B ≥ last year's B − 3 points.
  Criterion 2: live B − A > 0.
- **Final read (after the 90-day window):** B − A ≥ 1 point with a bootstrap CI
  excluding 0, and B within 3 points of its test value.
