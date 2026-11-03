# Signal (Sinyal) — Does web & app behaviour improve repurchase prediction?

> **🇹🇷 Kısa özet.** Sinyal (Signal), bir online moda perakendecisinin müşterilerinin önümüzdeki
> 90 günde tekrar alım yapıp yapmayacağını tahmin eden bir makine öğrenmesi projesidir.
> Asıl soru "en iyi model hangisi?" değil, **"site ve uygulama davranışı, sipariş
> geçmişine ne katar?"** sorusudur. Bunu kontrollü bir deneyle (A: sipariş geçmişi,
> B: + davranış) ve önceden kayda geçirilmiş başarı ölçütleriyle sınar. Zamana göre
> bölme, sızıntıya karşı zaman kuralı, tek seferlik test, eşleştirilmiş bootstrap,
> kalibrasyon, SHAP, ileri test ve mühürlü canlı tahmin projenin yöntemsel
> omurgasıdır. **Bu depodaki veri tamamen sentetiktir**; gerçek şirket verisi, sonuç
> sayıları ve şirkete özgü bilgiler çıkarılmıştır. Kod Türkçe adlandırmalı; belgeler
> İngilizce.

Workintech Data Science & AI capstone project. The original work was done on real
data from an omnichannel women's fashion retailer (online customers only). **This
repository contains only the method and code, running on fully synthetic data.**
No company data, no real results and no company-specific identifiers are included.

---

## The question

At the start of every month, for every customer who ordered online in the last
12 months:

> *What is the probability that this customer places at least one online order in
> the next 90 days — and does the last 30 days of web/app behaviour improve that
> forecast beyond what the order history already says?*

This is framed as a **controlled experiment**, not a model contest:

| Arm | Features | Purpose |
|---|---|---|
| **A** | 24 order-history & customer features | What can be done with data that already exists |
| **A + flag** | A + "has any behaviour data in the last 30 days" | Separates *having* behaviour data from its *content* |
| **B** | A + 14 behaviour features (38) | The treatment |

Same model, same settings, same customers, same dates — only the feature list changes.

## What the original project found (qualitatively)

- Behaviour data gave a **material improvement** that met all three pre-registered
  criteria (minimum effect size, bootstrap CI excluding zero, consistency across
  validation folds). The flag alone added little: the gain came from the *content*
  of behaviour.
- The gain was **largest for single-order and long-lapsed customers**, whose order
  history says little — the pre-registered secondary hypothesis held.
- A logistic regression showed the same B − A difference as LightGBM: the finding
  does not depend on the model family.
- A model frozen at the test date kept the same advantage on four later months it
  had never seen (forward test).
- A live forecast was sealed (SHA-256, write-once table) before its outcome existed;
  the interim read met its pre-registered criteria.

Actual numbers are confidential and are not published here.

## Methods worth stealing

| Problem | What this project does | Where |
|---|---|---|
| Random splits leak the future when customers repeat across months | Monthly cutoffs; train → buffer → test in time order; **gap rule** `cutoff + 90 days ≤ target` | `src/degerlendirme.py` (`kayan_katlar`, `hedef_egitim_indeksleri`) |
| Identity resolution leaks purchases into past behaviour (a device is linked to a customer *because* they bought later) | **Time rule:** a device→customer link counts only if its evidence (login or purchase) happened before the cutoff | `sql/22_egitim_b.sql` |
| "We tried 20 things and reported the best" | **Pre-registration**: hypotheses, metrics and the definition of "material" written and dated before results; versions appended, never edited | `docs/preregistration.md` |
| Peeking at the test set | Dry run on proxy targets, then the test is opened **once**; a lock file blocks reruns | `notebooks/03_test.ipynb` |
| Is the difference real? | **Paired, customer-level bootstrap** (1,000 resamples) + sign consistency across 4 rolling folds + a difference-of-differences test for the subgroup hypothesis | `src/degerlendirme.py` |
| Behaviour levels drift with season and tracking coverage | Behaviour counts → **percentile rank within the cutoff, among active customers only** | `src/ozellikler.py` (`goreli_davranis`) |
| Inflation in monetary features | Divide by the cutoff's median unit price (index) | `src/ozellikler.py` (`endeksle`) |
| Ranking is fine but probabilities drift by season | Isotonic + **level shift** to last year's same-month rate | `src/kalibrasyon.py` |
| Leakage regressions creep back in | 34 `pytest` tests (label windows, target columns, raw currency, feature sets, a model-training smoke test…) | `tests/` |
| "Did you change the forecast after seeing the outcome?" | Live scores sealed with SHA-256; verification script | `notebooks/06_…`, `scripts/07_ara_okuma.py` |
| Accuracy is meaningless at ~15 % positives | Business metric **capture@10** (share of all buyers in the top-scored 10 %), primary metric PR-AUC; ROC-AUC reported only | `src/degerlendirme.py` |

Baselines: recency rule, RFM, BG/NBD (`lifetimes`). Models: LightGBM (main) and
logistic regression (reference). Explanation: LightGBM's built-in TreeSHAP
(`pred_contrib=True`). Business layer: probability × value action matrix,
scenario profit curves and a control-group power analysis — explicitly labelled as
hypotheses, because the model predicts *who will buy*, not *whom contact will change*.

## Quickstart

Python 3.10 or 3.11. Run every command **inside the repository folder**
(`signal-repurchase-prediction`); the tests rely on `pytest.ini` there.

**Windows (PowerShell)**

```powershell
git clone https://github.com/ertugrulyildizz/signal-repurchase-prediction.git
cd signal-repurchase-prediction
py -3.10 -m venv .venv               # or: python -m venv .venv
.venv\Scripts\Activate.ps1           # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt

python -m sentetik.uret              # synthetic data → data\*.parquet (~10 s, 60k customers)
pytest -q                            # expect: 34 passed
```

**macOS / Linux**

```bash
git clone https://github.com/ertugrulyildizz/signal-repurchase-prediction.git
cd signal-repurchase-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m sentetik.uret              # synthetic data → data/*.parquet (~10 s, 60k customers)
pytest -q                            # expect: 34 passed
```

`ModuleNotFoundError: No module named 'src'` means you are one folder too high: `cd` into
`signal-repurchase-prediction` and run again.

Then run the notebooks **in order** (`jupyter lab`, open `notebooks/`):

| # | Notebook | What it does | Time* |
|---|---|---|---|
| 01 | `01_eda.ipynb` | EDA on training cutoffs only; correlation pruning; drift checks | < 1 min |
| 02 | `02_bazlar_ve_deney.ipynb` | Baselines + experiment on 4 rolling folds; transform choice. Set `HIZLI = False` for the full run | ~2 min |
| 03 | `03_test.ipynb` | **One-shot test.** Run once with `KURU_CALISMA = True` (dry run), then set it to `False`, restart, Run All. Writes a lock file | ~2 min |
| 04 | `04_kalibrasyon_ve_aciklama.ipynb` | Calibration, cumulative gain, subgroup forest plot, SHAP | ~1.5 min |
| 05 | `05_aksiyon_ve_kar.ipynb` | Action matrix, profit-curve scenarios, control-group sizing | < 1 min |
| 06 | `06_ileri_ve_canli.ipynb` | Forward test (one shot, lock file) + live forecast with SHA-256 seal | ~1 min |
| 07 | `python scripts/07_ara_okuma.py` | Verifies the seal, then the interim read against pre-registered criteria | seconds |

\* On a laptop with the default synthetic size.

To start over, delete `docs/sonuclar/*` and `data/modeller_03/` (the lock files live there).

## Synthetic data

`sentetik/uret.py` simulates customers with a latent purchase rate and churn
probability (BG/NBD-like), seasonal Poisson orders with inflation and discounts,
and web/app visits made of general interest plus **intent visits before purchases**.
Visits are attributed to a customer only after their identity link exists, and
tracking coverage and channel mix drift over time — so the time rule, the
within-cutoff percentile transform and the calibration step all have something
real to do.

The generator reproduces the **schema** of the original tables, not their
distributions. Effect sizes, rates and every number you will see are artefacts of
the simulation. Do not read business conclusions into them.

## Using your own data (BigQuery)

`sql/` contains the original feature/label queries as templates with placeholders
(`PROJE_ID.sinyal`) and documents the input views they expect (`sql/README.md`).
After building the tables:

```bash
pip install -r requirements-bigquery.txt
gcloud auth application-default login
export SINYAL_VERI_KAYNAGI=bigquery SINYAL_BQ_PROJE=<your-project> SINYAL_BQ_VERI_SETI=<your-dataset>
```

`src.veri.tablo_yukle()` then reads from BigQuery and caches to `data/`, which is
git-ignored. Row-level data should never be committed.

## Repository layout

```
sentetik/      synthetic data generator
src/           features, models & baselines, evaluation (metrics, folds, bootstrap), calibration, data access
tests/         leakage and consistency tests
notebooks/     01–06, run in order
scripts/       interim read with seal verification
sql/           BigQuery templates for the original pipeline
docs/          pre-registration (definitions, versioned); sonuclar/ = generated results (git-ignored)
```

Identifiers are in Turkish (e.g. `kesim` = cutoff, `etiket` = label, `ozellik` =
feature, `musteri_id` = customer id, `davranis` = behaviour, `capture@10`).

## Limitations

- No returns data: the label is gross purchase intent.
- Online customers only; marketplace purchases are invisible.
- Bootstrap covers customer sampling, not model seed variance; no hyperparameter
  search (A and B compared with identical defaults by design); no permutation test.
- The model is not causal. Measuring the effect of contacting customers needs a
  randomised control group, which the project designs but does not yet have.

## License

No open-source license: all rights reserved. The code is shared for portfolio and
educational viewing; please ask before reusing it.

Author: **Ertuğrul Yıldız**
