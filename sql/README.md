# SQL templates (BigQuery)

These are the feature/label queries from the original project, with the
project and dataset names replaced by `PROJE_ID.sinyal`. They are **not needed
for the synthetic run** — `python -m sentetik.uret` produces the same tables
directly. Use them if you want to rebuild the pipeline on your own GA4 +
order data.

| File | Builds | Key idea |
|---|---|---|
| `20_egitim_a.sql` | `egitim_a` | One row per customer × monthly cutoff. Every feature uses only `order_date < kesim`; label = ≥ 1 order in `[kesim, kesim + 90 days)`. Monetary features from a 270-day window, indexed later in Python. |
| `22_egitim_b.sql` | `egitim_b` | 7/30-day web & app behaviour before the cutoff. **Time rule:** a device→customer link is used only if its evidence (login or purchase) happened *before* the cutoff. Without it, identity resolution leaks future purchases into past behaviour. |
| `24_ara_okuma.sql` | `ilk_alim` | First order date after a cutoff, for the interim read of a live forecast. |
| `26_segment_listesi.sql` | `segment_canli_…` | Turns sealed live scores into top-10 % / 10–20 % segments with a deterministic random 15 % holdout. |

## Expected input views

Column names are what the templates use; map your own tables onto them.

| View | Columns |
|---|---|
| `v_siparis` | `musteri_id, order_id, order_date, platform ('App'/'Mobil'/'Web'), net_tutar, kdv, indirim, adet, kalem_sayisi, indirim_derinligi` |
| `v_kalem` | `musteri_id, order_date, category ('TEKSTIL'/'AKSESUAR'/'AYAKKABI'), category2, adet, net_tutar, kdv` |
| `v_musteri` | `musteri_id, uyelik_tarihi, dogum_tarihi` |
| `ga4_gunluk_eslesik` | `user_pseudo_id, musteri_id, tarih, platform, urun_goruntuleme, liste_goruntuleme, arama, sepete_ekleme, favori_ekleme, odeme_baslatma, toplam_event` (daily per device) |
| `kimlik_baglanti` | `user_pseudo_id, musteri_id, ilk_giris_kaniti, ilk_satin_alma_kaniti` (timestamps of the first login / purchase that link a device to a customer) |

Cleaning rules (which orders count, which accounts are excluded) live in the
views and are business-specific; they are not part of this repository.
