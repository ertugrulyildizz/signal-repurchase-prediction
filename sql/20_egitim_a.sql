-- =====================================================================
-- Sinyal | sql/20_egitim_a.sql   (v3 — 2026-10-17: ikinci test kesimi 2026-01-01)
-- Amaç : Kesim tarihli eğitim tablosu. Her satır = müşteri x kesim.
--        Model A özellikleri (yalnızca sipariş + müşteri) ve hedefler.
-- Tanımlar: docs/preregistration.md
--   Kesimler : 2024-10-01 → 2025-10-01 aylık (13 kesim) + 2026-01-01
--              egitim ≤ 2025-07-01 · tampon 2025-08, 2025-09 · test 2025-10-01
--              test2 2026-01-01 (sağlamlık; eğitimi ≤ 2025-10-01 kesimlerle)
--   Popülasyon: kesimden önceki 365 günde ≥ 1 temiz sipariş
--   Etiket    : [kesim, kesim + 90 gün) içinde ≥ 1 temiz sipariş
-- v2 değişiklikleri:
--   * Erken dönemde tutar verisi eksik → PARASAL özellikler 270 günlük
--     pencereden. Sayım/yakınlık 12 ay.
--   * Kategori payları ADET üzerinden.
--   * brut_marj_orani çıkarıldı (maliyet dönemler arası tutarsız).
--   * Ciro = ortalama sepet × sipariş sayısı (tutarı boş siparişler
--     ortalamayla temsil edilir; kısmi toplam yok).
--   * Tutarlar nominal TL (enflasyon). Python'da kesim medyanına bölünerek
--     endekslenecek; ham TL değerleri modele girmeyecek.
-- SIZINTI KURALI: Her özellik yalnızca order_date < kesim olan kayıtlardan.
--   "y_" ile başlayan kolonlar HEDEFTİR, özellik olarak kullanılmaz.
-- =====================================================================

CREATE OR REPLACE TABLE `PROJE_ID.sinyal.egitim_a`
CLUSTER BY kesim, musteri_id
OPTIONS (description = 'Sinyal v2: müşteri x kesim; Model A özellikleri + hedefler (y_*)')
AS
WITH kesimler AS (
  SELECT k AS kesim
  FROM UNNEST(ARRAY_CONCAT(
         GENERATE_DATE_ARRAY('2024-10-01', '2025-10-01', INTERVAL 1 MONTH),
         [DATE '2026-01-01'])) AS k
),
s AS (
  SELECT musteri_id, order_id, order_date, platform,
         net_tutar - kdv AS tutar,          -- KDV hariç, kargo hariç; boşsa NULL
         net_tutar, indirim, adet, kalem_sayisi, indirim_derinligi
  FROM `PROJE_ID.sinyal.v_siparis`
),

-- 1) Sipariş düzeyi özellikler
o AS (
  SELECT
    kz.kesim, s.musteri_id,
    -- Sayım ve yakınlık: 12 ay (sipariş tarihleri her dönemde tam)
    COUNT(*)                                                          AS siparis_12ay,
    COUNTIF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 30 DAY))      AS siparis_30g,
    COUNTIF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 90 DAY))      AS siparis_90g,
    COUNTIF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 180 DAY))     AS siparis_180g,
    COUNTIF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 270 DAY))     AS siparis_9ay,
    DATE_DIFF(kz.kesim, MAX(s.order_date), DAY)                       AS gun_son_siparis,
    DATE_DIFF(kz.kesim, MIN(s.order_date), DAY)                       AS gun_ilk_siparis_12ay,
    SAFE_DIVIDE(DATE_DIFF(MAX(s.order_date), MIN(s.order_date), DAY),
                COUNT(*) - 1)                                         AS ort_siparis_arasi_gun,
    SUM(s.adet)                                                       AS adet_12ay,
    AVG(s.kalem_sayisi)                                               AS ort_kalem,
    AVG(IF(s.platform = 'App',   1, 0))                               AS pay_app,
    AVG(IF(s.platform = 'Mobil', 1, 0))                               AS pay_mobil_web,
    AVG(IF(s.platform = 'Web',   1, 0))                               AS pay_masaustu,
    -- Parasal: 270 gün
    AVG(IF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 270 DAY), s.tutar, NULL)) AS ort_sepet_9ay,
    AVG(IF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 90 DAY),  s.tutar, NULL)) AS ort_sepet_90g,
    SAFE_DIVIDE(
      SUM(IF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 270 DAY) AND s.net_tutar IS NOT NULL,
             s.indirim, NULL)),
      SUM(IF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 270 DAY) AND s.net_tutar IS NOT NULL,
             s.net_tutar + s.indirim, NULL)))                          AS indirim_derinligi_9ay,
    AVG(IF(s.order_date >= DATE_SUB(kz.kesim, INTERVAL 270 DAY),
           IF(s.tutar IS NULL, 1, 0), NULL))                          AS tutar_eksik_oran_9ay,
    ARRAY_AGG(STRUCT(s.indirim_derinligi AS indirim_derinligi,
                     s.platform          AS platform,
                     s.tutar             AS tutar)
              ORDER BY s.order_date DESC, s.order_id DESC LIMIT 1)[OFFSET(0)] AS son
  FROM kesimler kz
  JOIN s ON s.order_date >= DATE_SUB(kz.kesim, INTERVAL 365 DAY)
        AND s.order_date <  kz.kesim
  GROUP BY kz.kesim, s.musteri_id
),

-- 2) Kalem düzeyi: kategori payları (adet), fiyat bandı (270 gün)
k AS (
  SELECT
    kz.kesim, kl.musteri_id,
    SAFE_DIVIDE(SUM(IF(kl.category = 'TEKSTIL',  kl.adet, 0)), SUM(kl.adet)) AS pay_tekstil,
    SAFE_DIVIDE(SUM(IF(kl.category = 'AKSESUAR', kl.adet, 0)), SUM(kl.adet)) AS pay_aksesuar,
    SAFE_DIVIDE(SUM(IF(kl.category = 'AYAKKABI', kl.adet, 0)), SUM(kl.adet)) AS pay_ayakkabi,
    COUNT(DISTINCT kl.category2)                                              AS farkli_kategori2,
    SAFE_DIVIDE(
      SUM(IF(kl.order_date >= DATE_SUB(kz.kesim, INTERVAL 270 DAY) AND kl.net_tutar IS NOT NULL,
             kl.net_tutar - kl.kdv, NULL)),
      SUM(IF(kl.order_date >= DATE_SUB(kz.kesim, INTERVAL 270 DAY) AND kl.net_tutar IS NOT NULL,
             kl.adet, NULL)))                                                 AS ort_birim_fiyat_9ay
  FROM kesimler kz
  JOIN `PROJE_ID.sinyal.v_kalem` kl
    ON kl.order_date >= DATE_SUB(kz.kesim, INTERVAL 365 DAY)
   AND kl.order_date <  kz.kesim
  GROUP BY kz.kesim, kl.musteri_id
),

-- 3) İlk sipariş (kıdem için; popülasyondaki herkes için < kesim)
ilk AS (
  SELECT musteri_id, MIN(order_date) AS ilk_siparis
  FROM `PROJE_ID.sinyal.v_siparis`
  GROUP BY musteri_id
),

-- 4) HEDEF: kesimden sonraki 90 gün
lab AS (
  SELECT kz.kesim, s.musteri_id,
         COUNT(*)     AS y_siparis_90g,
         SUM(s.tutar) AS y_ciro_90g
  FROM kesimler kz
  JOIN s ON s.order_date >= kz.kesim
        AND s.order_date <  DATE_ADD(kz.kesim, INTERVAL 90 DAY)
  GROUP BY kz.kesim, s.musteri_id
)

SELECT
  o.kesim,
  o.musteri_id,
  CASE WHEN o.kesim <= DATE '2025-07-01' THEN 'egitim'
       WHEN o.kesim =  DATE '2025-10-01' THEN 'test'
       WHEN o.kesim =  DATE '2026-01-01' THEN 'test2'
       ELSE 'tampon' END                                   AS kume,
  EXTRACT(MONTH FROM o.kesim)                              AS kesim_ay,   -- yalnızca analiz için; modele girmeyecek
  o.* EXCEPT (kesim, musteri_id, son),
  o.ort_sepet_9ay * o.siparis_9ay                          AS ciro_9ay,
  IF(o.siparis_90g = 0, 0, o.ort_sepet_90g * o.siparis_90g) AS ciro_90g,
  o.son.indirim_derinligi                                  AS son_indirim_derinligi,
  o.son.platform                                           AS son_platform,
  o.son.tutar                                              AS son_sepet,
  k.* EXCEPT (kesim, musteri_id),
  DATE_DIFF(o.kesim, LEAST(IFNULL(m.uyelik_tarihi, i.ilk_siparis), i.ilk_siparis), DAY) AS uyelik_gun,
  CASE WHEN m.dogum_tarihi BETWEEN DATE '1930-01-01' AND DATE '2010-12-31'
       THEN DATE_DIFF(o.kesim, m.dogum_tarihi, YEAR) END   AS yas,
  -- ---------------- HEDEF (özellik DEĞİL) ----------------
  IFNULL(l.y_siparis_90g, 0) > 0                           AS y_etiket,
  IFNULL(l.y_siparis_90g, 0)                               AS y_siparis_90g,
  IFNULL(l.y_ciro_90g, 0)                                  AS y_ciro_90g
FROM o
LEFT JOIN k                                          ON k.kesim = o.kesim AND k.musteri_id = o.musteri_id
LEFT JOIN ilk i                                      ON i.musteri_id = o.musteri_id
LEFT JOIN `PROJE_ID.sinyal.v_musteri` m ON m.musteri_id = o.musteri_id
LEFT JOIN lab l                                      ON l.kesim = o.kesim AND l.musteri_id = o.musteri_id;


-- [K8 v2] Tutarlılık, eksiklik ve enflasyon kayması ---------------------
-- Beklenen: satir/etiket_orani önceki K8 ile aynı; sepet_eksik tüm
-- kesimlerde düşük ve benzer; medyan sepet zamanla artar (enflasyon).
SELECT
  kesim, kume,
  COUNT(*)                                                       AS satir,
  ROUND(AVG(IF(y_etiket, 1, 0)), 4)                              AS etiket_orani,
  MIN(gun_son_siparis)                                           AS min_gun_son,
  MAX(gun_son_siparis)                                           AS max_gun_son,
  ROUND(SAFE_DIVIDE(COUNTIF(siparis_9ay > 0 AND ort_sepet_9ay IS NULL),
                    COUNTIF(siparis_9ay > 0)), 3)                AS sepet_eksik,
  ROUND(AVG(tutar_eksik_oran_9ay), 3)                            AS ort_tutar_eksik_oran,
  ROUND(AVG(IF(pay_tekstil IS NULL, 1, 0)), 3)                   AS kategori_eksik,
  ROUND(AVG(IF(siparis_9ay = 0, 1, 0)), 3)                       AS son_9ayda_siparissiz,
  ROUND(APPROX_QUANTILES(ort_sepet_9ay, 2)[OFFSET(1)], 0)        AS medyan_sepet_tl,
  ROUND(APPROX_QUANTILES(ort_birim_fiyat_9ay, 2)[OFFSET(1)], 0)  AS medyan_birim_fiyat_tl,
  ROUND(AVG(indirim_derinligi_9ay), 3)                           AS ort_indirim_derinligi
FROM `PROJE_ID.sinyal.egitim_a`
GROUP BY kesim, kume
ORDER BY kesim;
