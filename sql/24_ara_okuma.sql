-- =====================================================================
-- Sinyal | sql/24_ara_okuma.sql   (tanimlar v1.3 — 2026-10-27)
-- Amaç : Canlı kesimin (2026-10-01) ARA OKUMASI ve kıyası için kesim sonrası
--        ilk temiz sipariş tarihi. Kıyas kesimi: 2025-10-01 (test).
-- NE ZAMAN: 2026-11-01 ya da 02 (Ekim 2026 tamamlandıktan sonra). Aralık
--           sonunda yeniden çalıştırılırsa kesin okuma için de kullanılır.
-- Ön koşul: egitim_a_ileri (20_egitim_a.sql ile aynı mantık, ileri kesimler için).
-- =====================================================================

CREATE OR REPLACE TABLE `PROJE_ID.sinyal.ilk_alim`
CLUSTER BY kesim, musteri_id
OPTIONS (description = 'Sinyal v1.3: kesim sonrası 90 gün içindeki ilk temiz sipariş tarihi (2025-10-01 kıyas, 2026-10-01 canlı)')
AS
WITH pop AS (
  SELECT kesim, musteri_id FROM `PROJE_ID.sinyal.egitim_a`       WHERE kesim = DATE '2025-10-01'
  UNION ALL
  SELECT kesim, musteri_id FROM `PROJE_ID.sinyal.egitim_a_ileri` WHERE kesim = DATE '2026-10-01'
)
SELECT p.kesim, p.musteri_id, MIN(s.order_date) AS ilk_alim_tarihi
FROM pop p
LEFT JOIN `PROJE_ID.sinyal.v_siparis` s
  ON s.musteri_id = p.musteri_id
 AND s.order_date >= p.kesim
 AND s.order_date <  DATE_ADD(p.kesim, INTERVAL 90 DAY)
GROUP BY p.kesim, p.musteri_id;

-- [K] Kontrol: 2025-10-01 için 90 günlük alım oranı egitim_a'daki etiket oranıyla aynı olmalı
SELECT kesim, COUNT(*) AS satir,
       ROUND(AVG(IF(ilk_alim_tarihi IS NOT NULL, 1, 0)), 4)                                      AS oran_90g_simdiye_kadar,
       ROUND(AVG(IF(ilk_alim_tarihi < DATE_ADD(kesim, INTERVAL 1 MONTH), 1, 0)), 4)            AS oran_ilk_ay,
       MAX(ilk_alim_tarihi)                                                                      AS son_alim_tarihi
FROM `PROJE_ID.sinyal.ilk_alim`
GROUP BY kesim ORDER BY kesim;
