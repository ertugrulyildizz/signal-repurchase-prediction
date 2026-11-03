-- =====================================================================
-- Sinyal | sql/26_segment_listesi.sql   (2026-11-02)
-- Amaç : Canlı skorlardan (kesim 2026-10-01) ilk %10 / %10–20 dilim listesi,
--        olasılık bandı, Ekim'de zaten alanlar ve %15 rastgele kontrol grubu.
-- Mühürlü tabloya (canli_skor_20261001) yalnızca OKUMA yapılır; yeni tablo yazılır.
-- Ön koşul: sql/24 (ilk_alim) çalışmış olmalı.
-- =====================================================================

CREATE OR REPLACE TABLE `PROJE_ID.sinyal.segment_canli_20261001`
OPTIONS (description = 'Sinyal: canlı skor dilimleri + %15 kontrol grubu (2026-11-02)')
AS
WITH s AS (
  SELECT musteri_id, skor_B, p_kalibre,
         PERCENT_RANK() OVER (ORDER BY skor_B DESC) AS sira_yuzde      -- 0 = en olası müşteri
  FROM `PROJE_ID.sinyal.canli_skor_20261001`
),
ekim AS (
  SELECT musteri_id, ilk_alim_tarihi
  FROM `PROJE_ID.sinyal.ilk_alim`
  WHERE kesim = DATE '2026-10-01'
)
SELECT
  s.musteri_id, s.skor_B, s.p_kalibre,
  CASE WHEN s.sira_yuzde < 0.10 THEN '1_ilk_10'
       WHEN s.sira_yuzde < 0.20 THEN '2_yuzde_10_20'
       ELSE '3_diger' END                                              AS dilim,
  CASE WHEN s.p_kalibre >= 0.30 THEN 'yuksek'
       WHEN s.p_kalibre >= 0.10 THEN 'orta'
       ELSE 'dusuk' END                                                AS bant,
  e.ilk_alim_tarihi IS NOT NULL
    AND e.ilk_alim_tarihi < DATE '2026-11-01'                          AS ekimde_aldi,
  -- Deterministik rastgele %15 kontrol: aynı müşteri her çalıştırmada aynı grupta kalır.
  IF(MOD(ABS(FARM_FINGERPRINT(CONCAT('sinyal-pilot-2026-11|', CAST(s.musteri_id AS STRING)))), 100) < 15,
     'kontrol', 'hedef')                                               AS grup
FROM s
LEFT JOIN ekim e USING (musteri_id);

-- [K1] Dilim × grup sayıları (ilk_10 ≈ müşterilerin %10'u; kontrol her dilimde ≈ %15 olmalı)
SELECT dilim, grup, COUNT(*) AS musteri,
       ROUND(AVG(p_kalibre), 3) AS ort_olasilik,
       ROUND(AVG(IF(ekimde_aldi, 1, 0)), 3) AS ekimde_alan_payi
FROM `PROJE_ID.sinyal.segment_canli_20261001`
GROUP BY dilim, grup ORDER BY dilim, grup;

-- [K2] Tutarlılık: canlı tablodaki ust_yuzde10 bayrağıyla örtüşme (≈ %100 olmalı)
SELECT COUNTIF(sg.dilim = '1_ilk_10' AND CAST(c.ust_yuzde10 AS BOOL)) / COUNTIF(sg.dilim = '1_ilk_10') AS ortusme
FROM `PROJE_ID.sinyal.segment_canli_20261001` sg
JOIN `PROJE_ID.sinyal.canli_skor_20261001` c USING (musteri_id);
