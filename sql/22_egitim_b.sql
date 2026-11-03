-- =====================================================================
-- Sinyal | sql/22_egitim_b.sql
-- Amaç : Model B davranış özellikleri. Her satır = egitim_a'daki
--        (kesim, musteri_id). Kaynak: ga4_gunluk_eslesik + kimlik_baglanti.
-- ZAMAN KURALI: Bir (pseudo_id, musteri_id)
--   bağı yalnızca kanıtı (giriş ya da satın alma) kesimden ÖNCE
--   oluşmuşsa kullanılır. Pencere: kesimden önceki 30 gün (ve 7 gün).
-- Satın alma event'leri özellik DEĞİL (sipariş bilgisini tekrarlar).
-- Python'da egitim_a ile (kesim, musteri_id) üzerinden birleştirilir.
-- =====================================================================

CREATE OR REPLACE TABLE `PROJE_ID.sinyal.egitim_b`
CLUSTER BY kesim, musteri_id
OPTIONS (description = 'Sinyal: müşteri x kesim; Model B davranış özellikleri (zaman kurallı, 30 gün)')
AS
WITH e AS (
  SELECT kesim, musteri_id
  FROM `PROJE_ID.sinyal.egitim_a`
),
d AS (   -- kurala uyan, kesim öncesi 30 günlük davranış satırları
  SELECT e.kesim, e.musteri_id, ga.tarih, ga.platform,
         ga.urun_goruntuleme, ga.liste_goruntuleme, ga.arama,
         ga.sepete_ekleme, ga.favori_ekleme, ga.odeme_baslatma, ga.toplam_event
  FROM e
  JOIN `PROJE_ID.sinyal.ga4_gunluk_eslesik` ga
    ON ga.musteri_id = e.musteri_id
   AND ga.tarih <  e.kesim
   AND ga.tarih >= DATE_SUB(e.kesim, INTERVAL 30 DAY)
  JOIN `PROJE_ID.sinyal.kimlik_baglanti` b
    ON b.user_pseudo_id = ga.user_pseudo_id
   AND b.musteri_id        = ga.musteri_id
  WHERE LEAST(IFNULL(b.ilk_giris_kaniti,      TIMESTAMP '2100-01-01'),
              IFNULL(b.ilk_satin_alma_kaniti, TIMESTAMP '2100-01-01'))
        < TIMESTAMP(e.kesim, 'Europe/Istanbul')
),
a AS (
  SELECT
    d.kesim, d.musteri_id,
    COUNT(DISTINCT d.tarih)                                                            AS aktif_gun_30g,
    COUNT(DISTINCT IF(d.tarih >= DATE_SUB(d.kesim, INTERVAL 7 DAY), d.tarih, NULL))    AS aktif_gun_7g,
    DATE_DIFF(d.kesim, MAX(d.tarih), DAY)                                              AS gun_son_aktivite,
    SUM(d.urun_goruntuleme)                                                            AS urun_gor_30g,
    SUM(IF(d.tarih >= DATE_SUB(d.kesim, INTERVAL 7 DAY), d.urun_goruntuleme, 0))       AS urun_gor_7g,
    SUM(d.liste_goruntuleme)                                                           AS liste_gor_30g,
    SUM(d.arama)                                                                       AS arama_30g,
    SUM(d.sepete_ekleme)                                                               AS sepete_ekle_30g,
    SUM(IF(d.tarih >= DATE_SUB(d.kesim, INTERVAL 7 DAY), d.sepete_ekleme, 0))          AS sepete_ekle_7g,
    SUM(d.favori_ekleme)                                                               AS favori_30g,
    SUM(d.odeme_baslatma)                                                              AS odeme_baslat_30g,
    SUM(d.toplam_event)                                                                AS toplam_event_30g,
    SAFE_DIVIDE(COUNT(DISTINCT IF(UPPER(d.platform) LIKE '%WEB%', d.tarih, NULL)),
                COUNT(DISTINCT d.tarih))                                               AS web_gun_payi
  FROM d
  GROUP BY d.kesim, d.musteri_id
)
SELECT
  e.kesim,
  e.musteri_id,
  a.musteri_id IS NOT NULL                 AS davranis_var_30g,     -- "A + bayrak" kolunun bayrağı
  IFNULL(a.aktif_gun_7g, 0)             AS aktif_gun_7g,
  IFNULL(a.aktif_gun_30g, 0)            AS aktif_gun_30g,
  a.gun_son_aktivite,                                            -- davranış yoksa NULL
  IFNULL(a.urun_gor_7g, 0)              AS urun_gor_7g,
  IFNULL(a.urun_gor_30g, 0)             AS urun_gor_30g,
  IFNULL(a.liste_gor_30g, 0)            AS liste_gor_30g,
  IFNULL(a.arama_30g, 0)                AS arama_30g,
  IFNULL(a.sepete_ekle_7g, 0)           AS sepete_ekle_7g,
  IFNULL(a.sepete_ekle_30g, 0)          AS sepete_ekle_30g,
  IFNULL(a.favori_30g, 0)               AS favori_30g,
  IFNULL(a.odeme_baslat_30g, 0)         AS odeme_baslat_30g,
  IFNULL(a.toplam_event_30g, 0)         AS toplam_event_30g,
  a.web_gun_payi                                                 -- davranış yoksa NULL
FROM e
LEFT JOIN a ON a.kesim = e.kesim AND a.musteri_id = e.musteri_id;


-- [K11] Kontrol: kapsam K10'daki "kapsam_kural" ile aynı olmalı -------------
SELECT
  kesim,
  COUNT(*)                                                         AS satir,
  ROUND(AVG(IF(davranis_var_30g, 1, 0)), 3)                        AS kapsam,
  ROUND(AVG(IF(davranis_var_30g, aktif_gun_30g, NULL)), 2)         AS ort_aktif_gun_aktiflerde,
  APPROX_QUANTILES(IF(davranis_var_30g, urun_gor_30g, NULL), 2)[OFFSET(1)] AS medyan_urun_gor_aktiflerde,
  ROUND(AVG(IF(davranis_var_30g, IF(sepete_ekle_30g > 0, 1, 0), NULL)), 3) AS sepete_ekleyen_pay_aktiflerde,
  ROUND(AVG(IF(davranis_var_30g, web_gun_payi, NULL)), 3)          AS ort_web_gun_payi
FROM `PROJE_ID.sinyal.egitim_b`
GROUP BY kesim
ORDER BY kesim;
