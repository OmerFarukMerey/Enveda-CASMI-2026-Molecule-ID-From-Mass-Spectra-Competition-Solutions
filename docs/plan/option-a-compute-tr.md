<!-- index: 2026-11-18 (TR) A seçeneği (yapı üretici) için hesaplama raporu: Kaggle ve Runpod gereklilikleri, donanım başına süre ve maliyet, öneri -->
# A seçeneği için hesaplama raporu: Kaggle ve Runpod (2026-11-18)

> **Güncelleme (aynı gün): GPU harcamadan önce okuyun.** Bu fikir repoda daha önce kurulup kapatılmış:
> `solutions/fusion-v4n/reviews/fpgen-2026-10-12.md` (parmak izi + formül koşullu üretici, kod
> `solutions/fusion-v4n/train/fpgen/`; tahmin edilen parmak iziyle ilk 25'te %10,6 / %0,0 / %4,0, gerçek parmak
> iziyle bile %32,7 / %0,3 / %15,3; karar "bu hattı kapat") ve `frigidpilot-2026-10-17.md`. Darboğaz üretici değil,
> parmak izi tahmininin kalitesi: literatürde üretim, tahmin-gerçek Tanimoto 0,5'in altında umutsuz, ancak
> 0,75'in üstünde işe yarıyor; bizim modellerimiz 0,22-0,41 aralığında. Yeniden deneme bir "kodlayıcı" (spektrum
> → parmak izi) projesi olur; tahmini başarı olasılığı ~%5. Aşağıdaki maliyet tablosu geçerli, ama eğitime ancak
> ücretsiz ilk iki adım (parmak izi kalitesi histogramı, gerçek parmak iziyle üst sınır) geçerse başlanmalı.

Runpod fiyatları ve stok durumu 2026-11-18 sabahı canlı okundu (saatlik, USD). Hız katsayıları ve iş yükü
büyüklükleri tahmindir; fizibilite ölçümü ve mimari seçimi bitince netleşir.

## İş yükü (T4-saat cinsinden, tahmin)

| Kalem | Küçük | Orta | Büyük |
|---|---|---|---|
| Üretici ön eğitimi (yapı verisi, gerçek parmak izi) | 20 | 40 | 80 |
| Tahmin edilen parmak izine uyarlama + denemeler | 8 | 20 | 40 |
| Sınama çıkarımı (500 molekül x 256 örnek, birkaç kez) | 2 | 5 | 10 |
| Boru hattı içinde düzenek çalıştırmaları (yalnızca Kaggle) | 10 | 15 | 20 |
| Toplam | 40 | 80 | 150 |

Repodaki önceki tahmin (research-2026-11-13, A4): eğitim için 20-40 Kaggle GPU saati.

## Kaggle

- Donanım: T4 x2 (2 x 16 GB), 4 vCPU, ~30 GB RAM. P100 yarışmada emekli.
- Ücret: yok. Sınır: hesap başına haftalık GPU kotası (genelde ~30 saat; bizimkini API'den okuyamadık,
  kaggle.com/settings sayfasında görünür). Oturum en fazla 12 saat; yarışma not defteri 9 saat.
- Bu hafta kullanılan: düzenek için ~8,5 saat.
- Gözlenen kuyruk: kernel başına 20-90 dakika. Çalışırken log okunmuyor; çıktı ancak bitince iniyor.
- Eğitim 12 saati aşarsa ara kayıtla (checkpoint) bölünmeli; ara kayıtlar özel veri seti olarak taşınır.
- Boru hattı sınamaları (düzenek, gönderim not defteri) zaten yalnızca Kaggle'da yapılabilir.
- Gerekenler: yardımcı oturumda Sami'nin doğrudan onayı (yeni GPU bütçesi, özel veri seti oluşturma).
- Süre: Orta iş yükü tek hesapla ~1,6-2,7 hafta (iki GPU birlikte kullanılırsa alt uç). Takım arkadaşlarının
  hesapları kendi kotalarını ekler, ama her biri kendi kurulumunu yapmalı.

## Runpod

- Hesap bağlı; şu an çalışan pod yok. Bakiye buradan okunmadı.
- Secure Cloud: sabit veri merkezi, daha güvenilir. Community Cloud: daha ucuz, stok ve süreklilik zayıf.
- Disk ayrıca ücretlenir (küçük); pod oluşturmadan önce toplam saatlik fiyat yazılacak.
- Veri: üretici ön eğitimi yalnızca serbest yapı verisi ister (COCONUT, LOTUS, PubChem alt kümesi), yani pod'a
  Kaggle anahtarı gerekmez. Parmak izi tahminleri Kaggle'da üretilip küçük bir dosya olarak taşınır.
- Çıktı: eğitilen ağırlıklar özel Kaggle veri seti olarak yüklenir (not defteri çevrimdışı çalışır).
- Kural: dış veri ve modeller serbest ve herkese açık olmalı; NIST türevi yasak. Kendi ağırlıklarımızın ve
  eğitim verisinin kaynağı README'ye yazılmalı; final seçiminden önce yayımlama gerekip gerekmediği takımca
  doğrulanmalı.

## Donanım karşılaştırması (Orta iş yükünün eğitim kısmı: 65 T4-saat; düzenek çalıştırmaları Kaggle'da kalır)

| Donanım | Bellek | Saatlik | Stok | Hız (T4 = 1) | Süre | Maliyet |
|---|---|---|---|---|---|---|
| Kaggle T4 x2 | 2 x 16 GB | ücretsiz | kota | 1 (tek) / ~1,7 (çift) | 38-65 saat, 2-3 hafta | 0 |
| RTX 3090 (community) | 24 GB | 0,22 | düşük | ~3 | ~22 saat | ~5 |
| RTX A5000 (secure) | 24 GB | 0,27 | düşük | ~2,5 | ~26 saat | ~7 |
| RTX 4000 Ada (secure) | 20 GB | 0,28 | düşük | ~2 | ~33 saat | ~9 |
| A40 (secure) | 48 GB | 0,59 | orta | ~3 | ~22 saat | ~13 |
| RTX PRO 4500 (secure) | 32 GB | 0,72 | orta | ~4 | ~16 saat | ~12 |
| RTX 4090 (secure) | 24 GB | 0,89 | yüksek | ~5 | ~13 saat | ~12 |
| L40S (secure) | 48 GB | 1,09 | düşük | ~5 | ~13 saat | ~14 |
| RTX 5090 (secure) | 32 GB | 1,39 | düşük | ~7 | ~9 saat | ~13 |
| A100 80 GB (secure) | 80 GB | 2,09 | düşük | ~6 | ~11 saat | ~23 |
| H100 80 GB (secure) | 80 GB | 4,39 | orta | ~10 | ~6,5 saat | ~29 |

Küçük iş yükü için süre ve maliyeti ~0,45 ile, büyük için ~2 ile çarpın. Deneme-yanılma payı olarak toplamı
2-3 katına hazırlamak gerçekçi (Orta için ~25-40 USD).

## Yollar

1. Yalnızca Kaggle: ücretsiz, ama 2-3 hafta; 14 Aralık'a kadar tek tur deneme sığar. Önerilmez.
2. Yalnızca Runpod eğitim + Kaggle sınama (önerilen): eğitim RTX 4090 secure üzerinde ~13 saat, ~12 USD;
   birkaç tur denemeyle 25-40 USD. Kaggle kotası düzenek ve gönderim sınamalarına kalır.
3. Büyük GPU (A100 / H100): bu model boyutunda gereksiz; 24 GB yeterli. Yalnızca çok büyük model ya da çok
   büyük yığın (batch) gerekirse.
4. Ucuz community (3090): en düşük maliyet (~5 USD), ama stok düşük ve kesinti riski var; ara kayıt şart.

## Gereken onaylar

- Runpod: pod oluşturmadan önce donanım ve toplam saatlik fiyat onayı.
- Kaggle: yardımcı oturumda yeni GPU bütçesi ve özel veri seti oluşturma için doğrudan onay.
- Eğitime ancak fizibilite ölçümü geçerse başlanır (tahmin edilen parmak izinin kalitesi eşiğin üstündeyse).

## Ücretsiz iki adımın sonucu (2026-11-18): A kapatıldı, eğitim başlatılmadı

Ölçüm: 250 doğal ürün paneli, modellerin eğitimde görmediği moleküller; tahminler
`dmitriigluzdov/casmi26-fold-safe-fpnet` içindeki `panel_logits.npz`, gerçek bitler boru hattının kendi koduyla.

| Model | Tahmin-gerçek Tanimoto, ortanca | >= 0,5 olan pay | >= 0,7 olan pay | Çubuk (>= 0,75'te %8 ya da >= 0,5'te %35) |
|---|---|---|---|---|
| fold-safe FPNet | 0,21 | %0 | %0 | geçmedi |
| FPNet A+B | 0,39 | %11 | %0 | geçmedi |

Karşılaştırma: aynı bitlerde en yakın BAŞKA panel molekülü ortanca 0,56'da; yani tahmin edilen parmak izi gerçeğe,
rastgele bir komşu doğal üründen daha uzak. Adayları sıralamaya yetiyor (bit AUC 0,83-0,91), yapıyı yazmaya yetmiyor.

2. adım (gerçek parmak iziyle üst sınır) Ekim raporunda zaten ölçülmüş: ilk 25'te %32,7 (N) / %0,3 (F) / %15,3
(iskelet ayrımı); yenilik benzeri F setinde açık başarısızlık.

Yan bulgular: formül darboğaz değil (doğru yapı listeden çıkarılınca bile 1. sıranın formülü %88-96 doğru; üst
sınır). n0'ın kendi listesi, doğru yapı çıkarıldığında bile doğal ürünlerde ortanca 0,73 benzerlikte bir analog
içeriyor; bir şey kurulacaksa serbest üretim değil, üst adayların doğru formül altında düzenlenmesi daha sağlam
temelli olurdu (kural tabanlı sürümü daha önce tabloda 0,000 / 0,007 almıştı). Açık bir üreticinin tabloda kazanç
gösterdiğine dair kanıt yok (heliosli'nin ağırlıkları 16 Kasım'da geri çekildi).
