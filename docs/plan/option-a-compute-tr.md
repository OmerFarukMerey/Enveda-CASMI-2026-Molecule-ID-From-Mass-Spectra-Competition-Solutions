<!-- index: 2026-11-18 (TR) A seçeneği (yapı üretici) için hesaplama raporu: Kaggle ve Runpod gereklilikleri, donanım başına süre ve maliyet, öneri -->
# A seçeneği için hesaplama raporu: Kaggle ve Runpod (2026-11-18)

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
