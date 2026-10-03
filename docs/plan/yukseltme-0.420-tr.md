<!-- index: Turkish explainer: how the solution went from 0.409 to 0.420 (PubChem join) and what it implies -->
# 0.409'dan 0.420'ye: PubChem join ne yaptı, neden işe yaradı, sırada ne var

v12a, adayların sırasını değil aday kümesini değiştiren ilk gönderimimiz: PubChem'den gelen adaylar artık sabit sıralara konmuyor, ana motorun ranker'ı (sıralayıcı model) tarafından yerleştiriliyor. Public LB 0.409'dan 0.420'ye çıktı. Kanıt şimdilik tek gönderim ve tek yerel settir.

## Önceki pipeline

- **v1 engine (ana motor):** yaklaşık 710.000 bilinen yapıdan oluşan pool'dan (aday havuzu) kütleye uyanları ranker ile sıralar, ilk 60'ı verir.
- **engine-2:** kendi aday kümesi ve modelleriyle ikinci bir liste.
- **PubChem kanalı:** pool'da olmayan 25 yapıyı listeler. `lib_max` (molekülün kütüphanedeki en iyi spektrum benzerliği) 0.9'un altındaysa bunların ilk beşi sabit sıralara girer: 4, 8, 12, 16, 20; PubChem skoru pool'un en iyisini açık farkla geçiyorsa 2, 4, 6, 8, 10.
- **Forward modeller (ICEBERG, GLACIER):** aynı formüllü izomerleri yeniden sıralar.
- **Fusion:** bütün listelerin tek listede birleştirilmesi.

## Denenenler: on gönderim, hepsi 0.390 ile 0.409 arasında

Bu pipeline'ın ilk on gönderimi (referans kopyası ve v4b dahil) hep listedeki adayların sırasını veya yerini değiştirdi; hiçbiri listeye yeni aday sokmadı.

| Gönderim | Değişiklik | Public LB |
|---|---|---|
| v1, v2, v3b, v4a, v5a, v5c | referans, ağırlıklar, popülerlik prior'ı, library gate | 0.399, 0.404, 0.390, 0.404, 0.393, 0.405 |
| v4b | v2 + library gate + popülerlik 0.25 | 0.409 |
| v7b | engine-2 yalnızca `lib_max` ≥ 0.5 iken | 0.405 |
| v10a | ranker'da sıradan bağımsız eşitlikler (tie-order düzeltmesi) | 0.408 |
| v11 | kendi ranker'ımız (blend) + erken slotlar + küçük ayarlar | 0.405 |
| **v12a** | **PubChem join** | **0.420** |
| v13a | v12a, 50 yerine 100 aday | bekliyor |

İki öğretici başarısızlık:

- **engine-2 gate (v7b):** yerelde +0.13 görünüyordu, sonuç 0.405. Sebep: yerel doğrulama doğru cevabı engine-2'nin aday kümesinden farkında olmadan siliyordu; cevap engine-2'nin ilk 40'ında yalnızca %0.9 ile %8 arasında vardı. Yalnızca yanlış aday ekleyebilen bir kanalı kapatmak kazanç gibi göründü.
- **Kendi ranker'ımız:** fusion'dan önce class-2 moleküllerde +0.046 ± 0.014; fusion'dan sonra +0.005, +0.007, −0.002, yani sıfırdan ayırt edilemiyor. Forward modeller ve fusion farkı yutuyor.

## Yön değiştiren teşhis

Mixed regime ölçümünde (380 yerel molekül) v4b'nin MRR'ı (doğru cevabın 1/sıra ortalaması) cevabın nerede olduğuna göre şöyle:

| Cevap nerede | v4b MRR |
|---|---|
| pool'da, referans spektrumu var | 0.83 – 0.85 |
| pool'da, spektrum yok | 0.74 |
| yalnızca engine-2'nin adaylarında | 0.28 – 0.40 |
| hiçbir aday kümesinde değil (yalnızca PubChem'de veya yeni yapı) | 0.09 – 0.13 |

LB skoru 0.41 olduğuna göre gizli moleküllerin yaklaşık yarısı son türden olmalı. Darboğaz sıralama değil, listede hiç bulunmayan cevaplardı. Bu bir çıkarımdır, ölçüm değil; ayrıca bu yarının hepsi PubChem'de değil: PubChem incelemesi yalnızca PubChem'de olan cevapları public moleküllerin yaklaşık %7–14'ü olarak tahmin ediyor.

## Yükseltme: v12a

PubChem listesinin en iyi 50 adayı v1 engine'e sıradan aday olarak veriliyor. Bir pool adayının aldığı bütün kanıtları alıyorlar (analog yayılımı, parçalanma, alt formül) ve yerlerini ranker belirliyor.

| | Önce (v4b) | Sonra (v12a) |
|---|---|---|
| PubChem adayının yeri | sabit slot | ranker'ın verdiği sıra |
| Cevap PubChem listesinde 1. ise | slot aşamasında en iyi 4. sıra (0.25), agresif durumda 2. sıra (0.5) | 1. sıra mümkün (1.0) |
| Cevap son listede 1. sırada (yerel, kapısız) | %1.5 | %16.3 |

Yalnızca `lib_max` < 0.7 iken uygulanıyor. Sebep: kapısız haliyle v1 tarafında class-1 moleküller −0.029 ± 0.010 kaybediyor (1 kazanç, 13 kayıp) ve kayıp `lib_max` 0.7–0.9 arasındaki listelerde. Kapı bu kaybı −0.002'ye indiriyor ve kazancın %83–87'sini koruyor.

Yerel ölçüm (N 50, public ranker, kapı 0.7, bugünkü slot kuralına karşı):

- cevabı yalnızca PubChem'de olan moleküller: +0.163 ± 0.018 (116 kazanç, 11 kayıp); gerçek ICEBERG skorlarıyla +0.098 ± 0.020;
- cevabı pool'da, spektrumsuz: +0.013 ± 0.005, ICEBERG ile −0.005 ± 0.009;
- class 1: +0.002 ± 0.004, ICEBERG ile −0.006 ± 0.004.

## Sonuç ne anlama geliyor, ne anlama gelmiyor

Public LB yaklaşık 130 molekül; bir molekülün 1. sıraya gelmesi 0.0076 eder. +0.011 yaklaşık bir buçuk molekül demek, tek gönderimle gürültüden kesin ayrılamaz.

Etkilenen moleküllerdeki yerel kazanç (+0.10 ile +0.16) LB'ye ulaşanın yaklaşık on katı. Bu bekleniyordu: join yalnızca cevabı PubChem'de olup pool'da olmayan moleküllere dokunuyor (tahminen %7–14); gizli testin kimyasına en yakın grup olan doğal ürün yapıları kapısız ölçümde +0.07 ile +0.11 kazanıyor, sentetikler +0.26 ile +0.54; yereldeki "silinmiş" cevaplar da kütüphanede akrabaları olan eğitim yapıları.

Önceden yazılan tahmin 0.415'ti (0.403–0.44); PubChem listesinin tek public ölçümüne (0.037) göre ölçeklenen aralık +0.002 / +0.010 / +0.03'tü. Gelen +0.011 merkez değerle uyuşuyor; mixed regime karışımlarının verdiği +0.02 ile +0.04 yüksek kaldı. Tek isabet kalibrasyon kanıtı değil: v7b, v10a ve v11 tahminlerinin üçü de iyimser çıkmıştı.

## Sırada ne var

- **Daha derin PubChem listesi (v13a, N = 100):** yerelde 50'den 100'e çıkmak yalnızca-PubChem türünde +0.015 ekliyor. Tahmin 0.423 (0.412–0.432); sonuç bekleniyor.
- **engine-2'nin adayları:** cevap yalnızca engine-2'deyse engine-2 onu %83–85 oranında ilk 40'ta, medyan 1.–2. sırada buluyor; ama fusion böyle bir adaya v1'in 3.7. sırasının ağırlığını veriyor ve aday medyan 7. sıraya düşüyor. Aynı mantıkla bu adayları da ranker'a vermek akla geliyor; dosyalarda bunun ölçümü veya planı yok, bu bir öneridir.
- **Sıralama için yeni model eğitmek öncelik değil:** v10a ve v11 pool içi sıralamanın public skoru oynatmadığını gösterdi; simüle spektrumlarla tam eğitim bu yüzden ertelendi. İstisna: ranker'ı join'li listelerle yeniden eğitmek incelemede sıradaki adım olarak geçiyor (yerelde join + blend +0.204, join tek başına +0.163).
- **0.45 ve ötesi:** join PubChem kanalının LB değerini yaklaşık 0.011'den 0.02–0.04'e taşıyabilir; PubChem'de olmayan cevaba ulaşamaz. Orada çalışan generator cevabı moleküllerin %15.5'i için üretebiliyor, %49'unda hiç aday üretmiyor. Roadmap'e göre 0.50 hem daha iyi izomer sıralaması hem de bu grubun kayda değer bir kısmını gerektiriyor; 0.60'ı destekleyen ölçümümüz yok. 0.45'e giden ölçülmüş bir yol dosyalarda yok.
