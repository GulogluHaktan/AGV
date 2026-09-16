# Proje Devir Notu — Symmetry-Exploiting DRL for AGV Path Planning

**Tarih:** 2026-09-15
**Durum:** Eğitim şu anda durduruldu (devralan kişi kaldığı yerden devam edecek).

## 1. Proje nedir

Hedef dergi: *Symmetry* (MDPI). Makale fikri: `Mission/Symmetry_Makale_Fikirleri.docx` içindeki
"Makale Fikri 5" — AGV navigasyonunda D4 simetrisini (4 rotasyon x {ayna var/yok}) kullanan
DRL politikası, 3 kolla karşılaştırılıyor:

1. **baseline** — düz PPO/SAC, simetri bilgisi yok
2. **symmetric_augmentation** — eğitim verisi D4 grubuyla rastgele dönüştürülerek çoğaltılıyor
3. **equivariant** — E(2)-steerable (escnn) D4-eşdeğişken CNN özellik çıkarıcı

Ölçülecek 3 şey: (a) örnek verimliliği, (b) eğitimde görülmemiş aynalı/döndürülmüş haritalarda
genelleme (G3), (c) sim-to-real transfer gap'i (henüz başlanmadı).

Literatür taraması tamamlandı: `Literatur-Taramasi/simetri-farkinda-agv-rl-literatur-taramasi.md`
ve `...-yontem-ve-gap-detay.md` (55 makale, G3/G5 boşlukları bu projenin motivasyonu).

## 2. Simülatör kararı ve geçmişi

- **Önce Isaac Sim + Isaac Lab denendi, TAMAMEN TERK EDİLDİ.** Sebep: RTX 5060 (Blackwell,
  çok yeni GPU) + sürücü 610.57.04 kombinasyonunda Isaac Sim'in RTX-only render motoru
  (`librtx.scenedb.plugin.so`) deterministik olarak segfault veriyor, raster/Storm fallback yok.
  Çözülemedi, silindi. Bir daha denenmeye gerek yok — donanım/sürücü uyumsuzluğu.
- **Şu an aktif: Gazebo Harmonic + ROS2 Jazzy, Docker içinde** (Arch Linux'ta native ROS2/Gazebo
  paketleme sorunlarından kaçınmak için). Çalışıyor, sağlam.

## 3. Kod nerede

Her şey `~/Projects/AGV_AGR_Paper/Simulations/gazebo_agv_nav/` altında:

```
docker/Dockerfile          # agv-gazebo image tanımı (ROS2 Jazzy + Gazebo + SB3 + escnn)
worlds/gen_world.py        # D4-simetrik harita -> Gazebo SDF (depo görünümlü: tuğla duvar,
                            # palet-rafı engeller, hedef işaretçisi, zemin şeridi)
worlds/agv_nav.sdf         # üretilmiş dünya (gen_world.py'nin çıktısı, git'e değil disk'e yazılı)
launch/agv_nav.launch.py   # gz sim + ros_gz_bridge başlatıyor
launch/bridge.yaml         # ROS2<->GZ Transport topic eşlemesi (odom, cmd_vel)
envs/map_generator.py      # simetrik/asimetrik harita üretici + start/goal örnekleyici
envs/gazebo_agv_env.py     # Gymnasium env: robotu Gazebo'da sürüyor, ödül/reset/teleport
envs/symmetry.py           # D4 grup dönüşümleri (harita + nokta)
envs/wrappers.py           # SymmetricAugmentationWrapper (arm 2 için)
envs/equivariant_extractor.py  # arm 3: escnn D4-eşdeğişken CNN feature extractor (SB3'e takılıyor)
scripts/curriculum_train.py    # ASIL EĞİTİM SCRIPT'İ — 5 aşamalı curriculum, SAC/PPO, tüm 3 kol
scripts/eval.py             # tekil model değerlendirme
scripts/snapshot.py         # matplotlib üstten-görünüm debug görseli
```

Docker image'lar: `agv-gazebo:latest` (ROS2+Gazebo+SB3), `agv-gazebo:escnn` (üstüne escnn/gfortran
eklenmiş, arm 3 için — henüz `curriculum_train.py`'nin kullandığı ana image değil, gerekirse
`docker run` komutundaki image adını `agv-gazebo:escnn` yapmak yeterli, zaten aynı temel üzerine
kurulu).

## 4. Nasıl çalıştırılır (temel komut kalıbı)

```bash
cd ~/Projects/AGV_AGR_Paper/Simulations/gazebo_agv_nav
docker run -d --name agv-sac --network host --gpus all -e GZ_IP=127.0.0.1 \
  -v "$PWD:/workspace" agv-gazebo bash -c "
source /opt/ros/jazzy/setup.bash
cd /workspace
ros2 launch launch/agv_nav.launch.py > /tmp/gz.log 2>&1 &
sleep 8
python3 scripts/curriculum_train.py --arm baseline --algo sac \
  --out_prefix /workspace/sac_baseline_v2 > /workspace/sac_baseline_v2.log 2>&1
echo DONE > /workspace/done.flag
"
```

- `--arm`: `baseline` | `symmetric_augmentation` | `equivariant`
- `--algo`: `sac` (önerilen) | `ppo`
- `--gpus all`: **ÖNEMLİ**, olmadan torch CPU'ya düşüyor, ~4-6x yavaşlıyor (bkz. Bölüm 6).
- İlerlemeyi izlemek için: `docker exec agv-sac bash -c "tail -f /workspace/sac_baseline_v2.log"`
- Canlı 3D görüntü (kendi ekranında, ayrı terminalde):
  ```bash
  xhost +local:docker
  docker run --rm --network host -e GZ_IP=127.0.0.1 -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/Projects/AGV_AGR_Paper/Simulations/gazebo_agv_nav:/workspace \
    agv-gazebo bash -c "source /opt/ros/jazzy/setup.bash && gz sim -g"
  ```

## 5. Curriculum tasarımı (`scripts/curriculum_train.py`)

5 aşama, hem hedef mesafesi hem adım bütçesi birlikte büyüyor (bu ikisini ayrı büyütmemek
büyük bir hataydı, bkz. Bölüm 6):

| Aşama | max_goal_dist | max_steps | timesteps |
|-------|---------------|-----------|-----------|
| s1    | 4              | 80        | 80,000    |
| s2    | 8              | 140       | 120,000   |
| s3    | 12             | 200       | 150,000   |
| s4    | 16             | 260       | 180,000   |
| s5    | tam-rastgele   | 320       | 250,000   |

Her aşama sonunda o aşamanın kendi zorluğunda 20 episode'luk **deterministik** değerlendirme
otomatik yapılıyor ve `=== STAGE sX DONE: success_rate=... ===` satırı log'a yazılıyor. En
sonda tam-rastgele görevde 30 episode'luk final değerlendirme var.

`--resume_from <ckpt.zip> --start_stage N` ile ortadan devam edilebilir (ama bkz. Bölüm 7,
şu an temiz bir baştan-başlatma önerilir).

## 6. Bulunan ve düzeltilen gerçek buglar/sorunlar (önemli — tekrar keşfetmeyin)

1. **`GZ_IP` ortam değişkeni şart.** Docker `--network host` altında GZ Transport'un UDP
   multicast keşfi bu olmadan tamamen kırılıyor — ROS->GZ publish çalışıyor gibi görünse de
   GZ->ROS (örn. `/odom`) sessizce hiç mesaj almıyor.
2. **DDS/ROS2 discovery ısınma gecikmesi:** taze oluşturulmuş bir rclpy publisher/subscriber,
   `ros_gz_bridge` ile eşleşmesi için ~2 saniyeye ihtiyaç duyuyor. `GazeboAGVEnv.__init__`
   içinde bu yüzden 2 saniyelik bir spin-warmup var, silmeyin.
3. **Episode reset'te gerçek ışınlama şart** — episode'ları "zincirlemek" (reset'te sadece
   hedefi değiştirip robotu olduğu yerde bırakmak) robotun bir engelin arkasına sıkışıp tüm
   sonraki episode'ları bozmasına yol açtı. `gz service /world/.../set_pose` ile gerçek
   teleport + odom-frame ofsetinin her reset'te yeniden hizalanması gerekiyor
   (`envs/gazebo_agv_env.py::teleport()` ve `reset()`).
4. **`max_steps` sabit bırakılırsa curriculum'un ileri aşamaları imkansız hale geliyor.**
   İlk denemede mesafe arttıkça adım bütçesi sabit (120) kaldı, robot fiziksel olarak
   yetişemedi (`ep_len_mean` sürekli tavanda kalıyordu). Şimdi ikisi birlikte büyüyor.
5. **`sample_goal_near` (map_generator.py) eski hali** bazen `max_goal_dist` sınırını tamamen
   yok sayıp sınırsız uzak bir hedef üretiyordu (50 denemede sınır-içi hücre bulunamazsa
   tam-rastgele bir hücreye düşüyordu). Düzeltildi: artık `start` etrafındaki kutudan örnekliyor
   ve en kötü ihtimalle "denenenler arasında en yakın" hücreyi döndürüyor, asla sınırsız değil.
6. **SAC `ent_coef="auto"` çöktü** (~0.0003'e düştü, 16k adımda) — keşif tamamen durdu, politika
   kötü bir yerel optimumda donup kaldı (`ep_len_mean` tavanda, başarı platoda). Sabit
   `ent_coef=0.1` denendi, bu sefer TERS yönde sorun: politika hiç "kesinleşmeden" sürekli
   yüksek entropi/gürültüde kaldı, başarı %22'ye çıkıp %10'a geriledi. **`ent_coef=0.02` sabit**
   ikisi arasında makul bir denge — şu anki ayar bu.
7. **`gradient_steps=4`** (örnek-başına 4x gradyan güncellemesi, gerçek-zamanlı simülasyonun
   pahalı olması nedeniyle denenmişti) **"primacy bias" belirtisi gösterdi** — başarı oranı
   yükselip sonra düşüyordu, entropy'den bağımsız olarak tekrarlanan bir desendi.
   `gradient_steps=1`'e geri dönüldü.
8. **CPU/GPU:** `torch` container'da varsayılan olarak CPU'ya düşüyordu (`Using cpu device`),
   fps 8-20 civarındaydı. `docker run --gpus all` eklenince (makinede RTX 5060 var, boşta
   duruyordu) `Using cuda device`'a geçti, fps ~45-50'ye çıktı. **Bunu unutmayın, çok kritik.**
9. **[EN ÖNEMLİSİ, TAM ÇÖZÜLMEDİ] Eğitim-zamanı başarı oranı ile deterministik
   değerlendirme arasında büyük uçurum var.** Stage s1'de eğitim sırasında rolling başarı
   ~%14-30 iken final deterministik eval **%5** çıktı; stage s3'te eğitim ~%6-14 iken final eval
   **%0** çıktı. İki kez tekrarlanan bir desen, gürültü değil. Muhtemel neden: ödül
   fonksiyonundaki **+20 terminal bonus**, adım-başı ödülün (~±0.1) ~200 katı büyüklükte —
   bu, seyrek başarı geçişlerinde devasa TD-error sıçramalarına (`critic_loss` 0.03'ten 1.4'e
   fırlıyordu) ve muhtemelen politikanın "ortalama" (deterministik) davranışının stokastik
   keşif kadar iyi kalibre olamamasına yol açıyor. **+5'e düşürüldü** (`envs/gazebo_agv_env.py`
   satır ~126). Bu düzeltmeyle **yeni bir koşu başlatıldı ama tamamlanmadan durduruldu** —
   devralan kişinin ilk işi bu yeni koşuyu (`sac_baseline_v2`) tamamlatmak ve bu uçurumun
   gerçekten kapanıp kapanmadığını doğrulamak olmalı.

## 7. Şu anki durum / kaldığım yer

- **Eski koşu (`sac_baseline*`, eski +20 ödülüyle):** s1=%5, s2=%10, s3=%0 (deterministik eval,
  kendi zorluk seviyesinde). Checkpoint'ler: `sac_baseline_s1.zip`, `_s2.zip`, `_s3.zip` — bunlar
  **eski, muhtemelen sorunlu ödül ölçeğiyle eğitilmiş**, yeni deneyler için başlangıç noktası
  olarak KULLANMAYIN.
- **Yeni koşu (`sac_baseline_v2`, +5 ödülüyle, `ent_coef=0.02`, `gradient_steps=1`, GPU'lu):**
  stage s1'in ortasında durduruldu (~28k/80k adım). Eğitim-zamanı başarı oranı sağlıklı
  görünüyordu (%10-24 bandı, `critic_loss` çok düşüktü: 0.0005 civarı — eski koşudaki
  spike'lardan çok daha iyi). **Henüz checkpoint/eval yok, container durduruldu, hiçbir ilerleme
  kaydedilmedi (SB3 sadece stage sonunda save ediyor).**

**Devralan kişinin ilk yapması gereken:** Bölüm 4'teki komutla `sac_baseline_v2`'yi **sıfırdan**
yeniden başlatmak (`--arm baseline --algo sac --out_prefix /workspace/sac_baseline_v2`,
resume_from vermeden) ve 5 aşamayı sonuna kadar götürüp, stage-sonu deterministik eval
skorlarının artık eğitim-zamanı rolling skorlarına yakın çıkıp çıkmadığını doğrulamak.

## 8. Sıradaki işler (öncelik sırasıyla)

1. `sac_baseline_v2`'yi tamamla, +5 ödül düzeltmesinin uçurumu kapatıp kapatmadığını doğrula.
2. Kapatmazsa: değerlendirmeyi `deterministic=False` ile de dener (karşılaştırma için), veya
   `learning_starts`i artır, veya toplam eğitim süresini uzat (mean action'ın olgunlaşması için).
3. Baseline gerçekten makul bir final skoruna ulaşınca (**hangi skor "makale için yeterli"
   sorusu henüz netleşmedi** — literatürdeki benzer çalışmalarla kıyaslanmalı), aynı
   `curriculum_train.py`'yi `--arm symmetric_augmentation` ile çalıştır.
4. Ardından `--arm equivariant` (kod hazır, `envs/equivariant_extractor.py`, test edildi —
   `agv-gazebo:escnn` image'ını kullanmak gerekiyor, `docker run` satırındaki image adını
   değiştirin). Not: escnn'in D4-eşdeğişkenliği ayrık kernel yaklaşımı yüzünden tam sıfır değil,
   ~%10 sayısal hata var — bu literatürde bilinen/kabul edilen bir durum, makalede metrik
   olarak raporlanabilir.
5. G3 genelleme testi: `envs/symmetry.py`'deki D4 dönüşümleriyle eğitilmemiş aynalı/döndürülmüş
   haritalarda zero-shot değerlendirme — henüz Gazebo pipeline'ında koşulmadı.
6. Çoklu tohum (seed) istatistiksel doğrulama — şu ana kadar hep `seed=3`.
7. Sim-to-real bileşeni (RQ3) — hiç başlanmadı.

## 9. Diğer notlar

- ~65GB'lık kullanılmayan Isaac Sim image'ları (`lunar-rocket-isaaclab`, `nvcr.io/nvidia/isaac-sim`)
  silindi, disk temiz.
- `jolly_spence` adında bir GUI-izleyici container arka planda açık kalmış olabilir (kullanıcı
  canlı sahneyi izlemek için açmıştı) — `docker rm -f jolly_spence` ile kapatılabilir, eğitime
  bir etkisi yok, sadece CPU'dan pay alıyor.
- Depo/görsel iyileştirmeleri (tuğla duvar, palet-rafı, hedef işaretçisi) tamamen kozmetik,
  collision geometrisi/eğitim hızı etkilenmiyor — `worlds/gen_world.py` içinde.

## 10. Devir sonrası güncelleme (2026-09-15, yeni makine)

**Yeni ortam:** Proje yeni bir makineye (Ubuntu 22.04, RTX 4060 Laptop) devralındı. Docker ve
şifresiz sudo yok; pipeline artık **native** çalışıyor: micromamba env `agv`
(`~/micromamba/envs/agv`) — RoboStack'ten ROS2 Jazzy + Gazebo Harmonic 8.10
(`-c robostack-jazzy -c conda-forge`), pip'ten torch cu130 + SB3 2.9. TurtleBot3 modelleri
`models/` altına vendor'landı. Çalıştırma: `scripts/run_native.sh <arm> <algo> <out_prefix>`
(Bölüm 4'teki docker komutunun karşılığı; GZ_IP/ROS_DOMAIN_ID/GZ_SIM_RESOURCE_PATH ayarlıyor).
Not: İkinci bir izole sim örneği gerekirse (eğitim sürerken eval) `GZ_PARTITION` +
farklı `ROS_DOMAIN_ID` kullan — `scripts/eval_diag.py` böyle koşuldu.

**[GÜNCELLENDİ — asıl kök neden bulundu, bir sonraki blok tarihçe olarak kalıyor]
Bölüm 6.9'daki uçurumun İLK teşhisi (kısmen doğru ama eksikti):** `sac_baseline_v2` yeni koşusunda s1 sonunda
deterministik eval yine %5 çıktı (eğitim-zamanı ~%14-22). Aynı s1 checkpoint'i
`eval_diag.py` ile iki modda değerlendirildi: **deterministic=True → %5,
deterministic=False → %25**. Yani +20→+5 ödül düzeltmesi kritiği stabilize etti ama uçurumun
asıl nedeni ödül ölçeği değilmiş: stokastik davranış (keşif gürültüsü dahil) hedefe ulaşıyor,
politikanın ortalama aksiyonu ise henüz olgunlaşmamış (mean-action miscalibration).
Sabit `ent_coef=0.02` entropiyi canlı tutarken ortalamanın keskinleşmesini geciktiriyor.
Bu yüzden `curriculum_train.py::evaluate()` artık her stage sonunda **iki metriği birden**
raporluyor (`success_rate=` deterministik, `stochastic=` stokastik).

## 11. ASIL KÖK NEDEN (2026-09-15 akşamı): env adımı sim-süresine bağlı değildi

`sac_baseline_v2` s3 sonunda deterministik eval yine %0 çıktı; s3 checkpoint'i izole
örnekte değerlendirilince **stokastik eval de %0** çıktı (40/40 zaman aşımı) — oysa eğitim
sırasında %6-16 görünüyordu. `eval_diag.py`'ye sim-süresi ölçümü eklenince gerçek ortaya
çıktı:

- Env `step()` süresi **duvar-saatine bağlıydı** (2× `spin_once` ≈ birkaç ms): hızlı eval'de
  adım başına ~0.04 sim-s, eğitimde (aradaki gradyan hesabı ~20ms sayesinde) ~0.10 sim-s.
- 0.22 m/s tavan hızla bu, eval'de episode başına ~1.8 m, eğitimde ~4.4 m menzil demek —
  yani **s3+ hedeflerinin çoğu fiziksel olarak ulaşılamazdı**; eğitimdeki "başarılar" kutu-içi
  örneklemede şans eseri yakın düşen hedeflerdi. §6.9 uçurumu, ödül ölçeği veya politika
  olgunluğu değil, **eğitim ile eval'in fiilen farklı fizik bütçeleri koşmasıydı.** CPU yükü
  bile başarı oranını değiştiriyordu.

**Düzeltme (uygulandı):** `gazebo_agv_env.py` artık her adımı **sabit sim-süresine**
kilitliyor: `control_dt=0.5` sim-s; `_wait_sim()` odom zaman damgası 0.5 s ilerleyene kadar
spinliyor (10 s duvar-saat guard'ı var). Dünya RTF'si 5→0 (sınırsız) yapıldı. Duman testi:
sim_dt/adım = 0.500±0.000 s, yer değiştirme 0.108 m/adım (teorik 0.11), ~42 fps boş, eğitimde
~27 fps. Artık: dist 4 → ~37+ adım (bütçe 80), dist 12 → ~110+ (bütçe 200), dist 16 → ~145+
(bütçe 260) — tüm aşamalar ulaşılabilir.

**Koşu durumu:** `sac_baseline_v2` s4 ortasında DURDURULDU (s3+ sonuçları imkânsız-görev
gürültüsü; `_s1.._s3.zip` checkpoint'leri ve log yalnızca tarihçe/pilot olarak saklanıyor,
YENİ DENEYLERDE KULLANMAYIN — eski kırık dinamikle eğitildiler). Düzeltilmiş dinamikle
`sac_baseline_v3` sıfırdan başlatıldı (27 fps, ~8 saat; çift-metrikli stage eval devrede).
Beklenti: eğitim-zamanı başarı ile stage-sonu eval artık tutarlı olmalı ve mutlak başarı
oranları belirgin yükselmeli. v3 de düşük kalırsa bakılacak ilk şeyler: ent_coef aşama-bazlı
düşürme, learning_starts, ödül şekillendirmesi — ama önce yeni dinamikte bir tam koşu görün.

## 12. GÖREV ÖĞRENİLEBİLİR DEĞİL: gözlem uzayında robot yönü (yaw) yok (2026-09-16)

`sac_baseline_v3` 5 aşamayı tamamladı (02:18). Sim-süresi düzeltmesi çalıştı — eğitim ile
eval artık aynı fizikte koşuyor — **ama politika hiç öğrenmedi:**

| Aşama | Deterministik | Stokastik |
|-------|---------------|-----------|
| s1 | %5 | %30 |
| s2 | %10 | %10 |
| s3 | %0 | %0 |
| s4 | %0 | %5 |
| s5 | %0 | %15 |
| FINAL (tam-rastgele) | **%0** | **%13.3** |

Belirleyici kanıt, 780k adım sonunda aksiyon istatistikleri: `linear mean=-0.08 std=0.57 |
angular mean=+0.01 std=0.57`. **Ortalama aksiyon sıfır, std ise tanh-sıkıştırılmış politika
için neredeyse tavanda.** Yani öğrenilen politika "yerinde dur"; deterministik eval %0 çünkü
robot hiç hareket etmiyor, stokastik eval'deki %13-15 ise tamamen **keşif gürültüsünün
rastgele yürüyüşü** — hedef yarıçapı 1.2 m olduğu için ara sıra üstüne denk geliyor.
`ep_rew_mean` de 0.65'ten -2.26'ya iniyor; bu, adım cezası (-0.01 × 320) dışında net mesafe
kapatılmadığı anlamına geliyor (~0.7 m / 320 adım).

**Neden:** Gözlem `{occupancy, goal_relative}`; `goal_relative = (goal - pose)/grid_size`
yani **dünya çerçevesinde**. Aksiyon ise `(linear_v, angular_v)` — **robot gövde
çerçevesinde**. Robotun yönü (yaw) gözlemde hiçbir yerde YOK (`envs/` altında
`yaw|orientation|theta|quat` geçen tek satır bile yok). "Hedef 3 m kuzeydoğunda, hangi gövde-
çerçevesi hızını vereyim?" sorusunun cevabı tamamen yaw'a bağlı ve yaw gizli. Bu, kritik
durumu eksik bir POMDP — **optimal davranış gerçekten de "sıfır ortalama + yüksek varyans"
ile riskten korunmak**, ki ajan tam olarak bunu buldu. Daha fazla eğitim, ödül ayarı,
entropi ayarı bunu düzeltemez.

Ek olarak `teleport()` yalnızca `position` gönderiyor, `orientation` göndermiyor — yaw
episode'lar arasında sıfırlanmıyor, önceki episode'dan devrediyor (kontrolsüz gizli değişken).

**Bu hata en baştan var:** terk edilen Isaac prototipi de aynı gözlem uzayını kullanıyor
(`isaac_agv_nav/envs/agv_nav_env.py:44-48`); `pose[2]=theta` sadece dinamik entegrasyonu için
tutuluyor, gözleme hiç konmuyor. Dolayısıyla **bugüne kadarki TÜM koşular
(`curriculum_*`, `baseline_*`, `sac_baseline*` v1/v2/v3) çözülemez bir görevde eğitildi;
hiçbirinin başarı sayısı anlamlı değil.**

### 12b. İkinci kusur (makale açısından daha kritik): occupancy ızgarası sabit

`curriculum_train.py:111` haritayı stage döngüsünün DIŞINDA bir kez üretiyor — tüm koşu
boyunca tek harita. Yani `occupancy` gözlemi **sabit bir girdi**; hiçbir bilgi taşımıyor.
Bu doğrudan makalenin kalbini vuruyor: hem simetrik veri artırımı hem de D4-eşdeğişken CNN
occupancy ızgarası üzerinde çalışıyor. Sabit bir ızgarada eşdeğişkenliğin üzerinde
çalışacağı bir şey yok — **üç kol arasındaki karşılaştırma bu haliyle boş bir karşılaştırma
olur.** (Isaac prototipinde episode başına `_new_map()` vardı; Gazebo'ya taşırken kayıp.)

### 12c. Üçüncü kusur: çarpışma cezası ve ego-merkezli algı yok

Ödül `-0.01 + progress (+5 hedef)`; çarpışma terimi yok. Robot modeli `/scan` (360-ışın
lidar) yayınlıyor ama `launch/bridge.yaml` yalnızca `odom` + `cmd_vel` köprülüyor — ajan
engelleri hiç görmüyor. Engelden kaçınma öğrenmesi için sinyal de algı da yok.

### 12d. Önerilen düzeltme sırası (henüz UYGULANMADI — karar bekliyor)

1. **Yaw'ı gözleme ekle ve hedefi gövde çerçevesine çevir.** Odom quaternion'undan θ çıkar;
   `goal_relative`'ı −θ ile döndür (veya `(cos θ, sin θ)` ekle). Gövde çerçevesi hem doğrudan
   uygulanabilir bilgi verir hem de D4 grubunun temsile temiz etki etmesini sağlar.
2. **Reset'te yaw'ı rastgele ata** (`teleport()`'a `orientation` ekle) — gizli değişkeni
   kontrol altına al.
3. **UCUZ KAPI TESTİ — yeni 8 saatlik koşudan ÖNCE yapın:** elle yazılmış bir kontrolcü
   (hedefe dön, sonra ilerle) bu görevde ~%90+ almalı. Almıyorsa env hâlâ bozuk demektir.
   Bu 5 dakikalık test, iki gece boşa giden GPU süresini baştan yakalardı.
4. **Haritayı episode başına değiştir** (prosedürel simetrik varyantlar) — occupancy'nin
   bilgi içeriğini ve dolayısıyla simetri karşılaştırmasının anlamını geri getir.
5. Çarpışma cezası + `/scan`'i köprüle (ego-merkezli engel algısı).
6. Ancak bundan sonra baseline / augmentation / equivariant üç kolunu koş.
