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

## 13. Düzeltme uygulandı, ortam artık çözülebilir (2026-09-16)

§12d'nin 1-3. maddeleri yapıldı (commit `44154ec`):

- **Gözlem:** `goal_body` (hedef ofseti robot çerçevesine döndürülmüş) +
  `heading` = (cos yaw, sin yaw) + `occupancy`. D4 dönüşümü altında temiz grup etkisi:
  occupancy ve heading eşdeğişken, goal_body değişmez — makalenin simetri kollarının
  ihtiyacı olan temsil bu.
- **Poz kaynağı:** Vendor'lanan burger modeline `OdometryPublisher` eklendi, `/gt_odom`
  olarak köprülendi. Tekerlek odometrisi `set_pose` teleport'unu hiç algılamıyordu; artık
  gerçek dünya pozu okunuyor (teleport takibi 0.000 m / 0.00° doğrulukla test edildi) ve
  kırılgan spawn-offset hilesi silindi. Reset'te başlangıç yaw'ı rastgeleleştiriliyor.
- **Açısal hız tavanı 2.84 → 1.0 rad/s.** Komut `control_dt` boyunca tutulduğu için 2.84
  adım başına 81° dönüş demekti; elle yazılmış kontrolcü hiçbir makul tolerans bandına
  oturamıyor, 18 adımda sıfır ileri hareketle salınıyordu (+105° → −5° → −55°). 1.0 rad/s
  ile adım başına 29°. Limitler artık env niteliği (`env.max_lin`, `env.max_ang`), böylece
  kontrolcüler sabitleri kopyalamıyor.

**KAPI TESTİ (`scripts/gate_test.py`) — bundan sonra her GPU harcamasından ÖNCE koşun:**

| Aşama | Başarı | Ortalama adım / bütçe |
|-------|--------|------------------------|
| s1 | %100 | 15 / 80 |
| s2 | %90 | 40 / 140 |
| s3 | %100 | 72 / 200 |
| s4 | %100 | 49 / 260 |
| s5 | %80 | 124 / 320 |
| **Genel** | **%94** | — |

(Önceki deterministik politika: %0.) s2/s5'teki tek tek başarısızlıklar kontrolcünün engele
sıkışması — kaçınma mantığı yok, bu yüzden geçme eşiği %80.

Yan gözlem: `sample_goal_near` kutu-örneklemesi ve 16×16 ızgara sınırı nedeniyle aşamaların
gerçek mesafe dağılımları birbirine çok yakın (s3 ort. 7.3 m, s4 5.9 m, s5 6.7 m) — yani
curriculum merdiveni sanıldığı kadar ayrışmıyor. Eğitim koşusundan sonra gözden geçirilmeli.

**§12d madde 4-5 de tamamlandı (commit `81b91c0`):** onaylanan tasarım — eğitim episode başına
24 kanonik yönelimli düzenden birini örnekliyor; D4 görüntüleri ve ayrık tohumlu yeni düzenler
G3 için ayrı tutuluyor (`scripts/eval_g3.py`, commit `4ae06c5`). Reset, `obs_*` modellerini
o haritanın hücrelerine ışınlayarak fiziği gözlemle uyumlu tutuyor. Ödüle yoğun engel-yakınlığı
cezası eklendi (lidar eklenmedi: ajan engelleri zaten occupancy'den görüyor).

Performans notları (hepsi ölçülmüş): teleport `gz service` subprocess'i 308 ms, köprülenmiş ROS
servisi 0.25 ms; 14 teleport'u **eşzamanlı** göndermek 1-2'sini onaysız bırakıyor ve kayıp
istek = kayıp teleport, yani harita gözlemden sessizce sapıyor → **sıralı gönderin**; raflar
`<static>true</static>` kalmalı, dinamik yapıldıklarında teleport'un artık hızını koruyup
episode ortasında haritada geziniyorlar. Net: reset 6676 ms → 21 ms.

2. ve 3. kol kodu yeni gözleme taşındı ve grup etkisi doğrulandı (commit `5a818f5`):
`goal_body` döndürmeler altında değişmez, aynalamada sol/sağ bileşeni işaret değiştirir;
`heading` dünya-çerçevesi gibi dönüşür. `scripts/test_symmetry.py` bunu 200 rastgele durum ×
8 grup elemanında (4808 kontrol) dünyayı bağımsızca dönüştürmekle karşılaştırarak sınıyor.

## 14. Curriculum merdiveni zayıf: üst üç aşama aynı zorlukta (2026-09-16)

`sample_goal_near` dağılımı 3000 örnekle ölçüldü (havuz: 24 harita, 16×16):

| Aşama | max_dist | ort. mesafe | medyan | p90 | hedef yarıçapı içinde başlayan | gerekli adım / bütçe |
|-------|----------|-------------|--------|-----|-------------------------------|----------------------|
| s1 | 4 | 2.32 m | 2.24 | 3.61 | **%19.0** | 21 / 80 |
| s2 | 8 | 4.30 m | 4.24 | 7.07 | %7.5 | 39 / 140 |
| s3 | 12 | 5.81 m | 5.83 | 9.85 | %5.9 | 53 / 200 |
| s4 | 16 | 6.74 m | 6.71 | 11.05 | %3.4 | 61 / 260 |
| s5 | yok | 6.26 m | 6.32 | 10.20 | %3.2 | 57 / 320 |

Üç sorun:

1. **Merdiven üstte sıkışıyor.** s3/s4/s5 ortalamaları 5.8-6.7 m, yani fiilen aynı zorluk.
   16×16 ızgara (kenar payıyla 12×12 kullanılabilir alan) mesafeyi doğal olarak sınırlıyor;
   `max_dist` 12'nin üstüne çıkmak hiçbir şey değiştirmiyor. 5 aşamalı curriculum gerçekte
   3 aşamalı (2.3 / 4.3 / ~6 m).
2. **s5 "en zor" değil, s4'ten daha kolay** (6.26 < 6.74). Tam-rastgele örnekleme 12×12 kutuda
   iki nokta arası ortalama ~6.3 m verir; s4'ün kutu-örneklemesi biraz daha uzağa düşüyor.
   Makalenin manşet "final full-random başarı oranı" metriği dolayısıyla en zor koşul DEĞİL —
   hakem bunu yakalar.
3. **s1'de episode'ların %19'u hedef yarıçapının İÇİNDE başlıyor**, yani hiç hareket
   gerektirmeyen bedava başarılar. s1'in %95'i bu kadarıyla şişmiş. Örnek-verimliliği iddiası
   ("hedef başarı oranına ulaşmak için gereken episode sayısı") doğrudan bundan etkilenir.

Öneri: hedef örneklemesini **bant** haline getirin (aşama başına [min, max] mesafe, min >
goal_radius + pay ≈ 2.0 m) ki hiçbir episode önceden çözülmüş olmasın ve merdiven gerçekten
monoton olsun; ızgarayı büyütmek (16 → 24/32) üst aşamaların ayrışması için gerekir.

**Bu sampler'ı değiştirmek baseline'ın yeniden koşulmasını gerektirir** (kollar arası
karşılaştırma ancak aynı sampler ile geçerli).

### 14b. Düzeltildi (commit `688fe99`) — bant örneklemesi + 24×24 ızgara

Karar: sampler şimdi düzeltildi, baseline yeniden koşuluyor. `sac_baseline_v4` s3'ün ortasında
durduruldu (kalan aşamaları değiştirilen sampler'ı kullanacaktı, hiçbir şeyle
karşılaştırılamazdı); s1 = %95/%95 ve s2 = %70/%90 sonuçları env düzeltmesini uçtan uca
doğrulama işini zaten yapmıştı. Checkpoint'ler: `sac_baseline_v4_s1.zip`, `_s2.zip`.

Yeni merdiven (`envs/curriculum.py` — aşama tablosu artık TEK yerde; eğitim ve kapı testinde
ayrı ayrı durunca sapma riski vardı ve farklı mesafelerde koşan bir kapı testi hiçbir şey
kanıtlamaz):

| Aşama | Bant | Ölçülen ort. | Bütçe |
|-------|------|--------------|-------|
| s1 | 2-5 m | 3.32 m | 90 |
| s2 | 5-9 m | 6.63 m | 180 |
| s3 | 9-13 m | 10.68 m | 270 |
| s4 | 13-18 m | 14.71 m | 380 |
| s5 | 18+ m | 19.19 m | 600 |

Kesin monoton, s5 gerçekten en zor, hedef yarıçapı içinde başlayan episode oranı **%0**.
Izgara 16→24 (16×16'da kenar payıyla mesafe ~6 m'de tavan yapıyordu), engel 12→28 (kalabalık
yoğunluğu sabit kalsın diye: önce 12/196 iç hücre, şimdi 28/484).

### 14c. Kapı testi artık başarısızlık nedenini sınıflandırıyor

Çıplak başarı oranı "ortam bozuk" ile "bu kontrolcü bu harita için fazla basit"i ayırt
edemiyor, oysa ikisi zıt tepki gerektiriyor. Yeni sınıflar:

- **wedged** — engele temas hâlinde durdu. Kontrolcünün tasarım gereği kaçınma mantığı yok;
  RL politikası occupancy ızgarasını görüyor ve etrafından dolanabilir. Ortam hatası DEĞİL.
- **no_progress** — hiçbir şeye temas etmeden neredeyse hiç mesafe kapatmadı. **Bozuk ortamın
  imzası**; yaw eksikliği hatası tam olarak böyle görünüyordu. Buradaki herhangi bir ciddi
  oran kapıyı geçirmez.
- **budget** — gerçek ilerleme var, süre bitiminde hâlâ hareket hâlinde. `max_steps` dar.

Yeni curriculum'da sonuç: success %80, wedged %20, **no_progress %0, budget %0**. Beş
aşamanın tamamındaki her başarısızlık sıkışma; episode bazında doğrulandı (engel mesafesi
0.44-0.51 m, son 25 adımda sıfır yer değiştirme). Yani ortam sağlam ve adım bütçeleri yeterli.

**Şu an koşan:** `sac_baseline_v5` (baseline, SAC, bant curriculum, 24×24, 830k adım,
~6-7 saat). Ardından: `eval_g3.py` ile G3 değerlendirmesi, sonra `--arm
symmetric_augmentation` ve `--arm equivariant`.

## 15. 3. kol D4-değişmez DEĞİL — ve nedeni escnn değil (2026-09-16)

`scripts/test_equivariance.py` yazıldı: kodlayıcının bir haritayı D4 görüntüsünden ayırt
edip etmediğini ölçüyor. Sonuç, asimetrik haritalarda 8 grup elemanının tamamı için
**bağıl hata ~%22-27**. Yani 3. kolun yapısal iddiası (ağ haritayı ve dönüşümünü ayırt
edemez) mevcut kodda **yanlış**.

**Bölüm 8.4'teki açıklama hatalıydı.** Orada "escnn'in D4-eşdeğişkenliği ayrık kernel
yaklaşımı yüzünden tam sıfır değil, ~%10 sayısal hata var, literatürde kabul edilen bir
durum" deniyordu. Ölçüm bunu çürütüyor — escnn burada makine hassasiyetinde tam. Kanıt
deneyi: `group_pool` çıktısına **uzamsal ortalama havuzlama** eklendiğinde hata sekiz
elemanın tamamında **tam 0.0000** oluyor; mevcut hâlde ~%20.

**Gerçek neden:** `GroupPooling` özellikleri yalnızca *grup kanalı* etkisine göre değişmez
kılıyor. Uzamsal H×W haritası hâlâ dönüşüyor. `equivariant_extractor.py` bu haritayı
`flatten(1)` ile düzleştirip `nn.Linear`'a veriyor — döndürülmüş bir uzamsal harita farklı
bir düzleştirme sırası ürettiği için çıktı değişiyor. Değişmezliği yok eden adım bu.

Ek olarak testin ilk sürümü de yanıltıcıydı: `symmetric_corridor` haritaları ayna-simetrik
olduğu için `mirror_x` haritayı kendisine götürüyor ve bedava 0.0000 alıyordu. Test artık
asimetrik haritalar kullanıyor, böylece sekiz elemanın hepsi anlamlı.

### 15b. Bu bir tasarım kararı gerektiriyor (KULLANICI ONAYI BEKLİYOR)

Düzeltmenin üç yolu var ve makalenin yöntem iddiasını farklı şekilde etkiliyorlar:

- **(A) Genel uzamsal havuzlama.** `head`'den önce H,W üzerinden ortalama al. Tam değişmezlik
  (ölçüldü: 0.0000), tek satırlık değişiklik. Ama özellik vektörü "ortalama kalabalık" gibi
  küresel bir özete dönüşür, uzamsal yerleşim tamamen kaybolur — politika haritayı engelden
  kaçınmak için kullanamaz. Occupancy girdisinin faydası büyük ölçüde gider.
- **(B) Ego-merkezli occupancy ızgarası — ELENDİ, kullanmayın.** Izgarayı robot çerçevesine
  döndürmek değişmezliği bedava verir ve uzamsal yerleşimi korur, bu yüzden ilk bakışta en
  cazip seçenekti. Ama sayısal olarak sınandı ve **G3'ü üç kol için birden anlamsız kılıyor:**
  ego(harita, poz, yaw) ile ego(rot90·harita, karşılık gelen poz/yaw) 300 örnekte **%100
  bit düzeyinde aynı** çıkıyor. Başlangıç poz/yaw'ı düzgün örneklendiği için döndürülmüş bir
  haritadaki gözlem DAĞILIMI orijinaliyle özdeş; yani baseline dahil her kol rotasyon
  genellemesini kendiliğinden çözer ve RQ34 ölçülemez hale gelir.

  Genel ders: **yapısal rotasyon-değişmezliği veren her temsil, o kol için G3 rotasyon
  testini kendiliğinden çözer.** 3. kol için bu zaten amaçlanan sonuç (beklenen-sonuçlar
  tablosu "yapısal garanti" diyor); test anlamlı kalıyor çünkü 1. ve 2. kollar dünya-çerçevesi
  ızgarayı bu garanti olmadan kullanıyor. Dolayısıyla **ızgara dünya çerçevesinde kalmalı**
  ve düzeltme 3. kolun içinde yapılmalı.
- **(C) Uçtan uca eşdeğişken politika.** Uzamsal haritayı aksiyon başlığına kadar eşdeğişken
  taşı; aksiyon dağılımı da dönüşsün (aynalamada açısal hız işaret değiştirir). Makalenin
  "E(2)-steerable politika ağı" ifadesine en sadık seçenek ve en güçlü katkı, ama SB3'ün
  politika başlıkları bunu desteklemiyor, özel başlık yazmak gerekir (Bölüm 8.4'te bu
  bilinçli olarak kapsam dışı bırakılmıştı).

**Karar durumu (2026-09-16):** (A) ile (C) arasındaki seçim kullanıcıya soruldu, kullanıcı
kararı erteledi. 3. kol kodu ŞU AN DÜZELTİLMEMİŞ durumda ve `scripts/test_equivariance.py`
bilerek FAIL veriyor — bu, düzeltme yapılmadan 3. kolun koşulmaması gerektiğinin kalıcı
hatırlatıcısı. `--arm equivariant` ile bir koşu başlatmadan önce bu karar verilmeli.

## 16. İki ortam kazası ve alınan dersler (2026-09-16)

### 16a. `sim_down.sh` eğitim simülasyonunu öldürdü

`sim_down.sh` ilk hâlinde süreç-adı desenine göre *bütün* `gz sim` süreçlerini öldürüyordu.
2. kolu izole bir örnekte (domain 42) test ettikten sonra temizlik için çağrıldığında,
domain 17'deki **eğitim simülasyonunu da** kapattı — v5 o anda 170k adımdaydı.

Belirti tam olarak `_wait_sim`'in durma korumasının bastığı uyarı oldu:
`sim time only advanced 0.000s of 0.5s within 10.0s wall`. Yani koruma işini yaptı ve
sessiz veri bozulmasını görünür kıldı; o koruma olmasa adımlar fiziksiz geçer ve hiçbir şey
fark edilmezdi. Hasar 8 adımla sınırlı kaldı (170.447 içinde ihmal edilebilir).

**Düzeltme:** `sim_up.sh` artık PID'leri domain başına bir dosyaya yazıyor
(`$TMPDIR/agv_sim_<domain>.pids`), `sim_down.sh` yalnızca o dosyadaki PID'leri öldürüyor
(`sim_down.sh <domain>` veya `--all`). `run_native.sh` de kendi içinde ayrı başlatma mantığı
tutmak yerine `sim_up.sh`/`sim_down.sh` çağırıyor — iki kopya mantık zamanla sapıyordu.
İzolasyon test edildi: domain 42 kapatılırken domain 17 ayakta kaldı.

### 16b. escnn kurulumu numpy'i düşürdü, checkpoint'ler okunamaz oldu

Eğitim sürerken `pip install escnn` yapıldı. escnn → `lie-learn` numpy<2 istediği için pip
**numpy 2.5.3'ü sessizce 1.26.4'e düşürdü**. Sonuç: numpy 2.x altında kaydedilmiş her
checkpoint okunamaz hâle geldi — SB3 cloudpickle ile açıyor ve numpy 1.26'da `numpy._core`
yok, `ModuleNotFoundError: No module named 'numpy._core.numeric'`.

v5'i s1 checkpoint'inden devam ettirme denemesi tam bu yüzden başarısız oldu.

**Karar:** 3. kol escnn gerektiriyor, escnn numpy<2 gerektiriyor, dolayısıyla **üç kolun
tamamı tek ve aynı numeric yığın altında koşmalı** — aşamaları farklı numpy sürümleri
arasında bölmek bir makale için savunulamaz. Bu yüzden baseline sıfırdan yeniden başlatıldı
(`sac_baseline_v6`). v3/v4/v5 checkpoint'leri artık açılamaz; zaten tarihçe.

**Ders:** ortam bağımlılıklarını eğitim başlamadan ÖNCE kesinleştirin. Yığın artık
`requirements-pip.txt` ile sabitlendi ve `ENVIRONMENT.md` kurulum sırasını + numpy<2 pinini
açıklıyor. Yeni yığın altında kapı testi (%75 success + %25 wedged, no_progress %0) ve
simetri testi (4808 kontrol) yeniden koşuldu ve geçti.

**Şu an koşan:** `sac_baseline_v6` — baseline, SAC, bant curriculum, 24×24, 830k adım,
18-21 fps, tahmini ~11 saat.

## 17. 3. kol düzeltildi: vektör-alanı okuması (B′) — 2026-09-16

Karar B′ oldu ve uygulandı. `envs/equivariant_extractor.py` yeniden yazıldı.

**Ne yapıyor:** eşdeğişken CNN yığınının son katmanı artık skaler yerine **vektör alanları**
(D4'ün 2 boyutlu irrep'i, `gspace.irrep(1,1)`) ve yanında değişmez skaler alanlar üretiyor.
Uzamsal ortalama alma vektör alanları için eşdeğişkendir (döndürülmüş vektörlerin ortalaması,
ortalamanın döndürülmüşüdür), dolayısıyla havuzlamadan sonra elde dünya-çerçevesi vektörler
kalıyor: "kalabalık şu yöne artıyor". Bunlar gözlemden kurtarılan iki referans yönle iç
çarpıma sokuluyor — robotun yönü `h` ve onun diki `h_perp`.

**Neden "tam değişmez" değil, kasıtlı olarak:** aynalama altında doğru aksiyon
`(doğrusal, -açısal)`. Ayna-değişmez özellikler bu çevrilmiş davranışı ifade EDEMEZ. Vektör
ile referans birlikte döndüğü için `v·h` tam dönme-değişmez; yansıma yönelimi tersine
çevirdiği için `h_perp` işaret alır, yani `v·h_perp` ayna-TEK. Böylece özellik kümesi
aksiyonun kendi dönüşüm kuralıyla aynı şekilde dönüşüyor — `goal_body`'nin sol/sağ bileşeni
gibi. Dönme-değişmezliği G3'ün ihtiyaç duyduğu şey (`d4_orbit` simetrik düzenlerde dejenere
aynayı attığı için ayrılmış set dönmelerden oluşuyor).

Özellikler `[even | odd]` sırasıyla diziliyor ki test işaret desenini doğrulayabilsin.

**Konvansiyon kalibrasyonu (tahmin edilmedi, ölçüldü):** escnn'in vektör tabanı ile bizim
dünya çerçevemiz iki bileşenin takası kadar farklı — escnn'in dönmeleri
`transform_direction`'ın tersi çıkıyor ve yansıma ekseni diğeri. Havuzlanmış alanın dönüşüm
matrisi sekiz grup elemanı için en küçük karelerle çıkarıldı ve `transform_direction` ile
karşılaştırıldı; `(x,y)` takası ikisini birden gideriyor (`_SWAP_XY`).

**Doğrulama:** `scripts/test_equivariance.py` artık dünyayı gerçekten dönüştürüp (ızgara,
heading, goal_body — augmentation wrapper'ının kullandığı grup etkisiyle aynı) çıktıyı
karşılaştırıyor. Sonuç: **sekiz elemanın tamamında bağıl hata 0.00000**. Ayrıca SB3 ile uçtan
uca koştu: cuda, 466.760 parametre, 150 adım eğitim sorunsuz.

Not: vektör alanı katmanından sonra noktasal ReLU yok — irrep alanlarında noktasal
doğrusal-olmayanlık eşdeğişken değildir.
