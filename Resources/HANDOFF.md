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
