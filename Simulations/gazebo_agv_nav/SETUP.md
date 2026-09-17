# Kurulum ve Çalıştırma — tek dosya

Yeni bir makinede sıfırdan eğitim koşmak için gereken her şey. Yukarıdan aşağıya
uygulanabilir; başka dosyaya bakmanız gerekmez.

Hedef: Linux + NVIDIA GPU. Docker ve root yetkisi **gerekmez** — her şey kullanıcı
alanında micromamba ile kurulur. (Devir notundaki Docker tabanlı kurulum artık
kullanılmıyor, bkz. `Resources/HANDOFF.md` §10.)

---

## 0. Hızlı yol: install.sh

Bölüm 1-3'ün tamamını tek komut yapar (depo + ROS2/Gazebo ortamı + Python paketleri),
hepsini tek bir konuma kurar, sisteme hiçbir şey yazmaz, root istemez:

```bash
git clone https://github.com/GulogluHaktan/AGV.git /tmp/AGV && cd /tmp/AGV
./install.sh                       # varsayilan: /media/noron/DISK02/agv
./install.sh --root /baska/yol     # baska konum
./install.sh --keep-repo           # depoyu tasima, sadece ortami kur
```

Betik ~25 GB boş alan ve GPU'yu kontrol eder, `micromamba` ikilisini `<root>/bin`'e,
conda ortamını `<root>/micromamba`'ya, depoyu `<root>/AGV`'ye kurar. Sonunda
`Simulations/gazebo_agv_nav/env.local.sh` yazar; `sim_up.sh` ve `run_native.sh` bunu
okuyup ortamı nerede olursa olsun bulur (bu dosya makineye özeldir, git'e girmez).
Yeniden çalıştırılabilir: var olan adımları atlar.

Kurulum bittiğinde **Bölüm 5'teki üç kontrolü koşun**, sonra Bölüm 6 ile eğitime geçin.
Aşağıdaki 1-4 numaralı bölümler elle kurmak ya da ne olduğunu görmek isterseniz diye.

---

## 1. Depoyu al

```bash
git clone https://github.com/GulogluHaktan/AGV.git ~/Projects/AGV
cd ~/Projects/AGV/Simulations/gazebo_agv_nav
```

TurtleBot3 modelleri `models/` altında depoda gömülü — ayrıca indirmeniz gerekmez.

## 2. micromamba + ROS2/Gazebo ortamı

```bash
cd ~
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
~/bin/micromamba create -y -n agv -c conda-forge -c robostack-jazzy \
    ros-jazzy-ros-base ros-jazzy-ros-gz python=3.12 pip
```

Bu, ROS2 Jazzy + Gazebo Harmonic 8.x + `ros_gz_bridge` getirir. Birkaç GB iner.

## 3. Python paketleri — SIRA ÖNEMLİ

Önce Fortran derleyicisi: `escnn` → `py3nj` kaynaktan derleniyor ve `gfortran`
istiyor. Yoksa pip `Unknown compiler(s): gfortran ... metadata-generation-failed`
ile düşer. Ortama kurmak sudo istemez ve ortamın `bin`'i PATH'te sistemden önce
geldiği için meson bunu bulur:

```bash
~/bin/micromamba install -y -n agv -c conda-forge gfortran
```

Sonra paketler:

```bash
cd ~/Projects/AGV/Simulations/gazebo_agv_nav
~/bin/micromamba run -n agv pip install -r requirements-pip.txt
```

**Tek komutla, hepsini birden kurun.** Sebebi: `escnn` → `lie-learn` numpy<2 istiyor.
Önce numpy 2.x'li bir ortam kurup sonra escnn eklerseniz pip numpy'i sessizce düşürür
ve **numpy 2.x altında kaydedilmiş bütün checkpoint'ler okunamaz hâle gelir**
(`ModuleNotFoundError: No module named 'numpy._core.numeric'`). Bu bir koşuyu
baştan başlatmaya yol açtı. `requirements-pip.txt` numpy'i 1.26.4'e sabitliyor;
üç kolun tamamı aynı numeric yığında koşmalı, yoksa karşılaştırma savunulamaz.

Doğrulama:

```bash
~/bin/micromamba run -n agv python3 -c "
import numpy, torch, escnn, stable_baselines3, gymnasium
print(numpy.__version__, torch.__version__, escnn.__version__)
print('cuda:', torch.cuda.is_available())"
```

`cuda: True` görmelisiniz. Değilse torch GPU'yu görmüyor; eğitim 4-6 kat yavaşlar.

## 4. Dünyayı üret (opsiyonel)

Depodaki `worlds/agv_nav.sdf` taşınabilir (`model://` URI kullanır) ve doğrudan
çalışır. Izgara boyutunu ya da engel sayısını değiştirdiyseniz yeniden üretin:

```bash
PYTHONPATH=$PWD ~/bin/micromamba run -n agv python3 worlds/gen_world.py \
    --grid_size 24 --seed 3 --n_obstacle_pairs 14 --out worlds/agv_nav.sdf
```

`--grid_size` ve `--n_obstacle_pairs`, `envs/curriculum.py` içindeki `GRID_SIZE` ve
`N_OBSTACLE_PAIRS` ile **aynı olmalı**: env her episode'da haritayı gerçekleştirmek
için `obs_0..obs_{2N-1}` modellerini adıyla ışınlıyor ve dünyada yeterli model yoksa
hata verir.

## 5. GPU harcamadan önce üç kontrol

Dakikalar sürer ve **her biri, atlanırsa bir geceye mal olacak bir hatayı daha önce
yakaladı**. Her ortam/gözlem/ödül değişikliğinden sonra tekrar koşun.

```bash
./scripts/sim_up.sh 17

. scripts/_activate.sh          # ortami aktive eder (env.local.sh'i okur)
export GZ_IP=127.0.0.1 ROS_DOMAIN_ID=17 AGV_WORKSPACE=$PWD

python3 scripts/gate_test.py          # ortam çözülebilir mi?
python3 scripts/test_symmetry.py      # D4 grup etkisi doğru mu?
python3 scripts/test_equivariance.py  # 3. kol doğru dönüşüyor mu?

./scripts/sim_down.sh 17
```

Beklenen çıktılar:

- **gate_test** — `GATE PASSED`. Başarısızlıkları sınıflandırır: `wedged` (engele
  sıkışma) kontrolcünün tasarım gereği kaçınma mantığı olmamasından gelir, ortam
  hatası değildir; **`no_progress` bozuk ortamın imzasıdır** ve sıfıra yakın olmalı;
  `budget` adım bütçesinin dar olduğunu söyler. Referans: success %75-80,
  wedged %20-25, no_progress %0, budget %0.
- **test_symmetry** — `PASS`, 4808 kontrol.
- **test_equivariance** — `PASS`, sekiz D4 elemanında hata 0.00000.

## 6. Eğitim

Tek kol:

```bash
./scripts/run_native.sh baseline sac                sac_baseline_v7
./scripts/run_native.sh symmetric_augmentation sac  sac_aug_v1
./scripts/run_native.sh equivariant sac             sac_equi_v1
```

Betik simülasyonu kendi başlatır/kapatır, log'u `<prefix>.log`'a yazar, her aşama
sonunda `<prefix>_sX.zip` checkpoint'i kaydeder ve bitince `<prefix>.done` bırakır.

İki kolu eşzamanlı koşmak için ikincisine farklı bir ROS domain verin (aksi hâlde iki
simülasyon birbirinin topic'lerini görür):

```bash
AGV_ROS_DOMAIN_ID=23 ./scripts/run_native.sh symmetric_augmentation sac sac_aug_v1
```

Darboğaz simülasyon (CPU) tarafında; 12 çekirdekli bir makinede iki kol yan yana
her biri ~%70 hızda gider. GPU tek olduğu için `equivariant` kolunu tek başına
koşmak daha mantıklı.

İlerlemeyi izleme:

```bash
tail -f sac_baseline_v7.log | grep -E "STAGE|success-rate|WARNING"
```

## 7. Süreç yönetimi — dikkat

```bash
./scripts/sim_down.sh 17      # yalnızca o domain'i kapatır
./scripts/sim_down.sh --all   # bu projenin başlattığı her örneği kapatır
```

**`pkill -f "gz sim"` KULLANMAYIN.** `pkill -f` süreçlerin tam komut satırına bakar,
dolayısıyla komutu yazan kabuğun kendisiyle de eşleşir ve eşzamanlı iki örneği
ayırt edemez. Bu bir kez 170k adımdaki eğitim simülasyonunu öldürdü; belirti
`sim time only advanced 0.000s of 0.5s` uyarısıydı ve o koruma olmasa adımlar
sessizce fiziksiz geçecekti.

## 8. Sonuçları çıkarma

```bash
python3 scripts/extract_results.py sac_baseline_v7.log     # öğrenme eğrileri (CSV+JSON)
python3 scripts/eval_g3.py --ckpt sac_baseline_v7_s5.zip   # G3 genelleme tablosu
```

`eval_g3.py` üç koşulu ayrı ölçer: eğitim havuzu (dağılım-içi referans), eğitim
haritalarının D4 görüntüleri (görülmemiş yönelimler) ve ayrık tohumlu yeni düzenler.
`--seed` ve `--n_maps`, eğitim koşusuyla aynı olmalı; yoksa "train" koşulu
politikanın gerçekten eğitildiği havuz olmaz.

İzole bir simülasyonda eğitim sürerken değerlendirme yapabilirsiniz:

```bash
./scripts/sim_up.sh 42 evalpart
ROS_DOMAIN_ID=42 GZ_PARTITION=evalpart python3 scripts/eval_g3.py --ckpt ...
./scripts/sim_down.sh 42
```

## 9. Bilinen tuzaklar

| Belirti | Sebep / çözüm |
|---|---|
| `/odom` ya da `/gt_odom` hiç mesaj almıyor | `GZ_IP=127.0.0.1` eksik. GZ Transport'un keşfi bu olmadan kırılıyor; betikler ayarlıyor, elle koşuyorsanız siz ayarlayın. |
| `CONDA_BUILD: unbound variable` | Betikte `set -u` var. RoboStack'in `activate.d` betikleri tanımsız değişkene bakıyor; bu yüzden betikler `set -eo pipefail` kullanıyor, `-u` eklemeyin. |
| `ModuleNotFoundError: numpy._core.numeric` | Checkpoint numpy 2.x ile kaydedilmiş, ortam numpy 1.26. Bölüm 3'e bakın; checkpoint kurtarılamaz. |
| `Unknown compiler(s): gfortran` / `metadata-generation-failed` (py3nj) | Fortran derleyicisi yok. `micromamba install -y -n agv -c conda-forge gfortran`, sonra pip'i tekrar koşun. install.sh bunu kendi yapar. |
| `git@github.com: Permission denied (publickey)` | O makinede SSH anahtarı yok. Depo HTTPS ile anonim okunabilir: `git remote set-url origin https://github.com/GulogluHaktan/AGV.git`. Eğitim makinesinde yalnızca pull gerektiği için bu yeterli; push yapacaksanız SSH anahtarı ya da token gerekir. |
| `sim time only advanced 0.000s` | Gazebo ölmüş. Eğitimi durdurun — o adımlar fizik almıyor ve bozuk geçiş kaydediyor. |
| `HATA: domain 17 icin bir simulasyon zaten canli` | Aynı domain'de ikinci bir eğitim başlatmaya çalıştınız; betik reddediyor (yoksa birincinin simülasyonunu kapatır ve ikisi aynı log'a yazar). Farklı domain verin: `AGV_ROS_DOMAIN_ID=23 ./scripts/run_native.sh ...` |
| `map needs N obstacle models but the world only has M` | Dünya ile `envs/curriculum.py` uyuşmuyor; Bölüm 4'e göre yeniden üretin. |
| Eğitim `Using cpu device` diyor | torch GPU'yu görmüyor. `requirements-pip.txt`'teki torch CUDA derlemesini ve sürücüyü kontrol edin. |

## 10. Beklenen performans

Geliştirme makinesi (RTX 4060 Laptop, i5-12450HX, 12 iş parçacığı): eğitim sırasında
**18-21 fps**, yani 830k adımlık curriculum **kol başına ~12 saat**. Simülasyon
darboğaz olduğu için hızlı çekirdekler büyük ekran kartından daha çok fark eder.

Adım bütçeleri `envs/curriculum.py` içinde. Toplam 830k adımın 500k'sı s4+s5'te,
yani kısaltma kaldıracı orada. Karar vermeden önce s3'ü bitmiş bir baseline koşusunun
eğrisine bakın: `extract_results.py` çıktısında bir aşama ilk çeyrekten son çeyreğe
anlamlı ilerlemiyorsa o bütçe kısaltılabilir.
