#!/usr/bin/env bash
# Kur: AGV simetri-DRL projesinin TAMAMINI tek bir konuma kurar.
#
#   ./install.sh                          # /media/noron/DISK02/agv altina kurar
#   ./install.sh --root /baska/yol        # baska bir konuma
#   ./install.sh --keep-repo              # depoyu tasima, sadece ortami kur
#
# Ne kurulur (hepsi --root altinda, sisteme hicbir sey yazilmaz):
#   <root>/bin/micromamba      micromamba ikilisi
#   <root>/micromamba          conda ortami (ROS2 Jazzy + Gazebo Harmonic + torch/escnn)
#   <root>/AGV                 depo (--keep-repo verilmezse buraya klonlanir)
#
# root ve sudo GEREKMEZ. Yeniden calistirilabilir: var olan adimlari atlar.
#
# Kurulum bitince <repo>/Simulations/gazebo_agv_nav/env.local.sh yazilir; sim_up.sh
# ve run_native.sh bunu okuyup ortami nerede olursa olsun bulur.
#
# -u YOK: RoboStack'in activate.d betikleri tanimsiz degiskene bakiyor (CONDA_BUILD).
set -eo pipefail

ROOT="/media/noron/DISK02/agv"
REPO_URL="git@github.com:GulogluHaktan/AGV.git"
ENV_NAME="agv"
KEEP_REPO=0
NEED_GB=25

while [ $# -gt 0 ]; do
  case "$1" in
    --root)       ROOT="$2"; shift 2 ;;
    --repo-url)   REPO_URL="$2"; shift 2 ;;
    --env-name)   ENV_NAME="$2"; shift 2 ;;
    --keep-repo)  KEEP_REPO=1; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "bilinmeyen secenek: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nHATA: %s\n' "$*" >&2; exit 1; }

SRC_REPO="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------- on kontroller
say "On kontroller"

PARENT="$(dirname "$ROOT")"
[ -d "$PARENT" ] || die "hedefin ust dizini yok: $PARENT
Disk bagli mi? 'lsblk' ile kontrol edin, gerekirse baska bir yol icin --root kullanin."
[ -w "$PARENT" ] || die "$PARENT yazilabilir degil (kullanici: $(whoami))"

AVAIL_GB=$(df -BG --output=avail "$PARENT" | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge "$NEED_GB" ] || die "yetersiz alan: ${AVAIL_GB}GB bos, ~${NEED_GB}GB gerekiyor
(conda ortami tek basina ~12GB: ROS2 + Gazebo + torch/CUDA kutuphaneleri)"
echo "  hedef      : $ROOT"
echo "  bos alan   : ${AVAIL_GB}GB (gereken ~${NEED_GB}GB)"

command -v curl >/dev/null || die "curl bulunamadi (apt install curl)"
command -v git  >/dev/null || die "git bulunamadi (apt install git)"
command -v tar  >/dev/null || die "tar bulunamadi"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
  echo "  GPU        : ${GPU:-nvidia-smi calisti ama GPU listelenemedi}"
else
  echo "  GPU        : nvidia-smi yok -- egitim CPU'ya duser ve 4-6 kat yavaslar"
fi

mkdir -p "$ROOT/bin"

# ------------------------------------------------------------------ micromamba
MM="$ROOT/bin/micromamba"
if [ -x "$MM" ]; then
  say "micromamba zaten var ($("$MM" --version))"
else
  say "micromamba indiriliyor"
  ( cd "$ROOT" && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj bin/micromamba )
  [ -x "$MM" ] || die "micromamba kurulamadi"
  echo "  surum: $("$MM" --version)"
fi

export MAMBA_ROOT_PREFIX="$ROOT/micromamba"

# ------------------------------------------------------------------------ depo
if [ "$KEEP_REPO" = "1" ]; then
  REPO="$SRC_REPO"
  say "depo oldugu yerde birakiliyor: $REPO"
elif [ "${SRC_REPO#"$ROOT"}" != "$SRC_REPO" ]; then
  REPO="$SRC_REPO"
  say "depo zaten hedef altinda: $REPO"
else
  REPO="$ROOT/AGV"
  if [ -d "$REPO/.git" ]; then
    say "depo zaten klonlanmis: $REPO"
  else
    say "depo klonlaniyor -> $REPO"
    # once uzaktan; SSH anahtari yoksa yereldeki klondan kopyala (gecmis korunur)
    if ! git clone "$REPO_URL" "$REPO" 2>/dev/null; then
      echo "  uzaktan klonlama basarisiz (SSH anahtari?), yerel klondan kopyalaniyor"
      git clone "$SRC_REPO" "$REPO"
      git -C "$REPO" remote set-url origin "$REPO_URL"
    fi
  fi
fi

WS="$REPO/Simulations/gazebo_agv_nav"
[ -f "$WS/requirements-pip.txt" ] || die "beklenen dosya yok: $WS/requirements-pip.txt"

# --------------------------------------------------------------- conda ortami
if "$MM" env list 2>/dev/null | grep -qE "^\s+$ENV_NAME\s"; then
  say "conda ortami '$ENV_NAME' zaten var, atlaniyor"
else
  say "ROS2 Jazzy + Gazebo Harmonic kuruluyor (birkac GB, uzun surer)"
  "$MM" create -y -n "$ENV_NAME" -c conda-forge -c robostack-jazzy \
      ros-jazzy-ros-base ros-jazzy-ros-gz python=3.12 pip
fi

# --------------------------------------------------------------- pip paketleri
# TEK komutla: escnn -> lie-learn numpy<2 istiyor. Once numpy 2.x kurup sonra
# escnn eklemek numpy'i sessizce dusurur ve numpy 2.x ile kaydedilmis butun
# checkpoint'ler okunamaz hale gelir. Bu bir kosuyu bastan baslatmaya yol acti.
say "Python paketleri kuruluyor (torch/CUDA dahil, ~3GB iner)"
"$MM" run -n "$ENV_NAME" pip install -r "$WS/requirements-pip.txt"

# ------------------------------------------------------------- env.local.sh
say "env.local.sh yaziliyor"
cat > "$WS/env.local.sh" <<EOF
# install.sh tarafindan uretildi -- bu makineye ozeldir, git'e girmez.
# scripts/_activate.sh bunu okur, boylece ortam nerede olursa olsun bulunur.
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX"
export MICROMAMBA_BIN="$MM"
export AGV_ENV_NAME="$ENV_NAME"
EOF
echo "  $WS/env.local.sh"

# --------------------------------------------------------------- dogrulama
say "Dogrulama"
"$MM" run -n "$ENV_NAME" python3 - <<'PY'
import numpy, torch, scipy, escnn, stable_baselines3 as sb3, gymnasium, matplotlib
print(f"  numpy {numpy.__version__}  torch {torch.__version__}  scipy {scipy.__version__}")
print(f"  escnn {escnn.__version__}  sb3 {sb3.__version__}  gymnasium {gymnasium.__version__}")
assert numpy.__version__.startswith("1."), "numpy 2.x kurulmus: escnn ile uyumsuz, checkpoint'ler kirilir"
if torch.cuda.is_available():
    print(f"  cuda: EVET ({torch.cuda.get_device_name(0)})")
else:
    print("  cuda: HAYIR -- egitim CPU'da 4-6 kat yavas kosar")
PY
"$MM" run -n "$ENV_NAME" bash -c 'command -v gz >/dev/null && gz sim --versions 2>/dev/null | head -1 | sed "s/^/  gazebo /" || echo "  UYARI: gz bulunamadi"'
"$MM" run -n "$ENV_NAME" bash -c 'ros2 pkg list 2>/dev/null | grep -q ros_gz_bridge && echo "  ros_gz_bridge: var" || echo "  UYARI: ros_gz_bridge yok"'

# ------------------------------------------------------------------- bitti
cat <<EOF

==> Kurulum tamam.

Depo   : $REPO
Ortam  : $MAMBA_ROOT_PREFIX (env: $ENV_NAME)

Siradaki adim -- GPU harcamadan once uc kontrolu kosun. Dakikalar surer ve her biri
daha once bir geceye mal olacak bir hatayi yakaladi:

  cd $WS
  ./scripts/sim_up.sh 17
  . scripts/_activate.sh
  export GZ_IP=127.0.0.1 ROS_DOMAIN_ID=17 AGV_WORKSPACE=\$PWD
  python3 scripts/gate_test.py          # ortam cozulebilir mi?
  python3 scripts/test_symmetry.py      # D4 grup etkisi dogru mu?
  python3 scripts/test_equivariance.py  # 3. kol dogru donusuyor mu?
  ./scripts/sim_down.sh 17

Uculu de gectikten sonra egitim:

  ./scripts/run_native.sh baseline sac sac_baseline_v7

Ayrintilar, beklenen cikti degerleri ve bilinen tuzaklar: $WS/SETUP.md
EOF
