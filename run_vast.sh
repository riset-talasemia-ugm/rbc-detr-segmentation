#!/usr/bin/env bash
# Jalankan seluruh pipeline di instance vast.ai.
#   bash run_vast.sh            # run penuh (5 fold)
#   bash run_vast.sh --smoke    # 1 fold, 1 epoch: cek seluruh jalur dengan biaya kecil
# Pipeline berjalan di nohup (koneksi SSH putus tidak mematikannya); log: outputs/run.log.
# Selesai bila log berakhir "PIPELINE SELESAI"; hasil dibungkus di outputs/results.tar.gz.
# Skrip ini TIDAK menghapus instance: unduh hasilnya (scp), lalu destroy instance sendiri agar tagihan berhenti.
set -euo pipefail
cd "$(dirname "$0")"

SMOKE=""
[ "${1:-}" = "--smoke" ] && SMOKE="--smoke"

# 1) GPU dulu: gagal cepat sebelum memasang apa pun
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "GPU tidak terdeteksi (nvidia-smi gagal). Jalankan skrip ini di instance GPU vast.ai." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2) uv + virtual environment + dependensi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
[ -d .venv ] || uv venv
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install -r requirements.txt

# 3) .env: pakai pemeriksa yang sama dengan train.py (pesan jelas bila ada yang kosong)
[ -f .env ] || { echo ".env tidak ada: salin .env.example ke .env dan isi." >&2; exit 1; }
python -c "from dotenv import load_dotenv; load_dotenv(); from train import load_env; load_env()"

# 4) pipeline di nohup; hasil dibungkus bila semua langkah berhasil
mkdir -p outputs
nohup bash -c "python train.py $SMOKE && python compress.py $SMOKE && python evaluate.py $SMOKE \
  && tar czf outputs/results.tar.gz -C outputs results && echo PIPELINE SELESAI || echo PIPELINE GAGAL" \
  > outputs/run.log 2>&1 &
echo "Pipeline berjalan di latar belakang (PID $!). Pantau: tail -f outputs/run.log"
echo "Setelah 'PIPELINE SELESAI': scp outputs/results.tar.gz ke komputer Anda, lalu destroy instance."
