#!/bin/sh
# Runs on a Hugging Face job.   sh job.sh FIRST LAST WIDTH SAMPLES NAME
# Builds the scene, renders the frames, encodes the video and uploads it to the dataset repo.
set -e
A=${1:-0}; B=${2:-959}; W=${3:-1920}; SPP=${4:-32}; NAME=${5:-voynich_journey}; DEST=${6:-}
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq ffmpeg libxxf86vm1 libxi6 libxkbcommon0 libgl1 libegl1 libsm6 libxrender1 libxfixes3 libgomp1 >/dev/null
pip install -q uv && uv venv -q --python 3.13 /venv && . /venv/bin/activate
uv pip install -q bpy pillow huggingface_hub
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"
cd "$(dirname "$0")"
export VW_WORK=/work; mkdir -p /work
python build.py 2>&1 | tail -1
T0=$(date +%s)
P=${VW_PARTS:-1}                      # render processes sharing the one graphics card
for k in $(seq 0 $((P-1))); do
  VW_PART=$k/$P python render.py $A $B $W $SPP /work/frames 2>&1 | grep --line-buffered -E "^frame|DEVICE|rror|Traceback" &
done
wait
T1=$(date +%s); N=$(ls /work/frames | wc -l); echo "RENDERED $N frames in $((T1-T0)) s"
ffmpeg -loglevel error -y -framerate 24 -start_number $A -i /work/frames/%04d.png -c:v libx264 -crf 15 -preset slow -pix_fmt yuv420p -movflags +faststart /work/$NAME.mp4
python - <<PY
import glob, os, time
from PIL import Image
from huggingface_hub import HfApi
fs = sorted(glob.glob('/work/frames/*.png')); pick = [fs[i * (len(fs) - 1) // 5] for i in range(6)]
s = Image.new('RGB', (1920, 720))
for i, f in enumerate(pick): s.paste(Image.open(f).convert('RGB').resize((640, 360)), ((i % 3) * 640, (i // 3) * 360))
s.save('/work/${NAME}_sheet.jpg', quality=88)
api = HfApi()
for f in ('/work/$NAME.mp4', '/work/${NAME}_sheet.jpg'):
    for attempt in range(8):                      # several jobs commit to the same repo at once
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo='$DEST' + os.path.basename(f), repo_id='Digitalgoldfish79/voynich-broll', repo_type='dataset'); break
        except Exception as e:
            print('upload retry', attempt, type(e).__name__); time.sleep(5 + 7 * attempt)
    print('UPLOADED', os.path.basename(f), os.path.getsize(f))
PY
