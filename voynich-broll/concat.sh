#!/bin/sh
# Joins the segment videos in the dataset repo into one file.   sh concat.sh NAME
set -e
NAME=${1:-voynich_journey_40s_1080p}
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null
pip install -q huggingface_hub pillow
python - <<PY
import subprocess, os
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
d = snapshot_download('Digitalgoldfish79/voynich-broll', repo_type='dataset', allow_patterns='segments/*', local_dir='/work')
segs = sorted(f for f in os.listdir('/work/segments') if f.endswith('.mp4')); print('SEGMENTS', segs)
open('/work/list.txt', 'w').write(''.join("file '/work/segments/%s'\n" % s for s in segs))
subprocess.check_call(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', '/work/list.txt', '-c', 'copy', '-movflags', '+faststart', '/work/$NAME.mp4'])
print(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height', '-of', 'csv=p=0', '/work/$NAME.mp4']).decode())
sheet = Image.new('RGB', (1920, 1440))
for i in range(12):
    subprocess.check_call(['ffmpeg', '-loglevel', 'error', '-y', '-ss', str(1.5 + i * 3.4), '-i', '/work/$NAME.mp4', '-frames:v', '1', '-vf', 'scale=640:360', '/work/s.png'])
    sheet.paste(Image.open('/work/s.png'), ((i % 3) * 640, (i // 3) * 360))
sheet.save('/work/${NAME}_sheet.jpg', quality=88)
api = HfApi()
for f in ('/work/$NAME.mp4', '/work/${NAME}_sheet.jpg'):
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id='Digitalgoldfish79/voynich-broll', repo_type='dataset'); print('UPLOADED', os.path.basename(f), os.path.getsize(f))
PY
