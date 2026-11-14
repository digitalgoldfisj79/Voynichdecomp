"""Renders frames A..B.   python render.py A B [WIDTH] [SAMPLES] [OUTDIR]
Uses a graphics card if Blender can find one, otherwise the processor. Frames on disk are skipped."""
import bpy, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from journey import *
from mathutils import Vector
WORK = os.environ.get('VW_WORK', HERE)
a, b = int(sys.argv[1]), int(sys.argv[2])
width = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
samples = int(sys.argv[4]) if len(sys.argv) > 4 else 16
out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(WORK, 'frames')
os.makedirs(out, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=os.path.join(WORK, 'scene.blend'))
S = bpy.context.scene; O = bpy.data.objects
S.render.resolution_x = width; S.render.resolution_y = width * 9 // 16; S.render.resolution_percentage = 100
S.cycles.samples = samples; S.render.image_settings.file_format = 'PNG'
S.render.use_persistent_data = True            # keep textures and geometry loaded between frames
PART, PARTS = [int(v) for v in os.environ.get('VW_PART', '0/1').split('/')]   # this process renders every PARTS-th frame

device = 'CPU'
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for kind in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = kind; prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices: d.use = d.type == kind
                S.cycles.device = 'GPU'; device = kind + ': ' + gpus[0].name; break
        except Exception as e: pass
except Exception as e: print('no cycles prefs', e)
if device == 'CPU': S.cycles.device = 'CPU'
print('RENDER DEVICE', device, flush=True)

plants = [o for o in O if o.name.startswith('PLANT_')]
nymphs = [o for o in O if o.name.startswith('NYMPH_')]

def face(o, pos, extra_roll=0.0, flip=1):
    d = pos - o.location; o.rotation_euler = (0, extra_roll, math.atan2(d.x, -d.y)); o.scale.x = flip

def pose(t):
    pos, tgt = camera_at(t)
    C = O['CAM_walk']; C.location = pos; C.rotation_euler = (tgt - pos).to_track_quat('-Z', 'Y').to_euler()
    for o in plants:
        o.rotation_euler = (0, math.radians(1.3) * math.sin(t * 1.1 + o['sway']) + math.radians(0.5) * math.sin(t * 2.7 + o['sway'] * 2), o['yaw'])
    for o in nymphs:
        x, y, dz, lean, facing = nymph_pose(o['i'], o['phase'], o['x0'], o['y0'], t)
        o.location = (x, y, WATER_Z - 0.4 + dz); face(o, pos, lean, facing)
    for i in range(N_FISH):
        x, y, z, tilt, h, vis = fish_pose(i, t)
        f = O['FISH_%d' % i]; f.location = (x, y, z); face(f, pos, -tilt * h, 1 if h < 0 else -1); f.hide_render = not vis   # the drawn fish faces left
    for k in range(len(RAMS)):
        x, y, z, tilt, vis = ram_pose(k, t)
        r = O['RAM_%d' % k]; r.location = (x, y, z); face(r, pos, -tilt, -1); r.hide_render = not vis                      # drawn facing left; they run to the right
    A = O['ARCHER']; face(A, pos, archer_pose(t), 1)
    x, y, z, vis = creature_pose(t)
    c = O['CREATURE_76v']; c.location = (x, y, z); c.hide_render = not vis
    bpy.data.materials['water'].node_tree.nodes['ripple'].inputs['W'].default_value = t * 0.35

for f in range(a, b + 1):
    path = '%s/%04d.png' % (out, f)
    if f % PARTS != PART or os.path.exists(path): continue
    pose(f / FPS); S.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print('frame', f, flush=True)
