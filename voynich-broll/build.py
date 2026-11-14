"""Builds the journey scene and saves scene.blend.   python build.py"""
import bpy, math, os, random, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from journey import *

ASSETS = os.environ.get('VW_ASSETS', os.path.join(HERE, '..', 'voynich-world'))
WORK = os.environ.get('VW_WORK', HERE)
rng = random.Random(408)

bpy.ops.wm.read_factory_settings(use_empty=True)
S = bpy.context.scene; col = S.collection

# ---------------------------------------------------------------- ground
X0, X1, Y0, Y1, STEP = -48, 48, -8, 118, 0.4
nx, ny = int((X1 - X0) / STEP) + 1, int((Y1 - Y0) / STEP) + 1
verts = [(X0 + i * STEP, Y0 + j * STEP, ground(X0 + i * STEP, Y0 + j * STEP)) for j in range(ny) for i in range(nx)]
faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i) for j in range(ny - 1) for i in range(nx - 1)]
me = bpy.data.meshes.new('ground'); me.from_pydata(verts, [], faces); me.shade_smooth()
G = bpy.data.objects.new('ENV_ground', me); col.objects.link(G)
m = bpy.data.materials.new('parchment_ground'); m.use_nodes = True
nt = m.node_tree; b = nt.nodes['Principled BSDF']
b.inputs['Roughness'].default_value = 1.0; b.inputs['Specular IOR Level'].default_value = 0.0
tc = nt.nodes.new('ShaderNodeTexCoord'); n1 = nt.nodes.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = 0.09; n1.inputs['Detail'].default_value = 6
n2 = nt.nodes.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = 3.0; n2.inputs['Detail'].default_value = 8
ramp = nt.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].color = (0.62, 0.46, 0.22, 1); ramp.color_ramp.elements[0].position = 0.3
ramp.color_ramp.elements[1].color = (0.86, 0.69, 0.38, 1); ramp.color_ramp.elements[1].position = 0.7
mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'; mix.inputs['Factor'].default_value = 0.25
nt.links.new(tc.outputs['Object'], n1.inputs['Vector']); nt.links.new(tc.outputs['Object'], n2.inputs['Vector'])
nt.links.new(n1.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], mix.inputs['A']); nt.links.new(n2.outputs['Color'], mix.inputs['B'])
nt.links.new(mix.outputs['Result'], b.inputs['Base Color'])
bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.15
nt.links.new(n2.outputs['Fac'], bump.inputs['Height']); nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
me.materials.append(m)

# ---------------------------------------------------------------- water
px, py, pr = POND
bpy.ops.mesh.primitive_circle_add(vertices=128, radius=pr + 0.3, fill_type='NGON', location=(px, py, WATER_Z))
W = bpy.context.object; W.name = 'ENV_water'
m = bpy.data.materials.new('water'); m.use_nodes = True; nt = m.node_tree; b = nt.nodes['Principled BSDF']
b.inputs['Base Color'].default_value = (0.16, 0.36, 0.42, 1); b.inputs['Roughness'].default_value = 0.08
wn = nt.nodes.new('ShaderNodeTexNoise'); wn.name = 'ripple'; wn.noise_dimensions = '4D'; wn.inputs['Scale'].default_value = 1.4; wn.inputs['Detail'].default_value = 3
wb = nt.nodes.new('ShaderNodeBump'); wb.inputs['Strength'].default_value = 0.14
nt.links.new(wn.outputs['Fac'], wb.inputs['Height']); nt.links.new(wb.outputs['Normal'], b.inputs['Normal'])
W.data.materials.append(m)

# ---------------------------------------------------------------- cards
def image_material(path):
    name = os.path.basename(path)[:-4]
    if name in bpy.data.materials: return bpy.data.materials[name]
    img = bpy.data.images.load(path); img.alpha_mode = 'STRAIGHT'; img.pack()
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; b = nt.nodes['Principled BSDF']
    t = nt.nodes.new('ShaderNodeTexImage'); t.image = img; t.extension = 'CLIP'
    nt.links.new(t.outputs['Color'], b.inputs['Base Color']); nt.links.new(t.outputs['Alpha'], b.inputs['Alpha'])
    nt.links.new(t.outputs['Color'], b.inputs['Emission Color']); b.inputs['Emission Strength'].default_value = 0.35
    b.inputs['Roughness'].default_value = 1.0; b.inputs['Specular IOR Level'].default_value = 0.0
    m['aspect'] = img.size[0] / img.size[1]
    return m

def card(name, path, height, loc, yaw=0.0, origin='bottom'):
    m = image_material(path); w = height * m['aspect']
    z0, z1 = (0, height) if origin == 'bottom' else (-height / 2, height / 2)
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-w / 2, 0, z0), (w / 2, 0, z0), (w / 2, 0, z1), (-w / 2, 0, z1)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name='UVMap')
    for i, c in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]): uv.data[i].uv = c
    me.materials.append(m)
    o = bpy.data.objects.new(name, me); o.location = loc; o.rotation_euler = (0, 0, yaw); col.objects.link(o)
    return o

plants = sorted(f for f in os.listdir(ASSETS + '/plants') if f.endswith('.png'))
placed, manifest = [], []
free = lambda x, y, r: all((x - a) ** 2 + (y - b) ** 2 > (r + c) ** 2 for a, b, c in placed)
placed.append((ARCHER[0], ARCHER[1], 1.6))                                   # keep the archer's knoll clear

def plant_at(x, y, h, f):
    yaw = rng.uniform(-0.3, 0.3)
    o = card('PLANT_%03d_%s' % (len(manifest), f[:-10]), ASSETS + '/plants/' + f, h, (x, y, ground(x, y) - 0.03 * h), yaw)
    o['sway'] = rng.uniform(0, 6.28); o['yaw'] = yaw
    placed.append((x, y, h * bpy.data.materials[f[:-4]]['aspect'] * 0.35)); manifest.append({'object': o.name, 'folio': f[1:-10], 'height_m': round(h, 2)})

plant_at(LANDMARK[0], LANDMARK[1], 26.0, 'f93r_plant.png')                   # the giant at the end of the walk
order = plants[:]; rng.shuffle(order); k = tries = 0
while len(manifest) < 430 and tries < 60000:
    tries += 1
    y = rng.uniform(-2, 104); side = rng.choice((-1, 1)); w = valley_half_width(y)
    d = max(w, 2.6) + abs(rng.gauss(0, 8)); x = path_x(y) + side * d
    if w > 6 and rng.random() < 0.18: d = rng.uniform(3.5, w); x = path_x(y) + side * d      # a few low plants dotted in the meadow
    if not (X0 + 3 < x < X1 - 3) or math.hypot(x - px, y - py) < pr + 0.8: continue
    if math.hypot(x - LANDMARK[0], y - LANDMARK[1]) < 9: continue
    edge = d - w
    h = rng.uniform(0.9, 2.0) if edge < 0 else rng.uniform(1.2, 2.8) if edge < 4 else rng.uniform(2.5, 5.5) if edge < 12 else rng.uniform(5, 11)
    f = order[k % len(order)]
    if not free(x, y, h * 0.2): continue
    plant_at(x, y, h, f); k += 1

NYMPHS = ['nymph_75r_1', 'nymph_75r_3', 'nymph_75r_4', 'nymph_76v_0', 'nymph_76v_1', 'nymph_76v_2', 'nymph_76v_3', 'nymph_77r_2']
for i, n in enumerate(NYMPHS):
    a = 2.0 + i * 0.5 + rng.uniform(-0.08, 0.08); r = pr * rng.uniform(0.4, 0.78)       # on the path side of the pond
    x, y = px + math.cos(a) * r, py + math.sin(a) * r
    o = card('NYMPH_%d_%s' % (i, n[6:]), ASSETS + '/nymphs/' + n + '.png', 1.5, (x, y, WATER_Z - 0.4))
    o['phase'] = rng.uniform(0, 6.28); o['x0'] = x; o['y0'] = y; o['i'] = i
    manifest.append({'object': o.name, 'folio': n.split('_')[1], 'height_m': 1.5})

A = ASSETS + '/animals/'
for i in range(N_FISH):
    card('FISH_%d' % i, A + 'fish_pisces_70v2.png', 0.55, (px, py, WATER_Z - 2), origin='centre')
    manifest.append({'object': 'FISH_%d' % i, 'folio': '70v2', 'height_m': 0.55})
for k, (which, t0, y, ph) in enumerate(RAMS):
    f = ('ram_aries_70v1.png', 'ram_aries_71r.png')[which]
    card('RAM_%d' % k, A + f, 1.25, (0, y, 0), origin='bottom'); manifest.append({'object': 'RAM_%d' % k, 'folio': ('70v1', '71r')[which], 'height_m': 1.25})
card('ARCHER', A + 'archer_sagittarius_73v.png', 1.9, (ARCHER[0], ARCHER[1], ground(*ARCHER) - 0.05)); manifest.append({'object': 'ARCHER', 'folio': '73v', 'height_m': 1.9})
card('CREATURE_76v', A + 'creature_76v.png', 0.85, (0, 15, 0)); manifest.append({'object': 'CREATURE_76v', 'folio': '76v', 'height_m': 0.85})

# ---------------------------------------------------------------- light, sky, camera
bpy.ops.object.light_add(type='SUN'); L = bpy.context.object; L.name = 'LGT_sun'
L.rotation_euler = (math.radians(52), 0, math.radians(-38)); L.data.energy = 3.2; L.data.angle = math.radians(2.5); L.data.color = (1, 0.95, 0.86)
wld = bpy.data.worlds.new('sky'); S.world = wld; wld.use_nodes = True; nt = wld.node_tree
bg = nt.nodes['Background']; bg.inputs['Strength'].default_value = 1.0
tc = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ'); ramp = nt.nodes.new('ShaderNodeValToRGB')
ramp.color_ramp.elements[0].position = 0.0; ramp.color_ramp.elements[0].color = (0.86, 0.80, 0.62, 1)
ramp.color_ramp.elements[1].position = 0.45; ramp.color_ramp.elements[1].color = (0.42, 0.60, 0.72, 1)
nt.links.new(tc.outputs['Generated'], sep.inputs['Vector']); nt.links.new(sep.outputs['Z'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
cam = bpy.data.cameras.new('cam'); cam.lens = 26; cam.sensor_width = 36; cam.clip_end = 500
C = bpy.data.objects.new('CAM_walk', cam); col.objects.link(C); S.camera = C
S.render.engine = 'CYCLES'; S.render.fps = FPS; S.view_settings.view_transform = 'Standard'
S.cycles.max_bounces = 4; S.cycles.transparent_max_bounces = 24; S.cycles.use_denoising = True
S.cycles.caustics_reflective = False; S.cycles.caustics_refractive = False
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORK, 'scene.blend'))
json.dump(manifest, open(os.path.join(WORK, 'manifest.json'), 'w'), indent=1)
print('built', len(manifest), 'cards;', len({m['folio'] for m in manifest}), 'folios', flush=True)
