// ============================================================================
// quality.js  -  shared pieces for the nicer picture and the steady 60 fps.
//
//   Q.noiseTex      a small tiling noise picture. The mist and the water read it
//                   instead of working noise out per pixel (much cheaper).
//   Q.paperTex      paper grain + fibres, laid over the ground up close.
//   Q.sun           where the sun is and how strong (kept up to date by sky.js);
//                   plants use it for light and for their shadows on the ground.
//   Q.heightTex     the ground heights as a picture, so plant shadows can lie
//                   on the slopes instead of floating.
//   Q.resolution    keeps the frame rate at settings.quality.targetFps by
//                   drawing at a slightly lower resolution when the computer
//                   struggles, and back up when it can.
// Everything here can be switched off in settings.js -> quality.
// ============================================================================

const Q = (function () {
  function makeNoiseTex() {                      // 128x128, four independent channels, tiles seamlessly
    const N = 128, data = new Uint8Array(N * N * 4), rng = makeRng(7, 'noise-texture');
    for (let i = 0; i < data.length; i++) data[i] = Math.floor(rng.next() * 256);
    const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.magFilter = THREE.LinearFilter; t.minFilter = THREE.LinearFilter;
    t.needsUpdate = true;
    return t;
  }

  function makePaperTex() {                      // grey around 1.0: multiplied over the ground colour
    const N = 512, c = document.createElement('canvas'); c.width = c.height = N;
    const x = c.getContext('2d'), rng = makeRng(11, 'paper');
    x.fillStyle = 'rgb(232,232,232)'; x.fillRect(0, 0, N, N);
    const img = x.getImageData(0, 0, N, N), d = img.data;
    for (let i = 0; i < d.length; i += 4) { const g = 222 + rng.next() * 22; d[i] = d[i + 1] = d[i + 2] = g; }   // fine grain
    x.putImageData(img, 0, 0);
    x.globalAlpha = 0.18; x.lineCap = 'round';
    for (let k = 0; k < 900; k++) {                                    // fibres (drawn three times so they wrap at the edges)
      const px = rng.next() * N, py = rng.next() * N, a = rng.next() * Math.PI, l = 6 + rng.next() * 26;
      x.strokeStyle = rng.next() < 0.5 ? '#a8a8a8' : '#ffffff'; x.lineWidth = 0.6 + rng.next() * 0.9;
      for (const ox of [-N, 0, N]) for (const oy of [-N, 0, N]) {
        if (px + ox < -40 || px + ox > N + 40 || py + oy < -40 || py + oy > N + 40) continue;
        x.beginPath(); x.moveTo(px + ox, py + oy); x.quadraticCurveTo(px + ox + Math.cos(a + 0.4) * l * 0.5, py + oy + Math.sin(a + 0.4) * l * 0.5, px + ox + Math.cos(a) * l, py + oy + Math.sin(a) * l); x.stroke();
      }
    }
    x.globalAlpha = 0.07;
    for (let k = 0; k < 60; k++) {                                     // faint blotches, like old paper
      const px = rng.next() * N, py = rng.next() * N, r = 10 + rng.next() * 50;
      const g = x.createRadialGradient(px, py, 0, px, py, r); g.addColorStop(0, '#806a50'); g.addColorStop(1, 'rgba(128,106,80,0)');
      x.fillStyle = g; x.fillRect(px - r, py - r, r * 2, r * 2);
    }
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 2;
    return t;
  }

  const sun = {
    dir: { value: new THREE.Vector3(0.3, 0.9, -0.3).normalize() },
    color: { value: new THREE.Color(1, 1, 1) },
    power: { value: 1 },                         // 0 at night
    ambient: { value: new THREE.Color(1, 1, 1) },
  };

  // Ground heights as a float picture (for plant shadows lying on the ground).
  const heightTex = { value: null }, heightInfo = { value: new THREE.Vector4(0, 0, 1, 1) };   // x0, z0, size, cells
  function setTerrain(T) {
    const n1 = T.N + 1;
    if (heightTex.value) heightTex.value.dispose();
    const t = new THREE.DataTexture(new Float32Array(T.H), n1, n1, THREE.RedFormat, THREE.FloatType);
    t.magFilter = t.minFilter = THREE.NearestFilter;            // float textures are not filterable everywhere; the shader blends
    t.needsUpdate = true;
    heightTex.value = t;
    heightInfo.value.set(-T.size / 2, -T.size / 2, T.size, T.N);
  }
  const heightGLSL = `
    uniform sampler2D uHeight; uniform vec4 uHeightInfo;
    float groundAt(vec2 xz) {
      vec2 g = clamp((xz - uHeightInfo.xy) / uHeightInfo.z * uHeightInfo.w, vec2(0.0), vec2(uHeightInfo.w - 0.001));
      vec2 i = floor(g), f = g - i; float n1 = uHeightInfo.w + 1.0;
      float h00 = texture2D(uHeight, (i + vec2(0.5, 0.5)) / n1).r, h10 = texture2D(uHeight, (i + vec2(1.5, 0.5)) / n1).r;
      float h01 = texture2D(uHeight, (i + vec2(0.5, 1.5)) / n1).r, h11 = texture2D(uHeight, (i + vec2(1.5, 1.5)) / n1).r;
      return (f.x + f.y <= 1.0) ? h00 + (h10 - h00) * f.x + (h01 - h00) * f.y : h11 + (h01 - h11) * (1.0 - f.x) + (h10 - h11) * (1.0 - f.y);
    }`;

  // ---- holding the frame rate ---------------------------------------------------------------
  // Measures how long frames take. If they take longer than the target allows, the picture is
  // drawn at a slightly lower resolution (the window size does not change); when there is room
  // again it creeps back up. The HUD shows fps and the resolution in use.
  const resolution = {
    scale: 1, fps: 60, _acc: 0, _n: 0, _good: 0, _t: 0,
    update(dt, renderer, S, applySize) {
      const QS = S.quality || {};
      this._acc += dt; this._n++; this._t += dt;
      if (this._t < 1) return;                                     // decide once a second
      const avg = this._acc / this._n; this.fps = 1 / avg;
      this._acc = 0; this._n = 0; this._t = 0;
      if (!QS.adaptiveResolution || (S.recording && S.recording.enabled)) { if (this.scale !== 1) { this.scale = 1; applySize(); } return; }
      const target = 1 / (QS.targetFps || 60);
      if (avg > target * 1.12 && this.scale > (QS.minResolution || 0.6)) {         // too slow: step down
        this.scale = Math.max(QS.minResolution || 0.6, this.scale * 0.88); this._good = 0; applySize();
      } else if (avg < target * 1.04) {                                           // on target: after a while, try a little sharper
        if (++this._good >= 4 && this.scale < 1) { this.scale = Math.min(1, this.scale * 1.06); this._good = 0; applySize(); }
      } else this._good = 0;
    },
  };

  // the camera position for our shaders (three.js only passes its own to some kinds of material)
  const camPos = { value: new THREE.Vector3() };
  return { noiseTex: makeNoiseTex(), paperTex: makePaperTex(), sun, heightTex, heightInfo, setTerrain, heightGLSL, resolution, camPos };
})();

// Paper grain on the ground: strong close by, fading out with distance (so far hills stay calm).
// strength 0..1 (settings.quality.paperGrain).
function addPaperGrain(material, strength) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uPaper = { value: Q.paperTex }; shader.uniforms.uPaperK = { value: strength }; shader.uniforms.uCamPos = Q.camPos;
    shader.vertexShader = 'varying vec3 vPaperW;\n' + shader.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vPaperW = (modelMatrix * vec4(position, 1.0)).xyz;');
    shader.fragmentShader = 'uniform sampler2D uPaper; uniform float uPaperK; uniform vec3 uCamPos; varying vec3 vPaperW;\n' + shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      {
        float dCam = length(vPaperW - uCamPos);
        float g1 = texture2D(uPaper, vPaperW.xz * 0.11).r;           // grain, fibres and blotches (one read; repeats every ~9 m)
        float near = 1.0 - smoothstep(25.0, 70.0, dCam);
        diffuseColor.rgb *= mix(1.0, (g1 / 0.91) * 0.8 + 0.2, uPaperK * near);
      }`);
  };
}

// Water: colour depends on the viewing angle (sky colour at a glance, pond colour looking down),
// with slow ripples. Uses the scene fog colour as "the sky".
function addWaterShading(material) {
  const u = { uTime: { value: 0 }, uNoise: { value: Q.noiseTex }, uSky: { value: new THREE.Color() } };
  material.userData.u = u;
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, u); shader.uniforms.uCamPos = Q.camPos;
    shader.vertexShader = 'varying vec3 vWaterW;\n' + shader.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\n vWaterW = (modelMatrix * vec4(position, 1.0)).xyz;');
    shader.fragmentShader = 'uniform float uTime; uniform sampler2D uNoise; uniform vec3 uCamPos; varying vec3 vWaterW;\n' + shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      {
        vec3 v = normalize(uCamPos - vWaterW);
        vec2 p = vWaterW.xz * 0.35;
        vec2 r = texture2D(uNoise, (p + vec2(uTime * 0.05, uTime * 0.03)) / 128.0 * 4.0).rg + texture2D(uNoise, (p * 1.7 - vec2(uTime * 0.04, -uTime * 0.06)) / 128.0 * 4.0).ba;
        float tilt = (r.x - r.y) * 0.12;                                  // ripples tilt the surface a little
        float fres = pow(1.0 - clamp(v.y + tilt, 0.0, 1.0), 4.0);
        vec3 sky = fogColor * 0.97;
        diffuseColor.rgb = mix(diffuseColor.rgb * (0.94 + tilt), sky, clamp(fres * 0.5, 0.0, 0.5));
        diffuseColor.rgb += smoothstep(0.62, 0.92, r.x * 0.5 + r.y * 0.5) * 0.05;   // glints
      }`);
  };
}

// Plants: a little light and shade. The card turned towards the sun is brighter, the lower part
// of each plant is darker (as if shaded by its own leaves and the ground), and every plant gets a
// slightly different tone so a clump of the same picture does not look stamped.
function addPlantShading(material) {
  const prev = material.onBeforeCompile;
  material.onBeforeCompile = (shader, r) => {
    if (prev) prev(shader, r);
    shader.uniforms.uSunDir = Q.sun.dir; shader.uniforms.uSunColor = Q.sun.color; shader.uniforms.uSunPower = Q.sun.power;
    shader.vertexShader = 'attribute float aQuad;\nuniform vec3 uSunDir; uniform vec3 uSunColor; uniform float uSunPower;\nvarying vec3 vShade;\n' +
      shader.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>
      {
        #ifdef USE_INSTANCING
          mat4 shM = instanceMatrix;
        #else
          mat4 shM = modelMatrix;
        #endif
        vec3 n = normalize(mix(shM[2].xyz, shM[0].xyz, aQuad));          // which way this card faces
        float facing = abs(dot(n, normalize(uSunDir)));
        float lit = mix(0.80, 1.07, facing * uSunPower + 0.35 * (1.0 - uSunPower));
        float ao = mix(0.60, 1.0, smoothstep(0.0, 0.42, clamp(position.y, 0.0, 1.0)));
        vec2 ip = shM[3].xz;
        float tone = fract(sin(dot(ip, vec2(41.3, 289.1))) * 15731.743);
        vec3 tint = mix(vec3(1.03, 1.0, 0.95), vec3(0.95, 0.99, 1.03), tone);  // warmer or cooler
        vShade = lit * ao * (0.95 + 0.1 * tone) * tint * mix(vec3(1.0), uSunColor, 0.2 * uSunPower);
      }`);
    shader.fragmentShader = 'varying vec3 vShade;\n' + shader.fragmentShader.replace('#include <map_fragment>', '#include <map_fragment>\n diffuseColor.rgb *= vShade;');
  };
}

// Plant shadows: the picture itself, flattened onto the ground away from the sun and darkened.
// It lies on the slopes (it reads the ground heights) and fades out with distance.
function addPlantShadow(material, S) {
  const prev = material.onBeforeCompile;
  material.onBeforeCompile = (shader, r) => {
    if (prev) prev(shader, r);
    shader.uniforms.uSunDir = Q.sun.dir; shader.uniforms.uHeight = Q.heightTex; shader.uniforms.uHeightInfo = Q.heightInfo;
    shader.uniforms.uFadeFar = { value: 0 }; shader.uniforms.uCamPos = Q.camPos;
    shader.uniforms.uFadeFar.value = (S.quality && S.quality.shadowDistance) || 60;
    shader.uniforms.uMinLen = { value: (S.quality && S.quality.shadowMinLength) ?? 0.7 };
    shader.vertexShader = 'uniform vec3 uSunDir; uniform float uFadeFar; uniform float uMinLen; uniform vec3 uCamPos; varying float vShadowFade;\n' + Q.heightGLSL + '\n' +
      shader.vertexShader.replace('#include <project_vertex>', `
      {
        mat4 im = instanceMatrix;
        vec3 root = (modelMatrix * im * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
        float w = length(im[0].xyz) * sign(determinant(mat3(im))), hgt = length(im[1].xyz);
        vec3 sd = normalize(uSunDir);
        vec2 away = -normalize(sd.xz + 1e-5) * clamp(length(sd.xz) / max(sd.y, 0.35), uMinLen, 2.6);   // low sun = long shadow; never shorter than uMinLen x height
        vec2 side = normalize(vec2(-sd.z, sd.x) + 1e-5);
        vec2 xz = root.xz + side * transformed.x * w + away * transformed.y * hgt;
        vec4 wp = vec4(xz.x, groundAt(xz) + 0.06, xz.y, 1.0);
        vec4 mvPosition = viewMatrix * wp;
        gl_Position = projectionMatrix * mvPosition;
        vShadowFade = (1.0 - smoothstep(uFadeFar * 0.6, uFadeFar, length(wp.xyz - uCamPos))) * mix(1.0, 0.45, clamp(transformed.y, 0.0, 1.0));
      }`);
    shader.fragmentShader = 'varying float vShadowFade;\n' + shader.fragmentShader.replace('#include <map_fragment>', '#include <map_fragment>\n diffuseColor.a *= vShadowFade;');
  };
}
