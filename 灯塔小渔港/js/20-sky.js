/**
 * 20-sky.js —— 天空穹顶、太阳光与环境光
 * 清晨：低角度暖色朝阳 + 粉橙到浅蓝的渐变 + 贴海面的薄云。
 * 天空的颜色函数以 GLSL 片段形式共享给海面，让海面反射的正是同一片天。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});
  var W = LH.W;

  var U = {
    uSunDir: { value: W.SUN_DIR.clone() },
    uTime: { value: 0 },
    uHorizonWarm: { value: LH.col('#ffc9a0') },
    uHorizonCool: { value: LH.col('#e6d6cc') },
    uZenith: { value: LH.col('#3a76c8') },
    uSunColor: { value: LH.col('#ffe1b4') },
    uFogColor: { value: LH.col('#ead2bd') }
  };

  LH.skyUniforms = U;

  // 共享给海面的天空颜色函数（依赖 uSunDir / uHorizonWarm / uHorizonCool / uZenith / uSunColor）
  LH.glsl = LH.glsl || {};
  LH.glsl.skyChunk = [
    'vec3 lhSkyColor(vec3 dir) {',
    '  dir = normalize(dir);',
    '  float h = clamp(dir.y, 0.0, 1.0);',
    '  float t = smoothstep(0.0, 0.46, h);',
    '  vec2 dxz = normalize(dir.xz + vec2(1e-5));',
    '  vec2 sxz = normalize(uSunDir.xz + vec2(1e-5));',
    '  float sunSide = max(dot(dxz, sxz), 0.0);',
    '  vec3 horizon = mix(uHorizonCool, uHorizonWarm, pow(sunSide, 1.6) * 0.82 + 0.18);',
    '  vec3 col = mix(horizon, uZenith, t);',
    '  float haze = exp(-h * 13.0);',
    '  col = mix(col, uHorizonWarm * 0.92 + vec3(0.03, 0.025, 0.02), haze * 0.38);',
    '  col = mix(uFogColor, col, smoothstep(-0.035, 0.075, dir.y));',
    '  float pinkBand = exp(-pow((h - 0.045) / 0.055, 2.0));',
    '  col = mix(col, vec3(1.0, 0.62, 0.55), pinkBand * 0.07);',
    '  float d = max(dot(dir, uSunDir), 0.0);',
    '  col += uSunColor * pow(d, 9.0) * 0.30;',
    '  col += uSunColor * pow(d, 90.0) * 0.55;',
    '  return col;',
    '}'
  ].join('\n');

  var VERT = [
    'varying vec3 vDir;',
    'void main() {',
    '  vDir = position;',
    '  vec4 mv = modelViewMatrix * vec4(position, 1.0);',
    '  gl_Position = projectionMatrix * mv;',
    '}'
  ].join('\n');

  var FRAG = [
    'precision highp float;',
    'varying vec3 vDir;',
    'uniform vec3 uSunDir;',
    'uniform vec3 uHorizonWarm;',
    'uniform vec3 uHorizonCool;',
    'uniform vec3 uZenith;',
    'uniform vec3 uSunColor;',
    'uniform vec3 uFogColor;',
    'uniform float uTime;',
    '',
    'float hash21(vec2 p) {',
    '  p = fract(p * vec2(123.34, 456.21));',
    '  p += dot(p, p + 45.32);',
    '  return fract(p.x * p.y);',
    '}',
    'float vnoise(vec2 p) {',
    '  vec2 i = floor(p);',
    '  vec2 f = fract(p);',
    '  f = f * f * (3.0 - 2.0 * f);',
    '  float a = hash21(i);',
    '  float b = hash21(i + vec2(1.0, 0.0));',
    '  float c = hash21(i + vec2(0.0, 1.0));',
    '  float d = hash21(i + vec2(1.0, 1.0));',
    '  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);',
    '}',
    'float fbm5(vec2 p) {',
    '  float s = 0.0, a = 0.5;',
    '  for (int i = 0; i < 5; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; }',
    '  return s;',
    '}',
    LH.glsl.skyChunk,
    '',
    'void main() {',
    '  vec3 dir = normalize(vDir);',
    '  vec3 col = lhSkyColor(dir);',
    '',
    '  // 太阳本体（高亮，交给泛光去晕开）',
    '  float disk = smoothstep(0.99955, 0.99982, dot(dir, uSunDir));',
    '  col += uSunColor * disk * 2.6;',
    '',
    '  // 低空云带',
    '  vec2 cp = dir.xz / max(dir.y + 0.14, 0.05);',
    '  float band = exp(-pow((dir.y - 0.075) / 0.085, 2.0));',
    '  float n = fbm5(cp * 0.55 + vec2(uTime * 0.0035, uTime * 0.0012));',
    '  float cloud = smoothstep(0.46, 0.78, n) * band;',
    '  float sunSide = pow(max(dot(normalize(dir.xz + vec2(1e-5)), normalize(uSunDir.xz + vec2(1e-5))), 0.0), 2.0);',
    '  vec3 cloudCol = mix(vec3(0.50, 0.47, 0.50), vec3(1.18, 0.80, 0.58), sunSide * 0.9);',
    '  col = mix(col, cloudCol, cloud * 0.72 * smoothstep(-0.02, 0.03, dir.y));',
    '',
    '  // 高空细云',
    '  float band2 = smoothstep(0.16, 0.30, dir.y) * (1.0 - smoothstep(0.42, 0.62, dir.y));',
    '  float n2 = fbm5(cp * 1.9 + vec2(uTime * 0.008, 0.0));',
    '  float cloud2 = smoothstep(0.62, 0.86, n2) * band2;',
    '  col = mix(col, vec3(0.80, 0.84, 0.90), cloud2 * 0.20);',
    '',
    '  gl_FragColor = vec4(col, 1.0);',
    '}'
  ].join('\n');

  LH.buildSky = function (scene) {
    LH.scene.fog = new THREE.FogExp2(U.uFogColor.value.clone(), 0.00036);

    var geo = new THREE.SphereGeometry(12000, 64, 32);
    var mat = new THREE.ShaderMaterial({
      uniforms: U,
      vertexShader: VERT,
      fragmentShader: FRAG,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false
    });
    var dome = new THREE.Mesh(geo, mat);
    dome.name = 'skyDome';
    dome.frustumCulled = false;
    dome.renderOrder = -1000;
    scene.add(dome);

    // 太阳光晕（远景 Sprite，参与泛光）
    var glowMat = new THREE.SpriteMaterial({
      map: LH.tex.glow,
      color: LH.col('#ffd9a6'),
      transparent: true,
      opacity: 0.75,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      fog: false
    });
    var glow = new THREE.Sprite(glowMat);
    glow.position.copy(W.SUN_DIR).multiplyScalar(9000);
    glow.scale.set(2600, 2600, 1);
    glow.name = 'sunGlow';
    scene.add(glow);

    // 太阳主光
    var sun = new THREE.DirectionalLight(LH.col('#ffd0a2'), 3.1);
    sun.position.copy(W.SUN_DIR).multiplyScalar(420);
    sun.target.position.set(0, 0, 0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 40;
    sun.shadow.camera.far = 900;
    sun.shadow.camera.left = -120;
    sun.shadow.camera.right = 120;
    sun.shadow.camera.top = 120;
    sun.shadow.camera.bottom = -120;
    sun.shadow.bias = -0.00035;
    sun.shadow.normalBias = 0.035;
    sun.shadow.radius = 3.5;
    sun.shadow.camera.updateProjectionMatrix();
    scene.add(sun);
    scene.add(sun.target);

    // 天光：冷蓝天顶 + 暖色地面反弹
    var hemi = new THREE.HemisphereLight(LH.col('#b8d6f2'), LH.col('#7a6248'), 0.72);
    scene.add(hemi);

    // 逆光补光，模拟海面反光打亮暗部
    var fill = new THREE.DirectionalLight(LH.col('#ffd9c0'), 0.35);
    fill.position.set(-160, 60, 180);
    scene.add(fill);

    LH.sky = { dome: dome, sun: sun, hemi: hemi, glow: glow };
    LH.onUpdate(function (dt, t) { U.uTime.value = t; });
  };
})();
