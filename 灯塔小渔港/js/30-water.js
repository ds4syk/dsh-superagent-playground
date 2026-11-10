/**
 * 30-water.js —— 海面
 * 近处（±300m）用高密度网格做真实起伏，远处平面靠法线细节与菲涅尔；
 * 颜色来自与天空穹顶共享的同一段 GLSL 天空函数，太阳高光形成“朝阳反光带”，
 * 浅水区按地形高度贴图生成岸边泡沫。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});
  var W = LH.W;

  // ---------------------------------------------------------------- 可平铺水波法线贴图
  function periodicNoise(x, y, period, seed) {
    var xi = Math.floor(x), yi = Math.floor(y);
    var fx = x - xi, fy = y - yi;
    function h(ix, iy) {
      var px = ((ix % period) + period) % period;
      var py = ((iy % period) + period) % period;
      return LH.hash2(px, py, seed);
    }
    var a = h(xi, yi), b = h(xi + 1, yi), c = h(xi, yi + 1), d = h(xi + 1, yi + 1);
    var ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
    return LH.lerp(LH.lerp(a, b, ux), LH.lerp(c, d, ux), uy);
  }

  function buildWaterNormalMap() {
    var S = 256;
    var cv = document.createElement('canvas');
    cv.width = cv.height = S;
    var ctx = cv.getContext('2d');
    var img = ctx.createImageData(S, S);
    var seed = 777;
    function height(x, y) {
      var v = 0, amp = 0.55, periods = [4, 8, 16, 32, 64];
      for (var i = 0; i < periods.length; i++) {
        var p = periods[i];
        v += amp * periodicNoise(x / S * p, y / S * p, p, seed + i * 13);
        amp *= 0.55;
      }
      return v;
    }
    var data = new Float32Array(S * S);
    for (var y = 0; y < S; y++) {
      for (var x = 0; x < S; x++) data[y * S + x] = height(x, y);
    }
    var strength = 2.2;
    for (var yy = 0; yy < S; yy++) {
      for (var xx = 0; xx < S; xx++) {
        var xl = data[yy * S + ((xx - 1 + S) % S)];
        var xr = data[yy * S + ((xx + 1 + S) % S)];
        var yt = data[((yy - 1 + S) % S) * S + xx];
        var yb = data[((yy + 1 + S) % S) * S + xx];
        var nx = (xl - xr) * strength;
        var ny = (yt - yb) * strength;
        var nz = 1.0;
        var len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        var i = (yy * S + xx) * 4;
        img.data[i] = (nx / len * 0.5 + 0.5) * 255;
        img.data[i + 1] = (ny / len * 0.5 + 0.5) * 255;
        img.data[i + 2] = (nz / len * 0.5 + 0.5) * 255;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    var t = new THREE.CanvasTexture(cv);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.encoding = THREE.LinearEncoding;
    t.anisotropy = 4;
    return t;
  }

  // ---------------------------------------------------------------- 着色器
  var VERT = [
    'uniform float uTime;',
    'uniform sampler2D uCoast;',
    'uniform vec4 uCoastRect;',
    'uniform float uWaveAmp;',
    'varying vec3 vWorld;',
    'varying float vCrest;',
    'varying float vShore;',
    '',
    'vec3 waveSum(vec2 p, float t) {',
    '  // 返回 (高度, dH/dx, dH/dz)',
    '  float h = 0.0; float gx = 0.0; float gz = 0.0;',
    '  vec2 d; float k; float amp; float w; float ph; float s; float c;',
    '',
    '  d = vec2(0.86, 0.51); k = 6.2831853 / 240.0; amp = 0.55; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w; s = sin(ph); c = cos(ph);',
    '  h += amp * s; gx += amp * k * d.x * c; gz += amp * k * d.y * c;',
    '',
    '  d = vec2(-0.42, 0.91); k = 6.2831853 / 130.0; amp = 0.30; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w + 1.3; s = sin(ph); c = cos(ph);',
    '  h += amp * s; gx += amp * k * d.x * c; gz += amp * k * d.y * c;',
    '',
    '  d = vec2(0.97, -0.24); k = 6.2831853 / 62.0; amp = 0.15; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w + 2.7; s = sin(ph); c = cos(ph);',
    '  h += amp * s; gx += amp * k * d.x * c; gz += amp * k * d.y * c;',
    '',
    '  d = vec2(0.15, 0.99); k = 6.2831853 / 34.0; amp = 0.075; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w + 4.1; s = sin(ph); c = cos(ph);',
    '  h += amp * s; gx += amp * k * d.x * c; gz += amp * k * d.y * c;',
    '',
    '  d = vec2(-0.66, -0.75); k = 6.2831853 / 19.0; amp = 0.035; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w + 0.7; s = sin(ph); c = cos(ph);',
    '  h += amp * s; gx += amp * k * d.x * c; gz += amp * k * d.y * c;',
    '',
    '  return vec3(h, gx, gz);',
    '}',
    '',
    'void main() {',
    '  vec3 wp = (modelMatrix * vec4(position, 1.0)).xyz;',
    '  vWorld = wp;',
    '  vec2 uv = (wp.xz - uCoastRect.xy) / uCoastRect.zw;',
    '  float shore = texture2D(uCoast, clamp(uv, 0.001, 0.999)).g;',
    '  vShore = shore;',
    '  vec3 ws = waveSum(wp.xz, uTime);',
    '  vCrest = ws.x;',
    '  wp.y += ws.x * uWaveAmp;',
    '  gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);',
    '}'
  ].join('\n');

  var FRAG = [
    'precision highp float;',
    'uniform vec3 uSunDir;',
    'uniform vec3 uHorizonWarm;',
    'uniform vec3 uHorizonCool;',
    'uniform vec3 uZenith;',
    'uniform vec3 uSunColor;',
    'uniform vec3 uCamPos;',
    'uniform vec3 uFogColor;',
    'uniform float uFogDensity;',
    'uniform vec3 uDeepColor;',
    'uniform vec3 uShallowColor;',
    'uniform float uTime;',
    'uniform sampler2D uNormalMap;',
    'uniform sampler2D uCoast;',
    'uniform vec4 uCoastRect;',
    'uniform float uWaveAmp;',
    'varying vec3 vWorld;',
    'varying float vCrest;',
    'varying float vShore;',
    '',
    LH.glsl.skyChunk,
    '',
    'vec3 waveNormal(vec2 p, float t) {',
    '  vec3 ws = vec3(0.0);',
    '  vec2 d; float k; float amp; float w; float ph;',
    '  d = vec2(0.86, 0.51); k = 6.2831853 / 240.0; amp = 0.55; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w; ws.x += amp * k * d.x * cos(ph); ws.z += amp * k * d.y * cos(ph);',
    '  d = vec2(-0.42, 0.91); k = 6.2831853 / 130.0; amp = 0.30; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w + 1.3; ws.x += amp * k * d.x * cos(ph); ws.z += amp * k * d.y * cos(ph);',
    '  d = vec2(0.97, -0.24); k = 6.2831853 / 62.0; amp = 0.15; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w + 2.7; ws.x += amp * k * d.x * cos(ph); ws.z += amp * k * d.y * cos(ph);',
    '  d = vec2(0.15, 0.99); k = 6.2831853 / 34.0; amp = 0.075; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w + 4.1; ws.x += amp * k * d.x * cos(ph); ws.z += amp * k * d.y * cos(ph);',
    '  d = vec2(-0.66, -0.75); k = 6.2831853 / 19.0; amp = 0.035; w = sqrt(9.81 * k);',
    '  ph = dot(d, p) * k - t * w + 0.7; ws.x += amp * k * d.x * cos(ph); ws.z += amp * k * d.y * cos(ph);',
    '  return normalize(vec3(-ws.x, 1.0, -ws.z));',
    '}',
    '',
    'void main() {',
    '  vec3 V = normalize(uCamPos - vWorld);',
    '  float dist = length(uCamPos - vWorld);',
    '  float distFade = 1.0 - smoothstep(160.0, 1200.0, dist);',
    '',
    '  // 大尺度波浪法线 + 细节法线（两层滚动）',
    '  vec3 N = waveNormal(vWorld.xz, uTime);',
    '  vec2 nuv = vWorld.xz * 0.045;',
    '  vec3 n1 = texture2D(uNormalMap, nuv * 1.0 + vec2(uTime * 0.010, uTime * 0.004)).xyz * 2.0 - 1.0;',
    '  vec3 n2 = texture2D(uNormalMap, nuv * 2.7 - vec2(uTime * 0.017, uTime * 0.009)).xyz * 2.0 - 1.0;',
    '  vec3 n3 = texture2D(uNormalMap, nuv * 6.3 + vec2(uTime * 0.026, -uTime * 0.015)).xyz * 2.0 - 1.0;',
    '  vec2 slope = (n1.xy * 0.55 + n2.xy * 0.32 + n3.xy * 0.18) * (0.55 + 0.45 * distFade);',
    '  N = normalize(vec3(N.x * distFade + slope.x * 0.55, 1.0, N.z * distFade + slope.y * 0.55));',
    '',
    '  float ndv = max(dot(N, V), 0.0);',
    '  float fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);',
    '  fres = clamp(fres * 1.05, 0.0, 0.92);',
    '',
    '  // 天空反射',
    '  vec3 R = reflect(-V, N);',
    '  R.y = abs(R.y);',
    '  vec3 skyRef = lhSkyColor(R);',
    '',
    '  // 水的体色：浅水偏青，深水偏蓝',
    '  float shallow = clamp(vShore * 1.15, 0.0, 1.0);',
    '  vec3 body = mix(uDeepColor, uShallowColor, shallow * 0.75);',
    '  body *= 0.9 + 0.1 * max(dot(N, uSunDir), 0.0);',
    '  vec3 col = mix(body, skyRef, fres);',
    '',
    '  // 太阳高光：窄镜面 + 宽海面反光带',
    '  vec3 H = normalize(V + uSunDir);',
    '  float nh = max(dot(N, H), 0.0);',
    '  float spec = pow(nh, 900.0) * 8.0 + pow(nh, 120.0) * 0.42 + pow(nh, 24.0) * 0.07;',
    '  col += uSunColor * spec * (0.35 + 0.65 * distFade);',
    '',
    '  // 细碎鳞光',
    '  float sparkle = texture2D(uNormalMap, nuv * 18.0 + vec2(-uTime * 0.05, uTime * 0.03)).x;',
    '  float glint = smoothstep(0.86, 1.0, sparkle) * pow(nh, 60.0) * distFade;',
    '  col += uSunColor * glint * 0.9;',
    '',
    '  // 岸边泡沫',
    '  vec2 cuv = (vWorld.xz - uCoastRect.xy) / uCoastRect.zw;',
    '  float shoreMask = texture2D(uCoast, clamp(cuv, 0.001, 0.999)).g;',
    '  float shoreNoise = texture2D(uNormalMap, vWorld.xz * 0.055 + vec2(uTime * 0.012, -uTime * 0.009)).x;',
    '  float waveBand = 0.5 + 0.5 * sin(vWorld.x * 0.42 + vWorld.z * 0.31 + uTime * 0.85 + shoreNoise * 4.0);',
    '  float foam = smoothstep(0.32, 0.78, shoreMask) * smoothstep(0.2, 0.9, waveBand);',
    '  float crestFoam = smoothstep(0.72, 0.98, abs(vCrest) / max(uWaveAmp, 0.001)) * 0.25;',
    '  float foamTex = texture2D(uNormalMap, vWorld.xz * 0.35 + vec2(uTime * 0.03, 0.0)).y;',
    '  float foamAmt = clamp(foam * (0.75 + 0.7 * foamTex) * 1.25 + crestFoam * foamTex, 0.0, 0.96);',
    '  vec3 foamCol = vec3(0.86, 0.88, 0.86) * (0.75 + 0.25 * max(dot(N, uSunDir), 0.0));',
    '  col = mix(col, foamCol, foamAmt);',
    '',
    '  // 大气雾',
    '  float fog = 1.0 - exp(-pow(uFogDensity * dist, 2.0));',
    '  col = mix(col, uFogColor, clamp(fog, 0.0, 0.985));',
    '',
    '  gl_FragColor = vec4(col, 1.0);',
    '}'
  ].join('\n');

  LH.buildWater = function (scene, coastTexture, coastRect) {
    var normalMap = buildWaterNormalMap();

    var uniforms = {
      uSunDir: LH.skyUniforms.uSunDir,
      uHorizonWarm: LH.skyUniforms.uHorizonWarm,
      uHorizonCool: LH.skyUniforms.uHorizonCool,
      uZenith: LH.skyUniforms.uZenith,
      uSunColor: LH.skyUniforms.uSunColor,
      uTime: { value: 0 },
      uCamPos: { value: new THREE.Vector3() },
      uFogColor: { value: scene.fog.color },
      uFogDensity: { value: scene.fog.density },
      uDeepColor: { value: LH.col('#123f57') },
      uShallowColor: { value: LH.col('#3a8a88') },
      uNormalMap: { value: normalMap },
      uCoast: { value: coastTexture },
      uCoastRect: { value: coastRect },
      uWaveAmp: { value: 1.0 },
      uNearFade: { value: 1.0 }
    };

    var mat = new THREE.ShaderMaterial({
      uniforms: uniforms,
      vertexShader: VERT,
      fragmentShader: FRAG,
      fog: false
    });

    // 近水面：细网格，真实起伏
    var nearGeo = new THREE.PlaneGeometry(620, 620, 300, 300);
    nearGeo.rotateX(-Math.PI / 2);
    var near = new THREE.Mesh(nearGeo, mat);
    near.name = 'waterNear';
    near.position.y = 0.02;
    near.receiveShadow = false;
    near.frustumCulled = false;
    scene.add(near);

    // 远水面：环形（中间挖空，避免与近水面重叠导致深度撕咬），只靠法线与菲涅尔
    var farGeo = new THREE.RingGeometry(308, 8200, 192, 8);
    farGeo.rotateX(-Math.PI / 2);
    var farMat = mat.clone();
    farMat.uniforms = uniforms;
    var far = new THREE.Mesh(farGeo, farMat);
    far.name = 'waterFar';
    far.position.y = -0.02;
    far.frustumCulled = false;
    scene.add(far);

    // 近水起伏在边缘淡出，与远水平面衔接
    var origVert = VERT;
    mat.vertexShader = [
      'uniform float uNearFade;',
      origVert.replace('  wp.y += ws.x * uWaveAmp;',
        '  float fade = 1.0 - smoothstep(190.0, 280.0, length(wp.xz));' +
        '  wp.y += ws.x * uWaveAmp * fade * uNearFade;')
    ].join('\n');
    mat.needsUpdate = true;

    // 远水面不位移（保持平坦，与近水面淡出后的平面一致）
    farMat.vertexShader = origVert.replace('  wp.y += ws.x * uWaveAmp;', '  wp.y += 0.0;');
    farMat.needsUpdate = true;

    LH.water = { near: near, far: far, mat: mat };
    LH.onUpdate(function (dt, t, camera) {
      uniforms.uTime.value = t;
      uniforms.uCamPos.value.copy(camera.position);
    });
  };
})();
