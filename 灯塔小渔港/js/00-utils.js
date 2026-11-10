/**
 * 00-utils.js —— 基础工具库
 * 固定种子随机数 / 连续噪声 / 数学与颜色辅助 / 小几何工具。
 * 全场景所有随机量都从这里取种子随机数，保证每次打开画面一致。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});

  LH.SEED = 20261110;

  // ---------- 数学 ----------
  LH.clamp = function (v, a, b) { return v < a ? a : (v > b ? b : v); };
  LH.lerp = function (a, b, t) { return a + (b - a) * t; };
  LH.smoothstep = function (e0, e1, x) {
    var t = LH.clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  };
  LH.smootherstep = function (e0, e1, x) {
    var t = LH.clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * t * (t * (t * 6 - 15) + 10);
  };
  LH.damp = function (a, b, lambda, dt) {
    return LH.lerp(a, b, 1 - Math.exp(-lambda * dt));
  };
  LH.TAU = Math.PI * 2;
  LH.DEG = Math.PI / 180;

  // ---------- 种子随机数 ----------
  LH.mulberry32 = function (seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  /** 生成一个带常用方法的随机源。 */
  LH.rng = function (seed) {
    var r = LH.mulberry32(seed);
    var api = {
      next: r,
      range: function (a, b) { return a + (b - a) * r(); },
      int: function (a, b) { return a + Math.floor(r() * (b - a + 1)); },
      pick: function (arr) { return arr[Math.min(arr.length - 1, Math.floor(r() * arr.length))]; },
      chance: function (p) { return r() < p; },
      sign: function () { return r() < 0.5 ? -1 : 1; }
    };
    return api;
  };

  // 全局固定种子随机源（顺序敏感：各模块按加载顺序取数，结果仍然确定）
  LH.R = LH.rng(LH.SEED);

  // ---------- 连续噪声 ----------
  function hash2(ix, iy, seed) {
    var h = ix * 374761393 + iy * 668265263 + seed * 1442695040;
    h = (h ^ (h >>> 13)) * 1274126177;
    h = h ^ (h >>> 16);
    return (h >>> 0) / 4294967296;
  }
  LH.hash2 = hash2;

  /** 值噪声：平滑插值，范围 0..1。 */
  LH.noise2 = function (x, y, seed) {
    seed = seed || 0;
    var x0 = Math.floor(x), y0 = Math.floor(y);
    var fx = x - x0, fy = y - y0;
    var ux = fx * fx * (3 - 2 * fx);
    var uy = fy * fy * (3 - 2 * fy);
    var a = hash2(x0, y0, seed);
    var b = hash2(x0 + 1, y0, seed);
    var c = hash2(x0, y0 + 1, seed);
    var d = hash2(x0 + 1, y0 + 1, seed);
    return LH.lerp(LH.lerp(a, b, ux), LH.lerp(c, d, ux), uy);
  };

  /** 分形噪声：范围约 0..1，中心 0.5。 */
  LH.fbm2 = function (x, y, octaves, seed, lacunarity, gain) {
    octaves = octaves || 4;
    lacunarity = lacunarity || 2.0;
    gain = gain || 0.5;
    var sum = 0, amp = 1, norm = 0, fx = x, fy = y;
    for (var i = 0; i < octaves; i++) {
      sum += amp * LH.noise2(fx, fy, seed + i * 101);
      norm += amp;
      amp *= gain;
      fx *= lacunarity;
      fy *= lacunarity;
    }
    return sum / norm;
  };

  /** 山脊噪声，用于山丘剪影。 */
  LH.ridge2 = function (x, y, octaves, seed) {
    octaves = octaves || 4;
    var sum = 0, amp = 1, norm = 0, fx = x, fy = y;
    for (var i = 0; i < octaves; i++) {
      var n = 1 - Math.abs(LH.noise2(fx, fy, seed + i * 71) * 2 - 1);
      sum += amp * n * n;
      norm += amp;
      amp *= 0.5;
      fx *= 2.05;
      fy *= 2.05;
    }
    return sum / norm;
  };

  // ---------- 颜色 ----------
  // 工作色彩空间为线性：十六进制按 sRGB 书写，转成线性后再交给材质，最终统一 gamma 输出。
  LH.col = function (hex) {
    return new THREE.Color(hex).convertSRGBToLinear();
  };
  LH.colHex = function (hex) {
    var c = new THREE.Color(hex).convertSRGBToLinear();
    return '#' + c.getHexString();
  };
  LH.colorLerp = function (a, b, t) {
    return a.clone().lerp(b, t);
  };
  LH.hsl = function (h, s, l) {
    var c = new THREE.Color();
    c.setHSL(h, s, l);
    // setHSL 产生的是 sRGB 数值，转成线性
    return c.convertSRGBToLinear();
  };

  // ---------- 统一材质/几何小工具 ----------
  LH.std = function (opts) {
    var p = {
      color: 0xffffff,
      roughness: 0.8,
      metalness: 0.0,
      flatShading: false
    };
    if (opts) for (var k in opts) p[k] = opts[k];
    if (typeof p.color === 'string') p.color = LH.col(p.color);
    return new THREE.MeshStandardMaterial(p);
  };

  LH.roundedBox = function (w, h, d, r, seg) {
    r = Math.min(r === undefined ? 0.08 : r, Math.min(w, h, d) * 0.49);
    return new THREE.RoundedBoxGeometry(w, h, d, seg || 2, r);
  };

  /** 把几何体顶点按噪声轻微扰动，得到自然起伏（用于礁石、树冠）。 */
  LH.displaceGeometry = function (geo, amount, freq, seed, yScale) {
    var pos = geo.attributes.position;
    var v = new THREE.Vector3();
    for (var i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      var n = LH.fbm2(v.x * freq + 13.1, v.z * freq + v.y * freq * 0.7, 3, seed) - 0.5;
      var n2 = LH.noise2(v.y * freq * 1.7 + 4.2, v.x * freq - v.z * freq, seed + 9) - 0.5;
      var len = v.length() || 1;
      var scale = 1 + (n * 1.6 + n2 * 0.6) * amount;
      v.multiplyScalar(scale);
      if (yScale) v.y *= yScale;
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    return geo;
  };

  /** 设置阴影标记。 */
  LH.shadows = function (obj, cast, receive) {
    obj.traverse(function (o) {
      if (o.isMesh) {
        o.castShadow = cast === true;
        o.receiveShadow = receive !== false;
      }
    });
    return obj;
  };

  LH.mesh = function (geo, mat, x, y, z) {
    var m = new THREE.Mesh(geo, mat);
    if (x !== undefined) m.position.set(x, y, z);
    return m;
  };

  /** 地面接触阴影（柔和的径向暗斑，弥补 SSAO 在远景/低端设备上的不足）。 */
  LH.contact = function (x, y, z, sx, sz, opacity, rotY) {
    var mat = new THREE.MeshBasicMaterial({
      map: LH.tex.contact,
      color: 0x000000,
      transparent: true,
      opacity: opacity === undefined ? 0.55 : opacity,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2
    });
    var geo = new THREE.PlaneGeometry(sx, sz);
    var m = new THREE.Mesh(geo, mat);
    m.rotation.x = -Math.PI / 2;
    m.rotation.z = rotY || 0;
    m.position.set(x, y, z);
    m.renderOrder = 2;
    return m;
  };

  // ---------- 每帧更新登记 ----------
  LH.updaters = [];
  LH.onUpdate = function (fn) { LH.updaters.push(fn); };

  // ---------- 世界常量 ----------
  LH.W = {
    // 世界坐标约定：+X 东（日出），-Z 北（灯塔岬角），+Z 南（渔港海湾），-X 西（村落山坡）
    SUN_DIR: new THREE.Vector3(0.982, 0.152, -0.116).normalize(),
    DECK_Y: 1.32,
    PIER_X0: -18.0,
    PIER_X1: 6.2,
    PIER_HALF_W: 1.55,
    EYE: 1.62,
    SHORE_MIN_X: -26.0,
    SHORE_MAX_X: -13.0,
    SHORE_HALF_Z: 12.0
  };

  // 场景根节点：所有可移动/静态物体都挂在这里
  LH.root = new THREE.Group();
  LH.root.name = 'worldRoot';
})();
