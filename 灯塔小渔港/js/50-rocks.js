/**
 * 50-rocks.js —— 礁石 / 岬角 / 防波堤
 * 低多边形但顶点带噪声起伏的圆润礁石；顶点色模拟底部潮气与顶部的干燥受光。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});

  var geoCache = [];
  var matCache = null;

  function rockGeo(seed, detail) {
    var g = new THREE.IcosahedronGeometry(1, detail || 3);
    LH.displaceGeometry(g, 0.30, 1.17, seed, 0.82);
    // 顶点色：底部压暗，顶部提亮
    var pos = g.attributes.position;
    var colors = new Float32Array(pos.count * 3);
    for (var i = 0; i < pos.count; i++) {
      var y = pos.getY(i);
      var n = LH.noise2(pos.getX(i) * 3.1, pos.getZ(i) * 3.1, seed + 7);
      var b = 0.60 + 0.40 * LH.clamp(y * 0.5 + 0.5, 0, 1);
      b *= 0.86 + 0.28 * n;
      colors[i * 3] = b;
      colors[i * 3 + 1] = b;
      colors[i * 3 + 2] = b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return g;
  }

  function materials() {
    if (matCache) return matCache;
    var base = {
      vertexColors: true,
      roughness: 0.95,
      metalness: 0.02,
      flatShading: true
    };
    matCache = [
      new THREE.MeshStandardMaterial(Object.assign({ map: LH.texRepeat(LH.tex.rock, 3.0, 3.0) }, base)),
      new THREE.MeshStandardMaterial(Object.assign({ map: LH.texRepeat(LH.tex.rockDark, 2.6, 2.6) }, base)),
      new THREE.MeshStandardMaterial(Object.assign({ map: LH.texRepeat(LH.tex.rock, 4.0, 4.0), color: LH.col('#9a948a') }, base))
    ];
    return matCache;
  }

  LH.makeRock = function (seed, sx, sy, sz, x, y, z, opts) {
    opts = opts || {};
    var idx = Math.floor(LH.R.next() * geoCache.length);
    var g = geoCache[idx];
    var m = new THREE.Mesh(g, materials()[opts.mat !== undefined ? opts.mat : idx % 3]);
    m.position.set(x, y, z);
    m.scale.set(sx, sy, sz);
    m.rotation.set(opts.rx || 0, opts.ry !== undefined ? opts.ry : LH.R.range(0, 6.28), opts.rz || 0);
    m.castShadow = opts.shadow !== false;
    m.receiveShadow = true;
    return m;
  };

  LH.buildRocks = function (scene) {
    for (var i = 0; i < 6; i++) geoCache.push(rockGeo(1000 + i * 37, i < 3 ? 3 : 2));

    var rocks = new THREE.Group();
    rocks.name = 'rocks';

    // ---------- 北侧岬角（灯塔所在）
    var cape = new THREE.Group();
    cape.name = 'cape';
    // 主体
    cape.add(LH.makeRock(1, 33, 12.5, 25, 10, -2.0, -176, { mat: 0, ry: 0.4 }));
    cape.add(LH.makeRock(2, 17, 8.5, 16, -6, -0.6, -160, { mat: 1, ry: 1.2, rx: -0.1 }));
    cape.add(LH.makeRock(3, 15, 7.5, 14, 30, -0.8, -188, { mat: 2, ry: 2.1 }));
    cape.add(LH.makeRock(4, 12, 5.0, 11, 2, 0.8, -198, { mat: 0, ry: 0.9 }));
    // 顶部平台
    cape.add(LH.makeRock(5, 12.5, 3.2, 10.5, 12, 9.6, -176, { mat: 2, ry: 0.2, rx: 0.02 }));
    cape.add(LH.makeRock(6, 7.0, 2.4, 6.5, 20.5, 8.4, -168, { mat: 0, ry: 1.7 }));
    // 伸向海面的低矮礁石
    cape.add(LH.makeRock(7, 6.5, 2.2, 5.5, 38, -0.8, -196, { mat: 1, ry: 0.6 }));
    cape.add(LH.makeRock(8, 4.5, 1.6, 4.0, 46, -0.6, -184, { mat: 0, ry: 2.6 }));
    cape.add(LH.makeRock(9, 5.0, 1.8, 4.2, 27, -0.4, -206, { mat: 1, ry: 1.1 }));
    // 岬角主体周围的大块礁石
    cape.add(LH.makeRock(911, 12, 7.5, 9, -6, 1.2, -182, { mat: 1, ry: 1.4 }));
    cape.add(LH.makeRock(912, 10, 6.5, 8, 24, 1.4, -170, { mat: 0, ry: 0.4 }));
    cape.add(LH.makeRock(913, 9, 5.5, 7, 20, 0.6, -194, { mat: 2, ry: 2.2 }));
    cape.add(LH.makeRock(914, 8, 5, 6.5, -2, 2.2, -158, { mat: 0, ry: 1.1 }));
    cape.add(LH.makeRock(915, 7, 4, 5.5, 34, 0.4, -178, { mat: 1, ry: 0.8 }));
    // 岬角外围碎礁
    for (var kr = 0; kr < 14; kr++) {
      var ka = (kr / 14) * Math.PI * 2;
      var kx = 10 + Math.cos(ka) * LH.R.range(24, 34);
      var kz = -176 + Math.sin(ka) * LH.R.range(18, 27);
      var ks = LH.R.range(1.0, 2.6);
      cape.add(LH.makeRock(950 + kr, ks * 1.4, ks * 0.9, ks, kx, LH.R.range(-0.8, 0.3), kz, { mat: kr % 3 }));
    }
    // 岬角顶部的草丛
    for (var kg = 0; kg < 7; kg++) {
      var gx = 10 + LH.R.range(-7, 7);
      var gz = -176 + LH.R.range(-6, 6);
      var tuft = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), LH.std({ color: LH.col('#5d7a42'), roughness: 0.95 }));
      tuft.scale.set(LH.R.range(0.5, 1.2), LH.R.range(0.25, 0.5), LH.R.range(0.5, 1.2));
      tuft.position.set(gx, 11.3 + LH.R.range(-0.5, 0.6), gz);
      cape.add(tuft);
    }
    scene.add(cape);
    LH.cape = { group: cape, top: 10.6 };

    // ---------- 防波堤（南侧海湾外沿，弧形）
    var bw = new THREE.Group();
    bw.name = 'breakwater';
    var pts = [];
    for (var t = 0; t <= 1.0001; t += 0.05) {
      var ang = 0.55 + t * 0.92;
      var r = 74 + Math.sin(t * 3.1) * 2.5;
      var bx = 5 + Math.cos(ang) * r;
      var bz = 10 + Math.sin(ang) * r;
      pts.push([bx, bz]);
    }
    pts.forEach(function (p, i) {
      var s = 2.6 + (i % 3) * 0.5;
      var y = -0.35 + (i % 4 === 0 ? 0.6 : 0) + (i % 5 === 0 ? 0.4 : 0);
      bw.add(LH.makeRock(200 + i, s * 1.3, s * 0.8, s * 1.1, p[0], y, p[1], { mat: i % 3 }));
      if (i % 2 === 0) {
        bw.add(LH.makeRock(300 + i, s * 0.85, s * 0.55, s * 0.9, p[0] + LH.R.range(-2.5, 2.5), -0.6, p[1] + LH.R.range(-2.5, 2.5), { mat: (i + 1) % 3 }));
      }
    });
    // 堤头灯桩
    var end = pts[pts.length - 1];
    var lightPost = new THREE.Group();
    var post = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.6, 5.2, 12), LH.std({ color: LH.col('#c8503e'), roughness: 0.6 }));
    post.position.y = 2.4;
    var lamp = new THREE.Mesh(new THREE.SphereGeometry(0.5, 12, 8), new THREE.MeshBasicMaterial({ color: LH.col('#ff6a4a') }));
    lamp.position.y = 5.1;
    lightPost.add(post, lamp);
    lightPost.position.set(end[0], 1.0, end[1]);
    bw.add(lightPost);
    scene.add(bw);
    LH.breakwater = bw;

    // ---------- 岸边碎礁
    var shoreRocks = new THREE.Group();
    shoreRocks.name = 'shoreRocks';
    for (var i = 0; i < 22; i++) {
      var z = -128 + i * 12 + LH.R.range(-4, 4);
      var x = LH.coastX(z) - LH.R.range(-2, 5);
      var s = LH.R.range(0.4, 1.1);
      if (Math.abs(z) < 24) s *= 0.5;
      shoreRocks.add(LH.makeRock(400 + i, s * 1.5, s * 0.85, s, x, LH.R.range(-0.7, 0.25), z, { mat: i % 3 }));
    }
    // 北滩零星礁石
    for (var i2 = 0; i2 < 10; i2++) {
      var z2 = -150 + i2 * 10 + LH.R.range(-3, 3);
      var x2 = LH.coastX(z2) - LH.R.range(1, 8);
      var s2 = LH.R.range(0.35, 0.9);
      shoreRocks.add(LH.makeRock(520 + i2, s2 * 1.4, s2 * 0.8, s2, x2, LH.R.range(-0.4, 0.3), z2, { mat: i2 % 3 }));
    }
    // 东侧海中的两处礁石堆（补全东向层次）
    shoreRocks.add(LH.makeRock(501, 7, 3.2, 6, 152, -0.8, -58, { mat: 2 }));
    shoreRocks.add(LH.makeRock(502, 4.5, 2.2, 4, 148, -0.5, -44, { mat: 1 }));
    shoreRocks.add(LH.makeRock(503, 10, 4.0, 8, 260, -1.0, 96, { mat: 0 }));
    scene.add(shoreRocks);

    buildSurf(scene);
  };

  // 礁石脚下的白色浪花（半透明贴片，缓慢呼吸）
  function buildSurf(scene) {
    var mat = new THREE.MeshBasicMaterial({
      map: LH.tex.foam,
      color: LH.col('#f2f4f0'),
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
      fog: true
    });
    var group = new THREE.Group();
    group.name = 'surf';
    var items = [];
    function addFoam(x, z, size, phase) {
      var geo = new THREE.PlaneGeometry(size, size);
      var m = new THREE.Mesh(geo, mat.clone());
      m.rotation.x = -Math.PI / 2;
      m.rotation.z = LH.R.range(0, 6.28);
      m.position.set(x, 0.055, z);
      m.renderOrder = 5;
      m.material.opacity = 0.35 + LH.R.range(0, 0.25);
      group.add(m);
      items.push({ mesh: m, phase: phase, base: m.material.opacity, baseScale: size });
    }
    // 岬角一圈
    for (var i = 0; i < 16; i++) {
      var a = (i / 16) * Math.PI * 2;
      addFoam(10 + Math.cos(a) * 27, -176 + Math.sin(a) * 21, LH.R.range(3, 7), i * 0.7);
    }
    // 防波堤
    for (var j = 0; j < 12; j++) {
      var ang = 0.55 + (j / 11) * 0.92;
      addFoam(5 + Math.cos(ang) * 74, 10 + Math.sin(ang) * 74, LH.R.range(2.5, 5), j * 0.9 + 1.0);
    }
    // 近岸几处
    [[-13, 8], [-14, -18], [-18, 22], [140, -52], [-15, 2], [-16, -6], [-13, 14], [-20, 26]].forEach(function (p, k) {
      addFoam(p[0], p[1], 3.2, k * 1.3 + 2.0);
    });
    scene.add(group);
    LH.onUpdate(function (dt, t) {
      for (var k = 0; k < items.length; k++) {
        var it = items[k];
        var pulse = 0.5 + 0.5 * Math.sin(t * 0.9 + it.phase);
        it.mesh.material.opacity = it.base * (0.55 + 0.65 * pulse);
        var sc = 1 + pulse * 0.16;
        it.mesh.scale.set(sc, sc, 1);
      }
    });
  }
})();
