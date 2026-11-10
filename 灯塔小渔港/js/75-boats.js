/**
 * 75-boats.js —— 渔船、随浪晃动、海鸥盘旋
 * 船体由参数化截面生成：圆润的船腹、收窄的船首，木纹材质 + 吃水线附近的磨损。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});

  // 参数化船体
  function hullGeometry(L, B, D, seed) {
    var N = 20;      // 纵向站数
    var M = 7;       // 截面点数（左舷→龙骨→右舷）
    var secX = [-1, -0.9, -0.55, 0, 0.55, 0.9, 1];
    var secY = [0, -0.38, -0.78, -1, -0.78, -0.38, 0];
    var pos = [];
    var uv = [];
    var index = [];

    function halfWidth(t) { // t 0..1 从船尾到船首
      var u = (t - 0.5) * 2;                       // -1..1
      var tip = Math.pow(Math.max(0, 1 - u * u), 0.55);
      var bow = 1 - Math.pow(Math.max(0, u), 3.2) * 0.55; // 船首稍窄
      return (B / 2) * tip * bow;
    }
    function depth(t) {
      var u = (t - 0.5) * 2;
      return D * (0.62 + 0.38 * Math.pow(Math.max(0, 1 - u * u), 0.6));
    }

    for (var i = 0; i <= N; i++) {
      var t = i / N;
      var z = (t - 0.5) * L;
      var w = halfWidth(t);
      var d = depth(t);
      for (var j = 0; j < M; j++) {
        var x = secX[j] * w;
        var y = secY[j] * d;
        // 船首上翘
        if (i > N * 0.78) y += Math.pow((t - 0.78) / 0.22, 2) * 0.35;
        pos.push(x, y, z);
        uv.push(j / (M - 1), t);
      }
    }
    for (var ii = 0; ii < N; ii++) {
      for (var jj = 0; jj < M - 1; jj++) {
        var a = ii * M + jj;
        var b = a + M;
        index.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(index);
    geo.computeVertexNormals();

    // 甲板
    var deckPos = [];
    var deckIdx = [];
    var deckUv = [];
    for (var k = 0; k <= N; k++) {
      var t2 = k / N;
      var w2 = halfWidth(t2) * 0.94;
      var z2 = (t2 - 0.5) * L;
      var y2 = 0.04 + (k > N * 0.78 ? Math.pow((t2 - 0.78) / 0.22, 2) * 0.35 : 0);
      deckPos.push(-w2, y2, z2, w2, y2, z2);
      deckUv.push(0, t2, 1, t2);
      if (k < N) {
        var da = k * 2;
        deckIdx.push(da, da + 1, da + 2, da + 1, da + 3, da + 2);
      }
    }
    var deckGeo = new THREE.BufferGeometry();
    deckGeo.setAttribute('position', new THREE.Float32BufferAttribute(deckPos, 3));
    deckGeo.setAttribute('uv', new THREE.Float32BufferAttribute(deckUv, 2));
    deckGeo.setIndex(deckIdx);
    deckGeo.computeVertexNormals();

    return { hull: geo, deck: deckGeo, halfWidth: halfWidth, depth: depth };
  }

  function tube(points, r, mat) {
    var curve = new THREE.CatmullRomCurve3(points);
    return new THREE.Mesh(new THREE.TubeGeometry(curve, 24, r, 5, false), mat);
  }

  // 单艘渔船
  function buildBoat(opts) {
    var L = opts.L, B = opts.B, D = opts.D;
    var g = new THREE.Group();
    var parts = hullGeometry(L, B, D, opts.seed);

    var paint = new THREE.MeshStandardMaterial({
      map: LH.texRepeat(LH.tex.woodBoat, 2, 2),
      color: LH.col(opts.paint || '#7fa3aa'),
      roughness: 0.82,
      metalness: 0.02,
      side: THREE.DoubleSide
    });
    var hull = new THREE.Mesh(parts.hull, paint);
    hull.castShadow = hull.receiveShadow = true;
    g.add(hull);

    var deckMat = new THREE.MeshStandardMaterial({
      map: LH.texRepeat(LH.tex.wood, 3, 3),
      color: LH.col('#c8b18c'),
      roughness: 0.9,
      side: THREE.DoubleSide
    });
    var deck = new THREE.Mesh(parts.deck, deckMat);
    deck.receiveShadow = true;
    g.add(deck);

    // 舷缘
    var rimMat = LH.std({ color: LH.col('#8c4a36'), roughness: 0.8 });
    var rimPts = [[], []];
    for (var i = 0; i <= 16; i++) {
      var t = i / 16;
      var w = parts.halfWidth(t) * 0.99;
      var z = (t - 0.5) * L;
      var y = 0.05 + (t > 0.78 ? Math.pow((t - 0.78) / 0.22, 2) * 0.35 : 0);
      rimPts[0].push(new THREE.Vector3(-w, y, z));
      rimPts[1].push(new THREE.Vector3(w, y, z));
    }
    g.add(tube(rimPts[0], 0.05, rimMat));
    g.add(tube(rimPts[1], 0.05, rimMat));

    if (!opts.simple) {
      // 驾驶室（偏船尾）
      var cabinZ = -L * 0.18;
      var cabin = new THREE.Mesh(
        LH.roundedBox(B * 0.58, 1.35, L * 0.24, 0.09, 2),
        LH.std({ color: LH.col(opts.cabin || '#e6e0cf'), roughness: 0.75 })
      );
      cabin.position.set(0, 0.78, cabinZ);
      cabin.castShadow = cabin.receiveShadow = true;
      g.add(cabin);

      var glassMat = new THREE.MeshBasicMaterial({ color: LH.col('#26333a'), toneMapped: true });
      [-1, 1].forEach(function (sd) {
        var win = new THREE.Mesh(new THREE.PlaneGeometry(L * 0.17, 0.42), glassMat);
        win.position.set(sd * (B * 0.29 + 0.01), 1.02, cabinZ);
        win.rotation.y = sd * Math.PI / 2;
        g.add(win);
      });
      var frontWin = new THREE.Mesh(new THREE.PlaneGeometry(B * 0.42, 0.4), glassMat);
      frontWin.position.set(0, 1.02, cabinZ - L * 0.121);
      frontWin.rotation.y = Math.PI;
      g.add(frontWin);

      var cabinRoof = new THREE.Mesh(LH.roundedBox(B * 0.64, 0.1, L * 0.27, 0.04, 1), LH.std({ color: LH.col('#a84432'), roughness: 0.8 }));
      cabinRoof.position.set(0, 1.5, cabinZ);
      cabinRoof.castShadow = true;
      g.add(cabinRoof);

      // 桅杆与斜撑
      var mastMat = LH.std({ color: LH.col('#7d6a52'), roughness: 0.9 });
      var mast = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.075, L * 0.52, 8), mastMat);
      mast.position.set(0, L * 0.26, cabinZ + L * 0.08);
      mast.castShadow = true;
      g.add(mast);
      var boom = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, L * 0.32, 6), mastMat);
      boom.rotation.x = Math.PI / 2;
      boom.position.set(0, L * 0.32, cabinZ + L * 0.22);
      g.add(boom);
      var top = new THREE.Vector3(0, L * 0.52, cabinZ + L * 0.08);
      var rigMat = LH.std({ color: LH.col('#2c2c2c'), roughness: 0.8 });
      [[0, 0.06, L * 0.48], [0, 0.06, -L * 0.44]].forEach(function (p) {
        g.add(tube([top, new THREE.Vector3(p[0], p[1], cabinZ + p[2])], 0.012, rigMat));
      });

      // 救生圈与轮胎护舷
      var ring = new THREE.Mesh(new THREE.TorusGeometry(0.26, 0.07, 6, 16), LH.std({ color: LH.col('#e07a44'), roughness: 0.7 }));
      ring.position.set(B * 0.28, 0.95, cabinZ + 0.4);
      ring.rotation.y = Math.PI / 2;
      g.add(ring);

      var fenderMat = LH.std({ color: LH.col('#2f3236'), roughness: 0.9 });
      [-0.32, 0, 0.32].forEach(function (fz) {
        var fender = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.05, 6, 10), fenderMat);
        fender.position.set(-B * 0.49, 0.32, fz * L);
        fender.rotation.y = Math.PI / 2;
        g.add(fender);
      });

      // 甲板上的渔获箱与网包
      var fishBox = new THREE.Mesh(LH.roundedBox(0.55, 0.35, 0.42, 0.03, 1), LH.std({ color: LH.col('#4f6f7a'), roughness: 0.85 }));
      fishBox.position.set(0.35, 0.22, L * 0.12);
      fishBox.castShadow = true;
      g.add(fishBox);
      var netBundle = new THREE.Mesh(new THREE.SphereGeometry(0.42, 10, 8), LH.std({ color: LH.col('#3f4a35'), roughness: 1 }));
      netBundle.scale.set(1, 0.55, 1.3);
      netBundle.position.set(-0.4, 0.24, -L * 0.06);
      g.add(netBundle);

      // 船名横幅（色块，非文字）
      var band = new THREE.Mesh(new THREE.PlaneGeometry(B * 0.7, 0.22), LH.std({ color: LH.col(opts.band || '#d8c36a'), roughness: 0.8 }));
      band.position.set(0, 0.32, -L / 2 + 0.04);
      band.rotation.y = Math.PI;
      g.add(band);


    }

    // 划艇的两支木桨
    if (opts.simple) {
      var oarMat2 = LH.std({ color: LH.col('#9a7c56'), roughness: 0.95 });
      [-0.22, 0.22].forEach(function (oz) {
        var oar = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.06, L * 0.72), oarMat2);
        oar.position.set(0.1, 0.14, oz * L);
        oar.rotation.y = (oz > 0 ? -1 : 1) * 0.12;
        g.add(oar);
      });
    }

    g.position.set(opts.x || 0, opts.y || 0, opts.z || 0);
    g.rotation.y = opts.rot || 0;
    g.userData.bob = {
      phase: opts.phase || 0,
      amp: opts.amp === undefined ? 0.055 : opts.amp,
      baseY: opts.y || 0
    };
    return g;
  }
  LH.buildBoat = buildBoat;

  // 海鸥
  function buildGull(seed) {
    var g = new THREE.Group();
    var rng = LH.rng(seed);
    var white = LH.std({ color: LH.col('#f1eee6'), roughness: 0.85 });
    var grey = LH.std({ color: LH.col('#a9b2b6'), roughness: 0.85 });
    var body = new THREE.Mesh(new THREE.SphereGeometry(0.16, 8, 6), white);
    body.scale.set(0.8, 0.75, 2.0);
    g.add(body);
    var head = new THREE.Mesh(new THREE.SphereGeometry(0.1, 7, 5), white);
    head.position.set(0, 0.05, 0.28);
    g.add(head);
    var beak = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.14, 5), LH.std({ color: LH.col('#e0a24a'), roughness: 0.7 }));
    beak.rotation.x = Math.PI / 2;
    beak.position.set(0, 0.035, 0.42);
    g.add(beak);

    var wingL = new THREE.Group();
    var wingMeshL = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.02, 0.3), grey);
    wingMeshL.position.x = -0.47;
    wingL.add(wingMeshL);
    var tipL = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.02, 0.22), LH.std({ color: LH.col('#3c4448'), roughness: 0.9 }));
    tipL.position.x = -0.95;
    wingL.add(tipL);
    wingL.position.set(-0.06, 0.04, 0);
    g.add(wingL);

    var wingR = new THREE.Group();
    var wingMeshR = wingMeshL.clone();
    wingMeshR.position.x = 0.47;
    wingR.add(wingMeshR);
    var tipR = tipL.clone();
    tipR.position.x = 0.95;
    wingR.add(tipR);
    wingR.position.set(0.06, 0.04, 0);
    g.add(wingR);

    g.userData.flap = { wingL: wingL, wingR: wingR, phase: rng.range(0, 6.28), rate: rng.range(3.4, 4.6) };
    return g;
  }

  LH.buildBoats = function (scene) {
    var group = new THREE.Group();
    group.name = 'boats';

    // 港内三艘渔船
    var specs = [
      { L: 8.4, B: 2.8, D: 1.05, x: -9.4, z: 40, rot: Math.PI / 2 + 0.05, paint: '#7fa3aa', cabin: '#e8e2d2', band: '#d8c36a', phase: 0.3 },
      { L: 7.2, B: 2.5, D: 0.95, x: -9.0, z: 54, rot: Math.PI / 2 - 0.04, paint: '#a5503f', cabin: '#efe6d2', band: '#e6e0cf', phase: 1.7 },
      { L: 9.2, B: 3.0, D: 1.15, x: -9.8, z: 68, rot: Math.PI / 2 + 0.03, paint: '#6f8f7f', cabin: '#e2d8c2', band: '#d8c36a', phase: 3.1 }
    ];
    var boats = [];
    specs.forEach(function (s, i) {
      var b = buildBoat(s);
      group.add(b);
      boats.push(b);
    });

    // 缆绳：船侧两根系到码头边缘
    var ropeMat = LH.std({ color: LH.col('#d5c4a0'), roughness: 0.95 });
    specs.forEach(function (s) {
      [[s.L * 0.32, 1.1], [-s.L * 0.3, 1.35]].forEach(function (o) {
        var cleat = new THREE.Vector3(s.x - B_HALF(s) + 0.2, 0.5, s.z + o[0]);
        var peg = new THREE.Vector3(-14.8, 1.78, s.z + o[0] * 1.02);
        var mid = cleat.clone().lerp(peg, 0.5);
        mid.y -= 0.45;
        group.add(tube([cleat, mid, peg], 0.02, ropeMat));
      });
    });
    function B_HALF(s) { return s.B / 2; }

    // 码头边的系缆桩
    specs.forEach(function (s) {
      [-1.1, 1.35].forEach(function (dz) {
        var b = new THREE.Group();
        var mat = LH.std({ color: LH.col('#4a5054'), roughness: 0.65, metalness: 0.25 });
        var body = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.17, 0.46, 10), mat);
        body.position.y = 0.23;
        var cap = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 8), mat);
        cap.position.y = 0.48;
        cap.scale.y = 0.7;
        b.add(body, cap);
        b.position.set(-14.8, 1.68, s.z + dz * s.L * 0.32);
        group.add(b);
      });
    });

    // 远处出海的渔船（东向）
    var distant = [
      { L: 13, B: 4.2, D: 1.5, x: 320, z: 26, rot: 0.9, paint: '#5f7c86', cabin: '#ded6c2', band: '#cbb268', phase: 0.9, amp: 0.1 },
      { L: 10, B: 3.4, D: 1.2, x: 470, z: -150, rot: -0.4, paint: '#8a5a46', cabin: '#e2d8c2', band: '#d8c36a', phase: 2.2, amp: 0.12 }
    ];
    distant.forEach(function (s) {
      var b = buildBoat(s);
      // 远处加一盏白色航行灯
      var lamp = new THREE.Sprite(new THREE.SpriteMaterial({
        map: LH.tex.glow, color: LH.col('#fff0cf'), transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending
      }));
      lamp.position.set(0, 5.2, 0);
      lamp.scale.set(9, 9, 1);
      b.add(lamp);
      group.add(b);
    });

    // ---------------- 红绿航标浮标（东侧近海）
    var buoySpecs = [
      { color: '#d8443a', x: 96, z: -28, phase: 0.0 },
      { color: '#3e9e5c', x: 158, z: 36, phase: 1.3 },
      { color: '#d8443a', x: 240, z: 82, phase: 2.4 }
    ];
    var buoys = [];
    buoySpecs.forEach(function (b) {
      var g = new THREE.Group();
      var body = new THREE.Mesh(
        new THREE.CylinderGeometry(0.55, 0.8, 0.9, 12),
        LH.std({ color: LH.col(b.color), roughness: 0.55 })
      );
      body.position.y = 0.25;
      g.add(body);
      var top = new THREE.Mesh(new THREE.SphereGeometry(0.42, 10, 8), LH.std({ color: LH.col(b.color), roughness: 0.5 }));
      top.position.y = 0.8;
      g.add(top);
      var pole = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.5, 5), LH.std({ color: LH.col('#5a5a58'), roughness: 0.7 }));
      pole.position.y = 1.7;
      g.add(pole);
      var lamp = new THREE.Sprite(new THREE.SpriteMaterial({
        map: LH.tex.glow, color: LH.col(b.color), transparent: true, opacity: 0.85, depthWrite: false, blending: THREE.AdditiveBlending
      }));
      lamp.position.y = 2.5;
      lamp.scale.set(3.4, 3.4, 1);
      g.add(lamp);
      g.position.set(b.x, 0, b.z);
      g.userData.buoy = { phase: b.phase, baseY: 0 };
      group.add(g);
      buoys.push(g);
    });

    // 栈桥旁拴着的小划艇（补足南向近景）
    var dinghy2 = buildBoat({ L: 3.4, B: 1.4, D: 0.5, x: -6.6, z: 17, rot: 0.9, paint: '#8a6a4a', band: '#e6e0cf', phase: 1.2, amp: 0.09, simple: true });
    group.add(dinghy2);
    var ropeMat2 = LH.std({ color: LH.col('#d5c4a0'), roughness: 0.95 });
    group.add(tube([new THREE.Vector3(-7.0, 0.4, 16.4), new THREE.Vector3(-5.6, 0.7, 12.0)], 0.016, ropeMat2));
    // 水面浮球
    var floatCols = ['#e07a44', '#e8d9b0', '#d8563c'];
    var floatBalls = [];
    [[-6, 12], [-2, 16], [-9, 20], [2, 26]].forEach(function (p, i) {
      var ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.22, 10, 8),
        LH.std({ color: LH.col(floatCols[i % floatCols.length]), roughness: 0.5 })
      );
      ball.position.set(p[0], 0.15, p[1]);
      ball.userData.phase = i * 1.7;
      ball.userData.baseY = 0.15;
      group.add(ball);
      floatBalls.push(ball);
    });

    // 北滩搁浅的小木船
    var dinghy = buildBoat({ L: 4.0, B: 1.5, D: 0.58, x: -14.2, z: -34, rot: 0.9, paint: '#a5503f', band: '#e6e0cf', phase: 0.0, amp: 0.0, simple: true });
    dinghy.position.y = LH.terrainHeight(-14.2, -34) + 0.35;
    dinghy.rotation.z = 0.16;
    group.add(dinghy);

    // ---------------- 海鸥
    var gulls = [];
    for (var i = 0; i < 10; i++) {
      var gull = buildGull(900 + i * 17);
      var center = new THREE.Vector3(
        LH.R.range(-40, 60),
        LH.R.range(9, 30),
        LH.R.range(-170, 90)
      );
      if (i % 3 === 0) center.set(LH.R.range(6, 30), LH.R.range(16, 34), LH.R.range(-200, -150));
      gull.userData.path = {
        center: center,
        radius: LH.R.range(12, 46),
        speed: LH.R.range(0.12, 0.3) * (LH.R.chance(0.5) ? 1 : -1),
        phase: LH.R.range(0, 6.28),
        bobAmp: LH.R.range(0.6, 2.2)
      };
      gull.scale.setScalar(LH.R.range(1.4, 2.2));
      group.add(gull);
      gulls.push(gull);
    }

    scene.add(group);

    LH.onUpdate(function (dt, t) {
      // 渔船随浪晃动
      boats.forEach(function (b) {
        var s = b.userData.bob;
        b.position.y = s.baseY + Math.sin(t * 0.85 + s.phase) * s.amp;
        b.rotation.z = Math.sin(t * 0.62 + s.phase * 1.7) * 0.026;
        b.rotation.x = Math.sin(t * 0.47 + s.phase) * 0.018;
      });
      floatBalls.forEach(function (ball) {
        ball.position.y = ball.userData.baseY + Math.sin(t * 1.1 + ball.userData.phase) * 0.09;
      });
      // 浮标随浪起伏
      buoys.forEach(function (g, i) {
        var ph = g.userData.buoy.phase + i * 0.4;
        g.position.y = Math.sin(t * 0.9 + ph) * 0.22;
        g.rotation.z = Math.sin(t * 0.7 + ph) * 0.08;
        g.rotation.x = Math.cos(t * 0.55 + ph) * 0.06;
      });
      // 海鸥绕圈 + 扇翅
      gulls.forEach(function (g) {
        var p = g.userData.path;
        var a = p.phase + t * p.speed;
        g.position.set(
          p.center.x + Math.cos(a) * p.radius,
          p.center.y + Math.sin(t * 0.5 + p.phase) * p.bobAmp,
          p.center.z + Math.sin(a) * p.radius
        );
        // 朝向切线方向
        var dirX = -Math.sin(a) * Math.sign(p.speed);
        var dirZ = Math.cos(a) * Math.sign(p.speed);
        g.rotation.y = Math.atan2(dirX, dirZ);
        var w = Math.sin(t * g.userData.flap.rate + g.userData.flap.phase);
        g.userData.flap.wingL.rotation.z = w * 0.5 + 0.1;
        g.userData.flap.wingR.rotation.z = -w * 0.5 - 0.1;
        g.rotation.z = w * 0.06;
      });
    });
  };
})();
