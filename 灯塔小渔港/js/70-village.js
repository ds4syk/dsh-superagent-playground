/**
 * 70-village.js —— 西侧村落：彩色海边小屋、松树、电线杆、自动贩卖机、小径
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});

  function addWindow(parent, x, y, z, w, h, lit, ry) {
    var frame = new THREE.Mesh(
      new THREE.BoxGeometry(w + 0.14, h + 0.14, 0.06),
      LH.std({ color: LH.col('#efe7d8'), roughness: 0.8 })
    );
    frame.position.set(x, y, z);
    frame.rotation.y = ry || 0;
    parent.add(frame);
    var pane = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({ color: LH.col(lit ? '#ffc682' : '#5b6a70'), toneMapped: !lit })
    );
    pane.position.set(x, y, z);
    pane.rotation.y = ry || 0;
    pane.translateZ(0.035);
    parent.add(pane);
    return pane;
  }

  function buildHouse(opts) {
    var g = new THREE.Group();
    var w = opts.w, d = opts.d, h = opts.h;
    var groundY = LH.terrainHeight(opts.x, opts.z);

    // 石砌地基
    var foundation = new THREE.Mesh(
      LH.roundedBox(w + 0.3, 1.4, d + 0.3, 0.12, 2),
      new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.stone, 2, 1), roughness: 0.96 })
    );
    foundation.position.y = -0.2;
    foundation.castShadow = foundation.receiveShadow = true;
    g.add(foundation);

    // 墙体
    var wallTex = LH.texRepeat(opts.plaster, 2, 1.4);
    var body = new THREE.Mesh(
      LH.roundedBox(w, h, d, 0.22, 3),
      new THREE.MeshStandardMaterial({ map: wallTex, color: LH.col(opts.tint || '#ffffff'), roughness: 0.92 })
    );
    body.position.y = h / 2 + 0.5;
    body.castShadow = body.receiveShadow = true;
    g.add(body);

    // 双坡屋顶
    var slope = 0.52;
    var rise = (d / 2) * Math.tan(slope);
    var roofMat = new THREE.MeshStandardMaterial({ map: LH.texRepeat(opts.roof, 3, 1.4), roughness: 0.85 });
    var slabLen = (d / 2) / Math.cos(slope) + 0.35;
    var roofY = 0.5 + h + rise / 2;
    [-1, 1].forEach(function (sd) {
      var slab = new THREE.Mesh(new THREE.BoxGeometry(w + 0.7, 0.16, slabLen), roofMat);
      slab.position.set(0, roofY, sd * (d / 4 + 0.03));
      slab.rotation.x = sd * slope;
      slab.castShadow = true;
      g.add(slab);
    });
    var ridge = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, w + 0.7, 6), LH.std({ color: LH.col('#7a3f30'), roughness: 0.85 }));
    ridge.rotation.z = Math.PI / 2;
    ridge.position.y = 0.5 + h + rise;
    g.add(ridge);

    // 山墙三角
    var triShape = new THREE.Shape();
    triShape.moveTo(-d / 2, 0);
    triShape.lineTo(d / 2, 0);
    triShape.lineTo(0, rise);
    triShape.lineTo(-d / 2, 0);
    var triGeo = new THREE.ShapeGeometry(triShape);
    var triMat = new THREE.MeshStandardMaterial({ map: wallTex, color: LH.col(opts.tint || '#ffffff'), roughness: 0.92, side: THREE.DoubleSide });
    [-1, 1].forEach(function (sd) {
      var tri = new THREE.Mesh(triGeo, triMat);
      tri.rotation.y = sd * Math.PI / 2;
      tri.position.set(sd * (w / 2 + 0.005), 0.5 + h, 0);
      g.add(tri);
    });

    // 门窗（朝东 +X 为正面）
    var faceX = w / 2 + 0.03;
    addWindow(g, faceX, 0.5 + h * 0.55, -d * 0.22, 0.95, 0.95, opts.lit, Math.PI / 2);
    addWindow(g, faceX, 0.5 + h * 0.55, d * 0.22, 0.95, 0.95, opts.lit, Math.PI / 2);
    var door = new THREE.Mesh(
      LH.roundedBox(0.95, 1.85, 0.08, 0.03, 1),
      LH.std({ color: LH.col(opts.door || '#7a4a32'), roughness: 0.8 })
    );
    door.position.set(faceX - 0.02, 1.42, 0);
    door.rotation.y = Math.PI / 2;
    g.add(door);
    var handle = new THREE.Mesh(new THREE.SphereGeometry(0.045, 6, 5), LH.std({ color: LH.col('#c9b070'), metalness: 0.6, roughness: 0.35 }));
    handle.position.set(faceX + 0.04, 1.4, 0.32);
    g.add(handle);

    // 背面小窗
    addWindow(g, -faceX, 0.5 + h * 0.55, 0, 0.8, 0.8, false, -Math.PI / 2);

    // 烟囱
    if (opts.chimney !== false) {
      var chimney = new THREE.Mesh(
        LH.roundedBox(0.55, 1.6, 0.55, 0.05, 1),
        LH.std({ color: LH.col('#b5aa9a'), roughness: 0.95 })
      );
      chimney.position.set(w * 0.26, 0.5 + h + rise + 0.5, -d * 0.18);
      chimney.castShadow = true;
      g.add(chimney);
    }

    // 地面接触阴影
    g.add(LH.contact(0, -0.1, 0, w + 3.0, d + 3.0, 0.5, 0));

    g.position.set(opts.x, groundY + 0.15, opts.z);
    g.rotation.y = opts.rot || 0;
    return g;
  }

  // ---------------------------------------------------------------- 松树
  function buildPine(seed, scale) {
    var g = new THREE.Group();
    var rng = LH.rng(seed);
    var trunkMat = LH.std({ color: LH.col('#5a4532'), roughness: 0.95 });
    var trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.16 * scale, 0.26 * scale, 2.4 * scale, 7), trunkMat);
    trunk.position.y = 1.2 * scale;
    trunk.castShadow = true;
    g.add(trunk);
    var greenA = LH.col('#3f5e35');
    var greenB = LH.col('#547a41');
    var mat = new THREE.MeshStandardMaterial({ color: greenA.clone().lerp(greenB, rng.next()), roughness: 0.95, flatShading: false });
    var tiers = 4;
    for (var i = 0; i < tiers; i++) {
      var t = i / tiers;
      var r = (1.55 - t * 0.95) * scale;
      var hh = (2.5 - t * 0.5) * scale;
      var cone = new THREE.Mesh(new THREE.ConeGeometry(r, hh, 9, 1), mat);
      LH.displaceGeometry(cone.geometry, 0.05, 1.6, seed + i, 1);
      cone.position.y = (2.2 + i * 0.95) * scale;
      cone.rotation.y = rng.range(0, 6.28);
      cone.castShadow = true;
      g.add(cone);
    }
    return g;
  }

  function bush(seed, scale) {
    var geo = new THREE.IcosahedronGeometry(1, 1);
    LH.displaceGeometry(geo, 0.12, 1.4, seed, 0.8);
    var col = LH.col('#4d6b3c');
    var m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: col, roughness: 0.95 }));
    m.scale.set(scale, scale * 0.62, scale);
    m.castShadow = true;
    m.receiveShadow = true;
    return m;
  }

  // ---------------------------------------------------------------- 小径
  function buildPath(points, width, mat) {
    var curve = new THREE.CatmullRomCurve3(points.map(function (p) { return new THREE.Vector3(p[0], 0, p[1]); }));
    var segs = 64;
    var pos = [];
    var idx = [];
    var up = new THREE.Vector3(0, 1, 0);
    for (var i = 0; i <= segs; i++) {
      var t = i / segs;
      var p = curve.getPoint(t);
      var tangent = curve.getTangent(t);
      var side = new THREE.Vector3(-tangent.z, 0, tangent.x).normalize().multiplyScalar(width / 2);
      [1, -1].forEach(function (sd) {
        var xx = p.x + side.x * sd;
        var zz = p.z + side.z * sd;
        pos.push(xx, LH.terrainHeight(xx, zz) + 0.055, zz);
      });
      if (i < segs) {
        var a = i * 2;
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    var mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true;
    return mesh;
  }

  // ---------------------------------------------------------------- 草丛（实例化交叉面片）
  function grassGeometry() {
    var w = 0.66, h = 0.62;
    var pos = [
      -w / 2, 0, 0, w / 2, 0, 0, w / 2, h, 0, -w / 2, h, 0,
      0, 0, -w / 2, 0, 0, w / 2, 0, h, w / 2, 0, h, -w / 2
    ];
    var uv = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1];
    var idx = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    return geo;
  }

  function buildGrass(scene) {
    var mat = new THREE.MeshStandardMaterial({
      map: LH.tex.grass,
      transparent: true,
      alphaTest: 0.32,
      side: THREE.DoubleSide,
      roughness: 0.95,
      metalness: 0
    });
    var geo = grassGeometry();
    var COUNT = 900;
    var mesh = new THREE.InstancedMesh(geo, mat, COUNT);
    var m4 = new THREE.Matrix4();
    var q = new THREE.Quaternion();
    var e = new THREE.Euler();
    var v = new THREE.Vector3();
    var sc = new THREE.Vector3();
    var color = new THREE.Color();
    var rng = LH.rng(424242);
    var n = 0, guard = 0;
    while (n < COUNT && guard < COUNT * 12) {
      guard++;
      var zone = rng.int(0, 2);
      var x, z;
      if (zone === 0) { x = rng.range(-78, -22); z = rng.range(-105, 105); }
      else if (zone === 1) { x = rng.range(-34, -11); z = rng.range(-155, -45); }
      else { x = rng.range(-34, -16); z = rng.range(2, 42); }
      var h = LH.terrainHeight(x, z);
      if (h < 0.9 || h > 26) continue;
      var slope = Math.abs(LH.terrainHeight(x + 1, z) - h) + Math.abs(LH.terrainHeight(x, z + 1) - h);
      if (slope > 1.1) continue;
      e.set(0, rng.range(0, 6.283), 0);
      q.setFromEuler(e);
      v.set(x, h - 0.02, z);
      var s = rng.range(0.6, 1.5);
      sc.set(s, s * rng.range(0.8, 1.25), s);
      m4.compose(v, q, sc);
      mesh.setMatrixAt(n, m4);
      color.copy(LH.col('#6f8f45')).lerp(LH.col('#b4ad62'), rng.range(0, 0.55));
      color.offsetHSL(0, 0, rng.range(-0.05, 0.05));
      mesh.setColorAt(n, color);
      n++;
    }
    mesh.count = n;
    mesh.receiveShadow = true;
    mesh.castShadow = false;
    mesh.frustumCulled = false;
    scene.add(mesh);
  }

  LH.buildVillage = function (scene) {
    var group = new THREE.Group();
    group.name = 'village';

    // ---------------- 彩色海边小屋沿山坡排开
    var houses = [
      { x: -33.5, z: -40, rot: 1.62, w: 5.6, d: 6.2, h: 3.0, plaster: LH.tex.plasterCream, tint: '#f6dca0', roof: LH.tex.roofRed, lit: true },
      { x: -39.5, z: -22, rot: 1.45, w: 6.0, d: 5.4, h: 3.2, plaster: LH.tex.plasterMint, tint: '#bfe3cd', roof: LH.tex.roofBrown, lit: false },
      { x: -35.0, z: 0.5, rot: 1.75, w: 5.2, d: 6.0, h: 2.9, plaster: LH.tex.plasterPink, tint: '#f0c4b2', roof: LH.tex.roofRed, lit: true },
      { x: -41.0, z: 20, rot: 1.38, w: 6.4, d: 5.8, h: 3.4, plaster: LH.tex.plasterBlue, tint: '#b9d8e8', roof: LH.tex.roofGrey, lit: false },
      { x: -46.5, z: 40, rot: 1.68, w: 5.8, d: 6.4, h: 3.1, plaster: LH.tex.plasterWhite, tint: '#f3e9d2', roof: LH.tex.roofRed, lit: true },
      { x: -49.0, z: -62, rot: 1.55, w: 5.4, d: 5.6, h: 3.0, plaster: LH.tex.plasterCream, tint: '#ead0a8', roof: LH.tex.roofBrown, lit: false }
    ];
    houses.forEach(function (o) { group.add(buildHouse(o)); });

    // ---------------- 山坡上的松树与灌木
    var treeZone = [
      [-58, -150], [-72, -110], [-90, -70], [-64, -30], [-58, 10],
      [-67, 55], [-82, 90], [-100, 20], [-120, -20], [-105, 120],
      [-50, -95], [-45, 70], [-70, -20], [-95, 60], [-130, 80],
      [-55, 120], [-80, -150], [-110, -100], [-88, 5], [-75, 140],
      [-42, -120], [-36, -70], [-48, 30], [-44, 95], [-60, 150],
      [-100, -60], [-140, 40], [-160, -40], [-85, 170], [-125, 150],
      [-30, -58], [-33, 58], [-52, -45], [-38, 118], [-150, 110],
      [-31, 30], [-36, 48], [-29, 80], [-41, 90], [-34, 6], [-45, 108]
    ];
    treeZone.forEach(function (p, i) {
      var x = p[0] + LH.R.range(-3, 3), z = p[1] + LH.R.range(-3, 3);
      var y = LH.terrainHeight(x, z);
      if (y < 2.0) return;
      var ts = LH.R.range(0.8, 1.5);
      var tree = buildPine(600 + i * 13, ts);
      tree.position.set(x, y - 0.15, z);
      group.add(tree);
      group.add(LH.contact(x, y + 0.03, z, 2.6 * ts, 2.6 * ts, 0.42, 0));
    });
    for (var b = 0; b < 26; b++) {
      var bx = -30 - LH.R.range(0, 45);
      var bz = LH.R.range(-90, 110);
      var by = LH.terrainHeight(bx, bz);
      if (by < 1.6) continue;
      var bu = bush(700 + b, LH.R.range(0.45, 1.1));
      bu.position.set(bx, by + 0.1, bz);
      group.add(bu);
    }

    // ---------------- 电线杆与下垂电线
    var poleMat = LH.std({ color: LH.col('#5f5245'), roughness: 0.95 });
    var poleZs = [-46, -10, 26, 62];
    var poleX = -29.5;
    var poles = [];
    poleZs.forEach(function (pz, i) {
      var py = LH.terrainHeight(poleX, pz);
      var pole = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.18, 8.2, 8), poleMat);
      pole.position.set(poleX, py + 4.1, pz);
      pole.castShadow = true;
      group.add(pole);
      var arm = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.1, 0.1), poleMat);
      arm.position.set(poleX, py + 7.4, pz);
      group.add(arm);
      var arm2 = arm.clone();
      arm2.position.y = py + 6.7;
      group.add(arm2);
      // 绝缘子
      [-0.85, 0, 0.85].forEach(function (dx) {
        var ins = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 0.16, 6), LH.std({ color: LH.col('#cfd8d8'), roughness: 0.3 }));
        ins.position.set(poleX + dx, py + 7.52, pz);
        group.add(ins);
      });
      poles.push({ x: poleX, y: py + 7.52, z: pz });
    });
    var wireMat = LH.std({ color: LH.col('#2f3236'), roughness: 0.8 });
    for (var pi = 0; pi < poles.length - 1; pi++) {
      var A = poles[pi], B = poles[pi + 1];
      [-0.85, 0, 0.85].forEach(function (dx, wi) {
        var pts = [];
        for (var k = 0; k <= 12; k++) {
          var t = k / 12;
          var sag = Math.sin(Math.PI * t) * 0.9;
          pts.push(new THREE.Vector3(
            LH.lerp(A.x + dx, B.x + dx, t),
            LH.lerp(A.y, B.y, t) - sag + (wi === 1 ? 0.32 : 0),
            LH.lerp(A.z, B.z, t)
          ));
        }
        var curve = new THREE.CatmullRomCurve3(pts);
        var tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 20, 0.018, 4, false), wireMat);
        group.add(tube);
      });
    }

    // ---------------- 自动贩卖机
    var vmX = -21.0, vmZ = 6.4;
    var vmY = LH.terrainHeight(vmX, vmZ);
    var vm = new THREE.Group();
    var vmBody = new THREE.Mesh(
      LH.roundedBox(0.95, 1.95, 0.8, 0.06, 2),
      LH.std({ color: LH.col('#2a3a46'), roughness: 0.55, metalness: 0.1 })
    );
    vmBody.position.y = 1.0;
    vmBody.castShadow = vmBody.receiveShadow = true;
    vm.add(vmBody);
    var panel = new THREE.Mesh(
      new THREE.PlaneGeometry(0.82, 1.72),
      new THREE.MeshBasicMaterial({ map: LH.tex.vending, toneMapped: false })
    );
    panel.position.set(0.485, 1.02, 0);
    panel.rotation.y = Math.PI / 2;
    vm.add(panel);
    var glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: LH.tex.glow, color: LH.col('#bfe6ff'), transparent: true, opacity: 0.28, depthWrite: false, blending: THREE.AdditiveBlending
    }));
    glow.position.set(0.7, 1.1, 0);
    glow.scale.set(2.6, 3.2, 1);
    vm.add(glow);
    vm.position.set(vmX, vmY, vmZ);
    vm.rotation.y = -0.18;
    group.add(vm);

    // ---------------- 木栅栏（沿小径与院子）
    function fenceAlong(points, height) {
      var fg = new THREE.Group();
      var postMat = LH.std({ color: LH.col('#7a6248'), roughness: 0.95 });
      var railMat = LH.std({ color: LH.col('#8f7456'), roughness: 0.95 });
      var verts = [];
      for (var pi2 = 0; pi2 < points.length - 1; pi2++) {
        var A = points[pi2], B = points[pi2 + 1];
        var segLen = Math.hypot(B[0] - A[0], B[1] - A[1]);
        var steps = Math.max(1, Math.round(segLen / 1.6));
        for (var si = 0; si <= steps; si++) {
          var t = si / steps;
          var x = LH.lerp(A[0], B[0], t);
          var z = LH.lerp(A[1], B[1], t);
          verts.push([x, z]);
        }
      }
      for (var vi = 0; vi < verts.length; vi++) {
        var p2 = verts[vi];
        var gy = LH.terrainHeight(p2[0], p2[1]);
        var post = new THREE.Mesh(new THREE.BoxGeometry(0.09, height, 0.09), postMat);
        post.position.set(p2[0], gy + height / 2, p2[1]);
        post.castShadow = true;
        fg.add(post);
        if (vi > 0) {
          var q = verts[vi - 1];
          var qy = LH.terrainHeight(q[0], q[1]);
          var mx = (p2[0] + q[0]) / 2, mz = (p2[1] + q[1]) / 2;
          var len = Math.hypot(p2[0] - q[0], p2[1] - q[1]);
          var my = (gy + qy) / 2;
          [0.32, 0.72].forEach(function (hFrac) {
            var rail = new THREE.Mesh(new THREE.BoxGeometry(len, 0.07, 0.05), railMat);
            rail.position.set(mx, my + height * hFrac, mz);
            rail.rotation.y = -Math.atan2(p2[1] - q[1], p2[0] - q[0]);
            fg.add(rail);
          });
        }
      }
      return fg;
    }
    group.add(fenceAlong([[-19.8, -6.5], [-22.2, -2.5], [-23.6, 3.5], [-23.2, 10.5]], 0.95));
    group.add(fenceAlong([[-30.5, -45], [-32.5, -36], [-31.6, -28]], 0.9));

    // ---------------- 小径
    var pathMat = new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.sand, 1, 8), color: LH.col('#cbb894'), roughness: 1 });
    group.add(buildPath([[-19.2, 0], [-22.5, 10], [-24.5, 20], [-25.5, 28]], 2.2, pathMat));
    group.add(buildPath([[-19.2, 0], [-26, -6], [-34, -14], [-44, -26], [-50, -40]], 2.0, pathMat));

    // ---------------- 沙滩上的漂流木与礁石点缀
    var driftMat = LH.std({ color: LH.col('#b7a68a'), roughness: 0.95 });
    var drift = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.18, 2.6, 7), driftMat);
    drift.rotation.set(0.08, 0.4, Math.PI / 2 - 0.05);
    drift.position.set(-16.5, LH.terrainHeight(-16.5, 15) + 0.18, 15);
    drift.castShadow = true;
    group.add(drift);

    scene.add(group);
    buildGrass(scene);
  };
})();
