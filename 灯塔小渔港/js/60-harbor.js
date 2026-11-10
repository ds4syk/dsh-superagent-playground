/**
 * 60-harbor.js —— 渔港：木栈桥、石砌码头、渔网架、渔具仓库、木箱浮球
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});
  var W = LH.W;

  function woodMat(tex, rx, ry, tint) {
    return new THREE.MeshStandardMaterial({
      map: LH.texRepeat(tex, rx, ry),
      color: tint ? LH.col(tint) : 0xffffff,
      roughness: 0.88,
      metalness: 0.02
    });
  }

  // 带下垂的绳索
  function rope(points, radius, color) {
    var curve = new THREE.CatmullRomCurve3(points);
    var geo = new THREE.TubeGeometry(curve, 24, radius || 0.03, 5, false);
    return new THREE.Mesh(geo, LH.std({ color: color || '#d8c9a8', roughness: 0.95 }));
  }

  function bollard() {
    var g = new THREE.Group();
    var mat = LH.std({ color: LH.col('#4a5054'), roughness: 0.65, metalness: 0.25 });
    var body = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.17, 0.46, 10), mat);
    body.position.y = 0.23;
    var cap = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 8), mat);
    cap.position.y = 0.48;
    cap.scale.y = 0.7;
    g.add(body, cap);
    g.traverse(function (o) { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    return g;
  }

  LH.buildHarbor = function (scene) {
    var group = new THREE.Group();
    group.name = 'harbor';

    var deckY = W.DECK_Y;
    var x0 = W.PIER_X0, x1 = W.PIER_X1;
    var halfW = W.PIER_HALF_W;

    // ---------------- 栈桥桥面（实例化木板）
    var plankGeo = new THREE.BoxGeometry(0.27, 0.11, halfW * 2 + 0.06);
    var plankMat = woodMat(LH.tex.wood, 1, 1);
    var count = Math.floor((x1 - x0) / 0.305);
    var planks = new THREE.InstancedMesh(plankGeo, plankMat, count);
    planks.name = 'pierPlanks';
    var m4 = new THREE.Matrix4();
    var q = new THREE.Quaternion();
    var e = new THREE.Euler();
    var v = new THREE.Vector3();
    var s = new THREE.Vector3(1, 1, 1);
    var tint = new THREE.Color();
    for (var i = 0; i < count; i++) {
      var x = x0 + 0.15 + i * 0.305;
      e.set(LH.R.range(-0.012, 0.012), LH.R.range(-0.01, 0.01), LH.R.range(-0.012, 0.012));
      q.setFromEuler(e);
      v.set(x, deckY - 0.06 + LH.R.range(-0.018, 0.018), LH.R.range(-0.02, 0.02));
      m4.compose(v, q, s);
      planks.setMatrixAt(i, m4);
      var b = 0.72 + LH.R.range(0, 0.5);
      tint.setRGB(b, b * 0.97, b * 0.94);
      planks.setColorAt(i, tint);
    }
    planks.castShadow = true;
    planks.receiveShadow = true;
    planks.instanceMatrix.needsUpdate = true;
    if (planks.instanceColor) planks.instanceColor.needsUpdate = true;
    group.add(planks);

    // 纵梁与横梁
    var beamMat = woodMat(LH.tex.woodDark, 6, 1);
    [-1, 1].forEach(function (sd) {
      var beam = new THREE.Mesh(new THREE.BoxGeometry(x1 - x0 + 0.6, 0.22, 0.24), beamMat);
      beam.position.set((x0 + x1) / 2, deckY - 0.23, sd * (halfW - 0.18));
      beam.castShadow = true;
      group.add(beam);
    });
    for (var cx = x0 + 0.4; cx <= x1; cx += 2.4) {
      var cross = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.18, halfW * 2), beamMat);
      cross.position.set(cx, deckY - 0.34, 0);
      cross.castShadow = true;
      group.add(cross);
    }

    // 桩柱（按海底深度逐个拉长）
    var pileGeo = new THREE.CylinderGeometry(0.165, 0.19, 1, 10);
    var pileMat = woodMat(LH.tex.woodV, 1, 3, '#8a7a66');
    var positions = [];
    for (var px = x0 + 0.8; px <= x1 + 0.2; px += 2.4) {
      positions.push([px, -halfW - 0.06], [px, halfW + 0.06]);
    }
    var piles = new THREE.InstancedMesh(pileGeo, pileMat, positions.length);
    var pv = new THREE.Vector3();
    for (var p = 0; p < positions.length; p++) {
      var bx = positions[p][0], bz = positions[p][1];
      var bed = Math.min(-1.2, LH.terrainHeight(bx, bz));
      var len = deckY - 0.25 - bed;
      pv.set(bx, bed + len / 2, bz);
      m4.compose(pv, new THREE.Quaternion(), new THREE.Vector3(1, len, 1));
      piles.setMatrixAt(p, m4);
    }
    piles.castShadow = true;
    piles.receiveShadow = true;
    piles.instanceMatrix.needsUpdate = true;
    group.add(piles);

    // 露出桥面的桩头
    var capMat = woodMat(LH.tex.woodV, 1, 2, '#8f7a60');
    for (var pc = 0; pc < positions.length; pc += 3) {
      var capPile = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.17, 0.55, 9), capMat);
      capPile.position.set(positions[pc][0], deckY + 0.22, positions[pc][1]);
      capPile.castShadow = capPile.receiveShadow = true;
      group.add(capPile);
    }

    // 桩头绳索缠绕
    var wrapMat = LH.std({ color: LH.col('#b9a888'), roughness: 0.95 });
    for (var pw = 0; pw < positions.length; pw += 3) {
      var torus = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.035, 5, 14), wrapMat);
      torus.rotation.x = Math.PI / 2;
      torus.position.set(positions[pw][0], deckY - 0.28, positions[pw][1]);
      group.add(torus);
    }

    // 系船桩
    var bollards = [];
    [-14, -9, -4, 1, 5.4].forEach(function (bx2) {
      var b = bollard();
      b.position.set(bx2, deckY, halfW - 0.12);
      group.add(b);
      bollards.push(b);
    });
    LH.pierBollards = bollards;

    // 栈桥尽头小梯子
    var ladder = new THREE.Group();
    var lmat = LH.std({ color: LH.col('#8b7a62'), roughness: 0.9 });
    [-0.32, 0.32].forEach(function (dz) {
      var rail = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 3.4, 6), lmat);
      rail.position.set(0, deckY - 1.6, dz);
      ladder.add(rail);
    });
    for (var lr = 0; lr < 7; lr++) {
      var rung = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.64, 6), lmat);
      rung.rotation.x = Math.PI / 2;
      rung.position.set(0, deckY - 0.35 - lr * 0.45, 0);
      ladder.add(rung);
    }
    ladder.position.set(x1 - 0.2, 0, -halfW - 0.1);
    group.add(ladder);

    // 救生圈
    var ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.2, 0.055, 8, 20),
      LH.std({ color: LH.col('#df6a4c'), roughness: 0.7 })
    );
    ring.position.set(-6.4, deckY + 0.62, -halfW - 0.05);
    ring.rotation.y = Math.PI / 2;
    group.add(ring);
    var ringPost = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.9, 0.09), lmat);
    ringPost.position.set(-6.4, deckY + 0.45, -halfW - 0.05);
    group.add(ringPost);
    var ringStripe = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.057, 8, 20, Math.PI * 0.7), LH.std({ color: LH.col('#f2ead6'), roughness: 0.7 }));
    ringStripe.position.copy(ring.position);
    ringStripe.rotation.copy(ring.rotation);
    group.add(ringStripe);

    // 栈桥上的木箱 / 木桶 / 缆绳盘
    var crateTex = woodMat(LH.tex.wood, 1.4, 1.4, '#b79a78');
    function crate(w, h, d, x, y, z, ry) {
      var c = new THREE.Mesh(LH.roundedBox(w, h, d, 0.04, 2), crateTex);
      c.position.set(x, y, z);
      c.rotation.y = ry || 0;
      c.castShadow = c.receiveShadow = true;
      // 箱角木条
      var trimMat = LH.std({ color: LH.col('#6d4c32'), roughness: 0.9 });
      var trim = new THREE.Mesh(new THREE.BoxGeometry(w + 0.02, 0.055, 0.055), trimMat);
      trim.position.set(0, h * 0.34, d / 2);
      c.add(trim);
      var trim2 = trim.clone();
      trim2.position.y = -h * 0.34;
      c.add(trim2);
      return c;
    }
    group.add(crate(0.75, 0.55, 0.6, -12.4, deckY + 0.28, 0.55, 0.2));
    group.add(crate(0.6, 0.45, 0.55, -12.4, deckY + 0.23, -0.5, -0.35));
    group.add(crate(0.55, 0.5, 0.5, -12.9, deckY + 0.26, 0.2, 1.1));

    var barrelGeo = new THREE.CylinderGeometry(0.3, 0.26, 0.82, 14);
    var barrel = new THREE.Mesh(barrelGeo, woodMat(LH.tex.woodV, 2, 1, '#8f6a48'));
    barrel.position.set(-8.2, deckY + 0.42, -0.6);
    barrel.castShadow = barrel.receiveShadow = true;
    group.add(barrel);
    var hoopMat = LH.std({ color: LH.col('#4e4a44'), metalness: 0.5, roughness: 0.5 });
    [-0.28, 0.02, 0.3].forEach(function (hy) {
      var hoop = new THREE.Mesh(new THREE.TorusGeometry(0.3, 0.02, 5, 16), hoopMat);
      hoop.rotation.x = Math.PI / 2;
      hoop.position.set(-8.2, deckY + 0.42 + hy, -0.6);
      group.add(hoop);
    });

    var coil = new THREE.Mesh(
      new THREE.TorusGeometry(0.32, 0.11, 7, 18),
      LH.std({ color: LH.col('#c9b58f'), roughness: 0.95 })
    );
    coil.rotation.x = Math.PI / 2;
    coil.position.set(-10.6, deckY + 0.11, 0.5);
    group.add(coil);

    // 岸边下栈桥的石阶
    var stepMat = new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.stone, 1, 1), roughness: 0.95 });
    for (var st = 0; st < 4; st++) {
      var step = new THREE.Mesh(LH.roundedBox(1.9, 0.3, 1.0, 0.08, 1), stepMat);
      step.position.set(x0 - 0.45 - st * 0.5, deckY - 0.12 - st * 0.28, 0);
      step.castShadow = step.receiveShadow = true;
      group.add(step);
    }

    // ---------------- 石砌码头（南侧海湾）
    var quay = new THREE.Group();
    quay.name = 'quay';
    var quayX0 = -27, quayX1 = -14.5, quayZ0 = 28, quayZ1 = 88;
    var quayTopY = 1.55;
    var quayBody = new THREE.Mesh(
      LH.roundedBox(quayX1 - quayX0, 3.0, quayZ1 - quayZ0, 0.35, 3),
      new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.stone, 3, 6), roughness: 0.96 })
    );
    quayBody.position.set((quayX0 + quayX1) / 2, quayTopY - 1.5, (quayZ0 + quayZ1) / 2);
    quayBody.castShadow = quayBody.receiveShadow = true;
    quay.add(quayBody);
    // 码头面石板
    var paving = new THREE.Mesh(
      LH.roundedBox(quayX1 - quayX0 - 0.3, 0.12, quayZ1 - quayZ0 - 0.3, 0.06, 2),
      new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.stone, 4, 8), color: LH.col('#c8c2b4'), roughness: 0.92 })
    );
    paving.position.set((quayX0 + quayX1) / 2, quayTopY + 0.06, (quayZ0 + quayZ1) / 2);
    paving.receiveShadow = true;
    quay.add(paving);

    [-24, -18.5].forEach(function (bz2) {
      for (var qz = quayZ0 + 6; qz <= quayZ1 - 4; qz += 12) {
        var b = bollard();
        b.position.set(bz2, quayTopY + 0.12, qz);
        quay.add(b);
      }
    });

    // 码头台阶
    for (var qs = 0; qs < 3; qs++) {
      var qstep = new THREE.Mesh(LH.roundedBox(1.6, 0.28, 2.0, 0.06, 1), stepMat);
      qstep.position.set(quayX1 + 0.2 + qs * 0.45, quayTopY + 0.05 - qs * 0.28, quayZ1 - 9);
      qstep.castShadow = qstep.receiveShadow = true;
      quay.add(qstep);
    }

    // 渔网晾晒架
    var rackWood = woodMat(LH.tex.woodDark, 2, 1);
    var netMat = new THREE.MeshStandardMaterial({
      map: LH.texRepeat(LH.tex.net, 3, 3),
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      roughness: 0.95,
      alphaTest: 0.15
    });
    [40, 52, 64].forEach(function (rz, ri) {
      var rack = new THREE.Group();
      var h = 3.0;
      [-1.5, 1.5].forEach(function (dz) {
        var pole = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.09, h, 7), rackWood);
        pole.position.set(0, h / 2, dz);
        pole.castShadow = true;
        rack.add(pole);
      });
      var topBar = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 3.5, 7), rackWood);
      topBar.rotation.x = Math.PI / 2;
      topBar.position.y = h - 0.1;
      rack.add(topBar);
      var net = new THREE.Mesh(new THREE.PlaneGeometry(3.3, 2.3), netMat);
      net.position.set(0, h - 1.25, 0);
      net.rotation.y = Math.PI / 2;
      rack.add(net);
      // 网上浮球
      for (var f = 0; f < 4; f++) {
        var float = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 6), LH.std({ color: LH.col('#e07a44'), roughness: 0.55 }));
        float.position.set(0.06, h - 0.45 - f * 0.5, -1.1 + f * 0.75);
        rack.add(float);
      }
      rack.position.set(-20.5 + (ri % 2) * 1.2, quayTopY + 0.12, rz);
      rack.rotation.y = Math.PI / 2 + (ri - 1) * 0.08;
      quay.add(rack);
    });

    // 码头上的木箱、浮球堆
    var quayCrateY = quayTopY + 0.12;
    quay.add(crate(0.9, 0.65, 0.7, -22.5, quayCrateY + 0.33, 76, 0.4));
    quay.add(crate(0.7, 0.5, 0.6, -23.2, quayCrateY + 0.25, 77.4, -0.2));
    quay.add(crate(0.8, 0.6, 0.65, -21.6, quayCrateY + 0.3, 82, 0.9));
    var floatColors = ['#e07a44', '#e8d9b0', '#4f7f92', '#d8563c'];
    for (var fb = 0; fb < 9; fb++) {
      var ball = new THREE.Mesh(
        new THREE.SphereGeometry(LH.R.range(0.16, 0.3), 10, 8),
        LH.std({ color: LH.col(floatColors[fb % floatColors.length]), roughness: 0.5 })
      );
      var fx = -18 + LH.R.range(-2, 2);
      ball.position.set(fx, quayCrateY + 0.25, 70 + LH.R.range(-2, 2));
      ball.castShadow = true;
      quay.add(ball);
    }

    // ---------------- 渔具仓库（岸上）
    var wh = new THREE.Group();
    var whX = -43, whZ = 74;
    var groundY = LH.terrainHeight(whX, whZ);
    var whBody = new THREE.Mesh(
      LH.roundedBox(9.5, 5.2, 7.5, 0.25, 3),
      new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.woodGrey, 4, 2), roughness: 0.9 })
    );
    whBody.position.set(whX, groundY + 2.6, whZ);
    whBody.castShadow = whBody.receiveShadow = true;
    wh.add(whBody);
    // 锈铁皮屋顶
    var whRoof = new THREE.Mesh(
      new THREE.BoxGeometry(10.6, 0.22, 8.6),
      new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.rust, 4, 3), roughness: 0.75, metalness: 0.25 })
    );
    whRoof.position.set(whX, groundY + 5.35, whZ);
    whRoof.rotation.z = 0.035;
    whRoof.castShadow = true;
    wh.add(whRoof);
    var whDoor = new THREE.Mesh(
      new THREE.PlaneGeometry(3.4, 3.4),
      LH.std({ color: LH.col('#4c4a42'), roughness: 0.9 })
    );
    whDoor.position.set(whX + 4.77, groundY + 1.9, whZ);
    whDoor.rotation.y = Math.PI / 2;
    wh.add(whDoor);
    var whWindow = windowPaneH(1.3, 0.9, '#ffc782');
    whWindow.position.set(whX + 4.78, groundY + 3.9, whZ + 2.2);
    whWindow.rotation.y = Math.PI / 2;
    wh.add(whWindow);
    quay.add(wh);

    function windowPaneH(w, h2, color) {
      return new THREE.Mesh(new THREE.PlaneGeometry(w, h2), new THREE.MeshBasicMaterial({ color: LH.col(color), toneMapped: false }));
    }

    // 仓库旁的小推车与木箱
    quay.add(crate(1.0, 0.7, 0.8, -36, LH.terrainHeight(-36, 68) + 0.35, 68, 0.3));
    quay.add(crate(0.7, 0.55, 0.6, -35.2, LH.terrainHeight(-35.2, 69.4) + 0.28, 69.4, -0.5));

    group.add(quay);

    scene.add(group);
    LH.harbor = { group: group, quayTopY: quayTopY, quayX0: quayX0, quayX1: quayX1, quayZ0: quayZ0, quayZ1: quayZ1 };
  };
})();
