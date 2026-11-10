/**
 * 55-lighthouse.js —— 红白条纹灯塔与守塔人小屋
 * 北侧岬角上：石基座 + 条纹塔身 + 灯室围栏阳台 + 旋转光束 + 亮窗小屋。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});

  function windowPane(w, h, color) {
    var mat = new THREE.MeshBasicMaterial({ color: LH.col(color || '#ffcf8a'), toneMapped: false });
    var m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    return m;
  }

  LH.buildLighthouse = function (scene) {
    var group = new THREE.Group();
    group.name = 'lighthouse';
    var base = new THREE.Vector3(12, 12.3, -176);

    // 石基座
    var plinth = new THREE.Mesh(
      new THREE.CylinderGeometry(3.5, 3.9, 1.5, 20),
      new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.stone, 2, 1), color: LH.col('#cfc8b8'), roughness: 0.95 })
    );
    plinth.position.set(base.x, base.y + 0.7, base.z);
    plinth.castShadow = plinth.receiveShadow = true;
    group.add(plinth);

    // 塔身（轻微收分）
    var towerH = 17.5;
    var tower = new THREE.Mesh(
      new THREE.CylinderGeometry(2.05, 2.9, towerH, 32, 1, false),
      new THREE.MeshStandardMaterial({
        map: LH.texRepeat(LH.tex.stripes, 1, 1),
        roughness: 0.62,
        metalness: 0.0
      })
    );
    tower.position.set(base.x, base.y + 1.4 + towerH / 2, base.z);
    tower.castShadow = tower.receiveShadow = true;
    group.add(tower);

    var topY = base.y + 1.4 + towerH; // 塔顶

    // 阳台平台
    var gallery = new THREE.Mesh(
      new THREE.CylinderGeometry(3.05, 2.6, 0.45, 28),
      LH.std({ color: LH.col('#4a4f52'), roughness: 0.75, metalness: 0.2 })
    );
    gallery.position.set(base.x, topY + 0.22, base.z);
    gallery.castShadow = gallery.receiveShadow = true;
    group.add(gallery);

    // 围栏
    var railMat = LH.std({ color: LH.col('#3c4144'), roughness: 0.6, metalness: 0.3 });
    var railTop = topY + 1.35;
    for (var i = 0; i < 20; i++) {
      var a = (i / 20) * Math.PI * 2;
      var post = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.15, 6), railMat);
      post.position.set(base.x + Math.cos(a) * 2.85, topY + 0.78, base.z + Math.sin(a) * 2.85);
      group.add(post);
    }
    var rail = new THREE.Mesh(new THREE.TorusGeometry(2.85, 0.06, 6, 40), railMat);
    rail.rotation.x = Math.PI / 2;
    rail.position.set(base.x, railTop, base.z);
    group.add(rail);
    var railMid = rail.clone();
    railMid.position.y = topY + 0.85;
    group.add(railMid);

    // 灯室
    var glassMat = new THREE.MeshStandardMaterial({
      color: LH.col('#bfe0ea'),
      roughness: 0.08,
      metalness: 0.05,
      transparent: true,
      opacity: 0.45,
      emissive: LH.col('#ffd6a0'),
      emissiveIntensity: 0.6
    });
    var lantern = new THREE.Mesh(new THREE.CylinderGeometry(1.55, 1.55, 2.4, 20), glassMat);
    lantern.position.set(base.x, topY + 1.85, base.z);
    group.add(lantern);

    // 灯室骨架
    for (var k = 0; k < 8; k++) {
      var ang = (k / 8) * Math.PI * 2;
      var bar = new THREE.Mesh(new THREE.BoxGeometry(0.07, 2.4, 0.07), railMat);
      bar.position.set(base.x + Math.cos(ang) * 1.55, topY + 1.85, base.z + Math.sin(ang) * 1.55);
      group.add(bar);
    }

    // 灯芯
    var lampCore = new THREE.Mesh(
      new THREE.SphereGeometry(0.55, 16, 12),
      new THREE.MeshBasicMaterial({ color: LH.col('#fff2cf'), toneMapped: false })
    );
    lampCore.position.set(base.x, topY + 1.9, base.z);
    group.add(lampCore);
    var lampGlow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: LH.tex.glow,
      color: LH.col('#ffe3b0'),
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    }));
    lampGlow.position.copy(lampCore.position);
    lampGlow.scale.set(9, 9, 1);
    group.add(lampGlow);

    // 灯室穹顶
    var roof = new THREE.Mesh(
      new THREE.SphereGeometry(1.85, 24, 14, 0, Math.PI * 2, 0, Math.PI / 2),
      LH.std({ color: LH.col('#a84432'), roughness: 0.55, metalness: 0.1 })
    );
    roof.position.set(base.x, topY + 3.05, base.z);
    roof.castShadow = true;
    group.add(roof);
    var finial = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 6), LH.std({ color: LH.col('#d9b46a'), metalness: 0.7, roughness: 0.3 }));
    finial.position.set(base.x, topY + 4.95, base.z);
    group.add(finial);
    var rod = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.2, 5), finial.material);
    rod.position.set(base.x, topY + 4.4, base.z);
    group.add(rod);

    // 旋转光束：两束相反方向的加色光锥
    var beamGroup = new THREE.Group();
    beamGroup.position.set(base.x, topY + 1.9, base.z);
    var beamMat = new THREE.MeshBasicMaterial({
      color: LH.col('#ffe6bd'),
      transparent: true,
      opacity: 0.02,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    for (var b = 0; b < 2; b++) {
      var cone = new THREE.Mesh(new THREE.ConeGeometry(3.2, 88, 18, 1, true), beamMat);
      cone.rotation.z = Math.PI / 2;
      cone.rotation.y = b * Math.PI;
      cone.position.set(b === 0 ? 47 : -47, 0, 0);
      cone.rotation.x = b === 0 ? -Math.PI / 2 : Math.PI / 2;
      beamGroup.add(cone);
    }
    group.add(beamGroup);

    // 小屋
    var cottage = new THREE.Group();
    cottage.name = 'keeperCottage';
    var shelf = LH.makeRock(901, 6.5, 2.2, 6.5, 3.0, 10.4, -166, { mat: 0 });
    group.add(shelf);
    var body = new THREE.Mesh(
      LH.roundedBox(5.4, 3.0, 4.2, 0.18, 3),
      new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.plasterWhite, 2, 1), roughness: 0.92 })
    );
    body.position.set(3.0, 12.2, -166);
    body.castShadow = body.receiveShadow = true;
    cottage.add(body);

    // 双坡屋顶（两片斜板，屋脊在中间）
    var roofMat = new THREE.MeshStandardMaterial({ map: LH.texRepeat(LH.tex.roofRed, 2, 1), roughness: 0.85 });
    var slope = 0.62;
    var r1 = new THREE.Mesh(new THREE.BoxGeometry(5.95, 0.2, 2.5), roofMat);
    r1.position.set(3.0, 14.32, -164.95);
    r1.rotation.x = slope;
    r1.castShadow = true;
    cottage.add(r1);
    var r2 = new THREE.Mesh(new THREE.BoxGeometry(5.95, 0.2, 2.5), roofMat);
    r2.position.set(3.0, 14.32, -167.05);
    r2.rotation.x = -slope;
    r2.castShadow = true;
    cottage.add(r2);
    // 屋脊
    var ridge = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 6.0, 8), LH.std({ color: LH.col('#8d3f2f'), roughness: 0.8 }));
    ridge.rotation.z = Math.PI / 2;
    ridge.position.set(3.0, 15.0, -166.0);
    cottage.add(ridge);

    // 烟囱
    var chimney = new THREE.Mesh(new THREE.BoxGeometry(0.62, 1.8, 0.62), LH.std({ color: LH.col('#b7ada0'), roughness: 0.95 }));
    chimney.position.set(5.1, 15.3, -167.2);
    chimney.castShadow = true;
    cottage.add(chimney);

    // 窗与门（暖光）
    var w1 = windowPane(0.85, 0.75, '#ffc47a');
    w1.position.set(1.3, 12.35, -163.87);
    cottage.add(w1);
    var w2 = w1.clone();
    w2.position.x = 4.6;
    cottage.add(w2);
    var w3 = windowPane(0.75, 0.7, '#ffcf8a');
    w3.rotation.y = Math.PI / 2;
    w3.position.set(0.27, 12.3, -166);
    cottage.add(w3);
    var door = windowPane(0.8, 1.5, '#8a5a3a');
    door.position.set(2.4, 11.9, -163.88);
    cottage.add(door);
    // 门廊灯
    var porch = new THREE.Sprite(new THREE.SpriteMaterial({
      map: LH.tex.glow, color: LH.col('#ffd39a'), transparent: true, opacity: 0.75, depthWrite: false, blending: THREE.AdditiveBlending
    }));
    porch.position.set(3.4, 13.2, -163.6);
    porch.scale.set(2.4, 2.4, 1);
    cottage.add(porch);

    group.add(cottage);

    // 烟囱白烟
    var smokeMat = new THREE.SpriteMaterial({
      map: LH.tex.mist, color: LH.col('#e8e2d8'), transparent: true, opacity: 0.24, depthWrite: false
    });
    var smoke = [];
    for (var s = 0; s < 9; s++) {
      var sp = new THREE.Sprite(smokeMat.clone());
      sp.position.set(4.6, 15.2 + s * 1.1, -166.6);
      sp.scale.setScalar(1.6 + s * 0.55);
      sp.material.opacity = 0.22 - s * 0.02;
      group.add(sp);
      smoke.push({ sp: sp, phase: s / 9 });
    }

    // 塔基周围礁石
    group.add(LH.makeRock(902, 3.2, 2.0, 3.0, 7.6, 11.0, -171, { mat: 1 }));
    group.add(LH.makeRock(903, 2.6, 1.6, 2.4, 16.6, 11.2, -171.5, { mat: 0 }));

    scene.add(group);
    LH.lighthouse = { group: group, beam: beamGroup, lampGlow: lampGlow };

    LH.onUpdate(function (dt, t) {
      beamGroup.rotation.y = t * 0.42;
      lampGlow.material.opacity = 0.75 + Math.sin(t * 2.1) * 0.12;
      for (var i = 0; i < smoke.length; i++) {
        var s = smoke[i];
        var p = (t * 0.05 + s.phase) % 1;
        s.sp.position.y = 15.2 + p * 9.0;
        s.sp.position.x = 4.6 + Math.sin(p * 5.0 + s.phase * 9.0) * 0.9;
        s.sp.material.opacity = 0.20 * (1 - p) * (1 - p);
      }
    });
  };
})();
