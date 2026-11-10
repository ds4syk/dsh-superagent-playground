/**
 * 80-mist.js —— 贴海面的薄雾
 * 多层水平半透明雾片缓慢漂移，配合指数雾把远景柔化。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});

  LH.buildMist = function (scene) {
    var group = new THREE.Group();
    group.name = 'mist';
    var layers = [];

    var matBase = {
      map: LH.tex.mist,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      fog: false
    };

    var specs = [
      { w: 420, h: 240, y: 0.55, x: -60, z: 40, op: 0.045, sp: 0.55, col: '#f6e6d4' },
      { w: 380, h: 220, y: 0.85, x: 150, z: -80, op: 0.04, sp: 0.85, col: '#e9eef2' },
      { w: 520, h: 280, y: 1.3, x: 100, z: 140, op: 0.035, sp: 0.4, col: '#f6e8d8' },
      { w: 360, h: 200, y: 0.65, x: -150, z: -120, op: 0.04, sp: 0.65, col: '#ece8e2' }
    ];

    specs.forEach(function (s, i) {
      var m = new THREE.MeshBasicMaterial(Object.assign({}, matBase, {
        color: LH.col(s.col),
        opacity: s.op
      }));
      var geo = new THREE.PlaneGeometry(s.w, s.h);
      var mesh = new THREE.Mesh(geo, m);
      mesh.rotation.x = -Math.PI / 2;
      mesh.rotation.z = (i * 1.3) % 6.28;
      mesh.position.set(s.x, s.y, s.z);
      mesh.renderOrder = 10 + i;
      group.add(mesh);
      layers.push({ mesh: mesh, speed: s.sp, baseX: s.x, baseZ: s.z, w: s.w });
    });

    scene.add(group);

    LH.onUpdate(function (dt, t) {
      for (var i = 0; i < layers.length; i++) {
        var l = layers[i];
        l.mesh.position.x = l.baseX + Math.sin(t * 0.02 * l.speed + i) * 14;
        l.mesh.position.z = l.baseZ + Math.cos(t * 0.015 * l.speed + i * 2) * 9;
        l.mesh.rotation.z = (i * 1.3 + t * 0.004 * l.speed) % 6.28;
      }
    });
  };
})();
