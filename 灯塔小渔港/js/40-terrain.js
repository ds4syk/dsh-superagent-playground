/**
 * 40-terrain.js —— 海岸地形
 * 解析高度场：西侧山坡与沙滩、南侧海湾内凹、北侧沙滩；同一函数供地形网格、
 * 散步范围、道具落地和水面浅滩泡沫贴图共用。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});

  // 海岸线：z 处的岸线 x 坐标（越负越靠西）
  LH.coastX = function (z) {
    var c = -12.5;
    c += 7.5 * Math.exp(-Math.pow((z + 95) / 80, 2));
    c -= 19.0 * Math.exp(-Math.pow((z - 78) / 46, 2));
    c += 2.2 * Math.sin(z * 0.045) + 1.1 * Math.sin(z * 0.11 + 1.9);
    return c;
  };

  LH.terrainHeight = function (x, z) {
    var cx = LH.coastX(z);
    var d = cx - x; // 向内陆为正
    var h;
    if (d <= 0) {
      h = Math.max(d * 0.30, -24);
    } else {
      h = 0.10 * d + 0.0016 * d * d;
    }

    // 岸线沙丘起伏
    h += 0.6 * Math.exp(-Math.pow((d - 7) / 4.5, 2)) * (0.4 + 0.6 * LH.fbm2(x * 0.06, z * 0.06, 3, 31));

    // 西侧山坡
    var inland = Math.max(0, d - 22);
    var rise = LH.smoothstep(0, 150, inland);
    h += rise * (LH.fbm2(x * 0.006, z * 0.006, 4, 41) * 36 + 5);

    // 中景起伏
    h += LH.fbm2(x * 0.021, z * 0.021, 3, 51) * 1.6 * LH.smoothstep(0, 30, inland);

    // 北侧沙滩略低平
    var north = LH.smoothstep(-60, -130, z);
    h *= (1 - north * 0.25);
    return h;
  };

  // 水面浅滩蒙版：R=高度，G=泡沫/浅滩强度
  function buildCoastTexture() {
    var S = 512;
    var rect = [-320, -280, 380, 620]; // minX, minZ, sizeX, sizeZ
    var cv = document.createElement('canvas');
    cv.width = cv.height = S;
    var ctx = cv.getContext('2d');
    var img = ctx.createImageData(S, S);
    for (var j = 0; j < S; j++) {
      var z = rect[1] + (j / (S - 1)) * rect[3];
      for (var i = 0; i < S; i++) {
        var x = rect[0] + (i / (S - 1)) * rect[2];
        var h = LH.terrainHeight(x, z);
        var r = LH.clamp((h + 25) / 75, 0, 1);
        var foam = 1.0 - LH.smoothstep(0.15, 2.2, Math.abs(h));
        foam += 0.30 * (1.0 - LH.smoothstep(0.15, 1.2, Math.abs(h)));
        var g = LH.clamp(foam, 0, 1);
        var idx = (j * S + i) * 4;
        img.data[idx] = r * 255;
        img.data[idx + 1] = g * 255;
        img.data[idx + 2] = 0;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    var t = new THREE.CanvasTexture(cv);
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    t.encoding = THREE.LinearEncoding;
    t.minFilter = THREE.LinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.needsUpdate = true;
    return {
      texture: t,
      rect: new THREE.Vector4(rect[0], rect[1], rect[2], rect[3])
    };
  }

  LH.buildTerrain = function (scene) {
    var X0 = -430, X1 = 46, Z0 = -360, Z1 = 360;
    var SEGX = 190, SEGZ = 280;
    var geo = new THREE.PlaneGeometry(X1 - X0, Z1 - Z0, SEGX, SEGZ);
    geo.rotateX(-Math.PI / 2);

    var pos = geo.attributes.position;
    var colors = new Float32Array(pos.count * 3);
    var sandDry = LH.col('#d8c49c');
    var sandWet = LH.col('#b39a6f');
    var sandDeep = LH.col('#8d7a58');
    var grassA = LH.col('#7e9346');
    var grassB = LH.col('#677c37');
    var grassDry = LH.col('#a09a58');
    var rock = LH.col('#79726a');
    var rockDark = LH.col('#5f5a52');
    var c = new THREE.Color();
    var minX = X0 + (X1 - X0) / 2, minZ = Z0 + (Z1 - Z0) / 2;

    for (var i = 0; i < pos.count; i++) {
      var x = pos.getX(i) + minX;
      var z = pos.getZ(i) + minZ;
      var h = LH.terrainHeight(x, z);
      pos.setXYZ(i, x, h, z);

      // 坡度（有限差分）
      var e = 1.2;
      var hx = LH.terrainHeight(x + e, z) - LH.terrainHeight(x - e, z);
      var hz = LH.terrainHeight(x, z + e) - LH.terrainHeight(x, z - e);
      var slope = Math.sqrt(hx * hx + hz * hz) / (2 * e);

      var n = LH.fbm2(x * 0.05, z * 0.05, 3, 61);
      var sandGate = 1.9 + (n - 0.5) * 1.7 + (LH.noise2(x * 0.3, z * 0.3, 67) - 0.5) * 0.6;
      var grassCol = grassA.clone().lerp(grassB, LH.clamp(n * 1.6 - 0.2, 0, 1));
      if (n > 0.62) grassCol.lerp(grassDry, (n - 0.62) * 1.6);
      if (h > 30) grassCol.lerp(rockDark, LH.clamp((h - 30) / 60, 0, 0.5));
      if (slope > 0.55) grassCol.lerp(rock, LH.clamp((slope - 0.55) * 0.7, 0, 0.85));
      if (slope > 0.9) grassCol.lerp(rockDark, 0.3);
      var sandCol;
      if (h < 0.35) {
        sandCol = sandWet.clone().lerp(sandDeep, LH.clamp(-h / 6, 0, 1));
      } else {
        sandCol = sandWet.clone().lerp(sandDry, LH.clamp((h - 0.35) / 1.55, 0, 1));
        sandCol.offsetHSL(0, 0, (n - 0.5) * 0.07);
      }
      var tMix = LH.smoothstep(sandGate - 0.7, sandGate + 0.7, h);
      c.copy(sandCol).lerp(grassCol, tMix);
      // 大尺度色斑 + 高频细碎变化
      var patch = LH.fbm2(x * 0.012, z * 0.012, 3, 71);
      var speck = LH.noise2(x * 0.35, z * 0.35, 83);
      c.offsetHSL(0, 0, (patch - 0.5) * 0.12 + (speck - 0.5) * 0.035);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();

    var detail = LH.texRepeat(LH.tex.detail, 150, 150);
    var mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      map: detail,
      roughness: 0.97,
      metalness: 0.0
    });
    var terrain = new THREE.Mesh(geo, mat);
    terrain.name = 'terrain';
    terrain.receiveShadow = true;
    terrain.castShadow = false;
    scene.add(terrain);

    // 浅滩蒙版交给海面使用
    var coast = buildCoastTexture();
    LH.coast = coast;
    LH.buildWater(scene, coast.texture, coast.rect);

    // 远景岛屿剪影
    buildIslands(scene);

    return terrain;
  };

  // ---------------------------------------------------------------- 远景岛屿
  function buildIslands(scene) {
    var mat = LH.std({ color: LH.col('#8291a0'), roughness: 1, metalness: 0 });
    var specs = [
      { x: 1650, z: -330, w: 1000, h: 52, seed: 1, ry: 0.28 },
      { x: 2450, z: 430, w: 1400, h: 74, seed: 3, ry: -0.2 }
    ];
    specs.forEach(function (s) {
      var seg = 48;
      var g = new THREE.PlaneGeometry(s.w, s.w * 0.42, seg, 16);
      g.rotateX(-Math.PI / 2);
      var p = g.attributes.position;
      var half = s.w / 2;
      var halfD = s.w * 0.21;
      for (var i = 0; i < p.count; i++) {
        var x = p.getX(i), z = p.getZ(i);
        var tx = Math.max(0, 1 - (x / half) * (x / half));
        var tz = Math.max(0, 1 - (z / halfD) * (z / halfD));
        var n = LH.fbm2((x + 800) * 0.0022, s.seed * 3.1, 3, s.seed * 97);
        var n2 = LH.fbm2((x + 800) * 0.007, 1.7, 2, s.seed * 53);
        var hh = (n * 0.75 + n2 * 0.25) * Math.pow(tx, 0.7) * Math.pow(tz, 0.8) * s.h;
        p.setY(i, hh - 2.5);
      }
      g.computeVertexNormals();
      var m = new THREE.Mesh(g, mat);
      m.position.set(s.x, 0, s.z);
      m.rotation.y = s.ry;
      m.name = 'distantIsland';
      scene.add(m);
    });
  }
})();
