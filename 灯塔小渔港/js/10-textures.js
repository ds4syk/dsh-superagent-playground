/**
 * 10-textures.js —— 全程序化纹理
 * 所有贴图都在运行时用 Canvas 2D 画出来（无外部图片），
 * 用固定种子的随机源，保证每次生成的纹理一致。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});
  LH.tex = {};

  function canvas(size, h) {
    var c = document.createElement('canvas');
    c.width = size;
    c.height = h || size;
    return c;
  }

  function finish(cv, opts) {
    var t = new THREE.CanvasTexture(cv);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 4;
    var o = opts || {};
    t.encoding = o.srgb === false ? THREE.LinearEncoding : THREE.sRGBEncoding;
    t.repeat.set(o.rx || 1, o.ry || 1);
    if (o.rotation) t.rotation = o.rotation;
    t.needsUpdate = true;
    return t;
  }

  function cloneRepeat(tex, rx, ry) {
    var t = tex.clone();
    t.needsUpdate = true;
    t.repeat.set(rx, ry);
    return t;
  }
  LH.texRepeat = cloneRepeat;

  function noiseFill(ctx, w, h, rng, base, amp, size) {
    var img = ctx.getImageData(0, 0, w, h);
    var d = img.data;
    for (var i = 0; i < d.length; i += 4) {
      var n = (rng.next() - 0.5) * amp;
      d[i] = LH.clamp(d[i] + n, 0, 255);
      d[i + 1] = LH.clamp(d[i + 1] + n, 0, 255);
      d[i + 2] = LH.clamp(d[i + 2] + n, 0, 255);
    }
    ctx.putImageData(img, 0, 0);
  }

  // ---------------------------------------------------------------- 木纹
  function wood(seed, opts) {
    var o = opts || {};
    var S = o.size || 512;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    var base = o.base || '#8a5a3c';
    var dark = o.dark || '#5f3a25';
    var light = o.light || '#b07c52';
    var vertical = !!o.vertical;

    ctx.fillStyle = base;
    ctx.fillRect(0, 0, S, S);

    // 大块明暗色带
    for (var i = 0; i < 26; i++) {
      var x = rng.range(0, S);
      var w = rng.range(S * 0.02, S * 0.09);
      var g = ctx.createLinearGradient(x, 0, x + w, 0);
      var a = rng.range(0.04, 0.12);
      g.addColorStop(0, 'rgba(0,0,0,0)');
      g.addColorStop(0.5, 'rgba(60,32,16,' + a.toFixed(3) + ')');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      if (vertical) ctx.fillRect(0, x, S, w); else ctx.fillRect(x, 0, w, S);
    }

    // 木纹线：沿 U 方向（或指定竖直）
    for (var k = 0; k < 190; k++) {
      var y0 = rng.range(0, S);
      var amp = rng.range(1.5, 7);
      var freq = rng.range(0.004, 0.02);
      var phase = rng.range(0, 6.283);
      ctx.beginPath();
      for (var t = 0; t <= S; t += 4) {
        var off = Math.sin(t * freq + phase) * amp + Math.sin(t * freq * 3.1 + phase * 2) * amp * 0.25;
        var xx = vertical ? t : t;
        var yy = (vertical ? y0 + off : y0 + off);
        if (t === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      var col = rng.chance(0.55) ? dark : light;
      ctx.strokeStyle = col;
      ctx.globalAlpha = rng.range(0.05, 0.2);
      ctx.lineWidth = rng.range(0.6, 2.2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // 木节
    for (var j = 0; j < 7; j++) {
      var kx = rng.range(0, S), ky = rng.range(0, S);
      var kr = rng.range(2, 6);
      ctx.strokeStyle = 'rgba(52,30,16,0.35)';
      for (var r = 0; r < 4; r++) {
        ctx.beginPath();
        ctx.ellipse(kx, ky, kr + r * kr * 0.9, (kr + r * kr * 0.9) * 0.45, rng.range(0, 3.14), 0, Math.PI * 2);
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
      ctx.fillStyle = 'rgba(45,26,14,0.45)';
      ctx.beginPath();
      ctx.ellipse(kx, ky, kr * 0.7, kr * 0.4, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // 风化：横向浅色磨损
    for (var m = 0; m < 40; m++) {
      ctx.strokeStyle = 'rgba(230,214,190,' + rng.range(0.02, 0.06).toFixed(3) + ')';
      ctx.lineWidth = rng.range(1, 5);
      var y = rng.range(0, S);
      ctx.beginPath();
      ctx.moveTo(rng.range(0, S * 0.4), y);
      ctx.lineTo(rng.range(S * 0.5, S), y + rng.range(-8, 8));
      ctx.stroke();
    }

    noiseFill(ctx, S, S, rng, null, 10);
    return finish(cv, { rx: o.rx || 1, ry: o.ry || 1 });
  }

  // ---------------------------------------------------------------- 岩石
  function rock(seed, tint) {
    var S = 512;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.fillStyle = tint || '#7d766c';
    ctx.fillRect(0, 0, S, S);

    // 大块斑驳
    for (var i = 0; i < 140; i++) {
      var x = rng.range(0, S), y = rng.range(0, S), r = rng.range(8, 80);
      var g = ctx.createRadialGradient(x, y, 0, x, y, r);
      var c = rng.chance(0.5) ? '60,58,54' : '168,164,154';
      g.addColorStop(0, 'rgba(' + c + ',' + rng.range(0.05, 0.16).toFixed(3) + ')');
      g.addColorStop(1, 'rgba(' + c + ',0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    // 裂纹
    for (var k = 0; k < 26; k++) {
      ctx.strokeStyle = 'rgba(38,35,32,' + rng.range(0.15, 0.4).toFixed(3) + ')';
      ctx.lineWidth = rng.range(0.7, 2.4);
      var px = rng.range(0, S), py = rng.range(0, S);
      ctx.beginPath();
      ctx.moveTo(px, py);
      for (var s = 0; s < 6; s++) {
        px += rng.range(-46, 46);
        py += rng.range(-46, 46);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    // 苔藓/海藻
    for (var m = 0; m < 26; m++) {
      var gx = rng.range(0, S), gy = rng.range(S * 0.45, S);
      var gr = rng.range(10, 44);
      var gg = ctx.createRadialGradient(gx, gy, 0, gx, gy, gr);
      gg.addColorStop(0, 'rgba(96,104,66,' + rng.range(0.12, 0.3).toFixed(3) + ')');
      gg.addColorStop(1, 'rgba(96,104,66,0)');
      ctx.fillStyle = gg;
      ctx.beginPath();
      ctx.arc(gx, gy, gr, 0, Math.PI * 2);
      ctx.fill();
    }
    noiseFill(ctx, S, S, rng, null, 22);
    return finish(cv, { rx: 2, ry: 2 });
  }

  // ---------------------------------------------------------------- 沙滩
  function sand(seed) {
    var S = 512;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.fillStyle = '#d9c49c';
    ctx.fillRect(0, 0, S, S);
    // 细沙颗粒
    for (var i = 0; i < 9000; i++) {
      var x = rng.range(0, S), y = rng.range(0, S);
      var v = rng.chance(0.5) ? 255 : 90;
      ctx.fillStyle = 'rgba(' + v + ',' + (v * 0.93 | 0) + ',' + (v * 0.78 | 0) + ',' + rng.range(0.05, 0.22).toFixed(3) + ')';
      ctx.fillRect(x, y, rng.range(0.6, 1.8), rng.range(0.6, 1.8));
    }
    // 潮汐波纹
    for (var k = 0; k < 16; k++) {
      ctx.strokeStyle = 'rgba(120,96,64,' + rng.range(0.04, 0.1).toFixed(3) + ')';
      ctx.lineWidth = rng.range(1, 3);
      var y0 = rng.range(0, S);
      ctx.beginPath();
      for (var t = 0; t <= S; t += 8) {
        var yy = y0 + Math.sin(t * 0.03 + k) * 8 + Math.sin(t * 0.011 + k * 2) * 14;
        if (t === 0) ctx.moveTo(t, yy); else ctx.lineTo(t, yy);
      }
      ctx.stroke();
    }
    noiseFill(ctx, S, S, rng, null, 14);
    return finish(cv, { rx: 8, ry: 8 });
  }

  // ---------------------------------------------------------------- 抹灰墙
  function plaster(seed, base) {
    var S = 256;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.fillStyle = base || '#e9e2d4';
    ctx.fillRect(0, 0, S, S);
    for (var i = 0; i < 60; i++) {
      var x = rng.range(0, S), y = rng.range(0, S), r = rng.range(6, 34);
      var g = ctx.createRadialGradient(x, y, 0, x, y, r);
      var dark = rng.chance(0.6);
      g.addColorStop(0, dark ? 'rgba(120,110,95,0.10)' : 'rgba(255,255,255,0.12)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    // 雨水痕迹
    for (var k = 0; k < 14; k++) {
      var sx = rng.range(0, S);
      var g2 = ctx.createLinearGradient(sx, 0, sx + rng.range(-8, 8), S);
      g2.addColorStop(0, 'rgba(90,84,72,0.10)');
      g2.addColorStop(1, 'rgba(90,84,72,0)');
      ctx.fillStyle = g2;
      ctx.fillRect(sx, 0, rng.range(2, 8), S);
    }
    noiseFill(ctx, S, S, rng, null, 9);
    return finish(cv, { rx: 2, ry: 2 });
  }

  // ---------------------------------------------------------------- 屋顶瓦片
  function roofTiles(seed, base) {
    var S = 512;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.fillStyle = base || '#9a4a37';
    ctx.fillRect(0, 0, S, S);
    var rows = 10, cols = 14;
    var rh = S / rows, cw = S / cols;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var x = c * cw + (r % 2 ? cw * 0.5 : 0);
        var y = r * rh;
        var shade = rng.range(-18, 18);
        ctx.fillStyle = 'rgba(' + (150 + shade) + ',' + (74 + shade * 0.4 | 0) + ',' + (56 + shade * 0.3 | 0) + ',1)';
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + cw * 0.92, y);
        ctx.lineTo(x + cw * 0.92, y + rh * 0.72);
        ctx.quadraticCurveTo(x + cw * 0.46, y + rh * 1.05, x, y + rh * 0.72);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = 'rgba(52,22,16,0.35)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
    // 苔藓与风化
    for (var i = 0; i < 40; i++) {
      var mx = rng.range(0, S), my = rng.range(0, S), mr = rng.range(5, 26);
      var g = ctx.createRadialGradient(mx, my, 0, mx, my, mr);
      g.addColorStop(0, 'rgba(88,92,58,' + rng.range(0.05, 0.16).toFixed(3) + ')');
      g.addColorStop(1, 'rgba(88,92,58,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(mx, my, mr, 0, Math.PI * 2);
      ctx.fill();
    }
    noiseFill(ctx, S, S, rng, null, 8);
    return finish(cv, { rx: 2, ry: 2 });
  }

  // ---------------------------------------------------------------- 生锈波纹铁皮
  function rustMetal(seed) {
    var S = 512;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.fillStyle = '#6f7370';
    ctx.fillRect(0, 0, S, S);
    // 竖向波纹
    for (var x = 0; x < S; x += 16) {
      var g = ctx.createLinearGradient(x, 0, x + 16, 0);
      g.addColorStop(0, 'rgba(255,255,255,0.10)');
      g.addColorStop(0.5, 'rgba(0,0,0,0.12)');
      g.addColorStop(1, 'rgba(255,255,255,0.06)');
      ctx.fillStyle = g;
      ctx.fillRect(x, 0, 16, S);
    }
    // 锈斑
    for (var i = 0; i < 90; i++) {
      var px = rng.range(0, S), py = rng.range(0, S), r = rng.range(4, 46);
      var gg = ctx.createRadialGradient(px, py, 0, px, py, r);
      gg.addColorStop(0, 'rgba(136,66,30,' + rng.range(0.12, 0.4).toFixed(3) + ')');
      gg.addColorStop(1, 'rgba(136,66,30,0)');
      ctx.fillStyle = gg;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
    // 锈水往下流
    for (var k = 0; k < 24; k++) {
      var sx = rng.range(0, S);
      var g2 = ctx.createLinearGradient(0, rng.range(0, S * 0.5), 0, S);
      g2.addColorStop(0, 'rgba(120,58,26,0.25)');
      g2.addColorStop(1, 'rgba(120,58,26,0)');
      ctx.fillStyle = g2;
      ctx.fillRect(sx, 0, rng.range(1, 4), S);
    }
    noiseFill(ctx, S, S, rng, null, 16);
    return finish(cv, { rx: 3, ry: 2 });
  }

  // ---------------------------------------------------------------- 石墙
  function stoneWall(seed) {
    var S = 512;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.fillStyle = '#6b665c';
    ctx.fillRect(0, 0, S, S);
    var y = 0, row = 0;
    while (y < S) {
      var h = rng.range(28, 52);
      var x = -rng.range(0, 60);
      while (x < S) {
        var w = rng.range(38, 96);
        var shade = rng.range(-20, 20);
        ctx.fillStyle = 'rgb(' + (118 + shade) + ',' + (112 + shade) + ',' + (100 + shade * 0.8) + ')';
        roundRect(ctx, x + 2, y + 2, w - 4, h - 4, rng.range(3, 9));
        ctx.fill();
        // 面层斑驳
        ctx.fillStyle = 'rgba(0,0,0,' + rng.range(0.02, 0.1).toFixed(3) + ')';
        ctx.fillRect(x + 6, y + 6, w - 12, h - 12);
        x += w;
      }
      y += h;
      row++;
    }
    // 勾缝阴影
    ctx.globalCompositeOperation = 'multiply';
    var g = ctx.createLinearGradient(0, 0, 0, S);
    g.addColorStop(0, 'rgba(255,255,255,0.9)');
    g.addColorStop(1, 'rgba(180,180,180,0.9)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, S, S);
    ctx.globalCompositeOperation = 'source-over';
    noiseFill(ctx, S, S, rng, null, 18);
    return finish(cv, { rx: 4, ry: 2 });
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  LH.roundRect = roundRect;

  // ---------------------------------------------------------------- 灯塔条纹
  function lighthouseStripes(seed) {
    var W = 256, H = 512;
    var cv = canvas(W, H);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    var bands = 8;
    for (var i = 0; i < bands; i++) {
      var t = i / bands;
      ctx.fillStyle = (i % 2 === 0) ? '#c9482f' : '#f1ece0';
      ctx.fillRect(0, t * H, W, H / bands + 1);
    }
    // 轻微脏污与褪色
    for (var k = 0; k < 120; k++) {
      var x = rng.range(0, W), y = rng.range(0, H);
      var r = rng.range(3, 30);
      var g = ctx.createRadialGradient(x, y, 0, x, y, r);
      var dust = rng.chance(0.6);
      g.addColorStop(0, (dust ? 'rgba(120,100,80,' : 'rgba(255,255,255,') + rng.range(0.03, 0.12).toFixed(3) + ')');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    // 底部潮气
    var g2 = ctx.createLinearGradient(0, H * 0.62, 0, H);
    g2.addColorStop(0, 'rgba(70,80,66,0)');
    g2.addColorStop(1, 'rgba(70,80,66,0.30)');
    ctx.fillStyle = g2;
    ctx.fillRect(0, H * 0.62, W, H * 0.38);
    noiseFill(ctx, W, H, rng, null, 8);
    return finish(cv, { rx: 1, ry: 1 });
  }

  // ---------------------------------------------------------------- 渔网（透明）
  function net(seed) {
    var S = 256;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.clearRect(0, 0, S, S);
    ctx.strokeStyle = 'rgba(228,224,206,0.95)';
    ctx.lineWidth = 2.2;
    var step = 16;
    for (var i = -S; i < S * 2; i += step) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i + S, S);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(i + S, 0);
      ctx.lineTo(i, S);
      ctx.stroke();
    }
    // 局部断线
    ctx.globalCompositeOperation = 'destination-out';
    for (var k = 0; k < 40; k++) {
      ctx.fillStyle = 'rgba(0,0,0,' + rng.range(0.2, 0.8).toFixed(2) + ')';
      ctx.beginPath();
      ctx.arc(rng.range(0, S), rng.range(0, S), rng.range(1, 5), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';
    var t = finish(cv, { srgb: false, rx: 6, ry: 6 });
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    return t;
  }

  // ---------------------------------------------------------------- 光晕
  function glow(seed) {
    var S = 256;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var g = ctx.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.18, 'rgba(255,250,235,0.85)');
    g.addColorStop(0.45, 'rgba(255,235,200,0.28)');
    g.addColorStop(1, 'rgba(255,220,180,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, S, S);
    return finish(cv, { srgb: false });
  }

  // ---------------------------------------------------------------- 雾气 / 泡沫（柔和斑块）
  function softBlobs(seed, scale, alpha) {
    var S = 512;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.clearRect(0, 0, S, S);
    for (var i = 0; i < 70; i++) {
      var x = rng.range(0, S), y = rng.range(0, S), r = rng.range(20, 120) * scale;
      var g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, 'rgba(255,255,255,' + (alpha * rng.range(0.4, 1)).toFixed(3) + ')');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    var t = finish(cv, { srgb: false, rx: 1, ry: 1 });
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    return t;
  }

  // ---------------------------------------------------------------- 接触阴影（透明黑）
  function contactShadow(seed) {
    var S = 256;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var g = ctx.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
    g.addColorStop(0, 'rgba(0,0,0,0.62)');
    g.addColorStop(0.45, 'rgba(0,0,0,0.38)');
    g.addColorStop(0.78, 'rgba(0,0,0,0.12)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, S, S);
    var t = finish(cv, { srgb: false });
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    return t;
  }

  // ---------------------------------------------------------------- 草叶（透明）
  function grassBlade(seed) {
    var S = 128;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.clearRect(0, 0, S, S);
    for (var i = 0; i < 46; i++) {
      var x = rng.range(6, S - 6);
      var h = rng.range(S * 0.35, S * 0.92);
      var bend = rng.range(-26, 26);
      var w = rng.range(1.2, 3.4);
      var g = ctx.createLinearGradient(0, S, 0, S - h);
      var light = rng.range(0.25, 0.6);
      g.addColorStop(0, 'rgba(66,84,40,' + (0.75 + light * 0.2).toFixed(2) + ')');
      g.addColorStop(0.55, 'rgba(' + (104 + light * 50 | 0) + ',' + (128 + light * 42 | 0) + ',56,0.9)');
      g.addColorStop(1, 'rgba(' + (150 + light * 60 | 0) + ',' + (164 + light * 42 | 0) + ',80,0)');
      ctx.strokeStyle = g;
      ctx.lineWidth = w;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x, S);
      ctx.quadraticCurveTo(x + bend * 0.3, S - h * 0.55, x + bend, S - h);
      ctx.stroke();
    }
    var t = finish(cv, { srgb: false });
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    return t;
  }

  // ---------------------------------------------------------------- 自动贩卖机面板
  function vendingPanel(seed) {
    var W = 256, H = 512;
    var cv = canvas(W, H);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    ctx.fillStyle = '#20465f';
    ctx.fillRect(0, 0, W, H);
    // 灯箱白条
    ctx.fillStyle = '#f2f8ff';
    ctx.fillRect(10, 10, W - 20, 54);
    ctx.fillStyle = '#245a7a';
    ctx.font = 'bold 26px sans-serif';
    ctx.fillText('DRINK', 24, 47);
    // 饮料排
    var cols = 5, rows = 3;
    var colors = ['#e04b3a', '#f2b53a', '#3f9e5c', '#3a7fd4', '#eef0f2', '#d46aa0'];
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var x = 18 + c * 44, y = 90 + r * 105;
        ctx.fillStyle = 'rgba(255,255,255,0.12)';
        roundRect(ctx, x - 4, y - 8, 40, 92, 8);
        ctx.fill();
        var col = colors[(r * cols + c) % colors.length];
        ctx.fillStyle = col;
        ctx.fillRect(x + 6, y, 20, 56);
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 6, y, 20, 56);
        ctx.fillStyle = 'rgba(255,255,255,0.75)';
        ctx.fillRect(x + 8, y + 4, 4, 46);
        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.fillRect(x + 2, y + 62, 34, 8);
      }
    }
    // 出货口与按钮
    ctx.fillStyle = '#0d1f2e';
    ctx.fillRect(20, H - 84, W - 40, 56);
    for (var b = 0; b < 6; b++) {
      ctx.fillStyle = 'rgba(120,200,240,0.8)';
      ctx.beginPath();
      ctx.arc(44 + b * 36, H - 104, 6, 0, Math.PI * 2);
      ctx.fill();
    }
    noiseFill(ctx, W, H, rng, null, 6);
    return finish(cv, { rx: 1, ry: 1 });
  }

  // ---------------------------------------------------------------- 地形细节（灰度乘算贴图）
  function detailNoise(seed) {
    var S = 256;
    var cv = canvas(S);
    var ctx = cv.getContext('2d');
    var rng = LH.rng(seed);
    var img = ctx.createImageData(S, S);
    for (var y = 0; y < S; y++) {
      for (var x = 0; x < S; x++) {
        var i = (y * S + x) * 4;
        var coarse = LH.fbm2(x * 0.03, y * 0.03, 3, seed);
        var fine = LH.noise2(x * 0.55, y * 0.55, seed + 5) * 0.55 + LH.noise2(x * 1.7, y * 1.7, seed + 9) * 0.45;
        var n = coarse * 0.45 + fine * 0.55;
        var v = 150 + n * 115;
        img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    return finish(cv, { rx: 24, ry: 24 });
  }

  // ---------------------------------------------------------------- 木牌/屋面板（灰蓝旧木）
  function paintedWood(seed, base, grainColor) {
    return wood(seed, { base: base, dark: grainColor || '#4a4034', light: '#c9c4b4', size: 512 });
  }

  // ---------------------------------------------------------------- 构建全部
  LH.buildTextures = function () {
    var T = LH.tex;
    T.wood = wood(11, { base: '#8a5a3c', dark: '#5a3620', light: '#b7835a', rx: 1, ry: 1 });
    T.woodV = wood(12, { base: '#7a4e34', dark: '#523120', light: '#a87350', vertical: true });
    T.woodDark = wood(13, { base: '#5e422e', dark: '#3d2a1c', light: '#8a6444' });
    T.woodGrey = paintedWood(14, '#8d8a7c', '#5b574a');
    T.woodBoat = wood(15, { base: '#5b7d84', dark: '#3d5860', light: '#9ab3b5' });
    T.woodBoatRed = wood(16, { base: '#a5503f', dark: '#6e3226', light: '#cf8a74' });
    T.rock = rock(21);
    T.rockDark = rock(22, '#6a645c');
    T.sand = sand(31);
    T.plasterWhite = plaster(41, '#efe9dc');
    T.plasterCream = plaster(42, '#efe0bd');
    T.plasterMint = plaster(43, '#cfe0d2');
    T.plasterPink = plaster(44, '#e8cdc2');
    T.plasterBlue = plaster(45, '#c9dbe6');
    T.roofRed = roofTiles(51, '#9a4a37');
    T.roofBrown = roofTiles(52, '#6f4a35');
    T.roofGrey = roofTiles(53, '#5c6266');
    T.rust = rustMetal(61);
    T.stone = stoneWall(71);
    T.stripes = lighthouseStripes(81);
    T.net = net(91);
    T.glow = glow(101);
    T.mist = softBlobs(111, 1.0, 0.5);
    T.foam = softBlobs(112, 0.55, 0.8);
    T.vending = vendingPanel(121);
    T.detail = detailNoise(131);
    T.grass = grassBlade(141);
    T.contact = contactShadow(151);
  };
})();
