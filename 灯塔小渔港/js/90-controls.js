/**
 * 90-controls.js —— 主观视角控制
 * 鼠标拖拽 / 单指滑动 360° 环视（水平不限，俯仰限幅）；
 * 滚轮调整视野远近；WASD / 方向键在栈桥与岸边小幅走动（限幅）；
 * 静止数秒后视角缓慢自动环视。页面本身没有任何 UI。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});
  var W = LH.W;

  function param(name, def) {
    var m = new RegExp('[?&]' + name + '=([^&]+)').exec(location.search);
    return m ? parseFloat(decodeURIComponent(m[1])) : def;
  }

  LH.buildControls = function (camera, dom) {
    var startX = param('x', -2.2);
    var startZ = param('z', 0.0);
    var state = {
      yaw: param('yaw', 0.0) * Math.PI / 180,
      pitch: param('pitch', -1.5) * Math.PI / 180,
      fov: param('fov', 58),
      fovTarget: param('fov', 58),
      eyeY: W.DECK_Y + W.EYE,
      pos: new THREE.Vector3(startX, W.DECK_Y + W.EYE, startZ),
      vel: new THREE.Vector3(),
      idle: 99,
      dragging: false,
      lastX: 0,
      lastY: 0,
      pointerId: null,
      keys: Object.create(null),
      bob: 0
    };
    state.eyeY = state.pos.y;

    var AUTO_PAN = !/[?&]auto=0/.test(location.search);
    var PITCH_MIN = -1.18;
    var PITCH_MAX = 0.95;
    var WALK_SPEED = 2.5;

    // ---------- 可行走区域：栈桥 + 岸边
    function inPier(x, z) {
      return x > -17.7 && x < W.PIER_X1 - 0.35 && Math.abs(z) < W.PIER_HALF_W - 0.28;
    }
    function inShore(x, z) {
      return x > -25.5 && x < -13.2 && Math.abs(z) < 10.5;
    }
    function groundEyeY(x, z) {
      if (inPier(x, z)) return W.DECK_Y + W.EYE;
      var h = LH.terrainHeight(x, z);
      return Math.max(h, 0.2) + W.EYE;
    }
    function canStand(x, z) { return inPier(x, z) || inShore(x, z); }

    // ---------- 指针拖拽
    function onDown(e) {
      if (state.pointerId !== null) return;
      state.pointerId = e.pointerId;
      state.dragging = true;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      state.idle = 0;
      if (dom.setPointerCapture) {
        try { dom.setPointerCapture(e.pointerId); } catch (err) { /* 忽略 */ }
      }
    }
    function onMove(e) {
      if (!state.dragging || e.pointerId !== state.pointerId) return;
      var dx = e.clientX - state.lastX;
      var dy = e.clientY - state.lastY;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      state.yaw -= dx * 0.0026;
      state.pitch = LH.clamp(state.pitch + dy * 0.0024, PITCH_MIN, PITCH_MAX);
      state.idle = 0;
    }
    function onUp(e) {
      if (e.pointerId !== state.pointerId) return;
      state.dragging = false;
      state.pointerId = null;
      if (dom.releasePointerCapture) {
        try { dom.releasePointerCapture(e.pointerId); } catch (err) { /* 忽略 */ }
      }
    }
    function onWheel(e) {
      e.preventDefault();
      state.fovTarget = LH.clamp(state.fovTarget + e.deltaY * 0.035, 30, 78);
      state.idle = 0;
    }

    dom.addEventListener('pointerdown', onDown);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    dom.addEventListener('wheel', onWheel, { passive: false });
    dom.addEventListener('contextmenu', function (e) { e.preventDefault(); });

    // ---------- 键盘
    window.addEventListener('keydown', function (e) {
      var k = e.key.toLowerCase();
      state.keys[k] = true;
      if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].indexOf(k) >= 0) {
        state.idle = 0;
        if (e.preventDefault) e.preventDefault();
      }
    });
    window.addEventListener('keyup', function (e) {
      state.keys[e.key.toLowerCase()] = false;
    });
    window.addEventListener('blur', function () { state.keys = Object.create(null); });

    // ---------- 每帧
    var fwd = new THREE.Vector3();
    var right = new THREE.Vector3();

    function update(dt) {
      var t = performance.now() / 1000;
      var k = state.keys;
      var ix = (k['d'] || k['arrowright'] ? 1 : 0) - (k['a'] || k['arrowleft'] ? 1 : 0);
      var iz = (k['w'] || k['arrowup'] ? 1 : 0) - (k['s'] || k['arrowdown'] ? 1 : 0);
      var moving = ix !== 0 || iz !== 0;

      if (moving) {
        fwd.set(Math.cos(state.yaw), 0, Math.sin(state.yaw));
        right.set(-Math.sin(state.yaw), 0, Math.cos(state.yaw));
        var dirX = fwd.x * iz + right.x * ix;
        var dirZ = fwd.z * iz + right.z * ix;
        var len = Math.hypot(dirX, dirZ) || 1;
        dirX /= len; dirZ /= len;
        var speed = WALK_SPEED * (k['shift'] ? 1.6 : 1.0);
        state.vel.x = LH.damp(state.vel.x, dirX * speed, 9, dt);
        state.vel.z = LH.damp(state.vel.z, dirZ * speed, 9, dt);
        state.bob += dt * 9;
        state.idle = 0;
      } else {
        state.vel.x = LH.damp(state.vel.x, 0, 8, dt);
        state.vel.z = LH.damp(state.vel.z, 0, 8, dt);
      }

      // 分轴尝试移动，边界处沿墙滑动
      var nx = state.pos.x + state.vel.x * dt;
      if (canStand(nx, state.pos.z)) state.pos.x = nx;
      var nz = state.pos.z + state.vel.z * dt;
      if (canStand(state.pos.x, nz)) state.pos.z = nz;

      // 视点高度平滑跟随地面/桥面
      var targetY = groundEyeY(state.pos.x, state.pos.z);
      state.eyeY = LH.damp(state.eyeY, targetY, 8, dt);
      var bobY = moving ? Math.sin(state.bob) * 0.022 : 0;
      camera.position.set(state.pos.x, state.eyeY + bobY, state.pos.z);

      // 静止自动环视
      state.idle += dt;
      if (AUTO_PAN && state.idle > 5.0 && !state.dragging) {
        var autoT = Math.min(1, (state.idle - 5.0) / 2.0);
        state.yaw += dt * 0.040 * autoT;
        state.pitch = LH.damp(state.pitch, -0.02 + Math.sin(t * 0.11) * 0.03, 0.8, dt);
      }

      // 视野远近
      state.fov = LH.damp(state.fov, state.fovTarget, 7, dt);
      if (Math.abs(camera.fov - state.fov) > 0.001) {
        camera.fov = state.fov;
        camera.updateProjectionMatrix();
      }

      // 朝向
      var cp = Math.cos(state.pitch);
      camera.lookAt(
        camera.position.x + Math.cos(state.yaw) * cp,
        camera.position.y + Math.sin(state.pitch),
        camera.position.z + Math.sin(state.yaw) * cp
      );
    }

    camera.rotation.order = 'YXZ';
    camera.position.copy(state.pos);
    update(0.016);

    return {
      update: update,
      state: state,
      setYawPitch: function (yawDeg, pitchDeg) {
        state.yaw = yawDeg * Math.PI / 180;
        state.pitch = LH.clamp(pitchDeg * Math.PI / 180, PITCH_MIN, PITCH_MAX);
      }
    };
  };
})();
