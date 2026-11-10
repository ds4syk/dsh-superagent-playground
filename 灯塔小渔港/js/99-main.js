/**
 * 99-main.js —— 启动
 * 渲染器、相机、场景装配、主循环。
 * 支持网址参数（仅用于截图/调试，不影响默认浏览）：?yaw=&pitch=&fov=&x=&z=&post=0&ao=0&dof=0&bloom=0
 */
(function () {
  'use strict';
  var LH = window.LH;
  var canvas = document.getElementById('scene');

  var renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance',
      stencil: false
    });
  } catch (err) {
    // 极少数不支持 WebGL 的环境：保持静默，不留任何页面文字
    console.error('WebGL unavailable', err);
    return;
  }

  var MAX_DPR = 1.5;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DPR));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  var QS = new URLSearchParams(location.search);
  renderer.shadowMap.enabled = QS.get('shadow') !== '0';
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;
  renderer.outputEncoding = THREE.LinearEncoding;

  var scene = new THREE.Scene();
  LH.scene = scene;

  var camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.2, 14000);

  // ---------------- 装配世界
  LH.buildTextures();
  LH.buildSky(scene);
  LH.buildTerrain(scene);
  LH.buildRocks(scene);
  LH.buildLighthouse(scene);
  LH.buildHarbor(scene);
  LH.buildVillage(scene);
  LH.buildBoats(scene);
  LH.buildMist(scene);

  // 场景整体挂到根节点（便于统一管理）
  scene.add(LH.root);

  // ---------------- 控制与后期
  var controls = LH.buildControls(camera, canvas);
  var postfx = LH.buildPostfx(renderer, scene, camera);

  var usePost = QS.get('post') !== '0';
  var clock = new THREE.Clock();
  var frames = 0;
  var elapsed = 0;
  var firstFrame = true;

  function frame() {
    requestAnimationFrame(frame);
    var dt = Math.min(clock.getDelta(), 0.05);
    elapsed += dt;
    for (var i = 0; i < LH.updaters.length; i++) {
      LH.updaters[i](dt, elapsed, camera);
    }
    controls.update(dt);

    if (usePost) postfx.render();
    else renderer.render(scene, camera);

    frames++;
    window.__LH_FRAMES__ = frames;
    if (firstFrame) {
      firstFrame = false;
      window.__LH_READY__ = true;
      document.documentElement.setAttribute('data-ready', '1');
    }
  }

  window.addEventListener('resize', function () {
    var w = window.innerWidth, h = window.innerHeight;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DPR));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    postfx.resize();
  });

  // 截图/调试状态
  window.__SHOT_STATE__ = function () {
    var yaw = controls.state.yaw * 180 / Math.PI;
    var pitch = controls.state.pitch * 180 / Math.PI;
    return {
      ready: !!window.__LH_READY__,
      position: [camera.position.x, camera.position.y, camera.position.z],
      yawDeg: yaw,
      pitchDeg: pitch,
      fov: camera.fov,
      triangles: renderer.info.render.triangles,
      calls: renderer.info.render.calls,
      textures: renderer.info.memory.textures
    };
  };
  window.__LH_SET_VIEW__ = function (yawDeg, pitchDeg) {
    controls.setYawPitch(yawDeg, pitchDeg);
  };

  LH.app = { renderer: renderer, scene: scene, camera: camera, controls: controls, postfx: postfx };
  window.__LH_APP__ = LH.app;

  frame();
})();
