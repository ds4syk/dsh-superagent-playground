/**
 * 95-postfx.js —— 后期处理链
 * 环境光遮蔽（SSAO）→ 泛光 → 远景景深 → 胶片颗粒 → 色彩分级/暗角 → 伽马输出。
 */
(function () {
  'use strict';
  var LH = (window.LH = window.LH || {});

  function param(name, def) {
    var m = new RegExp('[?&]' + name + '=([^&]+)').exec(location.search);
    return m ? m[1] : def;
  }

  var VignetteGradeShader = {
    uniforms: {
      tDiffuse: { value: null },
      uVignette: { value: 0.62 },
      uOffset: { value: 1.15 },
      uSaturation: { value: 1.04 }
    },
    vertexShader: [
      'varying vec2 vUv;',
      'void main() {',
      '  vUv = uv;',
      '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
      '}'
    ].join('\n'),
    fragmentShader: [
      'uniform sampler2D tDiffuse;',
      'uniform float uVignette;',
      'uniform float uOffset;',
      'uniform float uSaturation;',
      'varying vec2 vUv;',
      'void main() {',
      '  vec3 color = texture2D(tDiffuse, vUv).rgb;',
      '  float l = dot(color, vec3(0.299, 0.587, 0.114));',
      '  // 暗部偏冷、亮部偏暖的清晨分级',
      '  vec3 shadowTint = vec3(0.93, 1.015, 1.06);',
      '  vec3 highTint = vec3(1.05, 1.0, 0.945);',
      '  color *= mix(shadowTint, highTint, smoothstep(0.12, 0.72, l));',
      '  // 轻微提饱和',
      '  color = mix(vec3(l), color, uSaturation);',
      '  // 暗角',
      '  vec2 d = (vUv - 0.5) * uOffset;',
      '  float vig = 1.0 - uVignette * dot(d, d);',
      '  color *= clamp(vig, 0.0, 1.0);',
      '  gl_FragColor = vec4(color, 1.0);',
      '}'
    ].join('\n')
  };

  LH.buildPostfx = function (renderer, scene, camera) {
    var size = new THREE.Vector2();
    renderer.getSize(size);
    var dpr = renderer.getPixelRatio();
    var w = Math.floor(size.x * dpr);
    var h = Math.floor(size.y * dpr);

    var target = new THREE.WebGLRenderTarget(w, h, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      stencilBuffer: false
    });
    target.texture.encoding = THREE.LinearEncoding;

    var composer = new THREE.EffectComposer(renderer, target);
    composer.setSize(w, h);

    var registry = {};
    var useAO = param('ao', '1') !== '0';
    var useDof = param('dof', '1') !== '0';
    var useBloom = param('bloom', '1') !== '0';

    if (useAO && THREE.SSAOPass) {
      var ssao = new THREE.SSAOPass(scene, camera, w, h);
      ssao.kernelRadius = 0.55;
      ssao.minDistance = 0.0015;
      ssao.maxDistance = 0.06;
      ssao.output = THREE.SSAOPass.OUTPUT.Default;
      composer.addPass(ssao);
      registry.ssao = ssao;
    } else {
      composer.addPass(new THREE.RenderPass(scene, camera));
    }

    if (useBloom) {
      var bloom = new THREE.UnrealBloomPass(new THREE.Vector2(w, h), 0.30, 0.7, 0.90);
      composer.addPass(bloom);
      registry.bloom = bloom;
    }

    if (useDof && THREE.BokehPass) {
      var bokeh = new THREE.BokehPass(scene, camera, {
        focus: 42.0,
        aperture: 0.00011,
        maxblur: 0.0065
      });
      composer.addPass(bokeh);
      registry.dof = bokeh;
    }

    var film = new THREE.FilmPass(0.12, 0.0, 0, false);
    composer.addPass(film);
    registry.film = film;

    var grade = new THREE.ShaderPass(VignetteGradeShader);
    composer.addPass(grade);
    registry.grade = grade;

    var gamma = new THREE.ShaderPass(THREE.GammaCorrectionShader);
    composer.addPass(gamma);
    registry.gamma = gamma;
    window.__LH_POST__ = registry;

    function resize() {
      var s = new THREE.Vector2();
      renderer.getSize(s);
      var pr = renderer.getPixelRatio();
      var nw = Math.floor(s.x * pr);
      var nh = Math.floor(s.y * pr);
      composer.setSize(nw, nh);
      camera.aspect = s.x / s.y;
      camera.updateProjectionMatrix();
    }
    window.addEventListener('resize', resize);

    return {
      composer: composer,
      resize: resize,
      render: function () { composer.render(); }
    };
  };
})();
