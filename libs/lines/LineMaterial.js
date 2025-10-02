class LineMaterial extends THREE.ShaderMaterial {
  constructor(parameters) {
    super({
      type: 'LineMaterial',
      uniforms: {
        diffuse: { value: new THREE.Color(0xffffff) },
        opacity: { value: 1.0 },
        linewidth: { value: 1 },
        resolution: { value: new THREE.Vector2(512, 512) },
        dashOffset: { value: 0 },
        dashSize: { value: 1 },
        gapSize: { value: 1 }
      },
      vertexShader: `
        #include <common>
        attribute vec3 instanceStart;
        attribute vec3 instanceEnd;
        attribute vec3 instanceColorStart;
        attribute vec3 instanceColorEnd;
        
        uniform float linewidth;
        uniform vec2 resolution;
        
        varying vec2 vUv;
        varying vec3 vColor;
        
        void main() {
          vUv = uv;
          vColor = instanceColorStart;
          
          vec4 start = modelViewMatrix * vec4(instanceStart, 1.0);
          vec4 end = modelViewMatrix * vec4(instanceEnd, 1.0);
          
          vec2 aspectVec = vec2(resolution.x / resolution.y, 1.0);
          vec2 dir = (end.xy - start.xy) * aspectVec;
          dir = normalize(dir);
          
          vec2 offset = vec2(-dir.y, dir.x) * linewidth / resolution.y;
          
          if (position.x < 0.0) {
            gl_Position = projectionMatrix * start;
            gl_Position.xy += offset * gl_Position.w;
          } else {
            gl_Position = projectionMatrix * end;
            gl_Position.xy += offset * gl_Position.w;
          }
        }
      `,
      fragmentShader: `
        uniform vec3 diffuse;
        uniform float opacity;
        
        varying vec2 vUv;
        varying vec3 vColor;
        
        void main() {
          gl_FragColor = vec4(diffuse * vColor, opacity);
        }
      `
    });

    this.isLineMaterial = true;

    Object.defineProperties(this, {
      color: {
        enumerable: true,
        get: function () {
          return this.uniforms.diffuse.value;
        },
        set: function (value) {
          this.uniforms.diffuse.value = value;
        }
      },
      linewidth: {
        enumerable: true,
        get: function () {
          return this.uniforms.linewidth.value;
        },
        set: function (value) {
          this.uniforms.linewidth.value = value;
        }
      },
      resolution: {
        enumerable: true,
        get: function () {
          return this.uniforms.resolution.value;
        },
        set: function (value) {
          this.uniforms.resolution.value.copy(value);
        }
      }
    });

    this.setValues(parameters);
  }
}

export { LineMaterial };
