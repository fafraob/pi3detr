import { LineGeometry } from './LineGeometry.js';
import { LineMaterial } from './LineMaterial.js';

class Line2 extends THREE.Mesh {
  constructor(geometry = new LineGeometry(), material = new LineMaterial()) {
    super(geometry, material);
    this.isLine2 = true;
    this.type = 'Line2';
  }

  computeLineDistances() {
    const geometry = this.geometry;
    const instanceStart = geometry.attributes.instanceStart;
    const instanceEnd = geometry.attributes.instanceEnd;
    const lineDistances = new Float32Array(2 * instanceStart.count);

    for (let i = 0, j = 0, l = instanceStart.count; i < l; i++, j += 2) {
      const start = new THREE.Vector3();
      const end = new THREE.Vector3();
      
      start.fromBufferAttribute(instanceStart, i);
      end.fromBufferAttribute(instanceEnd, i);
      
      lineDistances[j] = (j === 0) ? 0 : lineDistances[j - 1];
      lineDistances[j + 1] = lineDistances[j] + start.distanceTo(end);
    }

    const instanceDistanceBuffer = new THREE.InstancedInterleavedBuffer(lineDistances, 2, 1);
    geometry.setAttribute('instanceDistanceStart', new THREE.InterleavedBufferAttribute(instanceDistanceBuffer, 1, 0));
    geometry.setAttribute('instanceDistanceEnd', new THREE.InterleavedBufferAttribute(instanceDistanceBuffer, 1, 1));

    return this;
  }
}

export { Line2 };
