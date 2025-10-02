class LineGeometry extends THREE.InstancedBufferGeometry {
  constructor() {
    super();
    this.isLineGeometry = true;
    this.type = 'LineGeometry';

    const positions = [-1, 2, 0, 1, 2, 0, -1, 1, 0, 1, 1, 0, -1, 0, 0, 1, 0, 0, -1, -1, 0, 1, -1, 0];
    const uvs = [-1, 2, 1, 2, -1, 1, 1, 1, -1, -1, 1, -1, -1, -2, 1, -2];
    const index = [0, 2, 1, 2, 3, 1, 2, 4, 3, 4, 5, 3, 4, 6, 5, 6, 7, 5];

    this.setIndex(index);
    this.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    this.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  }

  setPositions(array) {
    let lineSegments;

    if (array instanceof Float32Array) {
      lineSegments = array;
    } else if (Array.isArray(array)) {
      lineSegments = new Float32Array(array);
    }

    const instanceBuffer = new THREE.InstancedInterleavedBuffer(lineSegments, 6, 1);

    this.setAttribute('instanceStart', new THREE.InterleavedBufferAttribute(instanceBuffer, 3, 0));
    this.setAttribute('instanceEnd', new THREE.InterleavedBufferAttribute(instanceBuffer, 3, 3));

    this.computeBoundingBox();
    this.computeBoundingSphere();

    return this;
  }

  computeBoundingBox() {
    if (this.boundingBox === null) {
      this.boundingBox = new THREE.Box3();
    }

    const start = this.attributes.instanceStart;
    const end = this.attributes.instanceEnd;

    if (start !== undefined && end !== undefined) {
      this.boundingBox.setFromBufferAttribute(start);

      const box = new THREE.Box3();
      box.setFromBufferAttribute(end);
      this.boundingBox.union(box);
    }
  }
}

export { LineGeometry };
