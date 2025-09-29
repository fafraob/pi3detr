function copyBibTeX() {
    var bibTexElement = document.querySelector(".bibtex-section pre code");
    var bibTexText = bibTexElement.innerText;
    navigator.clipboard.writeText(bibTexText);
    alert("BibTeX citation copied to clipboard!");
  }
  function toggleDarkMode() {
    document.body.classList.toggle("dark-mode");
    document.querySelector(".nav").classList.toggle("dark-mode");
  }
  window.onscroll = function () {
    const scrollUpBtn = document.getElementById("scrollUpBtn");
    if (
      document.body.scrollTop > 100 ||
      document.documentElement.scrollTop > 100
    ) {
      scrollUpBtn.style.display = "block";
    } else {
      scrollUpBtn.style.display = "none";
    }
  };
  function scrollToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  class Carousel {
    constructor(element, interval = 3000) {
      this.container = element;
      this.track = element.querySelector(".carousel-track");
      this.slides = Array.from(element.querySelectorAll(".carousel-slide"));
      this.indicators = element.querySelector(".carousel-indicators");

      this.currentIndex = 0;
      this.slidesPerView = 3;
      this.totalSlides = Math.ceil(this.slides.length / this.slidesPerView);
      this.interval = interval;
      this.autoPlayTimer = null;

      this.createIndicators();
      this.setupEventListeners();
      this.startAutoPlay();
      this.updateCarousel();
    }

    createIndicators() {
      for (let i = 0; i < this.totalSlides; i++) {
        const button = document.createElement("button");
        button.classList.add("indicator");
        if (i === 0) button.classList.add("active");
        button.addEventListener("click", () => {
          this.goToSlide(i);
        });
        this.indicators.appendChild(button);
      }
    }

    setupEventListeners() {
      this.container
        .querySelector(".prev")
        .addEventListener("click", (e) => {
          e.preventDefault();
          this.prevSlide();
        });

      this.container
        .querySelector(".next")
        .addEventListener("click", (e) => {
          e.preventDefault();
          this.nextSlide();
        });

      this.container.addEventListener("mouseenter", () => {
        this.stopAutoPlay();
      });

      this.container.addEventListener("mouseleave", () => {
        this.startAutoPlay();
      });

      document.addEventListener("keydown", (e) => {
        if (this.container.matches(":hover")) {
          if (e.key === "ArrowLeft") {
            this.prevSlide();
          } else if (e.key === "ArrowRight") {
            this.nextSlide();
          }
        }
      });
    }

    updateCarousel() {
      const offset =
        -this.currentIndex *
        (100 / this.slidesPerView) *
        this.slidesPerView;
      this.track.style.transform = `translateX(${offset}%)`;

      const indicators = Array.from(this.indicators.children);
      indicators.forEach((indicator, index) => {
        indicator.classList.toggle("active", index === this.currentIndex);
      });
    }

    nextSlide() {
      this.currentIndex = (this.currentIndex + 1) % this.totalSlides;
      this.updateCarousel();
      this.resetAutoPlay();
    }

    prevSlide() {
      this.currentIndex =
        (this.currentIndex - 1 + this.totalSlides) % this.totalSlides;
      this.updateCarousel();
      this.resetAutoPlay();
    }

    goToSlide(index) {
      if (index !== this.currentIndex) {
        this.currentIndex = index;
        this.updateCarousel();
        this.resetAutoPlay();
      }
    }

    startAutoPlay() {
      if (this.autoPlayTimer) {
        clearInterval(this.autoPlayTimer);
      }
      this.autoPlayTimer = setInterval(() => {
        this.nextSlide();
      }, this.interval);
    }

    stopAutoPlay() {
      if (this.autoPlayTimer) {
        clearInterval(this.autoPlayTimer);
        this.autoPlayTimer = null;
      }
    }

    resetAutoPlay() {
      this.stopAutoPlay();
      this.startAutoPlay();
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const imageCarousel = new Carousel(
      document.querySelector("#imageCarousel"),
      3000
    );
    const videoCarousel = new Carousel(
      document.querySelector("#videoCarousel"),
      5000
    );

    // Initialize Point Cloud
    initPointCloud();

    // Initialize Advanced Point Cloud
    initAdvancedPointCloud();

    // Add touch support
    const carousels = [imageCarousel, videoCarousel];
    carousels.forEach((carousel) => {
      let touchStartX = 0;
      let touchEndX = 0;

      carousel.container.addEventListener(
        "touchstart",
        (e) => {
          touchStartX = e.changedTouches[0].screenX;
        },
        { passive: true }
      );

      carousel.container.addEventListener(
        "touchend",
        (e) => {
          touchEndX = e.changedTouches[0].screenX;
          handleSwipe(carousel);
        },
        { passive: true }
      );

      function handleSwipe(carousel) {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
          if (diff > 0) {
            carousel.nextSlide();
          } else {
            carousel.prevSlide();
          }
        }
      }
    });
  });

// Three.js Point Cloud Visualization
function initPointCloud() {
  const canvas = document.getElementById('pointcloud-canvas');
  if (!canvas) return;

  // Scene setup
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  // Camera setup
  const camera = new THREE.PerspectiveCamera(
    75,
    canvas.offsetWidth / canvas.offsetHeight,
    0.1,
    1000
  );
  camera.position.set(0, 0, 5);

  // Renderer setup
  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setSize(canvas.offsetWidth, canvas.offsetHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  // Create point cloud geometry
  const pointCount = 100;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(pointCount * 3);
  const colors = new Float32Array(pointCount * 3);

  // Generate random points
  for (let i = 0; i < pointCount; i++) {
    const i3 = i * 3;
    
    // Random positions in a cube
    positions[i3] = (Math.random() - 0.5) * 4;     // x
    positions[i3 + 1] = (Math.random() - 0.5) * 4; // y
    positions[i3 + 2] = (Math.random() - 0.5) * 4; // z

    // Random colors
    colors[i3] = Math.random();     // r
    colors[i3 + 1] = Math.random(); // g
    colors[i3 + 2] = Math.random(); // b
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  // Create point material
  const material = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true,
    transparent: true,
    opacity: 0.8
  });

  // Create points mesh
  const points = new THREE.Points(geometry, material);
  scene.add(points);

  // Add orbit controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = true;
  controls.enablePan = true;

  // Add ambient light
  const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
  scene.add(ambientLight);

  // Add directional light
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(1, 1, 1);
  scene.add(directionalLight);

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    
    // Rotate the point cloud slowly
    points.rotation.y += 0.005;
    
    controls.update();
    renderer.render(scene, camera);
  }

  // Handle window resize
  function onWindowResize() {
    camera.aspect = canvas.offsetWidth / canvas.offsetHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.offsetWidth, canvas.offsetHeight);
  }

  window.addEventListener('resize', onWindowResize);

  // Start animation
  animate();
}

// Advanced Point Cloud Visualization with Controls
function initAdvancedPointCloud() {
  const canvas = document.getElementById('advanced-pointcloud-canvas');
  if (!canvas) return;

  // Scene setup
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  // Camera setup
  const camera = new THREE.PerspectiveCamera(
    75,
    canvas.offsetWidth / canvas.offsetHeight,
    0.1,
    1000
  );
  camera.position.set(3, 3, 5);

  // Renderer setup
  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setSize(canvas.offsetWidth, canvas.offsetHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  // Variables for dynamic control
  let pointCount = 100;
  let pointSize = 0.05;
  let autoRotate = true;
  let showConnections = false;
  let points, lines;
  let pointsGeometry, linesGeometry;

  // Create initial point cloud
  function createPointCloud() {
    // Remove existing points and lines
    if (points) scene.remove(points);
    if (lines) scene.remove(lines);

    // Create points geometry
    pointsGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(pointCount * 3);
    const colors = new Float32Array(pointCount * 3);

    // Generate random points with more interesting distribution
    for (let i = 0; i < pointCount; i++) {
      const i3 = i * 3;
      
      // Create clustered random distribution
      const cluster = Math.floor(Math.random() * 3);
      const clusterOffset = [
        [0, 0, 0],
        [2, 1, -1],
        [-1, -2, 1]
      ][cluster];

      positions[i3] = (Math.random() - 0.5) * 2 + clusterOffset[0];
      positions[i3 + 1] = (Math.random() - 0.5) * 2 + clusterOffset[1];
      positions[i3 + 2] = (Math.random() - 0.5) * 2 + clusterOffset[2];

      // Color based on position and cluster
      const hue = (cluster * 0.3 + Math.random() * 0.2) % 1;
      const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;
    }

    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create points material
    const pointsMaterial = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true
    });

    // Create points mesh
    points = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(points);

    // Create connections if enabled
    if (showConnections) {
      createConnections();
    }
  }

  function createConnections() {
    if (lines) scene.remove(lines);

    const positions = pointsGeometry.attributes.position.array;
    const linePositions = [];
    const lineColors = [];

    // Connect nearby points
    for (let i = 0; i < pointCount; i++) {
      const i3 = i * 3;
      const pos1 = new THREE.Vector3(positions[i3], positions[i3 + 1], positions[i3 + 2]);

      let connections = 0;
      for (let j = i + 1; j < pointCount && connections < 3; j++) {
        const j3 = j * 3;
        const pos2 = new THREE.Vector3(positions[j3], positions[j3 + 1], positions[j3 + 2]);
        
        if (pos1.distanceTo(pos2) < 1.5) {
          linePositions.push(pos1.x, pos1.y, pos1.z);
          linePositions.push(pos2.x, pos2.y, pos2.z);
          
          // Line color based on distance
          const distance = pos1.distanceTo(pos2);
          const intensity = 1 - (distance / 1.5);
          lineColors.push(intensity, intensity * 0.5, intensity * 0.8);
          lineColors.push(intensity, intensity * 0.5, intensity * 0.8);
          
          connections++;
        }
      }
    }

    linesGeometry = new THREE.BufferGeometry();
    linesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
    linesGeometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

    const linesMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.3
    });

    lines = new THREE.LineSegments(linesGeometry, linesMaterial);
    scene.add(lines);
  }

  // Add orbit controls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = true;
  controls.enablePan = true;

  // Add lighting
  const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
  directionalLight.position.set(2, 2, 2);
  scene.add(directionalLight);

  // UI Controls
  const pointCountSlider = document.getElementById('point-count-slider');
  const pointCountDisplay = document.getElementById('point-count-display');
  const pointSizeSlider = document.getElementById('point-size-slider');
  const pointSizeDisplay = document.getElementById('point-size-display');
  const autoRotateCheckbox = document.getElementById('auto-rotate-checkbox');
  const connectionsCheckbox = document.getElementById('show-connections-checkbox');
  const regenerateButton = document.getElementById('regenerate-points');
  const resetCameraButton = document.getElementById('reset-camera');

  // Event listeners
  if (pointCountSlider) {
    pointCountSlider.addEventListener('input', (e) => {
      pointCount = parseInt(e.target.value);
      pointCountDisplay.textContent = pointCount;
      createPointCloud();
    });
  }

  if (pointSizeSlider) {
    pointSizeSlider.addEventListener('input', (e) => {
      pointSize = parseFloat(e.target.value);
      pointSizeDisplay.textContent = pointSize.toFixed(2);
      if (points) {
        points.material.size = pointSize;
      }
    });
  }

  if (autoRotateCheckbox) {
    autoRotateCheckbox.addEventListener('change', (e) => {
      autoRotate = e.target.checked;
    });
  }

  if (connectionsCheckbox) {
    connectionsCheckbox.addEventListener('change', (e) => {
      showConnections = e.target.checked;
      createPointCloud();
    });
  }

  if (regenerateButton) {
    regenerateButton.addEventListener('click', () => {
      createPointCloud();
    });
  }

  if (resetCameraButton) {
    resetCameraButton.addEventListener('click', () => {
      camera.position.set(3, 3, 5);
      camera.lookAt(0, 0, 0);
      controls.reset();
    });
  }

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    
    if (autoRotate && points) {
      points.rotation.y += 0.003;
      if (lines) lines.rotation.y += 0.003;
    }
    
    controls.update();
    renderer.render(scene, camera);
  }

  // Handle window resize
  function onWindowResize() {
    camera.aspect = canvas.offsetWidth / canvas.offsetHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.offsetWidth, canvas.offsetHeight);
  }

  window.addEventListener('resize', onWindowResize);

  // Initialize
  createPointCloud();
  animate();
}
