// First load Three.js and make it globally available
import './libs/three.min.js';

// Now import OrbitControls
import { OrbitControls } from './libs/OrbitControls.js';

// XYZ File Loader
class XYZLoader {
load(url, onLoad, onProgress, onError) {
    const loader = new THREE.FileLoader(THREE.DefaultLoadingManager);
    loader.setResponseType('text');
    loader.load(url, (text) => {
    try {
        const geometry = this.parse(text);
        onLoad(geometry);
    } catch (error) {
        if (onError) onError(error);
    }
    }, onProgress, onError);
}

parse(text) {
    const lines = text.split('\n');
    const positions = [];
    const colors = [];

    for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === '' || line.startsWith('#')) continue;

    const parts = line.split(/\s+/);
    if (parts.length >= 3) {
        // Position (x, y, z)
        positions.push(
        parseFloat(parts[0]),
        parseFloat(parts[1]),
        parseFloat(parts[2])
        );

        // Color (r, g, b) - if available, otherwise use default
        if (parts.length >= 6) {
        colors.push(
            parseFloat(parts[3]) / 255.0,
            parseFloat(parts[4]) / 255.0,
            parseFloat(parts[5]) / 255.0
        );
        } else {
        // Default color based on height (y-coordinate)
        const y = parseFloat(parts[1]);
        const normalizedY = (y + 1) * 0.5; // Normalize to 0-1
        colors.push(normalizedY, 0.5, 1.0 - normalizedY);
        }
    }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    // Center the geometry
    geometry.computeBoundingBox();
    const center = geometry.boundingBox.getCenter(new THREE.Vector3());
    //geometry.translate(-center.x, -center.y, -center.z);
    
    return geometry;
}
}

// Point Cloud Carousel Class
class PointCloudCarousel {
  constructor() {
    this.pointClouds = [];
    this.polylineData = [];
    this.viewers = new Map();
    this.currentIndex = 0;
    this.carouselInstance = null;
    this.showGroundTruth = false;
    this.showPredictions = true;

    this.classColors = {
      1: 0xffa300,
      2: 0x0000ff,
      3: 0x00ff00,
      4: 0xff0000
    };
    
    this.isLoading = true;
    this.loadingElements = new Map();
    
    // Memory management properties
    this.geometryCache = new Map();
    this.materialCache = new Map();
    this.loadedCount = 0;
    this.maxActiveViewers = 3;
    
    // Viewport detection properties
    this.visibleSlides = 1; // Default to mobile
    this.isMobile = false;
    
    this.detectViewport();
    this.loadPointClouds();
    
    // Listen for viewport changes
    window.addEventListener('resize', () => {
      this.detectViewport();
      this.updateActiveViewers(this.currentIndex);
    });
  }

  detectViewport() {
    const carousel = document.getElementById('results-carousel');
    if (!carousel) {
      // Fallback detection based on screen width
      this.isMobile = window.innerWidth <= 768;
      this.visibleSlides = this.isMobile ? 1 : this.getDesktopVisibleSlides();
      return;
    }

    // Try to detect from carousel configuration or computed styles
    const carouselWidth = carousel.offsetWidth;
    const itemWidth = this.getCarouselItemWidth();
    
    if (itemWidth > 0) {
      this.visibleSlides = Math.floor(carouselWidth / itemWidth);
    } else {
      // Fallback to breakpoint detection
      this.isMobile = window.innerWidth <= 768;
      this.visibleSlides = this.isMobile ? 1 : this.getDesktopVisibleSlides();
    }

    // Ensure minimum of 1 and maximum reasonable number
    this.visibleSlides = Math.max(1, Math.min(this.visibleSlides, 4));
    
    console.log(`Detected viewport: ${this.visibleSlides} visible slides (mobile: ${this.isMobile})`);
  }

  getCarouselItemWidth() {
    const firstItem = document.querySelector('.carousel-item');
    if (firstItem) {
      const styles = window.getComputedStyle(firstItem);
      const width = parseFloat(styles.width);
      const marginLeft = parseFloat(styles.marginLeft || 0);
      const marginRight = parseFloat(styles.marginRight || 0);
      return width + marginLeft + marginRight;
    }
    return 0;
  }

  getDesktopVisibleSlides() {
    // Estimate based on common responsive breakpoints
    const width = window.innerWidth;
    if (width >= 1200) return 3; // Large desktop
    if (width >= 992) return 2;  // Desktop
    if (width >= 768) return 2;  // Tablet
    return 1; // Mobile fallback
  }

  async loadPointClouds() {
    this.showLoadingForAllSlots();
    
    const loader = new XYZLoader();
    const baseUrl = './pointcloud/pointcloud_';
    
    // Create array of promises for parallel loading
    const loadPromises = [];

    // count number of files in folder
    const maxFiles = 40;
    
    for (let i = 1; i <= maxFiles; i++) {
      const paddedNumber = i.toString().padStart(2, '0');
      const url = `${baseUrl}${paddedNumber}.xyz`;
      
      loadPromises.push(
        this.loadSinglePointCloud(loader, url, i - 1)
          .catch(() => null)
      );
    }

    // Load all point clouds in parallel
    const results = await Promise.allSettled(loadPromises);
    
    // Filter successful loads and maintain order
    results.forEach((result, index) => {
      if (result.status === 'fulfilled' && result.value) {
        this.pointClouds[index] = result.value;
        this.loadedCount++;
      }
    });

    // Remove empty slots and compact array
    this.pointClouds = this.pointClouds.filter(Boolean);
    
    console.log(`Loaded ${this.pointClouds.length} XYZ files in parallel`);

    // If no XYZ files were loaded, create sample point clouds
    if (this.pointClouds.length === 0) {
      console.log('No XYZ files found, creating sample point clouds...');
      this.createSamplePointClouds();
    }

    // Hide unused slots immediately
    this.hideUnusedSlots();

    // Initialize only the first 3 viewers
    this.initActiveViewers();

    // Load polyline data in background
    this.loadPolylineDataAsync();
    
    // Setup carousel controls
    this.setupCarouselControls();
    this.addDisplayModeToggleButton();
    
    this.isLoading = false;
  }

  hideUnusedSlots() {
    const totalSlots = Math.min(this.pointClouds.length, 40);
    
    for (let i = totalSlots + 1; i <= 40; i++) {
      const item = document.querySelector(`.item-pointcloud-${i}`);
      if (item) {
        item.style.display = 'none';
      }
      this.hideLoading(i);
    }
  }

  initActiveViewers() {
    // Initialize viewers for current (0), next (1), and previous (not applicable for index 0)
    this.updateActiveViewers(0);
  }

  getActiveIndices(currentIndex) {
    const indices = [];
    const maxItems = Math.min(this.pointClouds.length, 40);
    
    // Calculate range based on visible slides
    const preloadRange = Math.max(1, Math.ceil(this.visibleSlides / 2));
    
    // Add current index
    if (currentIndex >= 0 && currentIndex < maxItems) {
      indices.push(currentIndex);
    }
    
    // Add visible range around current index
    for (let i = 1; i <= preloadRange; i++) {
      // Add previous indices
      const prevIndex = currentIndex - i;
      if (prevIndex >= 0 && prevIndex < maxItems) {
        indices.push(prevIndex);
      }
      
      // Add next indices
      const nextIndex = currentIndex + i;
      if (nextIndex >= 0 && nextIndex < maxItems) {
        indices.push(nextIndex);
      }
    }
    
    // For desktop with multiple visible slides, ensure we load all currently visible ones
    if (this.visibleSlides > 1) {
      for (let i = 0; i < this.visibleSlides; i++) {
        const visibleIndex = currentIndex + i;
        if (visibleIndex >= 0 && visibleIndex < maxItems && !indices.includes(visibleIndex)) {
          indices.push(visibleIndex);
        }
      }
    }
    
    // Sort and ensure all indices are within bounds
    return indices
      .filter(index => index >= 0 && index < maxItems)
      .sort((a, b) => a - b);
  }

  updateActiveViewers(newIndex) {
    const prevIndex = this.currentIndex;
    
    // Calculate proper bounds
    const maxItems = Math.min(this.pointClouds.length, 40);
    const maxIndex = this.visibleSlides === 1 ? maxItems - 1 : Math.max(0, maxItems - this.visibleSlides);
    
    // Handle wrapping: if below 0, wrap to end; still enforce upper bound
    if (newIndex < 0) {
      this.currentIndex = maxIndex;
    } else {
      this.currentIndex = Math.min(newIndex, maxIndex);
    }

    // Calculate which viewers should be active based on viewport
    const activeIndices = this.getActiveIndices(this.currentIndex);
    
    console.log(`Updating viewers for index ${this.currentIndex} (bounded from ${newIndex}), maxIndex: ${maxIndex}, visible slides: ${this.visibleSlides}, active indices:`, activeIndices);
    
    // Dispose viewers that are no longer needed
    this.viewers.forEach((viewer, index) => {
      if (!activeIndices.includes(index)) {
        this.disposeViewer(index);
      }
    });

    // Initialize new viewers that are needed
    activeIndices.forEach(index => {
      if (!this.viewers.has(index) && index < this.pointClouds.length && index < maxItems) {
        // Use requestIdleCallback for non-blocking initialization
        if (window.requestIdleCallback) {
          window.requestIdleCallback(() => {
            this.initViewerForIndex(index);
          });
        } else {
          setTimeout(() => {
            this.initViewerForIndex(index);
          }, 16);
        }
      }
    });

    // Update navigation buttons after index change
    this.updateNavigationButtons();
  }

  updateNavigationButtons() {
    const prevButton = document.querySelector('#results-carousel .carousel-nav-left');
    const nextButton = document.querySelector('#results-carousel .carousel-nav-right');
    
    if (prevButton && nextButton) {
      // Always show previous button (allow wrapping to end)
      prevButton.style.display = 'block';
      prevButton.style.opacity = '1';
      prevButton.style.pointerEvents = 'auto';
      prevButton.disabled = false;

      // Calculate max index - ensure we can see all point clouds
      const maxItems = Math.min(this.pointClouds.length, 40);
      const maxIndex = this.visibleSlides === 1 ? maxItems - 1 : Math.max(0, maxItems - this.visibleSlides);
      
      // Hide next button only at max index
      if (this.currentIndex >= maxIndex) {
        nextButton.style.display = 'none';
        nextButton.style.opacity = '0';
        nextButton.style.pointerEvents = 'none';
        nextButton.disabled = true;
      } else {
        nextButton.style.display = 'block';
        nextButton.style.opacity = '1';
        nextButton.style.pointerEvents = 'auto';
        nextButton.disabled = false;
      }
      
      console.log(`Navigation: currentIndex=${this.currentIndex}, maxIndex=${maxIndex}, visibleSlides=${this.visibleSlides}, totalItems=${maxItems}`);
    }
  }

  addCustomNavigationHandlers() {
    // Add additional event handlers to the navigation buttons for stricter control
    setTimeout(() => {
      const prevButton = document.querySelector('#results-carousel .carousel-nav-left');
      const nextButton = document.querySelector('#results-carousel .carousel-nav-right');
      
      // Remove the previous button restriction - allow wrapping
      if (prevButton) {
        prevButton.addEventListener('click', (e) => {
          // Allow going below 0 - will wrap to end
          console.log('Previous button clicked, allowing wrap to end');
        }, true);
      }
      
      if (nextButton) {
        nextButton.addEventListener('click', (e) => {
          const maxItems = Math.min(this.pointClouds.length, 40);
          const maxIndex = this.visibleSlides === 1 ? maxItems - 1 : Math.max(0, maxItems - this.visibleSlides);
          if (this.currentIndex >= maxIndex) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            return false;
          }
        }, true);
      }
    }, 1500);
  }

  setupCarouselControls() {
    setTimeout(() => {
      const carousel = document.getElementById('results-carousel');
      if (carousel) {
        // Detect carousel configuration based on viewport
        const slidesToShow = this.visibleSlides;
        
        const carouselInstances = bulmaCarousel.attach('#results-carousel', {
          slidesToScroll: 1,
          slidesToShow: slidesToShow,
          infinite: false,
          pagination: false,
          navigation: true,
          navigationKeys: true,
          navigationSwipe: true,
          effect: 'slide',
          duration: 600,
          timing: 'ease',
          breakpoints: [
            {
              changePoint: 768,
              slidesToShow: 1,
              slidesToScroll: 1
            },
            {
              changePoint: 992,
              slidesToShow: Math.min(2, this.pointClouds.length),
              slidesToScroll: 1
            },
            {
              changePoint: 1200,
              slidesToShow: Math.min(3, this.pointClouds.length),
              slidesToScroll: 1
            }
          ]
        });

        if (carouselInstances.length > 0) {
          this.carouselInstance = carouselInstances[0];
          
          // Update viewport detection when carousel changes
          this.carouselInstance.on('before:show', (state) => {
            const maxItems = Math.min(this.pointClouds.length, 40);
            const maxIndex = this.visibleSlides === 1 ? maxItems - 1 : Math.max(0, maxItems - this.visibleSlides);
            
            // Handle wrapping: if going below 0, wrap to end
            let newIndex = state.next;
            if (state.next < 0) {
              newIndex = maxIndex; // Wrap to the last valid index
              console.log(`Wrapping from ${state.next} to ${newIndex}`);
            }
            
            // Still prevent going above max
            if (state.next > maxIndex) {
              console.log(`Preventing navigation above ${maxIndex} (requested: ${state.next})`);
              return false;
            }
            
            console.log(`Carousel navigation: ${this.currentIndex} -> ${newIndex} (requested: ${state.next})`);
            
            this.detectViewport();
            this.updateActiveViewers(newIndex);
            this.updateNavigationButtons();
          });

          // Listen for carousel responsive changes
          this.carouselInstance.on('refresh', () => {
            this.detectViewport();
            this.updateActiveViewers(this.currentIndex);
            this.updateNavigationButtons();
          });

          // Add custom navigation button event handlers for additional control
          this.addCustomNavigationHandlers();
          
          // Initial navigation button state
          this.updateNavigationButtons();
        }
      }
    }, 1000);
  }

  async loadSinglePointCloud(loader, url, index) {
    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (geometry) => {
          // Cache geometry for potential reuse
          this.cacheGeometry(geometry, index);
          resolve(geometry);
        },
        undefined,
        reject
      );
    });
  }

  cacheGeometry(geometry, index) {
    // Store in cache with index as key
    this.geometryCache.set(index, geometry);
    
    // Pre-compute bounding box for faster viewer initialization
    geometry.computeBoundingBox();
  }

  async loadPolylineDataAsync() {
    // Load polyline data in background without blocking UI
    const baseUrl = './pointcloud/pointcloud_';
    const loadPromises = [];
    
    for (let i = 0; i < this.pointClouds.length; i++) {
      const paddedNumber = (i + 1).toString().padStart(2, '0');
      const jsonUrl = `${baseUrl}${paddedNumber}.json`;
      
      loadPromises.push(
        fetch(jsonUrl)
          .then(response => response.ok ? response.json() : null)
          .catch(() => null)
      );
    }

    const results = await Promise.allSettled(loadPromises);
    
    results.forEach((result, index) => {
      const data = result.status === 'fulfilled' ? result.value : null;
      this.polylineData[index] = data;
    });

    // Update viewers with polyline data when ready
    this.updateAllViewersAsync();
  }

  updateAllViewersAsync() {
    // Update viewers in small batches to avoid blocking
    const batchSize = 3;
    let currentBatch = 0;

    const updateBatch = () => {
      const start = currentBatch * batchSize;
      const end = Math.min(start + batchSize, this.viewers.size);
      const viewerEntries = Array.from(this.viewers.entries());

      for (let i = start; i < end; i++) {
        if (viewerEntries[i]) {
          const [index, viewer] = viewerEntries[i];
          this.updateViewerDisplay(viewer, index);
        }
      }

      currentBatch++;
      
      if (end < this.viewers.size) {
        // Use requestAnimationFrame for smooth updates
        requestAnimationFrame(updateBatch);
      }
    };

    if (this.viewers.size > 0) {
      updateBatch();
    }
  }

  getCachedMaterial(key, materialConfig) {
    if (!this.materialCache.has(key)) {
      this.materialCache.set(key, new THREE.PointsMaterial(materialConfig));
    }
    return this.materialCache.get(key);
  }

  // Add cleanup method for better memory management
  cleanup() {
    // Dispose all viewers
    this.viewers.forEach((viewer, index) => {
      this.disposeViewer(index);
    });
    
    // Clear all caches
    this.geometryCache.clear();
    this.materialCache.clear();
    
    console.log('PointCloudCarousel cleaned up');
  }

  addDisplayModeToggleButton() {
    setTimeout(() => {
      const carouselContainer = document.getElementById('results-carousel').parentElement;
      
      const buttonContainer = document.createElement('div');
      buttonContainer.className = 'field has-addons mb-3';
      buttonContainer.style.marginBottom = '10px';
      
      const buttonGroup = document.createElement('div');
      buttonGroup.className = 'control';
      
      const predButton = document.createElement('button');
      predButton.className = 'button is-info is-selected';
      predButton.id = 'toggle-predictions-btn';
      predButton.innerHTML = `
        <span class="icon">
          <i class="fas fa-brain"></i>
        </span>
        <span>Predictions</span>
      `;
      
      const gtButton = document.createElement('button');
      gtButton.className = 'button is-light';
      gtButton.id = 'toggle-ground-truth-btn';
      gtButton.innerHTML = `
        <span class="icon">
          <i class="fas fa-check-circle"></i>
        </span>
        <span>Ground Truth</span>
      `;
      
      buttonGroup.appendChild(predButton);
      buttonGroup.appendChild(gtButton);
      buttonContainer.appendChild(buttonGroup);
      
      carouselContainer.insertBefore(buttonContainer, carouselContainer.firstChild);
      
      predButton.addEventListener('click', () => {
        if (this.showPredictions) {
          this.showPredictions = false;
          predButton.className = 'button is-light';
        } else {
          this.showPredictions = true;
          this.showGroundTruth = false;
          predButton.className = 'button is-info is-selected';
          gtButton.className = 'button is-light';
        }
        this.updateAllViewers();
      });
      
      gtButton.addEventListener('click', () => {
        if (this.showGroundTruth) {
          this.showGroundTruth = false;
          gtButton.className = 'button is-light';
        } else {
          this.showGroundTruth = true;
          this.showPredictions = false;
          gtButton.className = 'button is-success is-selected';
          predButton.className = 'button is-light';
        }
        this.updateAllViewers();
      });
    }, 500);
  }

  initViewerForIndex(index) {
    const slotNumber = index + 1;
    const container = document.getElementById(`pointcloud-viewer-${slotNumber}`);
    
    if (!container || !this.pointClouds[index]) {
      console.warn(`Cannot initialize viewer for index ${index}`);
      return;
    }

    console.log(`Initializing viewer for index ${index}`);
    
    const geometry = this.pointClouds[index];
    const viewer = this.initViewer(container, geometry, slotNumber);
    
    if (viewer) {
      this.viewers.set(index, viewer);
      
      // Update display if polyline data is ready
      if (this.polylineData[index]) {
        this.updateViewerDisplay(viewer, index);
      }
    }
  }

  disposeViewer(index) {
    const viewer = this.viewers.get(index);
    if (!viewer) return;

    console.log(`Disposing viewer for index ${index}`);

    // Cancel animation
    if (viewer.animationId) {
      cancelAnimationFrame(viewer.animationId);
    }

    // Dispose Three.js objects
    if (viewer.renderer) {
      viewer.renderer.dispose();
    }

    // Remove polylines
    if (viewer.polylines) {
      viewer.polylines.forEach(line => {
        if (line.geometry) line.geometry.dispose();
        if (line.material) line.material.dispose();
        viewer.scene.remove(line);
      });
    }

    // Dispose scene objects
    viewer.scene.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach(material => material.dispose());
        } else {
          object.material.dispose();
        }
      }
    });

    // Clear container and show loading
    const slotNumber = index + 1;
    const container = document.getElementById(`pointcloud-viewer-${slotNumber}`);
    if (container) {
      container.innerHTML = '';
      this.showLoading(container, slotNumber);
    }

    // Remove from viewers map
    this.viewers.delete(index);
  }

  initViewer(container, geometry, index) {
    try {
      this.hideLoading(index);
      container.innerHTML = '';
      
      if (!geometry || !geometry.attributes || !geometry.attributes.position) {
        console.error(`Invalid geometry for viewer ${index}`);
        return null;
      }
      
      // Use cached bounding box if available
      const boundingBox = geometry.boundingBox;
      const center = boundingBox.getCenter(new THREE.Vector3());
      const size = boundingBox.getSize(new THREE.Vector3());
      const maxDimension = Math.max(size.x, size.y, size.z);
      
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf8f9fa);

      const aspect = container.offsetWidth / container.offsetHeight || 1;
      const camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
      
      const distance = maxDimension;
      camera.position.set(
        center.x + distance,
        center.y + distance,
        center.z + distance
      );
      camera.lookAt(center);

      const renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true,
        powerPreference: "high-performance"
      });
      renderer.setSize(container.offsetWidth || 400, container.offsetHeight || 400);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
      renderer.shadowMap.enabled = false;
      
      container.appendChild(renderer.domElement);

      const pointSize = Math.max(0.01, maxDimension * 0.01);
      const material = this.getCachedMaterial('points', {
        size: pointSize,
        color: 0x969696,
        transparent: true,
        opacity: 0.7,
        sizeAttenuation: true
      });

      const points = new THREE.Points(geometry, material.clone());
      points.material.size = pointSize;
      scene.add(points);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;
      controls.enableZoom = true;
      controls.enablePan = true;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 3;
      controls.target.copy(center);
      
      controls.minDistance = maxDimension * 0.5;
      controls.maxDistance = maxDimension * 10;

      const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
      scene.add(ambientLight);

      let animationId;
      const animate = () => {
        animationId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };

      animate();

      const viewer = { 
        scene, 
        camera, 
        renderer, 
        controls, 
        container, 
        index: index - 1,
        pointsMesh: points,
        polylines: [],
        animationId
      };

      // Optimized resize handler
      let resizeTimeout;
      const handleResize = () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
          const width = container.offsetWidth || 400;
          const height = container.offsetHeight || 400;
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }, 100);
      };

      const resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(container);

      console.log(`Optimized viewer ${index} initialized (${maxDimension.toFixed(2)})`);
      
      return viewer;
      
    } catch (error) {
      console.error(`Error initializing viewer ${index}:`, error);
      this.showError(container, index, error.message);
      return null;
    }
  }

  updateAllViewers() {
    // Only update active viewers
    this.viewers.forEach((viewer, index) => {
      this.updateViewerDisplay(viewer, index);
    });
  }

  updateViewerDisplay(viewer, index) {
    if (viewer.polylines) {
      viewer.polylines.forEach(line => viewer.scene.remove(line));
      viewer.polylines = [];
    }
    
    if (viewer.pointsMesh) {
      viewer.pointsMesh.visible = true;
    }
    
    if (this.polylineData[index]) {
      const polylinesToAdd = [];
      
      if (this.showGroundTruth) {
        const gtPolylines = this.createPolylines(this.polylineData[index], 'gt');
        polylinesToAdd.push(...gtPolylines);
      }
      
      if (this.showPredictions) {
        const predPolylines = this.createPolylines(this.polylineData[index], 'pred');
        polylinesToAdd.push(...predPolylines);
      }
      
      viewer.polylines = polylinesToAdd;
      viewer.polylines.forEach(line => viewer.scene.add(line));
    }
  }

  createPolylines(data, mode) {
    const polylines = [];
    const polylineData = data[mode];
    
    if (!polylineData) return polylines;
    
    polylineData.forEach(item => {
      if (item.points && item.points.length > 1) {
        const points = item.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const color = this.classColors[item.class] || 0xffffff;

        const curve = new THREE.CatmullRomCurve3(points);
        const tubeGeometry = new THREE.TubeGeometry(curve, 100, 0.01, 8, false);
        const tubeMaterial = new THREE.MeshBasicMaterial({ color: color });
        const tube = new THREE.Mesh(tubeGeometry, tubeMaterial);
        polylines.push(tube);

        const startCap = new THREE.CircleGeometry(0.01, 8);
        const endCap   = new THREE.CircleGeometry(0.01, 8);

        const startCapMesh = new THREE.Mesh(startCap, tubeMaterial);
        const endCapMesh   = new THREE.Mesh(endCap, tubeMaterial);

        const startDir = new THREE.Vector3().subVectors(points[1], points[0]).normalize();
        const endDir   = new THREE.Vector3().subVectors(points[points.length - 2], points[points.length - 1]).normalize();

        startCapMesh.position.copy(points[0]);
        startCapMesh.lookAt(points[0].clone().sub(startDir));

        endCapMesh.position.copy(points[points.length - 1]);
        endCapMesh.lookAt(points[points.length - 1].clone().sub(endDir));

         polylines.push(startCapMesh);
         polylines.push(endCapMesh);
      }
    });
    
    return polylines;
  }

  showLoadingForAllSlots() {
    // Show loading for up to 20 slots
    for (let i = 1; i <= 40; i++) {
      const container = document.getElementById(`pointcloud-viewer-${i}`);
      if (container) {
        this.showLoading(container, i);
      }
    }
  }

  showLoading(container, slotNumber) {
    // Clear existing content
    container.innerHTML = '';
    
    // Create loading overlay
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = `
      <div class="loading-content">
        <div class="loading-spinner"></div>
        <div class="loading-text">Loading Point Cloud ${slotNumber}...</div>
      </div>
    `;
    
    container.appendChild(loadingOverlay);
    this.loadingElements.set(slotNumber, loadingOverlay);
  }

  hideLoading(slotNumber) {
    const loadingElement = this.loadingElements.get(slotNumber);
    if (loadingElement && loadingElement.parentNode) {
      loadingElement.remove();
      this.loadingElements.delete(slotNumber);
    }
  }

  ensureMinimumPointClouds(minCount) {
    if (this.pointClouds.length === 0) {
      this.createSamplePointClouds();
    }

    // If we still don't have enough, duplicate existing ones
    while (this.pointClouds.length < minCount) {
      const originalCount = this.pointClouds.length;
      for (let i = 0; i < originalCount && this.pointClouds.length < minCount; i++) {
        // Clone the geometry to avoid conflicts
        const originalGeometry = this.pointClouds[i];
        const clonedGeometry = originalGeometry.clone();
        this.pointClouds.push(clonedGeometry);
      }
    }
    
    console.log(`Ensured ${this.pointClouds.length} point clouds available for carousel`);
  }

  createSamplePointClouds() {
    const sampleConfigs = [
      { type: 'sphere', count: 500 },
      { type: 'cube', count: 400 },
      { type: 'spiral', count: 700 },
      { type: 'torus', count: 800 },
      { type: 'helix', count: 900 },
      { type: 'cone', count: 550 },
      { type: 'wave', count: 800 },
      { type: 'clusters', count: 600 }
    ];

    sampleConfigs.forEach(config => {
      const geometry = this.generateSampleGeometry(config);
      this.pointClouds.push(geometry);
    });
  }

  generateSampleGeometry(config) {
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    for (let i = 0; i < config.count; i++) {
      let x, y, z;
      
      switch (config.type) {
        case 'sphere':
          const phi = Math.random() * Math.PI * 2;
          const costheta = Math.random() * 2 - 1;
          const u = Math.random();
          const theta = Math.acos(costheta);
          const r = 2 * Math.cbrt(u);
          x = r * Math.sin(theta) * Math.cos(phi);
          y = r * Math.sin(theta) * Math.sin(phi);
          z = r * Math.cos(theta);
          break;
          
        case 'cube':
          x = (Math.random() - 0.5) * 4;
          y = (Math.random() - 0.5) * 4;
          z = (Math.random() - 0.5) * 4;
          break;
          
        case 'spiral':
          const t = i / config.count * Math.PI * 4;
          const radius = t * 0.3;
          x = radius * Math.cos(t);
          y = t * 0.5 - 3;
          z = radius * Math.sin(t);
          break;
          
        case 'torus':
          const u_tor = Math.random() * Math.PI * 2;
          const v_tor = Math.random() * Math.PI * 2;
          const R = 2, r_tor = 0.8;
          x = (R + r_tor * Math.cos(v_tor)) * Math.cos(u_tor);
          y = (R + r_tor * Math.cos(v_tor)) * Math.sin(u_tor);
          z = r_tor * Math.sin(v_tor);
          break;
          
        case 'helix':
          const t_helix = i / config.count * Math.PI * 6;
          x = 2 * Math.cos(t_helix);
          y = t_helix * 0.3 - 3;
          z = 2 * Math.sin(t_helix);
          break;
          
        case 'cone':
          const h = Math.random() * 4;
          const r_cone = (4 - h) * 0.5;
          const a_cone = Math.random() * Math.PI * 2;
          x = r_cone * Math.cos(a_cone);
          z = r_cone * Math.sin(a_cone);
          y = h - 2;
          break;
          
        case 'wave':
          x = (Math.random() - 0.5) * 6;
          z = (Math.random() - 0.5) * 6;
          y = Math.sin(x) * Math.cos(z);
          break;
          
        case 'clusters':
          const cluster = Math.floor(Math.random() * 3);
          const offsets = [[0, 0, 0], [3, 0, 0], [-1.5, 2.6, 0]];
          x = (Math.random() - 0.5) * 1.5 + offsets[cluster][0];
          y = (Math.random() - 0.5) * 1.5 + offsets[cluster][1];
          z = (Math.random() - 0.5) * 1.5 + offsets[cluster][2];
          break;
          
        default:
          x = (Math.random() - 0.5) * 4;
          y = (Math.random() - 0.5) * 4;
          z = (Math.random() - 0.5) * 4;
      }

      positions.push(x, y, z);

      // Generate vibrant colors based on position
      const hue = (x + y + z + 10) / 20;
      const color = new THREE.Color().setHSL((hue % 1), 0.8, 0.6);
      colors.push(color.r, color.g, color.b);
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    // Ensure bounding box is computed
    geometry.computeBoundingBox();
    
    return geometry;
  }

  getMemoryInfo() {
    return {
      activeViewers: this.viewers.size,
      totalPointClouds: this.pointClouds.length,
      visibleSlides: this.visibleSlides,
      isMobile: this.isMobile,
      currentIndex: this.currentIndex,
      activeIndices: this.getActiveIndices(this.currentIndex)
    };
  }

  showError(container, slotNumber, errorMessage) {
    // Hide loading first
    this.hideLoading(slotNumber);
    
    // Clear container
    container.innerHTML = '';
    
    // Create error overlay
    const errorOverlay = document.createElement('div');
    errorOverlay.className = 'loading-overlay error-overlay';
    errorOverlay.innerHTML = `
      <div class="loading-content">
        <div class="error-icon">⚠️</div>
        <div class="loading-text">Failed to load Point Cloud ${slotNumber}</div>
        <div class="error-message">${errorMessage}</div>
      </div>
    `;
    
    container.appendChild(errorOverlay);
  }
}

// Add global function to check memory usage (for debugging)
window.getCarouselMemoryInfo = function() {
  const carousel = window.pointCloudCarousel;
  if (carousel && carousel.getMemoryInfo) {
    console.table(carousel.getMemoryInfo());
  }
};

// Point Cloud Initialization (existing function)
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
  const controls = new OrbitControls(camera, renderer.domElement);
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

function initAdvancedPointCloud() {
  const canvas = document.getElementById('advanced-pointcloud-canvas');
  if (!canvas) return;

  // ...existing advanced point cloud code...
}

// Initialize Point Cloud Carousel
function initPointCloudCarousel() {
  window.pointCloudCarousel = new PointCloudCarousel();
}

// Initialize when DOM is loaded and THREE.js is available
document.addEventListener('DOMContentLoaded', () => {
  // Add loading styles to the document
  addLoadingStyles();
  
  // Only initialize the carousel for now
  initPointCloudCarousel();
});

// Add CSS styles for loading overlay
function addLoadingStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 8px;
      z-index: 1000;
    }

    .loading-content {
      text-align: center;
      color: white;
    }

    .loading-spinner {
      width: 40px;
      height: 40px;
      border: 4px solid rgba(255, 255, 255, 0.3);
      border-top: 4px solid white;
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin: 0 auto 15px;
    }

    .loading-text {
      font-size: 14px;
      font-weight: 500;
      margin-bottom: 5px;
    }

    .error-overlay {
      background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
    }

    .error-icon {
      font-size: 32px;
      margin-bottom: 10px;
    }

    .error-message {
      font-size: 12px;
      opacity: 0.9;
      max-width: 200px;
      word-wrap: break-word;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
      .loading-spinner {
        width: 30px;
        height: 30px;
        border-width: 3px;
      }
      
      .loading-text {
        font-size: 12px;
      }
      
      .error-message {
        font-size: 10px;
      }
    }
  `;
  document.head.appendChild(style);
}