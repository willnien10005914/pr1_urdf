import * as THREE from './vendor/three/three.module.js';
import { OrbitControls } from './vendor/three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from './vendor/three/examples/jsm/loaders/STLLoader.js';
import URDFLoader from './vendor/urdf-loader/URDFLoader.js';

/** Geometric URDF preview. Joint motion here is not a physics rollout. */
export function createViewer(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#e7edf2');
  const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 80);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.style.cssText = 'width:100%;height:100%;display:block';
  container.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0.7);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8ea0b5, 1.15));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(2.4, -3.2, 5);
  scene.add(key);
  const grid = new THREE.GridHelper(8, 16, 0xc5d0da, 0xd5dee6);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  let robot = null;
  let disposed = false;
  let plant = 0;
  const status = container.appendChild(document.createElement('p'));
  status.className = 'viewer-status';

  function resize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(container);

  function jointStates() {
    if (!robot) return [];
    return Object.values(robot.joints).filter((joint) => joint.jointType !== 'fixed').map((joint) => {
      const lower = Number.isFinite(joint.limit?.lower) ? joint.limit.lower : -Math.PI;
      const upper = Number.isFinite(joint.limit?.upper) ? joint.limit.upper : Math.PI;
      const value = Array.isArray(joint.jointValue) ? joint.jointValue[0] : Number(joint.angle) || 0;
      return { name: joint.name, type: joint.jointType, lower, upper, value };
    });
  }

  function setJoints(values) {
    if (!robot) return;
    for (const [name, value] of Object.entries(values)) {
      const joint = robot.joints[name];
      if (!joint || !Number.isFinite(value)) continue;
      const lower = Number.isFinite(joint.limit?.lower) ? joint.limit.lower : -Infinity;
      const upper = Number.isFinite(joint.limit?.upper) ? joint.limit.upper : Infinity;
      robot.setJointValue(name, THREE.MathUtils.clamp(value, lower, upper));
    }
  }

  function soleHeight(name) {
    const link = (robot.links && robot.links[name]) || robot.getObjectByName(name);
    if (!link) return Number.POSITIVE_INFINITY;
    const box = new THREE.Box3().setFromObject(link);
    return box.isEmpty() ? Number.POSITIVE_INFINITY : box.min.z;
  }

  function correctStance(com = {}) {
    if (!robot) return { support: 'none' };
    const x = Number.isFinite(Number(com.x)) ? Number(com.x) : 0;
    const y = Number.isFinite(Number(com.y)) ? Number(com.y) : 0;
    robot.position.x = x;
    robot.position.y = y;
    robot.updateMatrixWorld(true);
    const left = soleHeight('left_ankle_roll_link');
    const right = soleHeight('right_ankle_roll_link');
    let lowest = Math.min(left, right);
    if (!Number.isFinite(lowest)) {
      const bounds = new THREE.Box3().setFromObject(robot);
      lowest = bounds.min.z;
    }
    if (Number.isFinite(lowest)) robot.position.z -= lowest;
    robot.updateMatrixWorld(true);
    const planted = Math.min(soleHeight('left_ankle_roll_link'), soleHeight('right_ankle_roll_link'));
    const gap = Math.abs(left - right);
    const support = !Number.isFinite(Math.min(left, right))
      ? 'none'
      : gap < 0.012 ? 'both' : (left <= right ? 'left' : 'right');
    container.dataset.support = support;
    container.dataset.sole = Number.isFinite(planted) ? planted.toFixed(4) : '';
    return { support };
  }

  function plantFeet() {
    robot.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(robot);
    if (Number.isFinite(bounds.min.z)) {
      robot.position.z -= bounds.min.z;
      plant = robot.position.z;
    }
  }

  function fit() {
    robot.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(robot);
    const size = bounds.getSize(new THREE.Vector3());
    if (!Number.isFinite(size.z) || size.z < 0.01) return;
    const center = bounds.getCenter(new THREE.Vector3());
    controls.target.set(center.x, center.y, bounds.min.z + size.z * 0.45);
    const distance = Math.max(size.z, size.x) / (2 * Math.tan(camera.fov * Math.PI / 360) * 0.7);
    camera.position.copy(controls.target).add(new THREE.Vector3(1, 0, 0.22).normalize().multiplyScalar(distance));
    controls.update();
  }

  async function load(url) {
    status.hidden = false;
    status.textContent = 'Loading URDF and meshes…';
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    loader.parseCollision = false;
    const meshLoads = [];
    loader.loadMeshCb = (path, loadingManager, done) => {
      meshLoads.push(new Promise((resolve) => {
        if (!/\.stl$/i.test(path)) {
          done(null, new Error(path));
          resolve(false);
          return;
        }
        new STLLoader(loadingManager).load(
          path,
          (geometry) => {
            done(new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({ color: 0xd5dbe3, shininess: 20 })));
            status.textContent = `Loading meshes ${meshLoads.length}`;
            resolve(true);
          },
          undefined,
          () => {
            done(null, new Error(path));
            resolve(false);
          },
        );
      }));
    };
    const loaded = await loader.loadAsync(url);
    const results = await Promise.all(meshLoads);
    if (!results.some(Boolean)) throw new Error('None of the STL meshes loaded.');
    if (disposed) return jointStates();
    if (robot) scene.remove(robot);
    robot = loaded;
    scene.add(robot);
    robot.position.set(0, 0, 0);
    robot.quaternion.identity();
    robot.updateMatrixWorld(true);
    plantFeet();
    fit();
    status.textContent = '';
    status.hidden = true;
    container.dataset.ready = 'true';
    container.dataset.meshes = String(results.filter(Boolean).length);
    return jointStates();
  }

  function showFrame(motion, index) {
    if (!robot || !motion) return;
    const origin = motion.root[0];
    const root = motion.root[index];
    const joints = {};
    motion.joint_names.forEach((name, column) => { joints[name] = motion.q[index][column]; });
    setJoints(joints);
    if (motion.plant_feet) {
      robot.quaternion.identity();
      correctStance({ x: root[0] - origin[0], y: root[1] - origin[1] });
      return;
    }
    robot.position.set(root[0] - origin[0], root[1] - origin[1], (root[2] - origin[2]) + plant);
    robot.quaternion.set(root[4], root[5], root[6], root[3]);
  }

  function frame() {
    if (disposed) return;
    requestAnimationFrame(frame);
    controls.update();
    renderer.render(scene, camera);
  }
  resize();
  requestAnimationFrame(frame);

  return {
    load,
    jointStates,
    setJoints,
    correctStance,
    showFrame,
    resetPose() {
      if (!robot) return;
      robot.traverse((node) => { if (node.setJointValue && node.jointType !== 'fixed') node.setJointValue(0); });
      robot.position.set(0, 0, 0);
      robot.quaternion.identity();
      plantFeet();
    },
  };
}
