import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import { PolicyController } from './policy.js';

// ---------------------------------------------------------------------------
// MuJoCo -> Three.js viewer
//
// The whole robot is described once by MuJoCo (the XML + meshes). Instead of
// hand-mapping each part (like the old servo demo did), we build the scene
// directly from the compiled `MjModel` arrays and sync the body transforms
// from `MjData` every frame. This works for any menagerie model.
// ---------------------------------------------------------------------------

// MuJoCo geom type enum (mjGeomType)
const GEOM_PLANE = 0;
const GEOM_HFIELD = 1;
const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;
const GEOM_ELLIPSOID = 4;
const GEOM_CYLINDER = 5;
const GEOM_BOX = 6;
const GEOM_MESH = 7;

// Collision geoms live in group 3 in the menagerie; groups < 3 are visual.
const COLLISION_GROUP = 3;

let mujoco, model, data;
let scene, camera, renderer, controls;
let bodyNodes = []; // one THREE.Group per MuJoCo body
let panel;
let controller = null; // PolicyController, or null when no .onnx is present
let statusEl = null;
let keyLight = null; // main directional light, kept centered on the robot

// Velocity command limits (m/s and rad/s) reached with the keyboard.
const MAX_VX = 1.0;
const MAX_VY = 0.5;
const MAX_YAW = 1.0;
const keys = Object.create(null);

// ---------------------------------------------------------------------------
// Loading: XML + assets through the emscripten VFS
// ---------------------------------------------------------------------------

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Falha ao buscar ${url}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function loadModel(basePath, xmlPath, assetFiles) {
  const xml = await (await fetch(`${basePath}${xmlPath}`)).text();

  const vfs = new mujoco.MjVFS();
  for (const file of assetFiles) {
    const bytes = await fetchBytes(`${basePath}${file}`);
    // The compiler resolves `meshdir`/`texturedir` prefixed paths, so register
    // both the prefixed path ("assets/x.obj") and the bare name ("x.obj").
    vfs.addBuffer(file, bytes);
    const bare = file.split('/').pop();
    if (bare !== file) vfs.addBuffer(bare, bytes);
  }

  const m = mujoco.MjModel.from_xml_string(xml, vfs);
  vfs.delete();
  return m;
}

// The menagerie scene.xml pulls in go2.xml via <include> and adds a ground
// plane, so the robot actually stands instead of free-falling.
const MODEL_BASE = '/models/unitree_go2/';
const MODEL_XML = 'scene.xml';
const MODEL_ASSETS = [
  'go2.xml', // resolved by <include file="go2.xml"/>
  'assets/base_0.obj',
  'assets/base_1.obj',
  'assets/base_2.obj',
  'assets/base_3.obj',
  'assets/base_4.obj',
  'assets/hip_0.obj',
  'assets/hip_1.obj',
  'assets/thigh_0.obj',
  'assets/thigh_1.obj',
  'assets/thigh_mirror_0.obj',
  'assets/thigh_mirror_1.obj',
  'assets/calf_0.obj',
  'assets/calf_1.obj',
  'assets/calf_mirror_0.obj',
  'assets/calf_mirror_1.obj',
  'assets/foot.obj',
];

// ---------------------------------------------------------------------------
// Geometry construction from MjModel
// ---------------------------------------------------------------------------

function meshGeometryFromModel(meshId) {
  const vertAdr = model.mesh_vertadr[meshId];
  const vertNum = model.mesh_vertnum[meshId];
  const faceAdr = model.mesh_faceadr[meshId];
  const faceNum = model.mesh_facenum[meshId];

  const verts = model.mesh_vert;
  const faces = model.mesh_face;

  const positions = new Float32Array(vertNum * 3);
  for (let i = 0; i < vertNum * 3; i++) {
    positions[i] = verts[vertAdr * 3 + i];
  }

  const indices = new Uint32Array(faceNum * 3);
  for (let i = 0; i < faceNum * 3; i++) {
    indices[i] = faces[faceAdr * 3 + i];
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  geom.computeVertexNormals();
  return geom;
}

function primitiveGeometryFromModel(geomId) {
  const type = model.geom_type[geomId];
  const s = model.geom_size;
  const o = geomId * 3; // geom_size is 3 values per geom

  switch (type) {
    case GEOM_SPHERE:
      return new THREE.SphereGeometry(s[o], 20, 14);

    case GEOM_BOX:
      return new THREE.BoxGeometry(2 * s[o], 2 * s[o + 1], 2 * s[o + 2]);

    case GEOM_CAPSULE: {
      // MuJoCo capsule is Z-aligned; Three's is Y-aligned.
      const g = new THREE.CapsuleGeometry(s[o], 2 * s[o + 1], 8, 16);
      g.rotateX(Math.PI / 2);
      return g;
    }

    case GEOM_CYLINDER: {
      const g = new THREE.CylinderGeometry(s[o], s[o], 2 * s[o + 1], 24);
      g.rotateX(Math.PI / 2);
      return g;
    }

    case GEOM_ELLIPSOID: {
      const g = new THREE.SphereGeometry(1, 20, 14);
      g.scale(s[o], s[o + 1], s[o + 2]);
      return g;
    }

    case GEOM_PLANE:
      // The viewer draws its own ground grid/floor, so skip MuJoCo's plane.
      return null;

    default:
      return null;
  }
}

function geomVisualMaterial(geomId) {
  // Use the model material color when available, otherwise a neutral grey.
  const matId = model.geom_matid ? model.geom_matid[geomId] : -1;
  if (matId >= 0 && model.mat_rgba) {
    const c = model.mat_rgba;
    return new THREE.MeshStandardMaterial({
      color: new THREE.Color(c[matId * 4], c[matId * 4 + 1], c[matId * 4 + 2]),
      metalness: 0.15,
      roughness: 0.55,
    });
  }
  return new THREE.MeshStandardMaterial({
    color: 0x9aa0a8,
    metalness: 0.15,
    roughness: 0.55,
  });
}

function buildSceneFromModel() {
  // One group per body, parented flat to the scene and positioned from world
  // transforms each frame (avoids recomputing the kinematic chain in JS).
  bodyNodes = new Array(model.nbody);
  for (let b = 1; b < model.nbody; b++) {
    const node = new THREE.Group();
    node.name = `body_${b}`;
    scene.add(node);
    bodyNodes[b] = node;
  }

  const geomPos = model.geom_pos;
  const geomQuat = model.geom_quat;
  const geomBodyId = model.geom_bodyid;
  const geomGroup = model.geom_group;
  const geomType = model.geom_type;
  const geomDataId = model.geom_dataid;

  let built = 0;
  for (let g = 0; g < model.ngeom; g++) {
    if (geomGroup[g] === COLLISION_GROUP) continue; // skip colliders
    const bodyId = geomBodyId[g];
    if (bodyId < 1 || !bodyNodes[bodyId]) continue;

    const type = geomType[g];
    let geometry = null;
    if (type === GEOM_MESH) {
      const meshId = geomDataId[g];
      if (meshId < 0) continue;
      geometry = meshGeometryFromModel(meshId);
    } else if (type !== GEOM_HFIELD) {
      geometry = primitiveGeometryFromModel(g);
    }
    if (!geometry) continue;

    const mesh = new THREE.Mesh(geometry, geomVisualMaterial(g));
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    // Geom local transform relative to its body.
    mesh.position.set(geomPos[g * 3], geomPos[g * 3 + 1], geomPos[g * 3 + 2]);
    mesh.quaternion.set(
      geomQuat[g * 4 + 1], // x
      geomQuat[g * 4 + 2], // y
      geomQuat[g * 4 + 3], // z
      geomQuat[g * 4 + 0]  // w
    );

    bodyNodes[bodyId].add(mesh);
    built++;
  }

  return built;
}

// ---------------------------------------------------------------------------
// Per-frame sync
// ---------------------------------------------------------------------------

function syncBodies() {
  const xpos = data.xpos;
  const xquat = data.xquat;
  for (let b = 1; b < model.nbody; b++) {
    const node = bodyNodes[b];
    if (!node) continue;
    node.position.set(xpos[b * 3], xpos[b * 3 + 1], xpos[b * 3 + 2]);
    node.quaternion.set(
      xquat[b * 4 + 1],
      xquat[b * 4 + 2],
      xquat[b * 4 + 3],
      xquat[b * 4 + 0]
    );
  }
}

// ---------------------------------------------------------------------------
// RL policy (optional)
// ---------------------------------------------------------------------------

async function setupPolicy() {
  let cfg;
  try {
    const res = await fetch(`${MODEL_BASE}policy.json`);
    if (!res.ok) throw new Error(`policy.json -> HTTP ${res.status}`);
    cfg = await res.json();
  } catch (e) {
    console.warn('Sem policy.json, rodando apenas com a pose home.', e);
    return null;
  }

  const onnxUrl = `${MODEL_BASE}${cfg.model}`;
  let bytes;
  try {
    const res = await fetch(onnxUrl);
    if (!res.ok) throw new Error(`policy.onnx -> HTTP ${res.status}`);
    bytes = new Uint8Array(await res.arrayBuffer());
    // Dev/preview servers fall back to index.html for unknown paths, so a 200
    // is not proof the file exists. Reject HTML payloads.
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('text/html') || bytes[0] === 0x3c /* '<' */) {
      throw new Error('policy.onnx não encontrado (resposta HTML)');
    }
  } catch (e) {
    console.warn('Sem policy.onnx, rodando apenas com a pose home.', e);
    return null;
  }

  // Models exported with external weights (mjlab / rsl_rl) keep the tensors in
  // a sibling `.data` file. ORT needs it explicitly when we pass raw bytes.
  const externalData = [];
  const dataName = cfg.externalData || `${cfg.model}.data`;
  try {
    const res = await fetch(`${MODEL_BASE}${dataName}`);
    if (res.ok) {
      const dbytes = new Uint8Array(await res.arrayBuffer());
      const ct = res.headers.get('content-type') || '';
      if (!ct.includes('text/html') && dbytes[0] !== 0x3c /* '<' */) {
        externalData.push({ path: dataName, data: dbytes });
        console.log(`Pesos externos carregados: ${dataName} (${dbytes.length} bytes)`);
      }
    }
  } catch (e) {
    console.warn(`Sem ${dataName}; o modelo pode falhar se usar pesos externos.`, e);
  }

  const c = new PolicyController(mujoco, model, data, cfg);
  await c.load(bytes, externalData);
  console.log(`Policy carregada: obs=${c.obsDim} dims, ${c.n} ações`);
  return c;
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

function setupInput() {
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    keys[k] = true;
    if (k === 'r') resetSim();
    if (['w', 'a', 's', 'd', 'q', 'e'].includes(k)) e.preventDefault();
  });
  window.addEventListener('keyup', (e) => {
    keys[e.key.toLowerCase()] = false;
  });
  window.addEventListener('blur', () => {
    for (const k in keys) keys[k] = false;
  });
}

function updateCommandFromKeys() {
  if (!controller) return;
  const vx = (keys['w'] ? 1 : 0) - (keys['s'] ? 1 : 0);
  const vy = (keys['a'] ? 1 : 0) - (keys['d'] ? 1 : 0);
  const yaw = (keys['q'] ? 1 : 0) - (keys['e'] ? 1 : 0);
  controller.setCommand(vx * MAX_VX, vy * MAX_VY, yaw * MAX_YAW);
}

function resetSim() {
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);
  if (controller) controller.reset();
}

// ---------------------------------------------------------------------------
// Environment (sky + floor)
// ---------------------------------------------------------------------------

// Gradient sky dome. The scene is Z-up, so the gradient is driven by the
// vertical (Z) component of the view direction.
function makeSkyDome() {
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      topColor: { value: new THREE.Color(0x6285b0) },
      midColor: { value: new THREE.Color(0x93aac6) },
      bottomColor: { value: new THREE.Color(0xbfcbd8) },
    },
    vertexShader: /* glsl */ `
      varying vec3 vWorldPos;
      void main() {
        vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 topColor;
      uniform vec3 midColor;
      uniform vec3 bottomColor;
      varying vec3 vWorldPos;
      void main() {
        float h = normalize(vWorldPos).z;                       // -1 .. 1 (Z-up)
        vec3 col = mix(bottomColor, midColor, smoothstep(-0.25, 0.08, h));
        col = mix(col, topColor, smoothstep(0.05, 0.75, h));
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(40, 32, 24), material);
  sky.frustumCulled = false;
  return sky;
}

// Procedural checkerboard with thin grid lines drawn in, so the floor reads as
// a clean technical surface instead of a flat dark plane.
function makeCheckerTexture() {
  const size = 512;
  const squares = 8;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const s = size / squares;

  for (let y = 0; y < squares; y++) {
    for (let x = 0; x < squares; x++) {
      ctx.fillStyle = (x + y) % 2 === 0 ? '#4d545f' : '#404750';
      ctx.fillRect(x * s, y * s, s, s);
    }
  }

  ctx.strokeStyle = 'rgba(138, 148, 163, 0.35)';
  ctx.lineWidth = Math.max(1, size / 320);
  for (let i = 0; i <= squares; i++) {
    ctx.beginPath();
    ctx.moveTo(i * s, 0);
    ctx.lineTo(i * s, size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * s);
    ctx.lineTo(size, i * s);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return tex;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  const container = document.getElementById('canvas-container');
  panel = document.getElementById('info');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xb9c6d4); // fallback beyond the dome

  // MuJoCo is Z-up, so keep the whole scene Z-up.
  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 60);
  camera.up.set(0, 0, 1);
  camera.position.set(1.2, -1.4, 0.9);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  // Sky
  scene.add(makeSkyDome());

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.25);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.update();

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  scene.add(new THREE.HemisphereLight(0xdce9f7, 0x8a939e, 0.65));

  const key = new THREE.DirectionalLight(0xfff6e8, 1.35);
  key.position.set(2, -3, 4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -3;
  key.shadow.camera.right = 3;
  key.shadow.camera.top = 3;
  key.shadow.camera.bottom = -3;
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 15;
  key.shadow.bias = -0.0006;
  key.shadow.normalBias = 0.01;
  scene.add(key);
  scene.add(key.target);
  keyLight = key;

  const fill = new THREE.DirectionalLight(0xdfeaff, 0.4);
  fill.position.set(-3, 2, 2);
  scene.add(fill);

  // Checkerboard floor on the XY plane (Z-up). 8 squares * repeat 5 over 20 m
  // => one square every 0.5 m.
  const floorTex = makeCheckerTexture();
  floorTex.repeat.set(5, 5);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.92, metalness: 0.0 })
  );
  floor.receiveShadow = true;
  scene.add(floor);

  // --- MuJoCo ---
  const mujocoReady = loadMujoco();
  mujoco = await mujocoReady;
  console.log('Módulo MuJoCo carregado:', mujoco);

  try {
    model = await loadModel(MODEL_BASE, MODEL_XML, MODEL_ASSETS);
  } catch (e) {
    console.error('Erro ao carregar o modelo MuJoCo:', e);
    if (panel) panel.textContent = `Erro ao carregar o modelo: ${e}`;
    return;
  }

  data = new mujoco.MjData(model);

  // Pose the robot in its "home" keyframe and hold it there via position actuators.
  mujoco.mj_resetDataKeyframe(model, data, 0);
  mujoco.mj_forward(model, data);

  const geomsBuilt = buildSceneFromModel();
  syncBodies();

  statusEl = document.getElementById('policy-status');
  controller = await setupPolicy();

  if (panel) {
    panel.innerHTML =
      `<strong>Unitree Go2</strong><br>` +
      `bodies: ${model.nbody} · geoms: ${model.ngeom} · meshes visíveis: ${geomsBuilt}`;
  }
  if (statusEl) {
    statusEl.textContent = controller
      ? `Policy RL ativa · obs ${controller.obsDim} dims · ${controller.n} ações`
      : 'Sem policy.onnx — apenas pose home (R para resetar)';
  }

  setupInput();

  // Control decimation: one policy step every `decimation` physics steps.
  physicsDt = model.opt?.timestep ?? 0.002;
  const controlHz = controller ? controller.cfg.controlHz : 50;
  decimation = Math.max(1, Math.round(1 / controlHz / physicsDt));
  console.log(`timestep=${physicsDt}s, controle a ${controlHz}Hz (${decimation} passos/controle)`);

  window.addEventListener('resize', onResize);
  onResize();
  requestAnimationFrame(animate);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

let physicsDt = 0.002;
let decimation = 10;
let simAccum = 0;
let stepCount = 0;
let lastTime = performance.now();

// Keep the shadow frustum centered on the robot, otherwise it loses its shadow
// once it walks away from the origin.
function followRobotWithLight() {
  if (!keyLight) return;
  const bx = data.xpos[3];
  const by = data.xpos[4];
  const bz = data.xpos[5];
  keyLight.target.position.set(bx, by, bz);
  keyLight.position.set(bx + 2, by - 3, bz + 4);
  keyLight.target.updateMatrixWorld();
}

function animate(now) {
  requestAnimationFrame(animate);
  if (now === undefined) now = performance.now();

  let elapsed = (now - lastTime) / 1000;
  lastTime = now;
  if (!Number.isFinite(elapsed) || elapsed < 0) elapsed = 0;
  elapsed = Math.min(elapsed, 0.05); // cap so a slow frame can't spiral
  simAccum += elapsed;

  while (simAccum >= physicsDt) {
    if (controller && controller.ready && stepCount % decimation === 0) {
      updateCommandFromKeys();
      controller.apply(); // write last action into ctrl
      controller.tick();  // async inference for the next action
    }
    mujoco.mj_step(model, data);
    simAccum -= physicsDt;
    stepCount++;
  }

  syncBodies();
  followRobotWithLight();
  controls.update();
  renderer.render(scene, camera);
}

init();