import * as THREE from 'three';

const FREE_JOINT = 0; // mjJNT_FREE

/**
 * Runs a pretrained locomotion policy (ONNX) against a MuJoCo model.
 *
 * The policy is treated as a black box: we only need to know
 *  - the order of the 12 leg joints it expects (`jointOrder`),
 *  - the observation layout (`obsOrder` + scales),
 *  - how to turn its output into actuator targets (`actionScale`, `defaultJointAngles`).
 *
 * All of that lives in `policy.json`, so a different checkpoint only requires
 * editing the config, not the code.
 */
export class PolicyController {
  constructor(mujoco, model, data, config) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.cfg = config;

    this.ort = null;
    this.session = null;
    this.inputName = null;
    this.outputName = null;
    this.busy = false;
    this.ready = false;

    this.n = config.jointOrder.length;
    this.action = new Float32Array(this.n);
    this.command = new Float32Array(3); // [vx, vy, yawRate]

    // Scratch objects (avoid per-frame allocations).
    this._q = new THREE.Quaternion();
    this._qInv = new THREE.Quaternion();
    this._v = new THREE.Vector3();

    this._buildIndexMaps();
    this._buildGains();
    this._buildObsLayout();
  }

  // --- setup ---------------------------------------------------------------

  _buildIndexMaps() {
    const { mujoco, model, cfg } = this;
    const JOINT = mujoco.mjtObj.mjOBJ_JOINT.value;

    this.qposAdr = new Int32Array(this.n);
    this.dofAdr = new Int32Array(this.n);
    this.actId = new Int32Array(this.n);
    this.defaultPos = new Float32Array(this.n);

    for (let i = 0; i < this.n; i++) {
      const name = cfg.jointOrder[i];
      const jid = mujoco.mj_name2id(model, JOINT, name);
      if (jid < 0) throw new Error(`Joint não encontrado no modelo: "${name}"`);

      this.qposAdr[i] = model.jnt_qposadr[jid];
      this.dofAdr[i] = model.jnt_dofadr[jid];

      // Find the actuator that drives this joint (actuator_trnid is 2 ints each).
      let aid = -1;
      for (let a = 0; a < model.nu; a++) {
        if (model.actuator_trnid[a * 2] === jid) {
          aid = a;
          break;
        }
      }
      if (aid < 0) throw new Error(`Nenhum atuador controla o joint "${name}"`);
      this.actId[i] = aid;

      const d = cfg.defaultJointAngles[name];
      this.defaultPos[i] = d === undefined ? 0 : d;
    }

    // Locate the floating base (free joint) for the observation.
    this.freeQposAdr = 0;
    this.freeDofAdr = 0;
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] === FREE_JOINT) {
        this.freeQposAdr = model.jnt_qposadr[j];
        this.freeDofAdr = model.jnt_dofadr[j];
        break;
      }
    }
  }

  /**
   * PD gains and actuator limits.
   *
   * `controlMode: "position"` (default) treats the policy output as a joint
   * position offset and computes the torque with a PD law — this is what most
   * legged-RL policies (legged_gym / unitree_rl_gym / mjlab) are trained with.
   * `controlMode: "torque"` feeds the scaled action straight to the actuator.
   */
  _buildGains() {
    const { model, cfg } = this;
    const n = this.n;

    this.kp = new Float32Array(n);
    this.kd = new Float32Array(n);
    this.ctrlRange = new Float32Array(n);

    const ctrlrange = model.actuator_ctrlrange;
    for (let i = 0; i < n; i++) {
      const name = cfg.jointOrder[i];
      const g = cfg.jointGains ? cfg.jointGains[name] : undefined;
      this.kp[i] = g?.kp ?? (Array.isArray(cfg.kp) ? cfg.kp[i] : cfg.kp ?? 0);
      this.kd[i] = g?.kd ?? (Array.isArray(cfg.kd) ? cfg.kd[i] : cfg.kd ?? 0);

      let lim = Infinity;
      if (ctrlrange) {
        const a = this.actId[i];
        const lo = ctrlrange[a * 2];
        const hi = ctrlrange[a * 2 + 1];
        if (hi > lo) lim = Math.max(Math.abs(lo), Math.abs(hi));
      }
      this.ctrlRange[i] = lim;
    }
  }

  _buildObsLayout() {
    const { cfg } = this;
    const n = this.n;
    const s = cfg.obsScales;

    const dims = {
      angVel: 3,
      linVel: 3,
      gravity: 3,
      command: 3,
      dofPos: n,
      dofVel: n,
      actions: n,
    };

    this.segments = cfg.obsOrder.map((name) => {
      if (!(name in dims)) throw new Error(`Segmento de observação desconhecido: "${name}"`);
      return { name, dim: dims[name] };
    });

    this.obsDim = this.segments.reduce((acc, seg) => acc + seg.dim, 0);
    this.obs = new Float32Array(this.obsDim);
    this._scales = s;
  }

  /**
   * @param {string|Uint8Array} source URL or raw .onnx bytes.
   * @param {Array<{path: string, data: Uint8Array}>} [externalData] Weights
   *   stored outside the .onnx (e.g. `policy.onnx.data`). Required when the
   *   model was exported with external data and `source` is raw bytes.
   */
  async load(source, externalData) {
    // Lazy import: the ONNX runtime is only fetched when a policy is actually
    // used. The wasm-only build runs on a single thread (no SharedArrayBuffer
    // / COOP-COEP headers required). We let the bundler resolve the .wasm
    // asset (Vite rewrites the `new URL(..., import.meta.url)` inside ORT),
    // so we must NOT point wasmPaths at /public — Vite forbids importing
    // modules from there.
    const ort = await import('onnxruntime-web/wasm');
    ort.env.wasm.numThreads = 1;

    const options = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    };
    if (externalData && externalData.length) options.externalData = externalData;

    this.session = await ort.InferenceSession.create(source, options);
    this.ort = ort;
    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
    this.ready = true;
    return this;
  }

  // --- runtime -------------------------------------------------------------

  reset() {
    this.action.fill(0);
    this.command.fill(0);
  }

  setCommand(vx, vy, yawRate) {
    this.command[0] = vx;
    this.command[1] = vy;
    this.command[2] = yawRate;
  }

  /** Build the observation vector from the current MuJoCo state. */
  buildObs() {
    const { data, cfg } = this;
    const q = data.qpos;
    const qv = data.qvel;
    const n = this.n;
    const obs = this.obs;
    let o = 0;

    // Base orientation -> quaternion (MuJoCo stores w, x, y, z).
    const fq = this.freeQposAdr;
    this._q.set(q[fq + 4], q[fq + 5], q[fq + 6], q[fq + 3]);
    this._qInv.copy(this._q).invert();

    // Projected gravity in the body frame.
    this._v.set(0, 0, -1).applyQuaternion(this._qInv);
    const gx = this._v.x;
    const gy = this._v.y;
    const gz = this._v.z;

    // Angular velocity is already expressed in the body frame.
    const fd = this.freeDofAdr;
    const wx = qv[fd + 3];
    const wy = qv[fd + 4];
    const wz = qv[fd + 5];

    // Linear velocity is in the world frame -> rotate into the body frame.
    this._v.set(qv[fd], qv[fd + 1], qv[fd + 2]).applyQuaternion(this._qInv);
    const lx = this._v.x;
    const ly = this._v.y;
    const lz = this._v.z;

    for (const seg of this.segments) {
      switch (seg.name) {
        case 'angVel': {
          const k = this._scales.angVel;
          obs[o++] = wx * k;
          obs[o++] = wy * k;
          obs[o++] = wz * k;
          break;
        }
        case 'linVel': {
          const k = this._scales.linVel;
          obs[o++] = lx * k;
          obs[o++] = ly * k;
          obs[o++] = lz * k;
          break;
        }
        case 'gravity':
          obs[o++] = gx;
          obs[o++] = gy;
          obs[o++] = gz;
          break;
        case 'command':
          obs[o++] = this.command[0] * cfg.commandScales.linVel;
          obs[o++] = this.command[1] * cfg.commandScales.linVel;
          obs[o++] = this.command[2] * cfg.commandScales.angVel;
          break;
        case 'dofPos': {
          const k = this._scales.dofPos;
          for (let i = 0; i < n; i++) {
            obs[o++] = (q[this.qposAdr[i]] - this.defaultPos[i]) * k;
          }
          break;
        }
        case 'dofVel': {
          const k = this._scales.dofVel;
          for (let i = 0; i < n; i++) obs[o++] = qv[this.dofAdr[i]] * k;
          break;
        }
        case 'actions':
          for (let i = 0; i < n; i++) obs[o++] = this.action[i];
          break;
      }
    }

    return obs;
  }

  /** Write the current action into the MuJoCo actuators. */
  apply() {
    const { data, cfg } = this;
    const clip = cfg.clipActions ?? 100;
    const torqueMode = cfg.controlMode === 'torque';

    for (let i = 0; i < this.n; i++) {
      const a = Math.max(-clip, Math.min(clip, this.action[i]));
      let u;

      if (torqueMode) {
        u = a * cfg.actionScale;
      } else {
        const target = this.defaultPos[i] + a * cfg.actionScale;
        const q = data.qpos[this.qposAdr[i]];
        const qd = data.qvel[this.dofAdr[i]];
        u = this.kp[i] * (target - q) - this.kd[i] * qd;
      }

      const lim = this.ctrlRange[i];
      data.ctrl[this.actId[i]] = Math.max(-lim, Math.min(lim, u));
    }
  }

  /**
   * Kick off one inference. Async: the result lands in `this.action` and is
   * applied on the next control tick (one-tick latency, keeps the render loop
   * from stalling).
   */
  async tick() {
    if (!this.ready || this.busy) return;
    this.busy = true;
    try {
      const obs = this.buildObs();
      const tensor = new this.ort.Tensor('float32', obs, [1, this.obsDim]);
      const out = await this.session.run({ [this.inputName]: tensor });
      const data = out[this.outputName].data;
      for (let i = 0; i < this.n; i++) this.action[i] = data[i];
    } catch (e) {
      console.error('Falha na inferência da policy:', e);
      this.ready = false;
    } finally {
      this.busy = false;
    }
  }
}