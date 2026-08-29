import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';
import { STLLoader } from './STLLoader.js';
import { api, postJSON, download, isServerless, onProgress, warmUp } from './transport.js';

// ---------------------------------------------------------------- state
//
// A tab can hold more than one uploaded model now (see the `models` section
// below). Everything that used to be a single flat field on `state` — the
// points, the dial-fed params, the stale flag, the lift, the meshes — is per
// model, so it lives on a `ModelEntry` instead: `state.models` holds one per
// upload, keyed by sid, and `state.activeSid` says which one every existing
// single-model function (rotate, lift, points, supports, export…) currently
// means when it reaches for "the" model. `active()` is that lookup. Nothing
// downstream of a stage call had to learn about other models at all — it
// still only ever touches the one entry `active()` hands it.

const state = {
  workspaceId: null,   // groups every model this tab has uploaded — see core.Workspace
  models: new Map(),   // sid -> ModelEntry
  order: [],           // sids in upload order, for the model chips and the layout
  activeSid: null,      // which model every stage call currently targets
  presets: null,        // every preset the server knows, by name — shared across models
  //: The preset the dropdown is currently naming. Shared rather than
  //: per-model, the same way the printer a preset describes is shared across
  //: whatever parts get dropped into one tab: choosing "0.6mm nozzle" for one
  //: model is choosing it for the next upload too, not just for the model on
  //: screen when the dropdown was touched.
  presetName: null,
  busy: false,
  //: A fixed spacing scheme, not a bin-packing one: each new model's group is
  //: shifted just far enough clear of the one before it. Good enough to keep
  //: several test models from overlapping in the viewport; it is a client-side
  //: display offset only and is never sent to the server, which still only
  //: ever sees each model in its own local frame (see switchModel/upload).
  layoutRight: 0,
};

const LAYOUT_GAP = 15; // mm between one model's footprint and the next

//: How many hand edits ctrl-Z can walk back, per model.
const UNDO_DEPTH = 32;

/** Everything the pipeline stages produce for one uploaded model, plus the
 *  three.js objects that show it. Kept alive for as long as the model is —
 *  switching the active model never disposes or re-fetches any of this, it
 *  only changes which entry the rest of the app reads and writes. That is
 *  what makes "click a model to make it active" cheap: the model you switch
 *  back to is still exactly as you left it. */
class ModelEntry {
  constructor(sid, name, summary) {
    this.sid = sid;
    this.name = name;
    this.summary = summary;
    //: Its own group so several models can share one scene without their
    //: geometry landing on top of each other — see layoutRight above. Every
    //: mesh this model owns is a child of this group and nothing else.
    this.group = new THREE.Group();
    world.add(this.group);
    this.modelMesh = null;
    this.supportMesh = null;
    this.overhangMesh = null;   // the checkered overlay — see loadOverhang/buildOverhangOverlay
    this.markers = [];   // one InstancedMesh per flavour; .userData.at maps back to points

    this.points = [];
    this.dropped = new Set();
    this.history = [];

    this.params = null;
    this.preset = null;
    this.pendingPreset = null;
    this.size = null;
    this.overhangArea = null;
    this.volume = null;
    //: The full stage-3 response from the last successful build, kept so
    //: switching back onto this model can redraw its stats line without a
    //: fresh build — see showStats and switchModel.
    this.lastBuild = null;

    //: What this model's view no longer answers for, worst case, since its
    //: last build. Same meaning as the old single-model `state.stale`.
    this.stale = null;
    //: The lift baked into this model's mesh on screen — see refloat.
    this.lift = null;
    //: The absolute rotation last applied to this model, so switching back to
    //: it shows the pose it is actually sitting in rather than 0/0/0.
    this.rotation = { rx: 0, ry: 0, rz: 0 };
  }
}

/** The model every existing single-model function currently means by "the"
 *  model — null before anything has been uploaded. */
function active() {
  return state.activeSid ? state.models.get(state.activeSid) : null;
}

/** Every element this file reaches for is one the page is expected to have, so
 *  a miss is never a case to handle — it means the script and the markup are
 *  not the same generation. That happens for real: a browser holding a cached
 *  `app.js` against a freshly deployed `index.html` (or the reverse) runs code
 *  against a page it was never written for. Left alone it surfaces a few frames
 *  later as "Cannot read properties of null", naming neither the element nor
 *  the cause; the bundle stamps its asset URLs to stop the mismatch happening
 *  at all, and this says so plainly if one ever slips through. */
const $ = (id) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`no #${id} on the page — reload to pick up a matching app.js`);
  return el;
};
const loader = new STLLoader();

// ---------------------------------------------------------------- scene

const canvas = $('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1115);

// Everything downstream is Z-up, because that is how print beds work. Telling
// three.js the same avoids transposing coordinates on every message.
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 4000);
camera.up.set(0, 0, 1);
camera.position.set(70, -95, 65);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Turned down from 1.5 + 1.9. The old pair drove the support orange to
// near-white on every face pointing at the key, which flattened a field of
// round shafts into one bright mass — the exact thing you are trying to read.
scene.add(new THREE.HemisphereLight(0xdfe6f2, 0x1b1f26, 1.0));
const key = new THREE.DirectionalLight(0xffffff, 1.2);
key.position.set(60, -80, 120);
scene.add(key);
const rim = new THREE.DirectionalLight(0x88a8ff, 0.5);
rim.position.set(-70, 60, 30);
scene.add(rim);

const grid = new THREE.GridHelper(240, 24, 0x3a4150, 0x23272f);
grid.rotation.x = Math.PI / 2;   // GridHelper is XZ by default; we want the XY bed
scene.add(grid);

// A grid alone gives depth but no floor: with the camera low there is nothing
// to say where the plate stops. A near-black quad just under the grid lines
// reads as the bed, and a brighter cross marks the origin the model is dropped
// onto. Both scale with the grid.
const plate = new THREE.Mesh(
  new THREE.PlaneGeometry(240, 240),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.025,
                                depthWrite: false }),
);
plate.position.z = -0.05;        // under the grid lines, so they stay crisp
plate.renderOrder = -1;
scene.add(plate);

const axes = new THREE.LineSegments(
  new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(
    [-12, 0, 0, 12, 0, 0, 0, -12, 0, 0, 12, 0], 3)),
  new THREE.LineBasicMaterial({ color: 0x59616f }),
);
scene.add(axes);

const world = new THREE.Group();
scene.add(world);

const MAT = {
  // Darker than it was (0x9aa3af): the model is the ground the scaffold stands
  // against, and at the old value it was the brightest thing on screen — it
  // competed with the panel text and left the supports nowhere to go.
  model: new THREE.MeshStandardMaterial({ color: 0x7f8794, roughness: 0.72, metalness: 0.04, flatShading: false }),
  // The other models sitting in the scene while one is active — dim and a
  // little translucent, so the eye settles on the active model and the
  // scaffold on it, and the rest read as "there, and clickable" rather than
  // competing for attention.
  modelInactive: new THREE.MeshStandardMaterial({
    color: 0x565c66, roughness: 0.85, metalness: 0.0, transparent: true, opacity: 0.55,
  }),
  supports: new THREE.MeshStandardMaterial({ color: 0xf2802e, roughness: 0.55, metalness: 0.05 }),
  // Contact points come in two flavours and the difference has to be readable
  // at a glance. Held ones are washed out on purpose — there are hundreds of
  // them and they are covering the model, so they have to stay out of the way
  // of the surface underneath. Unheld ones are solid and a touch bigger: those
  // are the ones worth looking at, and there are usually only a handful.
  // Cyan, not red. Red used to mean both "a normal contact you can click off"
  // and "nothing could reach this" — two meanings a shade apart, sitting on
  // orange supports, which is why the legend needed two lines to separate them.
  // Cyan is as far from both the grey model and the orange scaffold as this
  // scene gets, and it leaves red saying only one thing.
  point: new THREE.MeshBasicMaterial({
    color: 0x35d6ff, transparent: true, opacity: 0.55, depthWrite: false,
  }),
  pointDropped: new THREE.MeshBasicMaterial({ color: 0xff3b30 }),
};

const show = { model: true, supports: true, points: false, wire: false, overhang: false };

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}
function tick() {
  resize();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

// ---------------------------------------------------------------- logging

function log(msg, cls = '') {
  const el = document.createElement('div');
  el.className = cls;
  el.textContent = msg;
  $('log').appendChild(el);
  $('log').scrollTop = 1e9;
  // The collapsed strip shows the newest line, so the panel can stay shut and
  // still be current. Expanded, the strip and the list say the same thing,
  // which is fine — it is one line of duplication for a section that is
  // usually closed.
  const tail = $('logtail');
  tail.textContent = msg;
  tail.className = cls;
  // While a run is going the generate button says which stage it is in rather
  // than a generic "working" — the stage names are already being written here,
  // and "placing support points" is a more useful thing to be told than that
  // the app is busy. Only a run: an upload or an export leaves the button
  // alone, because neither is something the button started.
  if (activeRun) setWorking(msg);
}

/** The generate button's label while it is running. The trailing ellipsis the
 *  log uses to mean "in progress" is redundant next to a spinner. */
function setWorking(msg) {
  $('genlabel').textContent = (msg || 'working').replace(/\s*…\s*$/, '');
}

/** Two flavours of wait, because they interrupt differently.
 *
 *  `heavy` washes the canvas out, and is for the times there is nothing on it
 *  worth looking at: the first load, and the Pyodide warm-up. Everything else
 *  — a build, an export — is a wait during which watching the scaffold change
 *  is the entire point, so it gets a bar under the panel header, a spinner on
 *  the generate button, and leaves the viewport alone. Dimming the stage on
 *  every rebuild read as a flicker and hid the answer. */
function busy(on, heavy = false) {
  state.busy = on;
  $('prog').classList.toggle('on', on);
  $('busy').classList.toggle('on', on && heavy);
  paintGenerate();
}

// ---------------------------------------------------------------- api

// `api`, `postJSON` and `download` come from transport.js, which is either
// fetch against the FastAPI app or postMessage to a Pyodide worker running the
// whole pipeline in this tab. Both answer the same routes, so nothing below
// this line knows or cares which is in use.

// ------------------------------------------------------------ geometry io

async function loadSTL(m, url, kind, signal) {
  const buf = await (await api(url, { signal })).arrayBuffer();
  const geom = loader.parse(buf);
  geom.computeVertexNormals();

  const mesh = new THREE.Mesh(geom, kind === 'model' ? MAT.model : MAT.supports);
  if (kind === 'model') {
    if (m.modelMesh) { m.group.remove(m.modelMesh); m.modelMesh.geometry.dispose(); }
    m.modelMesh = mesh;
    // The overlay is built from this geometry's own vertex positions (see
    // buildOverhangOverlay), so a new model mesh strands the old one — face
    // indices from the previous pose mean nothing against this one.
    clearOverhangOverlay(m);
  } else {
    if (m.supportMesh) { m.group.remove(m.supportMesh); m.supportMesh.geometry.dispose(); }
    m.supportMesh = mesh;
  }
  m.group.add(mesh);
  applyVisibility();
  // The overlay is not a build stage and has no scope of its own — it just
  // has to survive the model mesh being replaced out from under it, so it is
  // refetched here rather than left stale until somebody happens to retoggle
  // it.
  if (kind === 'model' && show.overhang) await loadOverhang(signal);
  // Same reasoning as the overlay: the gizmo is built off this mesh's own
  // bounding box (see buildGizmo), so a replaced mesh strands it exactly the
  // way a replaced pose strands the overlay — refresh it here rather than
  // leaving it pointing at geometry that no longer exists. This is also what
  // picks the gizmo back up after a rotation commit's own reload.
  if (kind === 'model' && m.sid === state.activeSid) refreshGizmo();
  return mesh;
}

function frameModel(m) {
  if (!m?.modelMesh) return;
  const box = new THREE.Box3().setFromObject(m.modelMesh);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const r = Math.max(size.length(), 10);

  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(r * 0.75, -r * 0.95, r * 0.6));
  camera.near = r / 200;
  camera.far = r * 40;
  camera.updateProjectionMatrix();
  controls.update();

  const g = Math.max(60, Math.ceil(r / 20) * 20 * 2);
  grid.scale.setScalar(g / 240);
  plate.scale.setScalar(g / 240);
  axes.scale.setScalar(Math.max(1, g / 240));
}

/** Where in the scene a freshly uploaded model's group should sit so it does
 *  not land on top of the ones already there. Purely a display offset — every
 *  point sent to the server is converted back into the model's own local
 *  frame first (see the shift-click handler below), so nothing about the
 *  pipeline ever sees it. The first model is left at its native position, so
 *  a single-model session looks exactly as it always did. */
function layoutNewModel(m) {
  const box = new THREE.Box3().setFromObject(m.modelMesh);
  if (state.order.length <= 1) {
    state.layoutRight = box.max.x;
    return;
  }
  const shift = (state.layoutRight + LAYOUT_GAP) - box.min.x;
  m.group.position.x = shift;
  state.layoutRight = box.max.x + shift;
}

function rebuildMarkers(m) {
  m.markers.forEach(mk => { m.group.remove(mk); mk.geometry.dispose(); });
  m.markers = [];
  if (!m.points.length) { applyVisibility(); return; }

  // Split by whether stage 3 managed to support the point. Two meshes rather
  // than one with per-instance colours: opacity is a property of the material,
  // and the two flavours differ in it — held contacts are washed out because
  // there are hundreds of them over the surface you are trying to see, unheld
  // ones are solid because there are usually a handful and they are the news.
  // Each mesh remembers which m.points index every instance came from, so
  // clicking one still deletes the right point.
  const held = [], unheld = [];
  m.points.forEach((_, i) => (m.dropped.has(i) ? unheld : held).push(i));

  const r = (m.params?.tip_diameter ?? 0.3) * 1.6;
  const mat4 = new THREE.Matrix4();
  for (const [idx, mat, scale] of [[held, MAT.point, 1], [unheld, MAT.pointDropped, 1.45]]) {
    if (!idx.length) continue;
    const mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(r * scale, 8, 6), mat, idx.length);
    idx.forEach((pi, i) => {
      const p = m.points[pi].position;
      mat4.makeTranslation(p[0], p[1], p[2]);
      mesh.setMatrixAt(i, mat4);
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.userData.at = idx;
    // Solid markers draw last, so a dropped point stays visible through the
    // cloud of translucent ones around it.
    mesh.renderOrder = mat === MAT.pointDropped ? 2 : 1;
    m.markers.push(mesh);
    m.group.add(mesh);
  }
  applyVisibility();
}

/** Every model's meshes are always in the scene — see ModelEntry — so this is
 *  the one place that decides what actually shows: the active model at full
 *  detail, following the view chips exactly as a single model always did, and
 *  every other uploaded model as a dim, click-to-select placeholder with its
 *  scaffold and points hidden regardless of the chips. */
function applyVisibility() {
  for (const m of state.models.values()) {
    const isActive = m.sid === state.activeSid;
    if (m.modelMesh) {
      m.modelMesh.visible = show.model;
      m.modelMesh.material = isActive ? MAT.model : MAT.modelInactive;
    }
    if (m.supportMesh) m.supportMesh.visible = isActive && show.supports;
    if (m.overhangMesh) m.overhangMesh.visible = isActive && show.overhang;
    m.markers.forEach(mk => { mk.visible = isActive && show.points; });
  }
  MAT.model.wireframe = show.wire;
}

// ------------------------------------------------------------- overhang view
//
// A read-only viewer aid, not a build stage: `rsupport.web.core.overhang_faces`
// just answers "which faces of the pose on screen right now are overhang", and
// nothing here touches DIAL_SCOPE or markStale. It is recomputed when the
// toggle is switched on and whenever the model mesh it was drawn against is
// replaced (a rotation, a lift change, a fresh upload) — never on a dial
// change, which is what would wire it into the staleness machinery this is
// deliberately kept out of.

/** mm per checker cell. A fixed viewer constant, not a support dimension —
 *  CLAUDE.md's "derive every millimetre from nozzle_diameter" rule is about
 *  geometry the generator builds, and this overlay builds nothing. */
const OVERHANG_CHECK_MM = 0.25;
const OVERHANG_YELLOW = [0.95, 0.78, 0.05];
const OVERHANG_BLACK = [0.04, 0.04, 0.04];

function clearOverhangOverlay(m) {
  if (!m || !m.overhangMesh) return;
  m.group.remove(m.overhangMesh);
  m.overhangMesh.geometry.dispose();
  m.overhangMesh.material.dispose();
  m.overhangMesh = null;
}

/** A hazard checker in the model's own object space (so it moves with that
 *  model's group like modelMesh and supportMesh do): each triangle is
 *  coloured solid (all three of its own vertices, not shared with any
 *  neighbour) by the parity of its centroid's cell on a 3D grid. Summing all
 *  three axes rather than just XY keeps the pattern reading as a checker on a
 *  steep or vertical face too, not a set of stripes running the wrong way. */
function buildOverhangOverlay(m, faceIndices) {
  clearOverhangOverlay(m);
  if (!m || !m.modelMesh || !faceIndices || !faceIndices.length) { applyVisibility(); return; }

  const src = m.modelMesh.geometry.attributes.position;
  const positions = new Float32Array(faceIndices.length * 9);
  const colors = new Float32Array(faceIndices.length * 9);

  for (let i = 0; i < faceIndices.length; i++) {
    const f = faceIndices[i];
    let cx = 0, cy = 0, cz = 0;
    for (let v = 0; v < 3; v++) {
      const si = f * 3 + v, di = (i * 3 + v) * 3;
      const x = src.getX(si), y = src.getY(si), z = src.getZ(si);
      positions[di] = x; positions[di + 1] = y; positions[di + 2] = z;
      cx += x; cy += y; cz += z;
    }
    const cell = Math.floor(cx / 3 / OVERHANG_CHECK_MM)
               + Math.floor(cy / 3 / OVERHANG_CHECK_MM)
               + Math.floor(cz / 3 / OVERHANG_CHECK_MM);
    const c = (cell & 1) ? OVERHANG_YELLOW : OVERHANG_BLACK;
    for (let v = 0; v < 3; v++) {
      const di = (i * 3 + v) * 3;
      colors[di] = c[0]; colors[di + 1] = c[1]; colors[di + 2] = c[2];
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geom.computeVertexNormals();
  const mat = new THREE.MeshBasicMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    // Coplanar with the model surface it is drawn over, so without this the
    // two fight for the same pixels and flicker between them.
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
  });
  m.overhangMesh = new THREE.Mesh(geom, mat);
  m.overhangMesh.position.copy(m.modelMesh.position);
  m.group.add(m.overhangMesh);
  applyVisibility();
}

/** Fetch the overhang faces for the active model's pose on screen and rebuild
 *  its overlay from them. Not part of any run: it is its own request, made
 *  only when the toggle asks for one, or when the active model switches
 *  while the toggle is already on. */
async function loadOverhang(signal) {
  const m = active();
  if (!m || !m.modelMesh) return;
  try {
    const angle = +$('overhang').value;
    const r = await (await api(`/api/overhang/${m.sid}?angle_deg=${angle}`, { signal })).json();
    buildOverhangOverlay(m, r.faces);
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    log(`could not draw the overhang overlay: ${err.message}`, 'e');
    show.overhang = false;
    document.querySelector('[data-toggle="overhang"]')?.classList.remove('on');
    applyVisibility();
  }
}

/** The overhang chip's own click handler — everything else on `[data-toggle]`
 *  just flips a visibility flag, but this one has to fetch the first time it
 *  is switched on. */
async function toggleOverhang(btn) {
  const m = active();
  if (!m || state.busy) return;
  show.overhang = !show.overhang;
  btn.classList.toggle('on', show.overhang);
  if (show.overhang && !m.overhangMesh) await loadOverhang();
  else applyVisibility();
}

// ---------------------------------------------------------------- params

function overrides() {
  return {
    tip_diameter: +$('tip').value,
    shaft_lower_diameter: +$('shaft').value,
    support_spacing: +$('spacing').value,
    overhang_angle_deg: +$('overhang').value,
    tip_style: $('tipstyle').value,
    brace_enabled: $('braces').checked,
    brace_diameter: +$('bracethick').value,
    brace_max_span: +$('bracespan').value,
    brace_angle_deg: +$('braceangle').value,
    brace_headroom: +$('braceheadroom').value,
    brace_interval: +$('bracespacing').value,
    brace_start_height: +$('bracestart').value,
    plate_only: $('plateonly').checked,
    strut_lean_deg: +$('lean').value,
    parenting: +$('parenting').value,
    lift_height: +$('lift').value,
    foot_diameter: +$('base').value,
    foot_height: +$('baseh').value,
    join_cone_diameter: +$('cone').value,
    join_cone_height: +$('coneh').value,
  };
}

function syncSliders(p) {
  if (!p) return;
  for (const [id, key] of [['tip', 'tip_diameter'], ['shaft', 'shaft_lower_diameter'],
                           ['spacing', 'support_spacing'], ['overhang', 'overhang_angle_deg'],
                           ['lift', 'lift_height'], ['base', 'foot_diameter'],
                           ['baseh', 'foot_height'], ['cone', 'join_cone_diameter'],
                           ['coneh', 'join_cone_height'], ['bracethick', 'brace_diameter'],
                           ['bracespan', 'brace_max_span'],
                           ['braceheadroom', 'brace_headroom'],
                           ['bracespacing', 'brace_interval'],
                           ['bracestart', 'brace_start_height']]) {
    if (p[key] != null) $(id).value = p[key];
  }
  // The link angle only exists inside the band that prints — 90 − limit up to
  // the limit itself — so the slider is given that range rather than a fixed
  // one, and the server reports the angle actually in force.
  if (p.printable_overhang_deg != null) {
    $('braceangle').min = Math.round(90 - p.printable_overhang_deg);
    $('braceangle').max = Math.round(p.printable_overhang_deg);
    mirrorRange('braceangle');
  }
  if (p.brace_angle_deg != null) $('braceangle').value = p.brace_angle_deg;
  $('tipstyle').value = p.tip_style;
  $('braces').checked = p.brace_enabled;
  if (p.plate_only != null) $('plateonly').checked = p.plate_only;
  if (p.strut_lean_deg != null) $('lean').value = p.strut_lean_deg;
  if (p.parenting != null) $('parenting').value = p.parenting;
  showSliderValues();
  markDrift();
}

/** Put the current model's last-applied rotation into the readout boxes —
 *  used on upload (0/0/0) and on switching the active model (whatever it was
 *  left turned to). The gizmo itself is refreshed separately (see
 *  refreshGizmo below) since it also depends on which model is active and
 *  whether rotation mode is even on. */
function syncRotation(m) {
  $('rotx').value = m.rotation.rx;
  $('roty').value = m.rotation.ry;
  $('rotz').value = m.rotation.rz;
  showRotationValues();
}

//
// Every slider has a number box beside it holding the same value, so a value
// can be typed as well as dragged. The slider stays the one thing the rest of
// the app reads; the box only ever writes into it.
//
const DECIMALS = {
  tip: 2, shaft: 1, spacing: 2, overhang: 0, lean: 0, parenting: 2, lift: 1,
  base: 1, baseh: 1, cone: 1, coneh: 1,
  bracethick: 2, bracespan: 1, braceangle: 0, braceheadroom: 1,
  bracespacing: 1, bracestart: 1,
  rotx: 0, roty: 0, rotz: 0,
};
const VALUE_IDS = ['tip', 'shaft', 'spacing', 'overhang', 'lean', 'parenting', 'lift',
                   'base', 'baseh', 'cone', 'coneh', 'bracethick', 'bracespan',
                   'braceangle', 'braceheadroom', 'bracespacing', 'bracestart'];
const ROTATION_IDS = ['rotx', 'roty', 'rotz'];

/** Put the slider's value in its box — unless the box is the thing being typed
 *  in, in which case leave what is half-typed alone. `was` rides along as the
 *  last value the app actually acted on, which is what a typed value has to
 *  differ from to be worth re-running. */
function showValue(id) {
  const box = $(id + '_v');
  if (box === document.activeElement) return;
  box.value = (+$(id).value).toFixed(DECIMALS[id]);
  box.dataset.was = $(id).value;
}

function showSliderValues() { VALUE_IDS.forEach(showValue); }
function showRotationValues() { ROTATION_IDS.forEach(showValue); }

/** The box borrows the slider's range so typing past either end is pulled back
 *  in, and so the arrow keys step by the same amount the slider does. */
function mirrorRange(id) {
  const s = $(id), box = $(id + '_v');
  box.min = s.min;
  box.max = s.max;
  box.step = s.dataset.step;
}

/** Take what was typed: clamp it into range, round it to the digits the box
 *  shows, hand it to the slider, and re-run whatever a dragged slider re-runs.
 *  A typed value is honoured exactly even between the slider's notches — the
 *  notches come back the moment the slider itself is touched again. */
function commitTyped(id) {
  const s = $(id), box = $(id + '_v');
  // The slider has been following along keystroke by keystroke, so it is no
  // use as a before-and-after — compare against the last value the app ran on.
  const before = box.dataset.was != null ? +box.dataset.was : +s.value;
  const typed = parseFloat(box.value);
  if (Number.isFinite(typed)) {
    s.step = 'any';
    s.value = +Math.min(Math.max(typed, +s.min), +s.max).toFixed(DECIMALS[id]);
  }
  box.value = (+s.value).toFixed(DECIMALS[id]);  // gibberish just reverts
  box.dataset.was = s.value;
  if (+s.value !== before) s.dispatchEvent(new Event('change'));
}

function bindValueBox(id) {
  const s = $(id), box = $(id + '_v');
  s.dataset.step = s.step;
  mirrorRange(id);
  // Touching the slider itself puts it back on its own increments.
  ['pointerdown', 'keydown'].forEach(ev => s.addEventListener(ev, () => { s.step = s.dataset.step; }));
  s.addEventListener('input', () => showValue(id));
  s.addEventListener('change', () => { box.dataset.was = s.value; });
  // While typing, the slider follows along — but nothing re-runs until commit.
  box.addEventListener('input', () => {
    const v = parseFloat(box.value);
    if (Number.isFinite(v) && v >= +s.min && v <= +s.max) { s.step = 'any'; s.value = v; }
  });
  box.addEventListener('change', () => commitTyped(id));
  box.addEventListener('blur', () => commitTyped(id));
  box.addEventListener('keydown', (e) => { if (e.key === 'Enter') box.blur(); });
  box.addEventListener('focus', () => box.select());
  showValue(id);
}

VALUE_IDS.concat(ROTATION_IDS).forEach(bindValueBox);

// -------------------------------------------------------------- gizmo mode
//
// Bambu-Studio-style: rotation and movement are viewport gizmos, not sidebar
// dials. `rotx`/`roty`/`rotz` are still exactly the three absolute-angle
// values every rotation function already read and wrote (`bindValueBox`
// above wires their number boxes up); dragging a ring just moves the same
// values a dragged wheel used to, and still applies on release through the
// same `rerotate`, still bound to the range's `change` event below. Movement
// has no value store at all — it is a pure client-side rearrangement of
// `ModelEntry.group.position.x/y`, the same field `layoutNewModel` already
// uses to keep freshly uploaded models apart (see its own comment for why
// that is never sent to the server). The pointer handling for both lives
// down in the interaction section, alongside the raycaster it shares with
// click-to-select; this section only builds/tears down the gizmo mesh and
// keeps it in step with which model is active.
//
// Colours: violet/lime/indigo rather than the traditional red/green/blue —
// red is already this app's "failure" (a dropped contact) and green reads as
// "success" in the log, so reusing either on a gizmo ring sitting in the same
// viewport would spend a hue this app has already promised to one meaning.
// See the mirrored --c-gizmo-* vars in index.html for the exact hex.
const GIZMO_COLOR = { x: 0xa06cff, y: 0xaee34a, z: 0x5a86ff };

let gizmoMode = null;      // null | 'rotate' | 'move'
let gizmoGroup = null;     // THREE.Group of the three rings, child of the active model's group
let gizmoRings = null;     // { x, y, z } meshes, keyed the same way ringDrag.axis is
let gizmoHover = null;     // which ring the pointer is over right now, or null

function disposeGizmo() {
  if (!gizmoGroup) return;
  gizmoGroup.parent?.remove(gizmoGroup);
  gizmoGroup.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
  gizmoGroup = null;
  gizmoRings = null;
  gizmoHover = null;
}

/** Build the three rings on `m`, sized off its own bounding box so the gizmo
 *  is usable whether the model is 8mm or 300mm — and centred on that box, not
 *  on the group origin, since a tall model's natural handle is at its own
 *  middle. Added as a child of `m.group` so it inherits the same layout/move
 *  offset the model mesh does, and moves with a manual drag for free. */
function buildGizmo(m) {
  disposeGizmo();
  if (!m || !m.modelMesh) return;

  const geom = m.modelMesh.geometry;
  geom.computeBoundingBox();
  const box = geom.boundingBox;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const radius = Math.max(Math.max(size.x, size.y, size.z) * 0.65, 5);
  const tube = Math.max(radius * 0.035, 0.15);

  gizmoGroup = new THREE.Group();
  gizmoGroup.position.copy(center);
  gizmoGroup.renderOrder = 999;
  m.group.add(gizmoGroup);

  gizmoRings = {};
  // Each ring's hole axis is the axis it turns the model about. A torus is
  // built with its hole along Z, so X and Y need a quarter turn to stand it
  // up in the other two planes; Z is already right.
  const defs = [
    ['x', GIZMO_COLOR.x, (o) => { o.rotation.y = Math.PI / 2; }],
    ['y', GIZMO_COLOR.y, (o) => { o.rotation.x = -Math.PI / 2; }],
    ['z', GIZMO_COLOR.z, () => {}],
  ];
  for (const [axis, color, orient] of defs) {
    const mesh = new THREE.Mesh(
      new THREE.TorusGeometry(radius, tube, 12, 64),
      // depthTest off (and a high renderOrder) so the ring stays grabbable
      // and visible even where it passes through the model, the same reason
      // Bambu's own gizmo draws in front of the object it is turning.
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .8, depthTest: false, depthWrite: false }),
    );
    orient(mesh);
    mesh.renderOrder = 999;
    mesh.userData.axis = axis;
    gizmoGroup.add(mesh);
    gizmoRings[axis] = mesh;
  }
}

/** Show or hide the little hover highlight on a ring — the only feedback
 *  besides the cursor that a drag is about to grab this particular axis. */
function setGizmoHover(axis) {
  if (axis === gizmoHover) return;
  if (gizmoHover && gizmoRings?.[gizmoHover]) gizmoRings[gizmoHover].material.opacity = .8;
  gizmoHover = axis;
  if (gizmoHover && gizmoRings?.[gizmoHover]) gizmoRings[gizmoHover].material.opacity = 1;
}

/** Keep the gizmo (and its readout panel) matching whichever model is active
 *  and whether rotation mode is even on. Called after every model-mesh
 *  reload (upload, a rotation commit, a refloat — see loadSTL) and every
 *  time the active model or the mode itself changes, since none of those
 *  reliably fire the other two. */
function refreshGizmo() {
  const m = active();
  if (gizmoMode === 'rotate' && m?.modelMesh) buildGizmo(m); else disposeGizmo();
  $('gizmoPanel').hidden = !(gizmoMode === 'rotate' && m);
}

/** The two top-left mode buttons — mutually exclusive, Bambu-style: turning
 *  one on turns the other off. Neither is a `show[]` visibility flag like the
 *  other viewport chips (see the generic `[data-toggle]` handler below) since
 *  each has real setup/teardown to do, the same reason `overhang` keeps its
 *  own handler instead of using that generic one. */
function setGizmoMode(next) {
  if (ringDrag) { controls.enabled = true; ringDrag = null; }
  if (moveDrag) { controls.enabled = true; moveDrag = null; }
  gizmoMode = gizmoMode === next ? null : next;
  document.querySelector('[data-toggle="rotatemode"]').classList.toggle('on', gizmoMode === 'rotate');
  document.querySelector('[data-toggle="movemode"]').classList.toggle('on', gizmoMode === 'move');
  refreshGizmo();
}

// ---------------------------------------------------------------- pipeline
//
// Nothing here starts on its own. A dial only records that what is on screen no
// longer answers to it — see markStale — and the generate button is the one
// thing that runs a stage. Dragging six sliders used to cost six rebuilds, five
// of which nobody wanted to see, and the sixth arrived only after the other
// five had been waited out.

/** How far back a build has to start. Ordered cheapest first: markStale keeps
 *  the worst of what has piled up since the last one, so a spacing change on
 *  top of a hand-deleted point still re-places the points. */
const SCOPES = ['geometry', 'points'];

function markStale(scope) {
  const m = active();
  if (!m) return;
  if (m.stale === null || SCOPES.indexOf(scope) > SCOPES.indexOf(m.stale)) {
    m.stale = scope;
  }
  paintGenerate();
}

/** The list of model chips in the panel, and the workspace header. Hidden
 *  until a second model exists — with only one, a strip of one chip is noise
 *  the sidebar already covers with the filename readout. */
function renderModelChips() {
  const box = $('modelChips');
  box.innerHTML = '';
  $('modelsSection').hidden = state.order.length < 2;
  for (const sid of state.order) {
    const m = state.models.get(sid);
    if (!m) continue;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chip' + (sid === state.activeSid ? ' on' : '');
    btn.textContent = m.name || 'model';
    btn.title = `switch to ${m.name || 'this model'}`;
    btn.addEventListener('click', () => switchModel(sid));
    box.appendChild(btn);
  }
}

function filenameText(m) {
  return `${m.name} — ${m.summary.faces.toLocaleString()} faces, ` +
    m.summary.size.map(v => v.toFixed(1)).join(' × ') + ' mm';
}

/** Make a different already-uploaded model the active one.
 *
 *  Every pipeline function still only ever knows about "the" active model —
 *  this is the one place that changes which model that is. Nothing about the
 *  model being switched to or away from is rebuilt or re-fetched: each one's
 *  meshes, points and dials have been sitting untouched in its own ModelEntry
 *  since the last time it was active, so switching back to it shows exactly
 *  what was left there. */
async function switchModel(sid) {
  if (sid === state.activeSid || state.busy) return;
  const m = state.models.get(sid);
  if (!m) return;
  await stopRun();

  state.activeSid = sid;
  $('filename').textContent = filenameText(m);
  $('asloaded').disabled = false;
  syncRotation(m);
  if (m.params) syncSliders(m.params);
  markDrift();
  frameModel(m);
  applyVisibility();
  // The overlay is per-model and built lazily, so a model that has never had
  // it toggled on while active doesn't have one yet even though the toggle
  // itself is still showing "on" from whichever model set it there.
  if (show.overhang && !m.overhangMesh) await loadOverhang();
  // The gizmo, if rotation mode is on, is a child of whichever model's group
  // it was last built for — switching the active model has to move it (or
  // hide it, if the new active model has no mesh yet).
  refreshGizmo();
  rebuildMarkers(m);
  $('resetpoints').disabled = !m.history.length;
  if (m.lastBuild) showStats(m, m.lastBuild); else $('stats').innerHTML = '&mdash;';
  $('download').disabled = !m.supportMesh;
  paintGenerate();
  renderModelChips();

  // Bookkeeping only: every other route already names its model explicitly
  // by sid, so this changes nothing about the pipeline — it just keeps the
  // server's own idea of "active" (used only for a future workspace view)
  // in step with the one the UI is now acting on.
  if (state.workspaceId) {
    postJSON(`/api/workspace/${state.workspaceId}/active`, { sid }).catch(() => { /* cosmetic */ });
  }
}

async function upload(file) {
  await stopRun();
  busy(true, true);
  let m;
  try {
    $('log').innerHTML = '';
    log(`reading ${file.name} …`);

    if (!state.workspaceId) {
      state.workspaceId = (await (await api('/api/workspace', { method: 'POST' })).json()).id;
    }

    const fd = new FormData();
    fd.append('file', file);
    const ws = await (await api(`/api/workspace/${state.workspaceId}/model`, {
      method: 'POST', body: fd,
    })).json();

    const sid = ws.active;
    const info = ws.models.find((x) => x.id === sid);
    m = new ModelEntry(sid, info.name, info.summary);
    // Every fresh session starts on the server's own default preset (see
    // core.load_model), so a model's baseline has to be told the same name
    // the panel is currently showing — otherwise its own first build would
    // quietly pick up the server's default nozzle and clearances instead of
    // whatever preset this tab has actually settled on.
    m.preset = state.presets?.[state.presetName] ?? null;
    m.pendingPreset = state.presetName;
    state.models.set(sid, m);
    state.order.push(sid);
    state.activeSid = sid;

    $('filename').textContent = filenameText(m);
    if (!m.summary.watertight) log('mesh is not watertight; results may be rough', 'w');
    $('drop').classList.add('hide');
    $('asloaded').disabled = false;
    syncRotation(m);

    // The file is taken as posed. It arrives already dropped onto the bed, so
    // there is nothing to decide.
    await loadSTL(m, `/api/mesh/${sid}/model`, 'model');
    layoutNewModel(m);
    frameModel(m);
    applyVisibility();
    renderModelChips();
    log('using the pose from the file', 'g');
  } catch (err) {
    log(`error: ${err.message}`, 'e');
    busy(false);
    return;
  }
  busy(false);

  // The one build nobody has to ask for. There is nothing shown for this
  // model yet, so there is nothing for a press to be a decision about — and
  // a tool whose whole output is visual should not open on an empty stage.
  // Every change after this one waits for the button.
  markStale('points');
  await generate();
}

/** Drop this model's scaffold and contact points, and everything else that
 *  was only true because of them. They belong to a pose that no longer
 *  exists, so leaving them up while the next build is asked for would be
 *  showing the answer to a question nobody is still asking.
 *
 *  The download buttons go with them, and that is not tidiness: re-posing the
 *  model clears the session's supports server-side, so an export taken in this
 *  window is a model and an empty scaffold, written out without complaint. */
function clearBuild(m) {
  if (m.supportMesh) {
    m.group.remove(m.supportMesh);
    m.supportMesh.geometry.dispose();
    m.supportMesh = null;
  }
  m.points = [];
  m.dropped = new Set();
  m.history = [];
  m.volume = null;
  // Otherwise switching onto this model before it is rebuilt would redraw the
  // stats line from a build that belongs to the pose just left behind — see
  // switchModel, which falls back to this whenever there is nothing fresher.
  m.lastBuild = null;
  if (m.sid === state.activeSid) {
    $('resetpoints').disabled = true;
    $('stats').innerHTML = '&mdash;';
    $('download').disabled = true;
  }
  rebuildMarkers(m);
}

/** Rotation is absolute, always applied from the file's own pose — so
 *  re-running with all three at 0 is exactly the file's pose. */
async function runRotate(m) {
  const rx = +$('rotx').value, ry = +$('roty').value, rz = +$('rotz').value;
  log('rotating the model …');
  const r = await postJSON(`/api/rotate/${m.sid}`, { rx, ry, rz, overrides: overrides() });
  m.rotation = { rx, ry, rz };
  await loadSTL(m, `/api/mesh/${m.sid}/model`, 'model');
  m.lift = +$('lift').value;
  frameModel(m);
  log(`rotated (${r.elapsed.toFixed(2)}s)`, 'g');
}

/** Turning the model is not a build, and it does not wait for the button.
 *
 *  Posing is done by eye — turn it, look, turn it back — and a rotation
 *  control that moves nothing until a second control is pressed cannot be
 *  used that way at all. So the pose is applied at once (the wheel commits on
 *  release, same as the slider it replaced committed on drag-end) and only
 *  the scaffold is left out of date. What was already built came off the old
 *  pose and would now be standing through the model rather than under it, so
 *  it goes rather than hanging about looking like a result. */
async function rerotate() {
  const m = active();
  if (!m || state.busy) return;
  busy(true);
  try {
    clearBuild(m);
    await runRotate(m);
    markStale('points');
  } catch (err) {
    log(`error: ${err.message}`, 'e');
  } finally {
    busy(false);
  }
}

ROTATION_IDS.forEach(id => $(id).addEventListener('change', rerotate));

$('asloaded').onclick = () => {
  $('rotx').value = 0; $('roty').value = 0; $('rotz').value = 0;
  showRotationValues();
  rerotate();
};

/** Take a stage-2 result, whoever asked for it. */
function absorbPoints(m, r) {
  m.points = r.points;
  m.dropped = new Set();
  // A fresh placement is not something ctrl-Z can walk back into: the edits in
  // the stack belong to a point list that no longer exists.
  m.history = [];
  m.size = r.size || null;
  m.overhangArea = r.overhang_area ?? null;
  if (m.sid === state.activeSid) $('resetpoints').disabled = true;
  rebuildMarkers(m);
  const forced = m.points.filter(p => p.forced).length;
  log(`${m.points.length} points (${forced} mandatory) in ${r.elapsed.toFixed(2)}s`, 'g');
}

async function runPoints(m, run) {
  log('placing support points …');
  const body = { overrides: overrides() };
  // A preset is a base the dials sit on top of, not a set of dial values: the
  // nozzle, the clearances and the printable limit come with it and have no
  // slider anywhere in the panel. Sending the name is the only way stage 2
  // hears about those.
  if (m.pendingPreset) body.preset = m.pendingPreset;
  const r = await postJSON(`/api/points/${m.sid}`, body, run?.ac.signal);
  halt(run);
  // Only now is the preset spent. A stopped run has not applied it, so it has
  // to still be owed to the next one.
  m.pendingPreset = null;
  absorbPoints(m, r);
}

/** Stage 2 is what floats the model, so a lift that has moved leaves the mesh
 *  on screen at the old height — either at the wrong Z outright, or carrying
 *  the local offset previewLift put on it. It is the largest thing on the wire,
 *  so it is fetched again only when one of those is actually true. */
async function refloat(m, run) {
  if (m.lift === +$('lift').value && m.modelMesh && m.modelMesh.position.z === 0) return;
  await loadSTL(m, `/api/mesh/${m.sid}/model`, 'model', run?.ac.signal);
  m.lift = +$('lift').value;
}

async function runSupports(m, run) {
  log('building support geometry …');
  const r = await postJSON(`/api/supports/${m.sid}`, {
    overrides: overrides(),
    points: m.points,
  }, run?.ac.signal);
  halt(run);
  m.params = r.params;
  m.dropped = new Set(r.dropped_points || []);
  syncSliders(r.params);
  await loadSTL(m, `/api/mesh/${m.sid}/supports`, 'supports', run?.ac.signal).catch((err) => {
    // A model needing no supports at all answers 404 here, which is not news.
    // A stopped run answers with an abort, which is — it must not be swallowed
    // into "no supports needed", or stopping would look like an empty result.
    if (err.name === 'AbortError') throw err;
    log('no supports needed', 'g');
  });
  rebuildMarkers(m);

  m.volume = r.volume ?? null;
  m.lastBuild = r;
  showStats(m, r);
  (r.warnings || []).slice(0, 5).forEach(w => log(w, 'w'));
  log(`built in ${r.elapsed.toFixed(2)}s`, 'g');

  if (m.sid === state.activeSid) {
    $('download').disabled = false;
  }
}

//: g/cm^3. PLA, because that is what an FDM scaffold gets printed in more often
//: than not, and a number an order of magnitude out is worse than no number.
const DENSITY = 1.24;

/** What the run cost, in the terms somebody deciding whether to re-pose the
 *  model actually thinks in.
 *
 *  Overhang area is the one that earns its place: it is the area the scaffold
 *  has to reach, it is unaffected by any dial in the panel, and turning the
 *  model is the only thing that meaningfully shrinks it. The mass is deliberately
 *  hedged — the scaffold is exported as overlapping closed shells, so its signed
 *  volume counts every junction twice and the true figure is somewhat under. */
function showStats(m, r) {
  const bits = [`<b>${r.points}</b> supports &middot; <b>${r.braces}</b> links`,
                `<b>${r.faces.toLocaleString()}</b> triangles`];

  if (m.size) {
    bits.push(`<b>${m.size.map(v => v.toFixed(1)).join(' × ')}</b> mm`);
  }
  if (m.overhangArea != null) {
    bits.push(`<b>${(m.overhangArea / 100).toFixed(1)}</b> cm&sup2; overhang`);
  }
  if (m.volume) {
    const grams = (m.volume / 1000) * DENSITY;
    bits.push(`under <b>${grams.toFixed(1)}</b> g of support`);
  }
  if (r.dropped) {
    bits.push(`<span style="color:var(--err)"><b>${r.dropped}</b> unsupported ` +
              `&mdash; shown in solid red</span>`);
  }
  if (m.sid === state.activeSid) $('stats').innerHTML = bits.join('<br>');
}

/** The lift is a translation in Z and nothing else, so the mesh on screen can
 *  follow the slider without asking anybody — otherwise dragging it would do
 *  nothing visible at all until the button was pressed, which reads as broken.
 *  The scaffold deliberately stays put: it no longer reaches the model, which
 *  is exactly what out of date looks like. The next build fetches the properly
 *  floated mesh back and the offset goes with it — see refloat. */
function previewLift() {
  const m = active();
  if (m?.modelMesh && m.lift !== null) {
    m.modelMesh.position.z = +$('lift').value - m.lift;
    // The overlay is drawn in the model's own object space and has no lift
    // logic of its own — it just has to ride along, or it visibly separates
    // from the surface it is supposed to be painted on.
    if (m.overhangMesh) m.overhangMesh.position.z = m.modelMesh.position.z;
  }
}

// ---------------------------------------------------------------- runs
//
// One build at a time, started and stopped by the same button.
//
// A run is a token rather than a flag, so the press that stops it and the code
// that notices are not the same piece of state: `stop` is set by the second
// press, `ac` aborts whatever request is in flight, and `halt` is where the run
// actually ends. Both halves are load-bearing. The abort is what makes stopping
// immediate on the served build; the checks are all there is under Pyodide,
// where a request is a synchronous call into a single-threaded interpreter and
// cannot be interrupted at all.
//
// Which is why `halt` sits after every result as well as before every stage. A
// stop pressed during the *last* stage has no next stage to be refused, so
// without the check after the request the run would finish and show its answer
// as though nothing had been pressed. A stopped run applies nothing: what was
// on screen stays there, and the button goes back to saying it is out of date.
//
// So stopping the serverless build waits out the stage already running, and the
// button says `stopping` until it does. That is honest, and no slower than
// pretending otherwise would be: the worker is busy either way, and freeing the
// button early would only let a second run queue up behind the first.

class Stopped extends Error {}

let activeRun = null; // the run in flight, or null
let running = null;   // its promise, for anyone who has to wait for it to let go

function halt(run) { if (run?.stop) throw new Stopped(); }

/** The button. A press builds; a press while it is building stops the build. */
function generate() {
  if (activeRun) return stopRun();
  const m = active();
  if (!m || state.busy) return Promise.resolve();

  // A press with nothing out of date rebuilds the scaffold anyway. It is a
  // second or two, it is what was asked for, and a button that ignores a press
  // is indistinguishable from a broken one.
  const scope = m.stale ?? 'geometry';
  activeRun = { stop: false, ac: new AbortController() };
  running = runAll(m, scope, activeRun);
  return running;
}

async function runAll(m, scope, run) {
  m.stale = null;
  busy(true);
  setWorking('working');
  try {
    if (scope !== 'geometry') {
      await runPoints(m, run);
      halt(run);
      await refloat(m, run);
      halt(run);
    }
    await runSupports(m, run);
  } catch (err) {
    // Whatever did not finish leaves the view where it already was: out of
    // date, and now saying so again.
    m.stale = scope;
    if (err instanceof Stopped || err.name === 'AbortError') log('stopped', 'w');
    else log(`error: ${err.message}`, 'e');
  } finally {
    activeRun = null;
    running = null;
    busy(false);
  }
}

/** Ask the run in flight to stop, and hand back something to wait on. It is not
 *  over when this returns — see the note above — so the button goes on saying
 *  what it is doing until it is. */
function stopRun() {
  if (!activeRun) return Promise.resolve();
  if (!activeRun.stop) {
    activeRun.stop = true;
    activeRun.ac.abort();
    log('stopping …', 'w');
    setWorking('stopping');
    paintGenerate();
  }
  return running ?? Promise.resolve();
}

const TIP_IDLE = 'Generate the supports using this button';
const TIP_RUNNING = 'click to stop support generating';
const TIP_STOPPING = 'stopping — the stage already running has to finish';

/** Everything the button says about itself, in one place: whether it can be
 *  pressed, whether the view a press would replace is out of date, and which of
 *  the two things a press would do. */
function paintGenerate() {
  const b = $('generate');
  const m = active();
  b.disabled = !m || (state.busy && !activeRun) || !!activeRun?.stop;
  b.classList.toggle('running', !!activeRun);
  b.classList.toggle('stale', !activeRun && !!m && m.stale !== null);
  if (!activeRun) $('genlabel').textContent = 'generate';
  $('gentip').textContent = activeRun ? (activeRun.stop ? TIP_STOPPING : TIP_RUNNING) : TIP_IDLE;
}

$('generate').addEventListener('click', generate);

// ------------------------------------------------------------------ panel

//
// The sections of dials fold away, and which ones are open is remembered — the
// panel you left is the panel you come back to. A first visit gets them all
// shut, which is the whole point: ten headings you can read at a glance rather
// than a hundred-row scroll.
//
const FOLDS_KEY = 'rsupport.folds';

function readFolds() {
  try { return new Set(JSON.parse(localStorage.getItem(FOLDS_KEY) || '[]')); }
  catch { return new Set(); }
}

const openFolds = readFolds();
document.querySelectorAll('details.fold').forEach((d) => {
  d.open = openFolds.has(d.dataset.fold);
  d.addEventListener('toggle', () => {
    if (d.open) openFolds.add(d.dataset.fold); else openFolds.delete(d.dataset.fold);
    try { localStorage.setItem(FOLDS_KEY, JSON.stringify([...openFolds])); } catch { /* private mode */ }
    hideHelp();
  });
});

//
// Each dial explains itself on hover. The bubble is placed out over the canvas,
// clear of the panel, because the description is no use if it covers the thing
// being described — you have to see what you are dragging while you read what
// it does. Only when there is no canvas to the left (the stacked layout) does
// it go above or below the row instead, still never on top of it.
//
const help = document.createElement('div');
help.id = 'help';
document.body.appendChild(help);

let helpRow = null;

function hideHelp() {
  helpRow = null;
  help.classList.remove('on');
}

function showHelp(row) {
  if (row === helpRow) return;
  helpRow = row;
  help.innerHTML = row.dataset.help;

  const GAP = 10;
  const r = row.getBoundingClientRect();
  const panel = $('panel').getBoundingClientRect();

  // Measure where it will not be clipped, then place it.
  help.className = 'on';
  help.style.left = '0px';
  help.style.top = '0px';
  const w = help.offsetWidth, h = help.offsetHeight;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));

  if (panel.left >= w + GAP * 2) {
    const top = clamp(r.top + r.height / 2 - h / 2, GAP, innerHeight - h - GAP);
    help.classList.add('at-left');
    help.style.left = `${panel.left - w - GAP}px`;
    help.style.top = `${top}px`;
    help.style.setProperty('--point', `${clamp(r.top + r.height / 2 - top, 10, h - 10)}px`);
  } else {
    const above = r.top - h - GAP >= GAP;
    const left = clamp(r.left, GAP, innerWidth - w - GAP);
    help.classList.add(above ? 'at-top' : 'at-bottom');
    help.style.left = `${left}px`;
    help.style.top = `${above ? r.top - h - GAP : r.bottom + GAP}px`;
    help.style.setProperty('--point', `${clamp(r.left + 26 - left, 12, w - 12)}px`);
  }
}

const panelEl = $('panel');
panelEl.addEventListener('pointerover', (e) => {
  const row = e.target.closest('.row[data-help]');
  if (row) showHelp(row); else hideHelp();
});
panelEl.addEventListener('pointerleave', hideHelp);
// Tabbing through the panel is the same journey without a mouse.
panelEl.addEventListener('focusin', (e) => {
  const row = e.target.closest('.row[data-help]');
  if (row) showHelp(row);
});
panelEl.addEventListener('focusout', (e) => {
  if (!e.relatedTarget || !e.relatedTarget.closest('.row[data-help]')) hideHelp();
});
// A bubble pinned to a row that has since scrolled away is pointing at nothing.
$('scroll').addEventListener('scroll', hideHelp);
addEventListener('resize', hideHelp);

// ------------------------------------------------------------- interaction

//
// Deleting a contact is one click and adding one is shift-click, so a misplaced
// click used to cost a support with no way back short of nudging spacing to
// force a whole fresh placement. Every hand edit now files the list it started
// from, and ctrl-Z walks back through them.
//
// A plain click that lands on a model other than the active one does neither
// of those — it switches which model is active, the viewport equivalent of
// the model chips in the panel. Shift-click and marker deletion still only
// ever act on the active model's own mesh and markers.
//
function pushHistory(m) {
  m.history.push(m.points.slice());
  if (m.history.length > UNDO_DEPTH) m.history.shift();
  if (m.sid === state.activeSid) $('resetpoints').disabled = false;
}

function undo() {
  const m = active();
  if (!m || state.busy || !m.history.length) return;
  m.points = m.history.pop();
  m.dropped = new Set();
  rebuildMarkers(m);
  log(`undone — ${m.points.length} points`);
  markStale('geometry');
}

addEventListener('keydown', (e) => {
  // Not while a number box has the caret: there ctrl-Z is the browser's, and
  // undoing a keystroke is what the user means.
  if (document.activeElement?.tagName === 'INPUT'
      && document.activeElement.type === 'number') return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
    e.preventDefault();
    undo();
  }
});

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downAt = null;

function setNdcFromEvent(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

// -------------------------------------------------------- gizmo dragging
//
// Both gizmo drags take over the gesture from OrbitControls: `controls`
// checks `enabled` at the top of its own pointermove handler (see
// OrbitControls.js), so switching it off here — even though OrbitControls'
// own pointerdown listener, registered first, already ran by the time this
// one does — still stops it before the camera actually moves on the very
// next move event. Both drags end the same way a dragged wheel ended: the
// range's `change` event still applies the rotation exactly once, on
// release, through `rerotate` below — nothing about that commit path
// changed, only what moves the value.

let ringDrag = null; // { axis, plane, pivot, u, v, startValue, prevAngle, accumDeg }
let moveDrag = null; // { m, plane, start: Vector3, startX, startY }

function axisVector(axis) {
  return axis === 'x' ? new THREE.Vector3(1, 0, 0)
       : axis === 'y' ? new THREE.Vector3(0, 1, 0)
       :                 new THREE.Vector3(0, 0, 1);
}

/** Two vectors spanning the plane perpendicular to `axis` — an arbitrary but
 *  fixed frame for measuring an angle in that plane. Only the *change* in
 *  that angle over a drag is ever used, so which direction reads as zero
 *  never matters. */
function planeBasis(axis) {
  const arbitrary = Math.abs(axis.y) < 0.99 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const u = new THREE.Vector3().crossVectors(arbitrary, axis).normalize();
  const v = new THREE.Vector3().crossVectors(axis, u).normalize();
  return [u, v];
}

function angleInPlane(point, pivot, u, v) {
  const rel = point.clone().sub(pivot);
  return THREE.MathUtils.radToDeg(Math.atan2(rel.dot(v), rel.dot(u)));
}

/** `a - b`, wrapped into (-180, 180] — so a drag that sweeps past ±180° reads
 *  as a small step in the right direction instead of a 360° jump. */
function shortestDelta(a, b) {
  return (((a - b + 180) % 360) + 360) % 360 - 180;
}

/** Rotate `m`'s mesh by `deg` about `axis`, pivoting on `pivot` (in the
 *  model's own group-local space — the gizmo's own centre, see buildGizmo)
 *  rather than the mesh's local origin, which is the plate, not the model's
 *  middle. This is a preview only: the mesh this is applied to is discarded
 *  and replaced wholesale the moment the commit's fresh geometry arrives (see
 *  loadSTL), so there is nothing to reset it back out of afterwards. */
function previewRotation(m, axis, pivot, deg) {
  const q = new THREE.Quaternion().setFromAxisAngle(axis, THREE.MathUtils.degToRad(deg));
  const rotatedPivot = pivot.clone().applyQuaternion(q);
  m.modelMesh.position.copy(pivot).sub(rotatedPivot);
  m.modelMesh.quaternion.copy(q);
  if (m.overhangMesh) {
    m.overhangMesh.position.copy(m.modelMesh.position);
    m.overhangMesh.quaternion.copy(q);
  }
}

function startRingDrag(mesh, e) {
  const m = active();
  if (!m || !gizmoGroup) return;
  const axisName = mesh.userData.axis;
  const axis = axisVector(axisName);
  const pivot = gizmoGroup.getWorldPosition(new THREE.Vector3());
  const [u, v] = planeBasis(axis);
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(axis, pivot);
  const hitPoint = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(plane, hitPoint)) return;
  const startAngle = angleInPlane(hitPoint, pivot, u, v);
  ringDrag = {
    axis: axisName, plane, pivot, u, v,
    startValue: +$('rot' + axisName).value,
    prevAngle: startAngle,
    accumDeg: 0,
    m, localPivot: gizmoGroup.position.clone(),
  };
  controls.enabled = false;
  try { renderer.domElement.setPointerCapture(e.pointerId); } catch { /* already gone */ }
}

function updateRingDrag(e) {
  setNdcFromEvent(e);
  raycaster.setFromCamera(ndc, camera);
  const hitPoint = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(ringDrag.plane, hitPoint)) return;
  const angle = angleInPlane(hitPoint, ringDrag.pivot, ringDrag.u, ringDrag.v);
  ringDrag.accumDeg += shortestDelta(angle, ringDrag.prevAngle);
  ringDrag.prevAngle = angle;
  // Whole degrees, same as the old wheel's own Math.round — the box shows no
  // decimals (see DECIMALS.rotx) and the range's own step is 1. Rounding here
  // rather than leaving it to `endRingDrag`'s step reset matters: a browser
  // re-clamps a range's *current* value to its step the moment `.step`
  // changes, so a small, still-fractional drag would otherwise silently snap
  // to 0 on release instead of committing the whole-degree turn it was.
  const value = Math.max(-180, Math.min(180, Math.round(ringDrag.startValue + ringDrag.accumDeg)));
  const s = $('rot' + ringDrag.axis);
  s.step = 'any';
  s.value = value;
  s.dispatchEvent(new Event('input')); // keeps the readout box in step, same as the old wheel
  previewRotation(ringDrag.m, axisVector(ringDrag.axis), ringDrag.localPivot, value - ringDrag.startValue);
}

function endRingDrag() {
  const s = $('rot' + ringDrag.axis);
  s.step = s.dataset.step;
  controls.enabled = true;
  ringDrag = null;
  // Applies on release, same as letting go of the old wheel — rotation is
  // one of the two dials that acts at once rather than waiting for the
  // generate button (see the note above `rerotate`).
  s.dispatchEvent(new Event('change'));
}

/** Movement mode has no gizmo mesh at all — dragging the model's own surface
 *  is the whole control, the same direct-body drag Bambu Studio itself
 *  supports. The drag plane is simply world Z=0: only the XY component of
 *  where it crosses that plane is ever read, so which Z it sits at makes no
 *  difference to the result. */
function startMoveDrag(m, e) {
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const start = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(plane, start)) return;
  moveDrag = { m, plane, start, startX: m.group.position.x, startY: m.group.position.y };
  controls.enabled = false;
  try { renderer.domElement.setPointerCapture(e.pointerId); } catch { /* already gone */ }
}

function updateMoveDrag(e) {
  setNdcFromEvent(e);
  raycaster.setFromCamera(ndc, camera);
  const point = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(moveDrag.plane, point)) return;
  // Overrides layoutNewModel's automatic spacing outright, same as it says
  // it would — there is nothing to reconcile, the group position is just
  // whichever of the two touched it last.
  moveDrag.m.group.position.x = moveDrag.startX + (point.x - moveDrag.start.x);
  moveDrag.m.group.position.y = moveDrag.startY + (point.y - moveDrag.start.y);
  // Z is never touched — lift_height on the panel is the only thing that
  // governs it, per CLAUDE.md.
}

function endMoveDrag() {
  controls.enabled = true;
  moveDrag = null;
}

renderer.domElement.addEventListener('pointerdown', (e) => {
  downAt = { x: e.clientX, y: e.clientY };
  if (state.busy) return;
  const m = active();
  if (!m) return;
  setNdcFromEvent(e);
  raycaster.setFromCamera(ndc, camera);

  if (gizmoMode === 'rotate' && gizmoRings) {
    const hit = raycaster.intersectObjects(Object.values(gizmoRings), false)[0];
    if (hit) { startRingDrag(hit.object, e); e.preventDefault(); return; }
  }
  if (gizmoMode === 'move' && m.modelMesh) {
    const hit = raycaster.intersectObject(m.modelMesh, false)[0];
    if (hit) { startMoveDrag(m, e); e.preventDefault(); return; }
  }
});

renderer.domElement.addEventListener('pointermove', (e) => {
  if (ringDrag) { updateRingDrag(e); return; }
  if (moveDrag) { updateMoveDrag(e); return; }
  // Idle hover: which ring, if any, would a click grab right now — the only
  // feedback besides the cursor that a ring is about to be draggable.
  if (gizmoMode === 'rotate' && gizmoRings) {
    setNdcFromEvent(e);
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObjects(Object.values(gizmoRings), false)[0];
    setGizmoHover(hit ? hit.object.userData.axis : null);
  }
});

renderer.domElement.addEventListener('pointerup', (e) => {
  if (ringDrag) { endRingDrag(); downAt = null; return; }
  if (moveDrag) { endMoveDrag(); downAt = null; return; }
  // Ignore the pointerup that ends an orbit drag.
  if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return;
  if (state.busy) return;
  const m = active();
  if (!m) return;

  setNdcFromEvent(e);
  raycaster.setFromCamera(ndc, camera);

  if (e.shiftKey) {
    if (!m.modelMesh) return;
    const hit = raycaster.intersectObject(m.modelMesh, false)[0];
    if (!hit) return;
    const n = hit.face.normal.clone().transformDirection(m.modelMesh.matrixWorld);
    // The group carries a purely visual layout offset (see layoutNewModel) —
    // the point sent to the server has to be back in the model's own local
    // frame, the one it thinks z=0 and the plate mean.
    const local = hit.point.clone().sub(m.group.position);
    pushHistory(m);
    m.points.push({
      position: local.toArray(),
      normal: n.toArray(),
      forced: true,          // a hand-placed support is never thinned away
      source: 'manual',
    });
    log('added a support');
    m.dropped = new Set();
    rebuildMarkers(m);
    markStale('geometry');
    return;
  }

  if (m.markers.length && show.points) {
    const hit = raycaster.intersectObjects(m.markers, false)[0];
    if (hit && hit.instanceId != null) {
      // Instances are grouped by flavour, so the instance index is not the
      // point index — the mesh carries the mapping back.
      pushHistory(m);
      m.points.splice(hit.object.userData.at[hit.instanceId], 1);
      log('deleted a support');
      m.dropped = new Set();
      rebuildMarkers(m);
      markStale('geometry');
      return;
    }
  }

  // Nothing above matched: a plain click on some model's surface. If it is a
  // model other than the active one, that click is the whole gesture — click
  // a model to make it active, same as clicking its chip in the panel.
  const meshes = [...state.models.values()].map((x) => x.modelMesh).filter(Boolean);
  if (!meshes.length) return;
  const hit = raycaster.intersectObjects(meshes, false)[0];
  if (!hit) return;
  const owner = [...state.models.values()].find((x) => x.modelMesh === hit.object);
  if (owner && owner.sid !== state.activeSid) switchModel(owner.sid);
});

// A cancelled gesture (an interrupted browser/OS drag) skips pointerup
// entirely — this drops the drag without committing it, same treatment
// pointercancel already got on the old wheel.
renderer.domElement.addEventListener('pointercancel', () => {
  if (ringDrag) { controls.enabled = true; $('rot' + ringDrag.axis).step = $('rot' + ringDrag.axis).dataset.step; ringDrag = null; }
  if (moveDrag) { controls.enabled = true; moveDrag = null; }
  downAt = null;
});

document.querySelectorAll('[data-toggle]').forEach(btn => {
  const k = btn.dataset.toggle;
  // Every other chip is a bare visibility flag; this one has to fetch data
  // the first time it is switched on, so it keeps its own handler above.
  if (k === 'overhang') { btn.onclick = () => toggleOverhang(btn); return; }
  // The two gizmo modes are mutually exclusive and have real scene setup and
  // teardown to do — see setGizmoMode above — so, same as overhang, they
  // keep their own handler instead of the bare show[] flip below.
  if (k === 'rotatemode') { btn.onclick = () => setGizmoMode('rotate'); return; }
  if (k === 'movemode') { btn.onclick = () => setGizmoMode('move'); return; }
  btn.onclick = () => {
    show[k] = !show[k];
    btn.classList.toggle('on', show[k]);
    applyVisibility();
  };
});

//
// What each dial costs the next build. Placement is the expensive stage and
// almost nothing needs it: only the two dials that decide where a contact goes,
// and the lift, which moves every contact by moving the model under them.
//
// This is a table read by one delegated listener rather than a listener per
// dial. Twenty-odd `addEventListener` lines had already lost one — the
// plate-only checkbox had none at all, so ticking it changed nothing until some
// other dial was touched — and a table cannot go quiet like that: a dial
// missing from it is a dial that visibly never marks the view out of date.
//
const DIAL_SCOPE = {};
for (const id of ['spacing', 'overhang', 'lift']) DIAL_SCOPE[id] = 'points';
for (const id of ['tip', 'shaft', 'tipstyle', 'plateonly', 'lean', 'parenting',
                  'braces', 'bracethick', 'bracespan', 'braceangle',
                  'braceheadroom', 'bracespacing', 'bracestart',
                  'base', 'baseh', 'cone', 'coneh']) DIAL_SCOPE[id] = 'geometry';
//
// Which dials belong to which fold, so a section can say that its own contents
// have wandered from the preset. The mapping is here rather than in the markup
// because the override names are here too — one list to keep in step, not two.
//
const FOLD_KEYS = {
  model:      ['lift_height'],
  structure:  ['parenting', 'strut_lean_deg'],
  supports:   ['tip_diameter', 'shaft_lower_diameter', 'support_spacing',
               'overhang_angle_deg', 'tip_style', 'plate_only'],
  crosslinks: ['brace_enabled', 'brace_diameter', 'brace_max_span', 'brace_angle_deg',
               'brace_headroom', 'brace_interval', 'brace_start_height'],
  base:       ['foot_diameter', 'foot_height'],
  joincone:   ['join_cone_diameter', 'join_cone_height'],
};

/** Has this dial been moved off the preset? Slider values are floats that have
 *  been through JSON and a toFixed, so an exact test would report drift that
 *  is not there. */
function differs(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) > 1e-6;
  return a !== b;
}

/** Put a dot on every section that no longer matches the preset the select is
 *  still naming, and offer the way back. Without this the panel simply lies
 *  after the first nudge: it goes on saying "ender3" whatever you do to it. */
function markDrift() {
  const m = active();
  if (!m?.preset) return;
  const now = overrides();
  let any = false;
  for (const [fold, keys] of Object.entries(FOLD_KEYS)) {
    const off = keys.some(k => m.preset[k] !== undefined && differs(now[k], m.preset[k]));
    any = any || off;
    const el = document.querySelector(`details.fold[data-fold="${fold}"] .drift`);
    if (el) el.classList.toggle('on', off);
  }
  $('revert').disabled = !any;
}

/** Adopt a preset wholesale: every dial it names, and the name itself.
 *
 *  Both halves matter. The dials matter because stage 3 is told the panel's own
 *  values, so leaving the sliders where they were meant a new preset placed its
 *  points and then had its geometry built out of the old preset's numbers. The
 *  name matters because half of a preset — the nozzle, the clearances, the
 *  printable limit — reaches no slider at all, and only stage 2 being told the
 *  name picks that half up. It rides on m.pendingPreset until it does. */
function applyPreset(name) {
  state.presetName = name;
  const m = active();
  const p = state.presets?.[name];
  if (p) {
    if (m) m.preset = p;
    syncSliders(p);
  }
  if (m) {
    m.pendingPreset = name;
    markDrift();
    markStale('points');
  }
}

$('preset').addEventListener('change', () => applyPreset($('preset').value));
$('revert').onclick = () => applyPreset($('preset').value);

// Anything moved anywhere in the panel can put a section out of step with the
// preset and put the view out of step with the panel, so both checks ride on
// the panel rather than on twenty-odd controls. The number boxes write through
// their slider, so a typed value arrives here as a change on the slider's id,
// which is what DIAL_SCOPE is keyed by.
$('panel').addEventListener('change', (e) => {
  markDrift();
  if (e.target.id === 'lift') previewLift();
  const scope = DIAL_SCOPE[e.target.id];
  if (scope) markStale(scope);
});

// Not a dial: an instruction, and one whose whole point is to throw the hand
// edits away. Marking the view out of date and waiting would leave the edits on
// screen with nothing saying they had been discarded, so this one runs.
$('resetpoints').onclick = () => {
  markStale('points');
  generate();
};

$('download').onclick = async () => {
  const m = active();
  if (!m) return;
  const fmt = $('fmt3mf').checked ? '3mf' : 'stl';
  const separate = $('dlsep').checked;
  // Serverless there is nothing to navigate to: the file is assembled in the
  // tab and handed over as a blob. `download` hides which of the two it is.
  try {
    busy(true);
    if (separate) {
      // Two plain files, not a zip — a browser can balk at saving two files
      // from one click at once, so the model download always finishes before
      // the supports one starts.
      await download(`/api/export/${m.sid}?fmt=${fmt}&separate=true&part=model`);
      try {
        await download(`/api/export/${m.sid}?fmt=${fmt}&separate=true&part=supports`);
      } catch (err) { log(`no supports to export yet: ${err.message}`, 'w'); }
    } else {
      await download(`/api/export/${m.sid}?fmt=${fmt}&separate=false`);
    }
  } catch (err) { log(`export failed: ${err.message}`, 'e'); }
  finally { busy(false); }
};

// ---------------------------------------------------------------- dropping

const drop = $('drop');
for (const ev of ['dragenter', 'dragover']) {
  document.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('hide'); drop.classList.add('over'); });
}
document.addEventListener('dragleave', (e) => {
  if (e.relatedTarget) return;
  drop.classList.remove('over');
  if (state.activeSid) drop.classList.add('hide');
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  const file = e.dataTransfer?.files?.[0];
  if (file) upload(file); else if (state.activeSid) drop.classList.add('hide');
});
drop.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.stl,.obj,.ply,.3mf,.off';
  input.onchange = () => input.files[0] && upload(input.files[0]);
  input.click();
});

// A first visit has nothing on the canvas and no way to get anything there
// without going and finding an STL, which is a poor advertisement for a tool
// whose whole output is visual. The sample ships with the page — it is a static
// asset, so it works the same served from Python or from a bundle on Pages.
const SAMPLE = 'synthetic_mini.stl';
$('sample').addEventListener('click', async (e) => {
  e.stopPropagation();          // not also the picker behind it
  try {
    const res = await fetch(new URL(`./samples/${SAMPLE}`, import.meta.url));
    if (!res.ok) throw new Error(`${res.status}`);
    await upload(new File([await res.blob()], SAMPLE, { type: 'model/stl' }));
  } catch (err) {
    log(`could not load the sample: ${err.message}`, 'e');
  }
});

//
// Stacked on a narrow window there is no canvas corner for the legend to sit
// in, so it moves into the panel instead of disappearing. It used to be hidden
// outright below 680px, which took the only explanation of click-to-delete and
// shift-click-to-add away from the layout where those gestures are hardest to
// discover.
//
const legendHome = $('legend').parentNode;
const legendSlot = document.createElement('section');
legendSlot.innerHTML = '<h2>viewport</h2>';
const stacked = matchMedia('(max-width: 680px)');

function placeLegend() {
  const legend = $('legend');
  if (stacked.matches) {
    if (!legendSlot.parentNode) $('scroll').appendChild(legendSlot);
    legendSlot.appendChild(legend);
    legend.classList.add('inpanel');
  } else {
    legendHome.appendChild(legend);
    legend.classList.remove('inpanel');
    legendSlot.remove();
  }
}
// Both, on purpose. The media query is the one that expresses the intent, but
// it does not fire under every way a viewport can change size, and a legend
// stranded in the wrong parent is a layout bug rather than a missed nicety.
// placeLegend is idempotent, so the duplicate call costs nothing.
stacked.addEventListener('change', placeLegend);
addEventListener('resize', placeLegend);
placeLegend();

//
// The legend's own fold — open by default (nothing here explains the two new
// gizmo modes at a glance without it), remembered after that the same way
// the panel's fold sections already are.
//
const LEGEND_KEY = 'rsupport.legendOpen';

function readLegendOpen() {
  try {
    const v = localStorage.getItem(LEGEND_KEY);
    return v === null ? true : v === '1';
  } catch { return true; }
}

function setLegendOpen(open) {
  $('legend').classList.toggle('collapsed', !open);
  $('legendToggle').textContent = open ? 'hide' : 'show';
  $('legendToggle').setAttribute('aria-expanded', String(open));
  try { localStorage.setItem(LEGEND_KEY, open ? '1' : '0'); } catch { /* private mode */ }
}

let legendOpen = readLegendOpen();
setLegendOpen(legendOpen);
$('legendToggle').addEventListener('click', () => { legendOpen = !legendOpen; setLegendOpen(legendOpen); });

// ---------------------------------------------------------------- startup

(async function init() {
  try {
    if (isServerless) {
      // Nothing is reachable until the runtime is up, and that is a few
      // seconds of downloading on a cold cache. Say what it is doing rather
      // than showing an inert page: the first impression of the hosted build
      // is this wait, and an unexplained one reads as broken.
      busy(true, true);
      log('starting the pipeline in your browser — nothing is uploaded anywhere', 'w');
      const stop = onProgress(({ phase, detail }) => {
        if (phase === 'fatal') log(`could not start: ${detail}`, 'e');
        else log(`  ${phase}${detail ? ': ' + detail : ''} …`);
      });
      await warmUp();
      stop();
      log('ready — drop an STL to begin');
      busy(false);
    }

    const { presets, default: def } = await (await api('/api/presets')).json();
    const sel = $('preset');
    for (const name of Object.keys(presets)) {
      const o = document.createElement('option');
      o.value = o.textContent = name;
      sel.appendChild(o);
    }
    sel.value = def;
    state.presets = presets;
    state.presetName = def;
    // Sliders reflect the chosen preset even before a model exists, so the
    // very first upload's stage-2 call sends sensible overrides rather than
    // whatever the bare HTML attributes happened to default to.
    syncSliders(presets[def]);
  } catch (err) {
    busy(false);
    log(isServerless
      ? `could not start the pipeline: ${err.message}`
      : `could not reach the server: ${err.message}`, 'e');
  }
})();
