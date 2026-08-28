import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';
import { STLLoader } from './STLLoader.js';
import { api, postJSON, download, isServerless, onProgress, warmUp } from './transport.js';

// ---------------------------------------------------------------- state

const state = {
  sid: null,
  points: [],          // [{position:[x,y,z], normal:[..], forced, source}]
  dropped: new Set(),  // indices into points that stage 3 could not support
  params: null,        // last params the server reported back
  presets: null,       // every preset the server knows, by name
  preset: null,        // params of the preset named in the select, for drift
  size: null,          // oriented bounding box, mm
  overhangArea: null,  // mm^2 of flagged downward face
  volume: null,        // mm^3 enclosed by the scaffold, overlaps counted twice
  history: [],         // point lists before each hand edit — see undo()
  busy: false,
  //: What the view no longer answers for, worst case, since the last build:
  //: null when it is current, otherwise a SCOPES name. Every dial writes this
  //: instead of starting a run — see markStale.
  stale: null,
  //: The lift baked into the model mesh on screen, so a run knows whether the
  //: mesh has to be fetched again. null means "not known", which is the state
  //: a fresh upload is in: the session floats the model by its own preset's
  //: lift and the payload does not say what that was.
  lift: null,
  //: A preset the select is naming but no stage has been told about yet. The
  //: dials cannot carry one on their own — half of what a preset sets (the
  //: nozzle, the clearances, the printable limit) has no slider — so the name
  //: rides along with the next placement and is spent there.
  pendingPreset: null,
};

//: How many hand edits ctrl-Z can walk back. Each entry is a shallow copy of a
//: point list, so this is bounded memory for an unbounded session.
const UNDO_DEPTH = 32;

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

let modelMesh = null;
let supportMesh = null;
let overhangMesh = null;   // the checkered overlay — see loadOverhang/buildOverhangOverlay
let markers = [];   // one InstancedMesh per flavour; .userData.at maps back to state.points

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
  if (active) setWorking(msg);
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

async function loadSTL(url, kind, signal) {
  const buf = await (await api(url, { signal })).arrayBuffer();
  const geom = loader.parse(buf);
  geom.computeVertexNormals();

  const mesh = new THREE.Mesh(geom, MAT[kind]);
  if (kind === 'model') {
    if (modelMesh) { world.remove(modelMesh); modelMesh.geometry.dispose(); }
    modelMesh = mesh;
    // The overlay is built from this geometry's own vertex positions (see
    // buildOverhangOverlay), so a new model mesh strands the old one — face
    // indices from the previous pose mean nothing against this one.
    clearOverhangOverlay();
  } else {
    if (supportMesh) { world.remove(supportMesh); supportMesh.geometry.dispose(); }
    supportMesh = mesh;
  }
  world.add(mesh);
  applyVisibility();
  // The overlay is not a build stage and has no scope of its own — it just
  // has to survive the model mesh being replaced out from under it, so it is
  // refetched here rather than left stale until somebody happens to retoggle
  // it.
  if (kind === 'model' && show.overhang) await loadOverhang(signal);
  return mesh;
}

function frameModel() {
  if (!modelMesh) return;
  const box = new THREE.Box3().setFromObject(modelMesh);
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

function rebuildMarkers() {
  markers.forEach(m => { world.remove(m); m.geometry.dispose(); });
  markers = [];
  if (!state.points.length) return;

  // Split by whether stage 3 managed to support the point. Two meshes rather
  // than one with per-instance colours: opacity is a property of the material,
  // and the two flavours differ in it — held contacts are washed out because
  // there are hundreds of them over the surface you are trying to see, unheld
  // ones are solid because there are usually a handful and they are the news.
  // Each mesh remembers which state.points index every instance came from, so
  // clicking one still deletes the right point.
  const held = [], unheld = [];
  state.points.forEach((_, i) => (state.dropped.has(i) ? unheld : held).push(i));

  const r = (state.params?.tip_diameter ?? 0.3) * 1.6;
  const m = new THREE.Matrix4();
  for (const [idx, mat, scale] of [[held, MAT.point, 1], [unheld, MAT.pointDropped, 1.45]]) {
    if (!idx.length) continue;
    const mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(r * scale, 8, 6), mat, idx.length);
    idx.forEach((pi, i) => {
      const p = state.points[pi].position;
      m.makeTranslation(p[0], p[1], p[2]);
      mesh.setMatrixAt(i, m);
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.userData.at = idx;
    // Solid markers draw last, so a dropped point stays visible through the
    // cloud of translucent ones around it.
    mesh.renderOrder = mat === MAT.pointDropped ? 2 : 1;
    markers.push(mesh);
    world.add(mesh);
  }
  applyVisibility();
}

function applyVisibility() {
  if (modelMesh) {
    modelMesh.visible = show.model;
    MAT.model.wireframe = show.wire;
  }
  if (supportMesh) supportMesh.visible = show.supports;
  if (overhangMesh) overhangMesh.visible = show.overhang;
  markers.forEach(m => { m.visible = show.points; });
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

function clearOverhangOverlay() {
  if (!overhangMesh) return;
  world.remove(overhangMesh);
  overhangMesh.geometry.dispose();
  overhangMesh.material.dispose();
  overhangMesh = null;
}

/** A hazard checker in world space: each triangle is coloured solid (all
 *  three of its own vertices, not shared with any neighbour) by the parity of
 *  its centroid's cell on a 3D grid. Summing all three axes rather than just
 *  XY keeps the pattern reading as a checker on a steep or vertical face too,
 *  not a set of stripes running the wrong way. */
function buildOverhangOverlay(faceIndices) {
  clearOverhangOverlay();
  if (!modelMesh || !faceIndices || !faceIndices.length) { applyVisibility(); return; }

  const src = modelMesh.geometry.attributes.position;
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
  overhangMesh = new THREE.Mesh(geom, mat);
  overhangMesh.position.copy(modelMesh.position);
  world.add(overhangMesh);
  applyVisibility();
}

/** Fetch the overhang faces for the pose on screen and rebuild the overlay
 *  from them. Not part of any run: it is its own request, made only when the
 *  toggle asks for one. */
async function loadOverhang(signal) {
  if (!state.sid || !modelMesh) return;
  try {
    const angle = +$('overhang').value;
    const r = await (await api(`/api/overhang/${state.sid}?angle_deg=${angle}`, { signal })).json();
    buildOverhangOverlay(r.faces);
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
  if (!state.sid || state.busy) return;
  show.overhang = !show.overhang;
  btn.classList.toggle('on', show.overhang);
  if (show.overhang && !overhangMesh) await loadOverhang();
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
  if (state.stale === null || SCOPES.indexOf(scope) > SCOPES.indexOf(state.stale)) {
    state.stale = scope;
  }
  paintGenerate();
}

async function upload(file) {
  await stopRun();
  busy(true, true);
  try {
    $('log').innerHTML = '';
    log(`reading ${file.name} …`);
    const fd = new FormData();
    fd.append('file', file);
    const info = await (await api('/api/model', { method: 'POST', body: fd })).json();

    state.sid = info.id;
    state.lift = null;
    clearBuild();
    $('filename').textContent = `${info.name} — ${info.summary.faces.toLocaleString()} faces, ` +
      info.summary.size.map(v => v.toFixed(1)).join(' × ') + ' mm';
    if (!info.summary.watertight) log('mesh is not watertight; results may be rough', 'w');
    $('drop').classList.add('hide');
    $('asloaded').disabled = false;
    $('rotx').value = 0; $('roty').value = 0; $('rotz').value = 0;
    showRotationValues();

    // The file is taken as posed. It arrives already dropped onto the bed, so
    // there is nothing to decide.
    await loadSTL(`/api/mesh/${state.sid}/model`, 'model');
    frameModel();
    log('using the pose from the file', 'g');
  } catch (err) {
    log(`error: ${err.message}`, 'e');
    busy(false);
    return;
  }
  busy(false);

  // The one build nobody has to ask for. There is nothing on the canvas yet, so
  // there is nothing for a press to be a decision about — and a tool whose whole
  // output is visual should not open on an empty stage. Every change after this
  // one waits for the button.
  markStale('points');
  await generate();
}

/** Drop the scaffold and the contact points, and everything else that was only
 *  true because of them. They belong to a pose or a model that no longer
 *  exists, so leaving them up while the next build is asked for would be
 *  showing the answer to a question nobody is still asking.
 *
 *  The download buttons go with them, and that is not tidiness: re-posing the
 *  model clears the session's supports server-side, so an export taken in this
 *  window is a model and an empty scaffold, written out without complaint. */
function clearBuild() {
  if (supportMesh) {
    world.remove(supportMesh);
    supportMesh.geometry.dispose();
    supportMesh = null;
  }
  state.points = [];
  state.dropped = new Set();
  state.history = [];
  state.volume = null;
  $('resetpoints').disabled = true;
  $('stats').innerHTML = '&mdash;';
  $('download').disabled = true;
  rebuildMarkers();
}

/** Rotation sliders are absolute, always applied from the file's own pose —
 *  so re-running with all three at 0 is exactly the file's pose. */
async function runRotate() {
  const rx = +$('rotx').value, ry = +$('roty').value, rz = +$('rotz').value;
  log('rotating the model …');
  const r = await postJSON(`/api/rotate/${state.sid}`, { rx, ry, rz, overrides: overrides() });
  await loadSTL(`/api/mesh/${state.sid}/model`, 'model');
  state.lift = +$('lift').value;
  frameModel();
  log(`rotated (${r.elapsed.toFixed(2)}s)`, 'g');
}

/** Turning the model is not a build, and it does not wait for the button.
 *
 *  Posing is done by eye — turn it, look, turn it back — and a rotation slider
 *  that moves nothing until a second control is pressed cannot be used that way
 *  at all. So the pose is applied at once and only the scaffold is left out of
 *  date. What was already built came off the old pose and would now be standing
 *  through the model rather than under it, so it goes rather than hanging about
 *  looking like a result. */
async function rerotate() {
  if (!state.sid || state.busy) return;
  busy(true);
  try {
    clearBuild();
    await runRotate();
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
function absorbPoints(r) {
  state.points = r.points;
  state.dropped = new Set();
  // A fresh placement is not something ctrl-Z can walk back into: the edits in
  // the stack belong to a point list that no longer exists.
  state.history = [];
  state.size = r.size || null;
  state.overhangArea = r.overhang_area ?? null;
  $('resetpoints').disabled = true;
  rebuildMarkers();
  const forced = state.points.filter(p => p.forced).length;
  log(`${state.points.length} points (${forced} mandatory) in ${r.elapsed.toFixed(2)}s`, 'g');
}

async function runPoints(run) {
  log('placing support points …');
  const body = { overrides: overrides() };
  // A preset is a base the dials sit on top of, not a set of dial values: the
  // nozzle, the clearances and the printable limit come with it and have no
  // slider anywhere in the panel. Sending the name is the only way stage 2
  // hears about those.
  if (state.pendingPreset) body.preset = state.pendingPreset;
  const r = await postJSON(`/api/points/${state.sid}`, body, run?.ac.signal);
  halt(run);
  // Only now is the preset spent. A stopped run has not applied it, so it has
  // to still be owed to the next one.
  state.pendingPreset = null;
  absorbPoints(r);
}

/** Stage 2 is what floats the model, so a lift that has moved leaves the mesh
 *  on screen at the old height — either at the wrong Z outright, or carrying
 *  the local offset previewLift put on it. It is the largest thing on the wire,
 *  so it is fetched again only when one of those is actually true. */
async function refloat(run) {
  if (state.lift === +$('lift').value && modelMesh && modelMesh.position.z === 0) return;
  await loadSTL(`/api/mesh/${state.sid}/model`, 'model', run?.ac.signal);
  state.lift = +$('lift').value;
}

async function runSupports(run) {
  log('building support geometry …');
  const r = await postJSON(`/api/supports/${state.sid}`, {
    overrides: overrides(),
    points: state.points,
  }, run?.ac.signal);
  halt(run);
  state.params = r.params;
  state.dropped = new Set(r.dropped_points || []);
  syncSliders(r.params);
  await loadSTL(`/api/mesh/${state.sid}/supports`, 'supports', run?.ac.signal).catch((err) => {
    // A model needing no supports at all answers 404 here, which is not news.
    // A stopped run answers with an abort, which is — it must not be swallowed
    // into "no supports needed", or stopping would look like an empty result.
    if (err.name === 'AbortError') throw err;
    log('no supports needed', 'g');
  });
  rebuildMarkers();

  state.volume = r.volume ?? null;
  showStats(r);
  (r.warnings || []).slice(0, 5).forEach(w => log(w, 'w'));
  log(`built in ${r.elapsed.toFixed(2)}s`, 'g');

  $('download').disabled = false;
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
function showStats(r) {
  const bits = [`<b>${r.points}</b> supports &middot; <b>${r.braces}</b> links`,
                `<b>${r.faces.toLocaleString()}</b> triangles`];

  if (state.size) {
    bits.push(`<b>${state.size.map(v => v.toFixed(1)).join(' × ')}</b> mm`);
  }
  if (state.overhangArea != null) {
    bits.push(`<b>${(state.overhangArea / 100).toFixed(1)}</b> cm&sup2; overhang`);
  }
  if (state.volume) {
    const grams = (state.volume / 1000) * DENSITY;
    bits.push(`under <b>${grams.toFixed(1)}</b> g of support`);
  }
  if (r.dropped) {
    bits.push(`<span style="color:var(--err)"><b>${r.dropped}</b> unsupported ` +
              `&mdash; shown in solid red</span>`);
  }
  $('stats').innerHTML = bits.join('<br>');
}

/** The lift is a translation in Z and nothing else, so the mesh on screen can
 *  follow the slider without asking anybody — otherwise dragging it would do
 *  nothing visible at all until the button was pressed, which reads as broken.
 *  The scaffold deliberately stays put: it no longer reaches the model, which
 *  is exactly what out of date looks like. The next build fetches the properly
 *  floated mesh back and the offset goes with it — see refloat. */
function previewLift() {
  if (modelMesh && state.lift !== null) {
    modelMesh.position.z = +$('lift').value - state.lift;
    // The overlay is drawn in the model's own object space and has no lift
    // logic of its own — it just has to ride along, or it visibly separates
    // from the surface it is supposed to be painted on.
    if (overhangMesh) overhangMesh.position.z = modelMesh.position.z;
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

let active = null;    // the run in flight, or null
let running = null;   // its promise, for anyone who has to wait for it to let go

function halt(run) { if (run?.stop) throw new Stopped(); }

/** The button. A press builds; a press while it is building stops the build. */
function generate() {
  if (active) return stopRun();
  if (!state.sid || state.busy) return Promise.resolve();

  // A press with nothing out of date rebuilds the scaffold anyway. It is a
  // second or two, it is what was asked for, and a button that ignores a press
  // is indistinguishable from a broken one.
  const scope = state.stale ?? 'geometry';
  active = { stop: false, ac: new AbortController() };
  running = runAll(scope, active);
  return running;
}

async function runAll(scope, run) {
  state.stale = null;
  busy(true);
  setWorking('working');
  try {
    if (scope !== 'geometry') {
      await runPoints(run);
      halt(run);
      await refloat(run);
      halt(run);
    }
    await runSupports(run);
  } catch (err) {
    // Whatever did not finish leaves the view where it already was: out of
    // date, and now saying so again.
    markStale(scope);
    if (err instanceof Stopped || err.name === 'AbortError') log('stopped', 'w');
    else log(`error: ${err.message}`, 'e');
  } finally {
    active = null;
    running = null;
    busy(false);
  }
}

/** Ask the run in flight to stop, and hand back something to wait on. It is not
 *  over when this returns — see the note above — so the button goes on saying
 *  what it is doing until it is. */
function stopRun() {
  if (!active) return Promise.resolve();
  if (!active.stop) {
    active.stop = true;
    active.ac.abort();
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
  b.disabled = !state.sid || (state.busy && !active) || !!active?.stop;
  b.classList.toggle('running', !!active);
  b.classList.toggle('stale', !active && state.stale !== null);
  if (!active) $('genlabel').textContent = 'generate';
  $('gentip').textContent = active ? (active.stop ? TIP_STOPPING : TIP_RUNNING) : TIP_IDLE;
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
function pushHistory() {
  state.history.push(state.points.slice());
  if (state.history.length > UNDO_DEPTH) state.history.shift();
  $('resetpoints').disabled = false;
}

function undo() {
  if (!state.sid || state.busy || !state.history.length) return;
  state.points = state.history.pop();
  state.dropped = new Set();
  rebuildMarkers();
  log(`undone — ${state.points.length} points`);
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

renderer.domElement.addEventListener('pointerdown', (e) => { downAt = { x: e.clientX, y: e.clientY }; });

renderer.domElement.addEventListener('pointerup', (e) => {
  // Ignore the pointerup that ends an orbit drag.
  if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return;
  if (!state.sid || state.busy) return;

  const rect = renderer.domElement.getBoundingClientRect();
  ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);

  if (e.shiftKey && modelMesh) {
    const hit = raycaster.intersectObject(modelMesh, false)[0];
    if (!hit) return;
    const n = hit.face.normal.clone().transformDirection(modelMesh.matrixWorld);
    pushHistory();
    state.points.push({
      position: hit.point.toArray(),
      normal: n.toArray(),
      forced: true,          // a hand-placed support is never thinned away
      source: 'manual',
    });
    log('added a support');
    state.dropped = new Set();
    rebuildMarkers();
    markStale('geometry');
    return;
  }

  if (markers.length && show.points) {
    const hit = raycaster.intersectObjects(markers, false)[0];
    if (hit && hit.instanceId != null) {
      // Instances are grouped by flavour, so the instance index is not the
      // point index — the mesh carries the mapping back.
      pushHistory();
      state.points.splice(hit.object.userData.at[hit.instanceId], 1);
      log('deleted a support');
      state.dropped = new Set();
      rebuildMarkers();
      markStale('geometry');
    }
  }
});

document.querySelectorAll('[data-toggle]').forEach(btn => {
  const k = btn.dataset.toggle;
  // Every other chip is a bare visibility flag; this one has to fetch data
  // the first time it is switched on, so it keeps its own handler above.
  if (k === 'overhang') { btn.onclick = () => toggleOverhang(btn); return; }
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
  if (!state.preset) return;
  const now = overrides();
  let any = false;
  for (const [fold, keys] of Object.entries(FOLD_KEYS)) {
    const off = keys.some(k => state.preset[k] !== undefined && differs(now[k], state.preset[k]));
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
 *  name picks that half up. It rides on state.pendingPreset until it does. */
function applyPreset(name) {
  const p = state.presets?.[name];
  if (p) {
    state.preset = p;
    syncSliders(p);
  }
  state.pendingPreset = name;
  markDrift();
  if (state.sid) markStale('points');
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
  if (!state.sid) return;
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
      await download(`/api/export/${state.sid}?fmt=${fmt}&separate=true&part=model`);
      try {
        await download(`/api/export/${state.sid}?fmt=${fmt}&separate=true&part=supports`);
      } catch (err) { log(`no supports to export yet: ${err.message}`, 'w'); }
    } else {
      await download(`/api/export/${state.sid}?fmt=${fmt}&separate=false`);
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
  if (state.sid) drop.classList.add('hide');
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  const file = e.dataTransfer?.files?.[0];
  if (file) upload(file); else if (state.sid) drop.classList.add('hide');
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
    state.preset = presets[def];
    syncSliders(presets[def]);
    state.params = presets[def];
  } catch (err) {
    busy(false);
    log(isServerless
      ? `could not start the pipeline: ${err.message}`
      : `could not reach the server: ${err.message}`, 'e');
  }
})();
