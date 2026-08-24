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
};

//: How many hand edits ctrl-Z can walk back. Each entry is a shallow copy of a
//: point list, so this is bounded memory for an unbounded session.
const UNDO_DEPTH = 32;

const $ = (id) => document.getElementById(id);
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
let markers = [];   // one InstancedMesh per flavour; .userData.at maps back to state.points

const show = { model: true, supports: true, points: false, wire: false };

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
}

/** Two flavours of wait, because they interrupt differently.
 *
 *  `heavy` washes the canvas out, and is for the times there is nothing on it
 *  worth looking at: the first load, and the Pyodide warm-up. Everything else
 *  — a slider released, a point deleted — is a second or two during which
 *  watching the scaffold change is the entire point, so it gets a bar under
 *  the panel header and leaves the viewport alone. Dimming the stage on every
 *  slider release read as a flicker and hid the answer. */
function busy(on, heavy = false) {
  state.busy = on;
  $('prog').classList.toggle('on', on);
  $('busy').classList.toggle('on', on && heavy);
  if (!on) $('busy').classList.remove('on');
}

// ---------------------------------------------------------------- api

// `api`, `postJSON` and `download` come from transport.js, which is either
// fetch against the FastAPI app or postMessage to a Pyodide worker running the
// whole pipeline in this tab. Both answer the same routes, so nothing below
// this line knows or cares which is in use.

// ------------------------------------------------------------ geometry io

async function loadSTL(url, kind) {
  const buf = await (await api(url)).arrayBuffer();
  const geom = loader.parse(buf);
  geom.computeVertexNormals();

  const mesh = new THREE.Mesh(geom, MAT[kind]);
  if (kind === 'model') {
    if (modelMesh) { world.remove(modelMesh); modelMesh.geometry.dispose(); }
    modelMesh = mesh;
  } else {
    if (supportMesh) { world.remove(supportMesh); supportMesh.geometry.dispose(); }
    supportMesh = mesh;
  }
  world.add(mesh);
  applyVisibility();
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
  markers.forEach(m => { m.visible = show.points; });
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

async function upload(file) {
  busy(true, true);
  try {
    $('log').innerHTML = '';
    log(`reading ${file.name} …`);
    const fd = new FormData();
    fd.append('file', file);
    const info = await (await api('/api/model', { method: 'POST', body: fd })).json();

    state.sid = info.id;
    state.points = [];
    $('filename').textContent = `${info.name} — ${info.summary.faces.toLocaleString()} faces, ` +
      info.summary.size.map(v => v.toFixed(1)).join(' × ') + ' mm';
    if (!info.summary.watertight) log('mesh is not watertight; results may be rough', 'w');
    $('drop').classList.add('hide');
    $('asloaded').disabled = false;
    $('rotx').value = 0; $('roty').value = 0; $('rotz').value = 0;
    showRotationValues();

    // The file is taken as posed. It arrives already dropped onto the bed, so
    // there is nothing to decide — go straight to working out where supports go.
    await loadSTL(`/api/mesh/${state.sid}/model`, 'model');
    frameModel();
    log('using the pose from the file', 'g');

    await runPoints();
    await runSupports();
  } catch (err) {
    log(`error: ${err.message}`, 'e');
  } finally {
    busy(false);
  }
}

/** Rotation sliders are absolute, always applied from the file's own pose —
 *  so re-running with all three at 0 is exactly the file's pose. */
async function runRotate() {
  const rx = +$('rotx').value, ry = +$('roty').value, rz = +$('rotz').value;
  log('rotating the model …');
  const r = await postJSON(`/api/rotate/${state.sid}`, { rx, ry, rz, overrides: overrides() });
  await loadSTL(`/api/mesh/${state.sid}/model`, 'model');
  frameModel();
  log(`rotated (${r.elapsed.toFixed(2)}s)`, 'g');
}

async function rerotate() {
  if (!state.sid || state.busy) return;
  busy(true);
  try {
    await runRotate();
    await runPoints();
    await runSupports();
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

async function runPoints() {
  log('placing support points …');
  absorbPoints(await postJSON(`/api/points/${state.sid}`, { overrides: overrides() }));
}

async function runSupports() {
  log('building support geometry …');
  const r = await postJSON(`/api/supports/${state.sid}`, {
    overrides: overrides(),
    points: state.points,
  });
  state.params = r.params;
  state.dropped = new Set(r.dropped_points || []);
  syncSliders(r.params);
  await loadSTL(`/api/mesh/${state.sid}/supports`, 'supports').catch(() => {
    log('no supports needed', 'g');
  });
  rebuildMarkers();

  state.volume = r.volume ?? null;
  showStats(r);
  (r.warnings || []).slice(0, 5).forEach(w => log(w, 'w'));
  log(`built in ${r.elapsed.toFixed(2)}s`, 'g');

  ['dl3mf', 'dlstl', 'dlsep'].forEach(id => $(id).disabled = false);
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

/** The lift moves the model itself, so the viewer's copy of it is stale too.
 *  Stage 2 re-floats it server-side; all this has to do is fetch it again. */
async function relift() {
  if (!state.sid || state.busy) return;
  busy(true);
  try {
    const mm = +$('lift').value;
    log(mm > 0 ? `floating the model ${mm.toFixed(1)} mm off the plate …` : 'setting the model down …');
    await runPoints();
    await loadSTL(`/api/mesh/${state.sid}/model`, 'model');
    await runSupports();
  } catch (err) {
    log(`error: ${err.message}`, 'e');
  } finally {
    busy(false);
  }
}

/** Re-run only what actually changed. Geometry params are cheap; placement is not. */
async function rerun(scope) {
  if (!state.sid || state.busy) return;
  busy(true);
  try {
    if (scope === 'points') await runPoints();
    await runSupports();
  } catch (err) {
    log(`error: ${err.message}`, 'e');
  } finally {
    busy(false);
  }
}

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

async function undo() {
  if (!state.sid || state.busy || !state.history.length) return;
  state.points = state.history.pop();
  state.dropped = new Set();
  rebuildMarkers();
  log(`undone — ${state.points.length} points`);
  await rerun('geometry');
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

renderer.domElement.addEventListener('pointerup', async (e) => {
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
    await rerun('geometry');
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
      await rerun('geometry');
    }
  }
});

document.querySelectorAll('[data-toggle]').forEach(btn => {
  btn.onclick = () => {
    const k = btn.dataset.toggle;
    show[k] = !show[k];
    btn.classList.toggle('on', show[k]);
    applyVisibility();
  };
});

// Slider release, not every pixel of drag — each change costs a server round trip.
['tip', 'shaft'].forEach(id => $(id).addEventListener('change', () => rerun('geometry')));
['spacing', 'overhang'].forEach(id => $(id).addEventListener('change', () => rerun('points')));
$('tipstyle').addEventListener('change', () => rerun('geometry'));
$('braces').addEventListener('change', () => rerun('geometry'));
['lean', 'parenting'].forEach(id => $(id).addEventListener('change', () => rerun('geometry')));
['base', 'baseh', 'cone', 'coneh'].forEach(id => $(id).addEventListener('change', () => rerun('geometry')));
['bracethick', 'bracespan', 'braceangle', 'braceheadroom', 'bracespacing', 'bracestart']
  .forEach(id => $(id).addEventListener('change', () => rerun('geometry')));
// The lift moves the model, so the point list has to be placed again on top of
// rebuilding the scaffold — see relift().
$('lift').addEventListener('change', relift);
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

/** Adopt a preset wholesale: dials first, then the pipeline.
 *
 *  The dials matter. Stage 2 is told the preset by name, but stage 3 is told
 *  the panel's own values — so leaving the sliders where they were meant a new
 *  preset placed its points and then had its geometry built out of the old
 *  preset's numbers. */
async function applyPreset(name) {
  const p = state.presets?.[name];
  if (p) {
    state.preset = p;
    syncSliders(p);
  }
  if (!state.sid || state.busy) { markDrift(); return; }
  busy(true);
  try {
    absorbPoints(await postJSON(`/api/points/${state.sid}`, { preset: name }));
    await runSupports();
  } catch (err) { log(`error: ${err.message}`, 'e'); }
  finally { busy(false); }
}

$('preset').addEventListener('change', () => applyPreset($('preset').value));
$('revert').onclick = () => applyPreset($('preset').value);

// Anything moved anywhere in the panel can put a section out of step with the
// preset, so the check rides on the panel rather than on twenty-odd controls.
$('panel').addEventListener('change', markDrift);

$('resetpoints').onclick = async () => {
  if (!state.sid || state.busy) return;
  busy(true);
  try {
    log('placing support points again …');
    await runPoints();
    await runSupports();
  } catch (err) { log(`error: ${err.message}`, 'e'); }
  finally { busy(false); }
};

for (const [id, mode] of [['dl3mf', '3mf'], ['dlstl', 'combined'], ['dlsep', 'separate']]) {
  $(id).onclick = async () => {
    if (!state.sid) return;
    // Serverless there is nothing to navigate to: the file is assembled in the
    // tab and handed over as a blob. `download` hides which of the two it is.
    try { busy(true); await download(`/api/export/${state.sid}?mode=${mode}`); }
    catch (err) { log(`export failed: ${err.message}`, 'e'); }
    finally { busy(false); }
  };
}

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
