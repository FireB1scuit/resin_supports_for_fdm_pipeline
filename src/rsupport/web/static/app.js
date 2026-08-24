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
  busy: false,
};

const $ = (id) => document.getElementById(id);
const loader = new STLLoader();

// ---------------------------------------------------------------- scene

const canvas = $('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);

// Everything downstream is Z-up, because that is how print beds work. Telling
// three.js the same avoids transposing coordinates on every message.
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 4000);
camera.up.set(0, 0, 1);
camera.position.set(70, -95, 65);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xdfe6f2, 0x1b1f26, 1.5));
const key = new THREE.DirectionalLight(0xffffff, 1.9);
key.position.set(60, -80, 120);
scene.add(key);
const rim = new THREE.DirectionalLight(0x88a8ff, 0.5);
rim.position.set(-70, 60, 30);
scene.add(rim);

const grid = new THREE.GridHelper(240, 24, 0x3a4150, 0x252a33);
grid.rotation.x = Math.PI / 2;   // GridHelper is XZ by default; we want the XY bed
scene.add(grid);

const world = new THREE.Group();
scene.add(world);

const MAT = {
  model: new THREE.MeshStandardMaterial({ color: 0x9aa3af, roughness: 0.72, metalness: 0.04, flatShading: false }),
  supports: new THREE.MeshStandardMaterial({ color: 0xff8a3d, roughness: 0.55, metalness: 0.05 }),
  // Contact points come in two flavours and the difference has to be readable
  // at a glance. Held ones are washed out on purpose — there are hundreds of
  // them and they are covering the model, so they have to stay out of the way
  // of the surface underneath. Unheld ones are solid and a touch bigger: those
  // are the ones worth looking at, and there are usually only a handful.
  point: new THREE.MeshBasicMaterial({
    color: 0xff4d4d, transparent: true, opacity: 0.45, depthWrite: false,
  }),
  pointDropped: new THREE.MeshBasicMaterial({ color: 0xb00020 }),
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
}

function busy(on) {
  state.busy = on;
  $('busy').classList.toggle('on', on);
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
}

function rebuildMarkers() {
  markers.forEach(m => { world.remove(m); m.geometry.dispose(); });
  markers = [];
  if (!state.points.length) return;

  // Split by whether stage 3 managed to support the point. Two meshes rather
  // than one with per-instance colours: opacity is a property of the material,
  // and a deep red at 45% over a dark background reads as washed out, not as
  // urgent. Each mesh remembers which state.points index every instance came
  // from, so clicking one still deletes the right point.
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
  busy(true);
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

async function runPoints() {
  log('placing support points …');
  const r = await postJSON(`/api/points/${state.sid}`, { overrides: overrides() });
  state.points = r.points;
  state.dropped = new Set();
  rebuildMarkers();
  const forced = state.points.filter(p => p.forced).length;
  log(`${state.points.length} points (${forced} mandatory) in ${r.elapsed.toFixed(2)}s`, 'g');
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

  $('stats').innerHTML =
    `<b>${r.points}</b> supports &middot; <b>${r.braces}</b> links<br>` +
    `<b>${r.faces.toLocaleString()}</b> triangles` +
    (r.dropped
      ? `<br><span style="color:var(--err)"><b>${r.dropped}</b> unsupported ` +
        `&mdash; shown in solid red</span>`
      : '');
  (r.warnings || []).slice(0, 5).forEach(w => log(w, 'w'));
  log(`built in ${r.elapsed.toFixed(2)}s`, 'g');

  ['dl3mf', 'dlstl', 'dlsep'].forEach(id => $(id).disabled = false);
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
$('preset').addEventListener('change', async () => {
  if (!state.sid || state.busy) return;
  busy(true);
  try {
    const r = await postJSON(`/api/points/${state.sid}`, { preset: $('preset').value });
    state.points = r.points;
    state.dropped = new Set();
    await runSupports();
  } catch (err) { log(`error: ${err.message}`, 'e'); }
  finally { busy(false); }
});

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

// ---------------------------------------------------------------- startup

(async function init() {
  try {
    if (isServerless) {
      // Nothing is reachable until the runtime is up, and that is a few
      // seconds of downloading on a cold cache. Say what it is doing rather
      // than showing an inert page: the first impression of the hosted build
      // is this wait, and an unexplained one reads as broken.
      busy(true);
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
    syncSliders(presets[def]);
    state.params = presets[def];
  } catch (err) {
    busy(false);
    log(isServerless
      ? `could not start the pipeline: ${err.message}`
      : `could not reach the server: ${err.message}`, 'e');
  }
})();
