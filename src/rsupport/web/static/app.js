import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';
import { STLLoader } from './STLLoader.js';

// ---------------------------------------------------------------- state

const state = {
  sid: null,
  points: [],          // [{position:[x,y,z], normal:[..], forced, source}]
  params: null,        // last params the server reported back
  orientations: [],
  applied: 0,
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
  point: new THREE.MeshBasicMaterial({ color: 0xff5a5a }),
};

let modelMesh = null;
let supportMesh = null;
let markers = null;

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

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return res;
}

async function postJSON(path, body) {
  const res = await api(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  return res.json();
}

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
  if (markers) { world.remove(markers); markers.geometry.dispose(); markers = null; }
  if (!state.points.length) return;

  const r = (state.params?.tip_diameter ?? 0.3) * 1.6;
  const geo = new THREE.SphereGeometry(r, 8, 6);
  markers = new THREE.InstancedMesh(geo, MAT.point, state.points.length);
  const m = new THREE.Matrix4();
  state.points.forEach((p, i) => {
    m.makeTranslation(p.position[0], p.position[1], p.position[2]);
    markers.setMatrixAt(i, m);
  });
  markers.instanceMatrix.needsUpdate = true;
  world.add(markers);
  applyVisibility();
}

function applyVisibility() {
  if (modelMesh) {
    modelMesh.visible = show.model;
    MAT.model.wireframe = show.wire;
  }
  if (supportMesh) supportMesh.visible = show.supports;
  if (markers) markers.visible = show.points;
}

// ---------------------------------------------------------------- params

function overrides() {
  return {
    tip_diameter: +$('tip').value,
    pillar_diameter: +$('pillar').value,
    support_spacing: +$('spacing').value,
    overhang_angle_deg: +$('overhang').value,
    tip_style: $('tipstyle').value,
    brace_enabled: $('braces').checked,
    support_style: $('style').value,
    branch_angle_deg: +$('branch').value,
    merge_strength: +$('merge').value,
  };
}

/** Branch controls mean nothing to the pillar generator, and vice versa. */
function syncStyleVisibility() {
  const tree = $('style').value === 'tree';
  document.querySelectorAll('.tree-only').forEach(el => el.style.display = tree ? '' : 'none');
  document.querySelectorAll('.pillar-only').forEach(el => el.style.display = tree ? 'none' : '');
}

function syncSliders(p) {
  if (!p) return;
  for (const [id, key] of [['tip', 'tip_diameter'], ['pillar', 'pillar_diameter'],
                           ['spacing', 'support_spacing'], ['overhang', 'overhang_angle_deg']]) {
    $(id).value = p[key];
  }
  $('tipstyle').value = p.tip_style;
  $('braces').checked = p.brace_enabled;
  if (p.support_style) $('style').value = p.support_style;
  if (p.branch_angle_deg != null) $('branch').value = p.branch_angle_deg;
  if (p.merge_strength != null) $('merge').value = p.merge_strength;
  syncStyleVisibility();
  showSliderValues();
}

function showSliderValues() {
  $('tip_v').textContent = (+$('tip').value).toFixed(2) + ' mm';
  $('pillar_v').textContent = (+$('pillar').value).toFixed(1) + ' mm';
  $('spacing_v').textContent = (+$('spacing').value).toFixed(2) + ' mm';
  $('overhang_v').textContent = $('overhang').value + '°';
  $('branch_v').textContent = $('branch').value + '°';
  $('merge_v').textContent = (+$('merge').value).toFixed(2);
}
['tip', 'pillar', 'spacing', 'overhang', 'branch', 'merge']
  .forEach(id => $(id).addEventListener('input', showSliderValues));

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
    $('autorotate').disabled = false;
    $('asloaded').disabled = false;

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

/** Opt-in only. Off the default path — see the orientation panel. */
async function runOrient(pick = 0, skip = false) {
  log(skip ? 'restoring the file’s pose …' : 'searching for a better orientation …');
  const r = await postJSON(`/api/orient/${state.sid}`, { pick, skip, overrides: overrides() });
  state.orientations = r.orientations;
  state.applied = r.applied;
  renderOrientations();
  await loadSTL(`/api/mesh/${state.sid}/model`, 'model');
  frameModel();
  const label = r.orientations[r.applied]?.label ?? 'the file’s pose';
  log(`using ${label} (${r.elapsed.toFixed(2)}s)`, 'g');
}

async function reorient(pick, skip) {
  if (!state.sid || state.busy) return;
  busy(true);
  try {
    await runOrient(pick, skip);
    await runPoints();
    await runSupports();
  } catch (err) {
    log(`error: ${err.message}`, 'e');
  } finally {
    busy(false);
  }
}

$('autorotate').onclick = () => reorient(0, false);
$('asloaded').onclick = () => reorient(0, true);

async function runPoints() {
  log('placing support points …');
  const r = await postJSON(`/api/points/${state.sid}`, { overrides: overrides() });
  state.points = r.points;
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
  syncSliders(r.params);
  await loadSTL(`/api/mesh/${state.sid}/supports`, 'supports').catch(() => {
    log('no supports needed', 'g');
  });
  rebuildMarkers();

  const joins = ($('style').value === 'tree') ? 'merges' : 'braces';
  $('stats').innerHTML =
    `<b>${r.points}</b> supports &middot; <b>${r.braces}</b> ${joins}<br>` +
    `<b>${r.faces.toLocaleString()}</b> triangles` +
    (r.dropped ? `<br><span style="color:var(--warn)"><b>${r.dropped}</b> dropped (no clear path)</span>` : '');
  (r.warnings || []).slice(0, 5).forEach(w => log(w, 'w'));
  log(`built in ${r.elapsed.toFixed(2)}s`, 'g');

  ['dl3mf', 'dlstl', 'dlsep'].forEach(id => $(id).disabled = false);
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

function renderOrientations() {
  const host = $('orients');
  host.innerHTML = '';
  if (!state.orientations.length) return;

  state.orientations.forEach((o, i) => {
    const b = document.createElement('button');
    b.className = 'orient' + (i === state.applied ? ' on' : '');
    b.innerHTML = `<span>${o.label || 'candidate ' + i}</span><small>${o.score.toFixed(3)}</small>`;
    b.title = Object.entries(o.terms).map(([k, v]) => `${k}: ${v.toFixed(3)}`).join('\n');
    b.onclick = () => reorient(i, false);
    host.appendChild(b);
  });
}

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
    rebuildMarkers();
    await rerun('geometry');
    return;
  }

  if (markers && show.points) {
    const hit = raycaster.intersectObject(markers, false)[0];
    if (hit && hit.instanceId != null) {
      state.points.splice(hit.instanceId, 1);
      log('deleted a support');
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
['tip', 'pillar'].forEach(id => $(id).addEventListener('change', () => rerun('geometry')));
['spacing', 'overhang'].forEach(id => $(id).addEventListener('change', () => rerun('points')));
$('tipstyle').addEventListener('change', () => rerun('geometry'));
$('braces').addEventListener('change', () => rerun('geometry'));
['branch', 'merge'].forEach(id => $(id).addEventListener('change', () => rerun('geometry')));
$('style').addEventListener('change', () => { syncStyleVisibility(); rerun('geometry'); });
$('preset').addEventListener('change', async () => {
  if (!state.sid || state.busy) return;
  busy(true);
  try {
    const r = await postJSON(`/api/points/${state.sid}`, { preset: $('preset').value });
    state.points = r.points;
    await runSupports();
  } catch (err) { log(`error: ${err.message}`, 'e'); }
  finally { busy(false); }
});

for (const [id, mode] of [['dl3mf', '3mf'], ['dlstl', 'combined'], ['dlsep', 'separate']]) {
  $(id).onclick = () => { location.href = `/api/export/${state.sid}?mode=${mode}`; };
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
    log(`could not reach the server: ${err.message}`, 'e');
  }
})();
