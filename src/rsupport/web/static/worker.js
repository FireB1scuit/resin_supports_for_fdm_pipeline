// The pipeline, running in this tab.
//
// Boots Pyodide, installs what rsupport needs, unpacks the package into the
// virtual filesystem, and then answers the same routes rsupport.web.app serves
// over HTTP — by calling rsupport.web.browser.route in process.
//
// It runs in a Worker because Pyodide is single-threaded: an orientation search
// or a stage-3 build takes seconds, and on the main thread that is seconds of
// frozen UI.
//
// Two things here are not obvious and both are load-bearing:
//
//   * `sys.platform` is "emscripten" under Pyodide, which is what makes
//     rsupport.avoidance select its raster collision backend. The polygon one
//     cannot run here — it trips a GEOS 3.12 overlay bug whose C++ exception
//     unwinds past the interpreter, killing the runtime outright. See
//     src/rsupport/avoidance.py.
//   * A fatal error therefore cannot be caught and retried. When one happens
//     the worker reports it and stops; the page has to reload to get a runtime
//     back. Expected failures — a bad STL, a missing session — are not this:
//     rsupport.web.browser turns those into ordinary status codes.

const PYODIDE_VERSION = 'v0.28.0';
const PYODIDE_BASE = (self.RSUPPORT_PYODIDE_BASE
  || `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`);

// Everything rsupport imports that Pyodide ships as a built package. trimesh is
// pure Python and comes from PyPI through micropip; mapbox-earcut is not
// available for wasm and is deliberately not required — see CLAUDE.md.
const PACKAGES = ['micropip', 'numpy', 'scipy', 'shapely', 'networkx', 'lxml'];

let pyodide = null;
let route = null;

const status = (phase, detail = '') => self.postMessage({ type: 'status', phase, detail });

async function boot() {
  status('runtime', 'downloading the Python runtime');
  const { loadPyodide } = await import(`${PYODIDE_BASE}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: PYODIDE_BASE });

  status('packages', 'numpy, scipy, shapely');
  await pyodide.loadPackage(PACKAGES);

  status('packages', 'trimesh');
  await pyodide.pyimport('micropip').install('trimesh');

  status('source', 'unpacking rsupport');
  const zip = await (await fetch(new URL('./rsupport_src.zip', import.meta.url))).arrayBuffer();
  await pyodide.unpackArchive(zip, 'zip', { extractDir: '/rsupport_pkg' });

  status('source', 'importing');
  pyodide.runPython(`
import sys
sys.path.insert(0, '/rsupport_pkg')
`);
  route = pyodide.runPython(`
from rsupport.web.browser import route
route
`);

  const backend = pyodide.runPython(`
from rsupport import avoidance
avoidance.backend_name()
`);
  status('ready', `collision backend: ${backend}`);
  self.postMessage({ type: 'ready' });
}

const booted = boot().catch((err) => {
  self.postMessage({ type: 'fatal', detail: String(err && err.message || err) });
  throw err;
});

self.addEventListener('message', async (ev) => {
  const { id, type, method, path, body, data, filename } = ev.data;
  if (type !== 'route') return;
  try {
    await booted;
    // A Python dict rather than a JSON string: Pyodide converts plain objects
    // on the way in, and the payloads here are small.
    const res = route.callKwargs(method, path, {
      body: body === null ? undefined : pyodide.toPy(body),
      data: data ? new Uint8Array(data) : undefined,
      filename: filename ?? undefined,
    });

    const ok = res.ok;
    const st = res.status;
    const payload = res.data ? res.data.toJs({ dict_converter: Object.fromEntries }) : null;
    const name = res.filename;
    let binary = null;
    if (res.body !== null && res.body !== undefined) {
      // Copied out of the wasm heap before the proxy is destroyed, then
      // transferred so the main thread does not pay for it twice.
      binary = res.body.toJs().buffer;
    }
    res.destroy();

    self.postMessage({ id, ok, status: st, data: payload, filename: name, binary },
                     binary ? [binary] : []);
  } catch (err) {
    // Reaching here means the runtime itself failed, not the request — an
    // expected failure would have come back as a status from browser.route.
    self.postMessage({
      id, ok: false, status: 500,
      data: { detail: String(err && err.message || err) },
    });
  }
});
