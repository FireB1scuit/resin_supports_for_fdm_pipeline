// How the UI reaches the pipeline. There are two ways and app.js knows neither.
//
//   http    fetch against the FastAPI app — `python -m rsupport.web`, Docker.
//   worker  postMessage to a Pyodide worker running the pipeline in this tab,
//           with no host behind it at all.
//
// Both answer the same routes with the same payloads, because the Python behind
// them is the same module (rsupport.web.core) reached through two front ends.
// That is what lets this file be a swap rather than a fork: `api()` returns
// something with .json() and .arrayBuffer() either way, so every call site in
// app.js is unchanged.
//
// The page picks by setting window.RSUPPORT_TRANSPORT before importing app.js;
// index.html leaves it unset and gets http, browser.html sets 'worker'.

const MODE = globalThis.RSUPPORT_TRANSPORT === 'worker' ? 'worker' : 'http';

export const isServerless = MODE === 'worker';

// ------------------------------------------------------------------- http

async function httpApi(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return res;
}

function httpDownload(path) {
  // The server sets Content-Disposition, so the browser saves rather than
  // navigates — but not via `location.href`. Setting it a second time (the
  // separate-files export downloads model then supports, one click) cancels
  // whichever request that first assignment was still fetching before it
  // ever reached a Content-Disposition header, so only the second file ever
  // landed. A download-anchor click starts an independent browser download
  // instead of a navigation, so two clicks in the same tick don't race.
  const a = document.createElement('a');
  a.href = path;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ----------------------------------------------------------------- worker

let worker = null;
let ready = null;
let seq = 0;
const pending = new Map();
const listeners = new Set();

/** Called with ({phase, detail}) while the runtime boots. */
export function onProgress(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function announce(msg) {
  for (const fn of listeners) {
    try { fn(msg); } catch { /* a bad listener must not stall the boot */ }
  }
}

function startWorker() {
  if (ready) return ready;
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

  ready = new Promise((resolve, reject) => {
    worker.addEventListener('message', (ev) => {
      const msg = ev.data;
      if (msg.type === 'status') { announce(msg); return; }
      if (msg.type === 'ready') { resolve(); return; }
      if (msg.type === 'fatal') {
        // Pyodide is single-threaded and a hard failure takes the interpreter
        // with it, so there is nothing to retry into — say so plainly rather
        // than leaving every later call hanging on a promise nobody resolves.
        const err = new Error(msg.detail);
        for (const { reject: rj } of pending.values()) rj(err);
        pending.clear();
        announce({ phase: 'fatal', detail: msg.detail });
        reject(err);
        return;
      }
      const slot = pending.get(msg.id);
      if (!slot) return;
      pending.delete(msg.id);
      slot.resolve(msg);
    });
    worker.addEventListener('error', (ev) => reject(new Error(ev.message || 'worker failed')));
  });
  return ready;
}

function callWorker(payload, transfer = []) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, ...payload }, transfer);
  });
}

async function request(path, opts = {}) {
  await startWorker();
  const method = (opts.method || 'GET').toUpperCase();
  let body = null;
  let data = null;
  let filename = null;

  if (opts.body instanceof FormData) {
    const file = opts.body.get('file');
    data = await file.arrayBuffer();
    filename = file.name;
  } else if (typeof opts.body === 'string') {
    body = JSON.parse(opts.body);
  }

  // The upload buffer is transferred, not copied — a 50 MB STL should not be
  // duplicated on its way across.
  const res = await callWorker({ type: 'route', method, path, body, data, filename },
                               data ? [data] : []);
  if (!res.ok) throw new Error(res.data?.detail ?? `status ${res.status}`);
  return res;
}

// `opts.signal` is honoured by the http front end and cannot be by this one:
// a request here is a synchronous call into a single-threaded interpreter, and
// there is nothing running alongside it to notice an abort. app.js stops
// between stages instead — see the note above `generate`.
async function workerApi(path, opts = {}) {
  const res = await request(path, opts);
  return {
    json: async () => res.data,
    arrayBuffer: async () => res.binary,
  };
}

async function workerDownload(path) {
  const res = await request(path);
  const blob = new Blob([res.binary], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = res.filename || 'supported.3mf';
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoked on the next turn: revoking synchronously can beat the click.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

// ------------------------------------------------------------------ exports

export const api = MODE === 'worker' ? workerApi : httpApi;
export const download = MODE === 'worker' ? workerDownload : httpDownload;

export async function postJSON(path, body, signal) {
  const res = await api(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
    signal,
  });
  return res.json();
}

/** Boot the runtime early, so the wait happens before a file is dropped. */
export function warmUp() {
  return MODE === 'worker' ? startWorker() : Promise.resolve();
}
