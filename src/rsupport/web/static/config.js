// Which transport the UI should use. Read by transport.js at import time.
//
// This copy says "http": it is the one `python -m rsupport.web` and the
// container serve, where there is a FastAPI app to talk to.
//
// `scripts/build_web.py` overwrites this single line with 'worker' when it
// assembles the static bundle, which is why index.html is one file rather than
// two nearly identical ones. A classic script, not a module, so it runs before
// the deferred module that reads it.
globalThis.RSUPPORT_TRANSPORT = 'http';
