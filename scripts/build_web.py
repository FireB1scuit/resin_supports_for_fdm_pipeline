"""Assemble the static bundle: the whole app as files you can drop on a host.

    python scripts/build_web.py dist/

The result is a directory that needs no server of its own — copy it to GitHub
Pages, Cloudflare Pages, Netlify, S3, anything that serves files — and the
pipeline runs in whoever's browser opens it. No API, no CPU, no per-user cost,
and a model never leaves the machine it was opened on.

This is **not a frontend build step**. There is no bundler, no transpiler, no
Node and no package.json — CLAUDE.md rules those out and this respects that.
It does three things a shell copy could do, and stays a script only because
doing them by hand is easy to get subtly wrong:

1. copies ``web/static`` across verbatim;
2. rewrites ``config.js`` to select the worker transport, which is the one line
   of difference between the served app and the hosted one;
3. zips the ``rsupport`` package so the Pyodide worker can unpack it into its
   virtual filesystem — the browser has no site-packages to install into.

The zip is the reason this exists at all: the browser needs the Python source as
one fetchable artefact, and keeping a second copy of the package in ``static/``
would be a copy to forget to update.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "rsupport"
STATIC = PACKAGE / "web" / "static"

#: Not shipped to the browser. `web/app.py` is the FastAPI front end, which the
#: hosted build has no use for and which would drag fastapi/pydantic into a
#: place they cannot be installed.
SKIP_MODULES = {"web/app.py"}

#: Nor is the static directory itself: it is being copied to the bundle root, so
#: including it in the source zip would ship every asset twice.
SKIP_DIRS = {"__pycache__", "web/static"}


def _skip(rel: Path) -> bool:
    posix = rel.as_posix()
    if posix in SKIP_MODULES:
        return True
    return any(part == d or posix.startswith(f"{d}/") for part in rel.parts for d in SKIP_DIRS)


def build_source_zip(dest: Path) -> int:
    """Zip the rsupport package, laid out so ``/rsupport_pkg`` is a sys.path root."""
    written = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(PACKAGE.rglob("*.py")):
            rel = path.relative_to(PACKAGE)
            if _skip(rel):
                continue
            z.write(path, Path("rsupport") / rel)
            written += 1
    return written


def build(out_dir: Path) -> None:
    if not STATIC.is_dir():
        raise SystemExit(f"no static directory at {STATIC}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(STATIC, out_dir)

    config = out_dir / "config.js"
    text = config.read_text(encoding="utf-8")
    swapped = text.replace("globalThis.RSUPPORT_TRANSPORT = 'http';",
                           "globalThis.RSUPPORT_TRANSPORT = 'worker';")
    if swapped == text:
        raise SystemExit(
            "config.js no longer contains the transport line this script rewrites — "
            "the bundle would silently ship the http transport and try to fetch an "
            "API that is not there. Fix the replacement in scripts/build_web.py."
        )
    config.write_text(swapped, encoding="utf-8")

    # GitHub Pages runs Jekyll over whatever it publishes unless this file is
    # present, and Jekyll silently drops anything whose name starts with an
    # underscore. Nothing here does today, but the failure mode is a file that
    # is simply absent in production with no error raised anywhere, so the
    # guard is worth its zero bytes. Written here rather than in the workflow
    # because the workflow is not the only way this bundle reaches a host.
    (out_dir / ".nojekyll").write_bytes(b"")

    modules = build_source_zip(out_dir / "rsupport_src.zip")

    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"wrote {out_dir}")
    print(f"  {modules} python modules zipped")
    print(f"  {total / 1e6:.1f} MB of static files")
    print()
    print("Serve it with any static host. To try it locally:")
    print(f"  python -m http.server -d {out_dir} 8000")
    print()
    print("The Python runtime itself (~35 MB) comes from the jsDelivr CDN on first")
    print("load and is then cached by the browser. To self-host it instead, drop a")
    print("Pyodide release into the bundle and set self.RSUPPORT_PYODIDE_BASE in")
    print("worker.js to point at it.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="dist", type=Path,
                    help="directory to write the bundle into (default: dist)")
    args = ap.parse_args(argv)
    build(args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
