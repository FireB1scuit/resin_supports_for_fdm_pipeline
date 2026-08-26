"""The static bundle assembler.

`scripts/build_web.py` is the only thing standing between the source tree and a
hosted build, and every way it can go wrong is silent: a bundle that still says
`transport = 'http'` looks fine until it tries to fetch an API that is not
there, and one missing module does not surface until Pyodide fails to import it
in somebody else's browser. So the failure modes get pinned here rather than
found in production.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_web import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "rsupport" / "web" / "static"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bundle") / "dist"
    build(out)
    return out


def test_the_bundle_selects_the_worker_transport(bundle):
    """The one line of difference between the served app and the hosted one.
    If this silently stops being rewritten the bundle fetches an API that does
    not exist, and every request fails after a page that looked fine."""
    text = (bundle / "config.js").read_text(encoding="utf-8")
    assert "'worker'" in text
    assert "'http'" not in text


def test_the_bundle_opts_out_of_jekyll(bundle):
    """GitHub Pages runs Jekyll unless told not to, and Jekyll drops anything
    starting with an underscore — without an error, so the file is just missing
    once it is live. `.nojekyll` is the opt-out."""
    assert (bundle / ".nojekyll").exists()


def test_every_static_file_is_carried_across(bundle):
    for src in STATIC.iterdir():
        if src.is_file():
            assert (bundle / src.name).exists(), f"{src.name} missing from the bundle"


def test_the_source_zip_carries_the_whole_package(bundle):
    """Anything importable at runtime has to be in the zip: the browser has no
    site-packages to fall back on."""
    names = set(zipfile.ZipFile(bundle / "rsupport_src.zip").namelist())
    expected = {
        f"rsupport/{p.relative_to(ROOT / 'src' / 'rsupport').as_posix()}"
        for p in (ROOT / "src" / "rsupport").rglob("*.py")
        if "__pycache__" not in p.parts and "static" not in p.parts
    }
    expected.discard("rsupport/web/app.py")  # deliberately excluded, see below
    assert expected <= names, f"missing from the bundle: {sorted(expected - names)}"


def test_the_fastapi_front_end_is_left_out(bundle):
    """`web/app.py` imports fastapi and pydantic, which cannot be installed in
    Pyodide. The hosted build reaches the same logic through `web/browser.py`,
    so shipping app.py would only add an import that must fail."""
    names = zipfile.ZipFile(bundle / "rsupport_src.zip").namelist()
    assert not any(n.endswith("web/app.py") for n in names)


def test_the_static_assets_are_not_shipped_twice(bundle):
    """They are copied to the bundle root already; putting them in the zip as
    well would double a three.js-sized payload."""
    names = zipfile.ZipFile(bundle / "rsupport_src.zip").namelist()
    assert not any("static/" in n for n in names)


def test_the_entry_point_loads_the_config_before_the_app(bundle):
    """config.js is a classic script and app.js a module, so config runs first.
    Reverse them and the transport is read before it is set."""
    html = (bundle / "index.html").read_text(encoding="utf-8")
    assert html.index("config.js") < html.index('type="module"')


def test_rebuilding_over_an_existing_bundle_is_clean(tmp_path):
    """Stale files in a re-used output directory would be served alongside the
    new ones."""
    out = tmp_path / "dist"
    build(out)
    litter = out / "stale.js"
    litter.write_text("// left over", encoding="utf-8")
    build(out)
    assert not litter.exists()


# The workflow is not the bundle, but it is the other half of "what the site
# serves". These two guards were written after the site spent weeks quietly
# serving README.md: `configure-pages`' `enablement` flag turns Pages *on* but
# will not move an already-enabled repo off "Deploy from a branch", so GitHub
# kept building README.md with Jekyll in parallel and that build landed last.
# Deleting either step below brings the whole failure back, and it is invisible
# from a green workflow — hence pinning them here, where the deploy gate runs.

PAGES_WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"


def test_the_workflow_asks_for_the_pages_source_to_be_this_workflow():
    """On a branch source Jekyll renders README.md, deploys it alongside the
    bundle, and usually wins the race. CI is not allowed to change that setting
    with GITHUB_TOKEN, so the request is best-effort — but it is what makes a
    PAT in PAGES_SOURCE_TOKEN fix the site without anyone editing the workflow,
    and it is where the log explains the manual fix when it is refused."""
    text = PAGES_WORKFLOW.read_text(encoding="utf-8")
    assert "build_type=workflow" in text


def test_the_workflow_checks_what_the_site_actually_serves():
    """A deploy going green says nothing about the URL — that is exactly how
    this went unnoticed. The published site has to be asked."""
    text = PAGES_WORKFLOW.read_text(encoding="utf-8")
    assert 'content="Jekyll' in text, "nothing detects a README being served"
    assert "RSUPPORT_TRANSPORT" in text, "nothing confirms the bundle is served"
