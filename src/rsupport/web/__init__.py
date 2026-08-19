"""Local browser app. `python -m rsupport.web` starts it."""

from __future__ import annotations


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    import threading
    import webbrowser

    import uvicorn

    from .app import app

    url = f"http://{host}:{port}/"
    if open_browser:
        # Fire once the server has had a moment to bind, otherwise the tab
        # opens on a connection refused.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"\n  resin supports for FDM  ->  {url}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["serve"]
