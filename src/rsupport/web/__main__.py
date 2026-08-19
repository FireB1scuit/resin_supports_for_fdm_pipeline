import argparse

from . import serve

p = argparse.ArgumentParser(prog="python -m rsupport.web", description="run the browser app")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--no-browser", action="store_true")
a = p.parse_args()

serve(host=a.host, port=a.port, open_browser=not a.no_browser)
