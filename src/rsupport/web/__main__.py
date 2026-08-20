import argparse
import os

from . import serve

p = argparse.ArgumentParser(prog="python -m rsupport.web", description="run the browser app")
p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
# PORT is how a container, a compose file or an IDE preview says where to bind.
# The flag still wins, and 8000 is still the default when nobody says anything.
p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
p.add_argument("--no-browser", action="store_true")
a = p.parse_args()

serve(host=a.host, port=a.port, open_browser=not a.no_browser)
