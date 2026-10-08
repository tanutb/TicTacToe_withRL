"""Serve the web demo (web/) and open it in the browser."""
import argparse
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    web = Path(__file__).resolve().parent / "web"
    handler = partial(QuietHandler, directory=str(web))
    url = f"http://127.0.0.1:{args.port}"
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Demo running at {url} (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped")
