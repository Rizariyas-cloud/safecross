"""Start the SafeCross dashboard demo in your browser.

    python app/run_demo.py

Serves app/web/index.html on http://127.0.0.1:8765 and opens it. Press Ctrl+C to stop.
The page works without internet; with internet it also loads the nicer fonts.
"""
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

PORT = 8765
WEB = Path(__file__).resolve().parent / "web"


def main():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    handler.log_message = lambda *a, **k: None
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/index.html"
        print(f"SafeCross demo running at {url}\nPress Ctrl+C to stop.")
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
