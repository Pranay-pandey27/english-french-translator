"""JSON API for English-to-French inference."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from translator import HybridTranslator  # noqa: E402


def handler_for(model: HybridTranslator):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._respond(200, {"status": "ok"}) if self.path == "/health" else self._respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/translate":
                self._respond(404, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                self._respond(200, model.translate(payload["text"]))
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._respond(400, {"error": str(exc)})

        def _respond(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/translator.joblib")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    translator = HybridTranslator.load(args.model)
    print(f"Serving translations on http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), handler_for(translator)).serve_forever()
