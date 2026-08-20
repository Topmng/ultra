"""Optional in-process HTTP runtime for local tests only.

The submitted image uses Synth's base serving loop
(``VHFT_MINER_ENTRYPOINT=synth_ultra.model``). This server is not copied
into the image.
"""

from __future__ import annotations

import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from synth_ultra.model import predict_percentiles
from synth_ultra.payload import make_sample_payload
from synth_ultra.validate import check_output

HOST = "0.0.0.0"
PORT = 8080


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys_stderr = __import__("sys").stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/"):
            self._send(200, b"ok\n", "text/plain")
            return
        self._send(404, b"not found\n", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send(404, b"not found\n", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = pickle.loads(raw)
            out = check_output(predict_percentiles(payload))
        except Exception as exc:  # noqa: BLE001
            self._send(400, str(exc).encode("utf-8"), "text/plain")
            return
        self._send(200, pickle.dumps(out, protocol=4), "application/octet-stream")


def warmup() -> None:
    payload = make_sample_payload(0)
    out = predict_percentiles(payload)
    check_output(np.asarray(out))


def main() -> None:
    warmup()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"synth-ultra runtime listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
