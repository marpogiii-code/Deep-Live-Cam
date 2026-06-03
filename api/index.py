"""Health / info endpoint for the Deep-Live-Cam Vercel deployment."""

import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        payload = {
            "status": "ok",
            "project": "Deep-Live-Cam",
            "version": "2.1.6",
            "description": (
                "Real-time face swap and video deepfake "
                "with a single click and only a single image."
            ),
            "repository": "https://github.com/marpogiii-code/Deep-Live-Cam",
            "features": [
                "Real-time face swapping",
                "Mouth mask retention",
                "Multi-face mapping",
                "Video deepfake processing",
                "Live camera support",
            ],
        }
        self.wfile.write(json.dumps(payload, indent=2).encode())
