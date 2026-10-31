from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            import base64
            import binascii
            import io
            import json
            from PIL import Image
            import mediscan_core as medscan

            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > 12 * 1024 * 1024:
                self._send(413, {"error": "Image request must be between 1 byte and 12 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            image_data = payload.get("image", "")
            if not isinstance(image_data, str) or not image_data:
                self._send(400, {"error": "The request must include a base64-encoded image."})
                return
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data, validate=True))).convert("RGB")
            values = medscan.analyze_medicine_workflow_v4(image)
            report = "\n\n".join(str(value) for value in values[1:9])
            self._send(200, {"report": report})
        except (ValueError, KeyError, TypeError, binascii.Error, OSError) as error:
            self._send(400, {"error": f"Invalid image request: {error}"})
        except Exception as error:
            self._send(500, {"error": f"Analysis failed: {error}"})

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
