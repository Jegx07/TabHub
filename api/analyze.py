import base64
import binascii
import io
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "medicine_metadata.json"
MAX_REQUEST_BYTES = 12 * 1024 * 1024


def load_catalog():
    with METADATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def image_signature(image):
    from PIL import ImageOps

    normalized = ImageOps.fit(image.convert("RGB"), (64, 64))
    return list(normalized.convert("L").getdata())


def similarity(left, right):
    difference = sum(abs(a - b) for a, b in zip(left, right))
    return max(0.0, 100.0 - (difference / (len(left) * 255.0) * 100.0))


def identify(image):
    catalog = load_catalog()
    target_signature = image_signature(image)
    matches = []
    for medicine_id, record in catalog.items():
        references = record.get("reference_images", [])
        if not references and record.get("reference_image"):
            references = [record["reference_image"]]
        best_score = 0.0
        for reference in references:
            path = ROOT / reference.replace("/", str(Path("/")))
            if not path.exists():
                continue
            with Image.open(path) as reference_image:
                best_score = max(best_score, similarity(target_signature, image_signature(reference_image)))
        if best_score >= 85.0:
            matches.append((best_score, medicine_id, record))

    matches.sort(reverse=True, key=lambda match: match[0])
    if not matches or (len(matches) > 1 and matches[0][0] - matches[1][0] < 5.0):
        return (
            "### SECTION A: IDENTIFICATION\n\n"
            "- **Status:** `UNIDENTIFIED`\n"
            "- **Message:** Medicine could not be identified reliably. "
            "Please upload a clearer photograph showing the packaging or printed label."
        )

    score, _, record = matches[0]
    composition = record.get("composition", [])
    composition_text = "\n".join(
        f"- {item.get('ingredient', 'Unknown')}: {item.get('strength', 'Unverified')}"
        for item in composition
    )
    interactions = "\n".join(
        f"- {item.get('interacting_drug', 'Unknown')}: {item.get('risk', 'See pharmacist')}"
        for item in record.get("interactions", [])
    ) or "- No catalog interactions listed."
    return f"""### SECTION A: IDENTIFICATION

- **Status:** `IDENTIFIED` ({score:.1f}/100 Reference Match)
- **Brand Name:** **{record.get("brand_name", "Unknown")}**
- **Generic Name:** {record.get("generic_name", "Unknown")}
- **Strength:** {", ".join(item.get("strength", "Unverified") for item in composition)}
- **Dosage Form:** {record.get("dosage_form", "Unverified")}
- **Manufacturer:** {record.get("manufacturer", "Unverified")}

### SECTION B: COMPOSITION

{composition_text or "- Composition unavailable."}

### SECTION C: USES & APPLICATIONS

**Uses:** {record.get("uses", "Verify approved indications with a clinician.")}

**Mechanism:** {record.get("mechanism_of_action", "Verify with an official monograph.")}

### SECTION D: SAFETY

**Side effects:** {record.get("side_effects", "See official product information.")}

**Serious warnings:** {record.get("serious_adverse_effects", "Seek medical advice for severe reactions.")}

**Precautions:** {record.get("precautions", "Use only as directed by a healthcare professional.")}

### SECTION E: INTERACTIONS

{interactions}

> Informational use only. Confirm the medicine and its suitability with a licensed pharmacist or physician.
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send(200, {"status": "ok", "service": "MediScan AI"})

    def do_POST(self):
        try:
            from PIL import Image

            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._send(413, {"error": "Image request must be between 1 byte and 12 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            image_data = payload.get("image", "")
            if not isinstance(image_data, str) or not image_data:
                self._send(400, {"error": "The request must include a base64-encoded image."})
                return
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data, validate=True)))
            self._send(200, {"report": identify(image)})
        except (ValueError, KeyError, TypeError, binascii.Error, OSError, json.JSONDecodeError) as error:
            self._send(400, {"error": f"Invalid image request: {error}"})
        except Exception as error:
            self._send(500, {"error": f"Analysis failed: {error}"})

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
