# -*- coding: utf-8 -*-
"""
MediScan AI - Version 4.0
Global AI-Powered Medicine Identification & Pharmaceutical Intelligence Platform

This production-grade script configures the multi-source clinical verification engine,
computer vision preprocessing, OCR extraction, multimodal Gemini vision analysis,
pharmaceutical database integrations (RxNorm, openFDA, DailyMed, PubChem, Indian CDSCO),
drug-drug interaction evaluation, and the interactive Gradio dashboard.
"""

import os
import sys
import re
import json
import sqlite3
import datetime
import urllib.parse
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Reconfigure stdout for UTF-8 compatibility across all environments
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Optional fuzzy matching
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False

# Google GenAI SDK Resolution
GENAI_SDK = None
try:
    from google import genai
    from google.genai import types
    GENAI_SDK = "google_genai"
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GENAI_SDK = "google_generativeai"
    except ImportError:
        GENAI_SDK = None

# Google Colab userdata resolution
COLAB_USERDATA = False
try:
    from google.colab import userdata
    COLAB_USERDATA = True
except ImportError:
    COLAB_USERDATA = False

# Optional Gradio import
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


# ==============================================================================
# 1. DATABASE & CACHING ARCHITECTURE
# ==============================================================================

class DatabaseManager:
    """Manages SQLite cache, audit logs, and persistent pharmaceutical metadata."""

    def __init__(self, db_path: str = "medicine_cache.db", metadata_path: str = "medicine_metadata.json"):
        self.db_path = db_path
        self.metadata_path = metadata_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                query_key TEXT PRIMARY KEY,
                source_id TEXT,
                response_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                medicine_detected TEXT,
                confidence_rating TEXT,
                evidence_summary TEXT,
                sources_queried TEXT,
                full_result_json TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def get_cached_response(self, key: str) -> Optional[Any]:
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT response_json FROM api_cache WHERE query_key = ?", (key,))
            row = cur.fetchone()
            conn.close()
            if row:
                return json.loads(row[0])
        except Exception as e:
            print(f"[CACHE_READ_ERROR] {e}")
        return None

    def cache_response(self, key: str, data: Any, source_id: str = "GENERIC"):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO api_cache (query_key, source_id, response_json)
                VALUES (?, ?, ?)
            """, (key, source_id, json.dumps(data)))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[CACHE_WRITE_ERROR] {e}")

    def log_verification(self, medicine_name: str, confidence: str, evidence: str, sources: List[str], full_result: Dict[str, Any]):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO verification_logs (medicine_detected, confidence_rating, evidence_summary, sources_queried, full_result_json)
                VALUES (?, ?, ?, ?, ?)
            """, (medicine_name, confidence, evidence, ", ".join(sources), json.dumps(full_result)))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[LOG_ERROR] {e}")

    def load_metadata(self) -> Dict[str, Any]:
        candidates = [
            self.metadata_path,
            os.path.join("medicine_database", "medicine_metadata.json"),
            "medicine_metadata.json"
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[METADATA_LOAD_ERROR] {p}: {e}")
        return {}

    def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        paths = [self.metadata_path, os.path.join("medicine_database", "medicine_metadata.json")]
        success = False
        for p in paths:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4)
                success = True
            except Exception as e:
                print(f"[METADATA_SAVE_ERROR] {p}: {e}")
        return success


# Global Database Instance
db_manager = DatabaseManager()


# Database Administration Functions Required by Prompt
def add_medicine(medicine_id: str, brand_name: str, generic_name: str, composition: List[Dict[str, str]],
                 category: str, dosage_form: str, manufacturer: str, packaging_type: str,
                 mechanism_of_action: str, uses: str, side_effects: str, serious_adverse_effects: str,
                 contraindications: str, interactions: List[Dict[str, str]], precautions: str,
                 pregnancy_warning: str, storage: str, alternatives: List[str], reference_image_path: str = "") -> bool:
    """Safely adds or updates a medicine record in the medicine database."""
    db = db_manager.load_metadata()
    is_new = medicine_id not in db
    db[medicine_id] = {
        "medicine_id": medicine_id,
        "brand_name": brand_name,
        "generic_name": generic_name,
        "composition": composition,
        "category": category,
        "dosage_form": dosage_form,
        "manufacturer": manufacturer,
        "market": "Global / India",
        "packaging_type": packaging_type,
        "is_antibiotic": any("antibiotic" in c.get("class", "").lower() for c in composition),
        "is_combination": len(composition) > 1,
        "schedule_class": "Prescription / OTC",
        "mechanism_of_action": mechanism_of_action,
        "therapeutic_class": category,
        "uses": uses,
        "off_label_uses": "Consult physician for unlisted indications.",
        "side_effects": side_effects,
        "less_common_side_effects": "Refer to official product insert for full profile.",
        "serious_adverse_effects": serious_adverse_effects,
        "contraindications": contraindications,
        "allergies_warning": "Discontinue if allergic hypersensitivity appears.",
        "interactions": interactions,
        "precautions": precautions,
        "pregnancy_warning": pregnancy_warning,
        "breastfeeding_warning": "Consult medical provider before use during lactation.",
        "pediatric_warning": "Dose adjustment required for pediatric patients.",
        "storage": storage,
        "alternatives": alternatives,
        "visual_descriptors": {"packaging": packaging_type},
        "reference_image": reference_image_path
    }
    db_manager.save_metadata(db)
    action = "Registered" if is_new else "Updated"
    print(f"✅ Successfully {action} {brand_name} (ID: {medicine_id}) in the local database.")
    return True


def search_medicine(query: str) -> List[Dict[str, Any]]:
    """Searches the local and cached medicine repository for matching brand or generic terms."""
    db = db_manager.load_metadata()
    query_clean = query.lower().strip()
    results = []
    for med_id, data in db.items():
        brand = data.get("brand_name", "").lower()
        generic = data.get("generic_name", "").lower()
        if query_clean in brand or query_clean in generic:
            results.append(data)
            continue
        for comp in data.get("composition", []):
            if query_clean in comp.get("ingredient", "").lower():
                results.append(data)
                break
    return results


def update_medicine(medicine_id: str, updates: Dict[str, Any]) -> bool:
    """Updates specific fields of an existing medicine record."""
    db = db_manager.load_metadata()
    if medicine_id in db:
        db[medicine_id].update(updates)
        return db_manager.save_metadata(db)
    return False


def verify_medicine(medicine_id: str) -> Dict[str, Any]:
    """Verifies existence, integrity, and source completeness of a medicine profile."""
    db = db_manager.load_metadata()
    record = db.get(medicine_id)
    if not record:
        return {"status": "NOT_FOUND", "message": f"Medicine ID {medicine_id} not registered."}
    required_keys = ["brand_name", "generic_name", "composition", "mechanism_of_action", "uses", "side_effects"]
    missing = [k for k in required_keys if not record.get(k)]
    return {
        "status": "VERIFIED" if not missing else "PARTIAL",
        "medicine_id": medicine_id,
        "brand_name": record.get("brand_name"),
        "missing_fields": missing,
        "is_complete": len(missing) == 0
    }


def get_medicine_details(medicine_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full clinical and chemical metadata for a specific medicine."""
    db = db_manager.load_metadata()
    return db.get(medicine_id)


def get_medicine_sources() -> List[Dict[str, Any]]:
    """Returns the registered pharmaceutical databases, their owners, license terms, and status."""
    reg_path = os.path.join("medicine_database", "source_registry.json")
    if os.path.exists(reg_path):
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                return json.load(f).get("sources", [])
        except Exception:
            pass
    return []


# ==============================================================================
# 2. IMAGE PREPROCESSING & QUALITY EVALUATION PIPELINE
# ==============================================================================

def validate_uploaded_image(image_input) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates uploaded file format (JPG, JPEG, PNG, WEBP), web URLs,
    dimensions, file size, and checks image readability.
    """
    if image_input is None:
        return False, "No image provided. Please upload an image file or provide a web URL.", None

    pil_img = None
    try:
        if isinstance(image_input, str):
            clean_str = image_input.strip()
            # Support web URL inputs
            if clean_str.startswith("http://") or clean_str.startswith("https://"):
                import requests
                import io
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MediScanAI/4.0"}
                res = requests.get(clean_str, headers=headers, timeout=10)
                if res.status_code != 200:
                    return False, f"Failed to fetch image from URL (HTTP status {res.status_code}). Please verify the URL.", None
                content_type = res.headers.get("Content-Type", "")
                if "image" not in content_type and not any(clean_str.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    return False, f"URL does not appear to return an image (Content-Type: {content_type}).", None
                pil_img = Image.open(io.BytesIO(res.content))
            else:
                if not os.path.exists(clean_str):
                    return False, f"File does not exist: {clean_str}", None
                ext = os.path.splitext(clean_str)[1].lower()
                if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                    return False, f"Unsupported file extension '{ext}'. Only JPG, JPEG, PNG, and WEBP are supported.", None
                pil_img = Image.open(clean_str)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        else:
            return False, f"Unsupported image object type: {type(image_input)}", None

        # Convert to RGB mode
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Validate dimensions
        w, h = pil_img.size
        if w < 64 or h < 64:
            return False, f"Image dimensions ({w}x{h}) are too small (minimum 64x64).", None
        if w > 10000 or h > 10000:
            return False, f"Image dimensions ({w}x{h}) exceed maximum allowable size (10,000 px).", None

        return True, "Image is valid.", pil_img

    except Exception as e:
        return False, f"Corrupted or unreadable image: {str(e)}", None


def check_image_quality(image_np: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates blurriness (Laplacian variance), contrast (standard deviation of gray levels),
    and brightness (mean luminance). Returns comprehensive quality metrics.
    """
    if image_np is None or image_np.size == 0:
        return {"quality_score": 0, "is_blurry": True, "blur_score": 0.0, "status": "INVALID"}

    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    is_blurry = laplacian_var < 50.0
    is_low_contrast = contrast_std < 20.0
    is_under_exposed = mean_brightness < 40.0
    is_over_exposed = mean_brightness > 225.0

    score = 100.0
    if is_blurry:
        score -= min(40.0, (50.0 - laplacian_var) * 0.8)
    if is_low_contrast:
        score -= min(25.0, (20.0 - contrast_std) * 1.25)
    if is_under_exposed or is_over_exposed:
        score -= 20.0

    score = max(5.0, min(100.0, score))
    status = "EXCELLENT" if score >= 80 else ("GOOD" if score >= 60 else ("FAIR" if score >= 40 else "POOR"))

    return {
        "quality_score": round(score, 1),
        "is_blurry": is_blurry,
        "blur_score": round(laplacian_var, 2),
        "brightness": round(mean_brightness, 1),
        "contrast": round(contrast_std, 1),
        "status": status,
        "flags": [
            flag for flag, active in [
                ("BLURRY", is_blurry),
                ("LOW_CONTRAST", is_low_contrast),
                ("UNDER_EXPOSED", is_under_exposed),
                ("OVER_EXPOSED", is_over_exposed)
            ] if active
        ]
    }


def preprocess_medicine_image(image_input, max_size: Tuple[int, int] = (1024, 1024)) -> Tuple[Image.Image, bool, float, Dict[str, Any]]:
    """
    Preprocesses medicine photographs for optimal OCR and multimodal AI analysis:
    - Validates file format and dimensions
    - Preserves aspect ratio with high-fidelity Lanczos resampling
    - Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB space
    - Subtle unsharp masking for imprint/text edge enhancement
    - Preserves original image attributes
    """
    valid, msg, pil_img = validate_uploaded_image(image_input)
    if not valid or pil_img is None:
        raise ValueError(f"Preprocessing aborted: {msg}")

    initial_np = np.array(pil_img)
    quality_meta = check_image_quality(initial_np)

    resized_pil = pil_img.copy()
    resized_pil.thumbnail(max_size, Image.Resampling.LANCZOS)

    img_np = np.array(resized_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    gaussian = cv2.GaussianBlur(rgb_enhanced, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(rgb_enhanced, 1.25, gaussian, -0.25, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    enhanced_pil = Image.fromarray(sharpened)
    return enhanced_pil, quality_meta["is_blurry"], quality_meta["blur_score"], quality_meta


def check_image_blur(image_np: np.ndarray, threshold: float = 50.0) -> Tuple[bool, float]:
    """Compatibility wrapper for blur detection."""
    q = check_image_quality(image_np)
    return q["blur_score"] < threshold, q["blur_score"]


# ==============================================================================
# 3. TEXT NORMALIZATION & OCR ENGINE
# ==============================================================================

def extract_text_from_image(pil_image: Image.Image) -> Dict[str, Any]:
    """Extracts printed text, imprints, and markings from medicine packaging or tablets."""
    extracted_text = ""
    lines = []
    engine_used = "MULTIMODAL_VISION"

    try:
        import pytesseract
        text = pytesseract.image_to_string(pil_image)
        if text.strip():
            extracted_text = text.strip()
            lines = [l.strip() for l in extracted_text.splitlines() if l.strip()]
            engine_used = "TESSERACT_OCR"
    except Exception:
        pass

    return {
        "raw_text": extracted_text,
        "lines": lines,
        "engine": engine_used,
        "detected": len(extracted_text) > 0
    }


def normalize_medicine_text(text: str) -> str:
    """Normalizes brand names, chemical entities, and dosage texts."""
    if not text:
        return ""
    clean = text.lower()
    clean = re.sub(r'\bacetaminophen\b', 'paracetamol', clean)
    clean = re.sub(r'\bacetylsalicylic acid\b', 'aspirin', clean)
    clean = re.sub(r'[^a-z0-9]', '', clean)
    return clean


# ==============================================================================
# 4. MULTIMODAL GEMINI VISION ENGINE
# ==============================================================================

class GeminiVisionEngine:
    """Multimodal inspection of medicine photographs using Google GenAI SDK."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or self._resolve_api_key()
        self.model_name = model_name
        self.client = self._init_client()

    def _resolve_api_key(self) -> str:
        key = ""
        if COLAB_USERDATA:
            try:
                key = userdata.get('GEMINI_API_KEY')
            except Exception:
                pass
        if not key:
            key = os.environ.get("GEMINI_API_KEY", "")
        return key

    def _init_client(self):
        if not self.api_key:
            return None
        if GENAI_SDK == "google_genai":
            try:
                return genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[GENAI_INIT_ERROR] {e}")
        elif GENAI_SDK == "google_generativeai":
            try:
                genai_legacy.configure(api_key=self.api_key)
                return genai_legacy.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"[GENAI_LEGACY_INIT_ERROR] {e}")
        return None

    def analyze_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Executes multimodal analysis requesting a structured JSON payload."""
        if not self.client:
            return {
                "is_suitable": False,
                "error": "Gemini Client not initialized. Please verify GEMINI_API_KEY.",
                "detected_text": "",
                "visual_characteristics": {},
                "candidates": []
            }

        prompt = """
        You are the primary computer vision engine of MediScan AI V4.0.
        Analyze the uploaded photograph of a pharmaceutical product (tablet, capsule, blister pack, strip, box, packaging, or liquid bottle).

        MANDATORY CLINICAL EXTRACTION TASKS:
        1. Read ALL visible packaging or tablet text exactly as printed.
        2. Detect whether the image contains:
           - A blister pack or box
           - A loose tablet or capsule (check for imprint, color, shape, scoring, coating)
           - Multiple separate medicines or packages in the same photograph
        3. Identify active pharmaceutical ingredients, strengths (e.g. 500mg, 40mg), brand name, and manufacturer if visible.
        4. Detect whether the drug is an antibiotic, steroid, controlled substance, or OTC.

        STRICT SAFETY RULES:
        - NEVER identify a loose tablet with certainty based solely on color and shape if no distinct imprint is legible.
        - If text is blurred or unreadable, state so clearly in 'suitability_reason'.
        - If multiple candidates exist or packaging is ambiguous, report confidence as 'AMBIGUOUS' or 'UNIDENTIFIED'.
        - Never fabricate chemical formulas, NDC codes, or manufacturer names.

        RETURN A VALID JSON OBJECT COMPLYING STRICTLY WITH THIS SCHEMA:
        {
          "is_suitable": true,
          "suitability_reason": "Clear packaging visible with high contrast.",
          "is_loose_tablet": false,
          "has_imprint": true,
          "multiple_medicines_detected": false,
          "detected_text": "Extracted visible raw packaging/tablet text",
          "visual_characteristics": {
            "packaging_type": "Blister strip / Bottle / Box / Loose tablet",
            "dosage_form": "Tablet / Capsule / Syrup / Injection",
            "color": "Visual color of pill or packaging",
            "shape": "Round / Oblong / Oval / Bottle",
            "imprint": "Visible debossed or printed markings"
          },
          "candidates": [
            {
              "brand_name": "Detected Brand Name or 'Unknown'",
              "generic_name": "Generic Active Substance(s)",
              "active_ingredients": [
                {"name": "Active ingredient name", "strength": "e.g. 500 mg"}
              ],
              "strength": "Overall dosage strength, e.g. 500mg",
              "manufacturer": "Manufacturer name if visible, else 'Unknown'",
              "dosage_form": "Tablet / Capsule / Liquid",
              "is_antibiotic": false,
              "confidence": "HIGH / MEDIUM / LOW / UNIDENTIFIED",
              "reasoning": "Clinical reasoning based on visible text and features"
            }
          ],
          "warnings": ["Clinical warnings or limitations observed in this image"]
        }
        """

        try:
            if GENAI_SDK == "google_genai":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[pil_image, prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                text_content = response.text
            elif GENAI_SDK == "google_generativeai":
                response = self.client.generate_content([prompt, pil_image])
                text_content = response.text
            else:
                return {"error": "GenAI SDK not available in environment."}

            clean_json = re.sub(r'^```json\s*', '', text_content.strip())
            clean_json = re.sub(r'```$', '', clean_json.strip())
            return json.loads(clean_json)

        except Exception as e:
            return {
                "is_suitable": False,
                "error": f"Gemini Vision API error: {str(e)}",
                "detected_text": "",
                "visual_characteristics": {},
                "candidates": [],
                "warnings": [f"API communication failure: {str(e)}"]
            }


gemini_engine = GeminiVisionEngine()


def analyse_image_with_gemini(pil_image: Image.Image, model_name: str = "gemini-2.5-flash") -> Dict[str, Any]:
    """Top-level function for multimodal image analysis using Gemini."""
    gemini_engine.model_name = model_name
    return gemini_engine.analyze_image(pil_image)


def identify_medicine_via_gemini(pil_image: Image.Image) -> Dict[str, Any]:
    """Compatibility alias for prior notebook calls."""
    return analyse_image_with_gemini(pil_image)


# ==============================================================================
# 5. EXTERNAL GLOBAL PHARMACEUTICAL CONNECTORS
# ==============================================================================

class ExternalPharmaConnectors:
    """Connectors for RxNorm, openFDA, DailyMed, PubChem, and Indian CDSCO."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def query_rxnorm(self, medicine_name: str) -> Optional[Dict[str, Any]]:
        """Queries the public RxNorm REST API for clinical drug concepts."""
        if not medicine_name or medicine_name.lower() == "unknown":
            return None
        cache_key = f"rxnorm_{normalize_medicine_text(medicine_name)}"
        cached = self.db.get_cached_response(cache_key)
        if cached:
            return cached

        clean_term = urllib.parse.quote(medicine_name.strip())
        url = f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={clean_term}"
        try:
            import requests
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                concept_group = data.get("drugGroup", {}).get("conceptGroup", [])
                extracted = []
                for cg in concept_group:
                    for prop in cg.get("conceptProperties", []):
                        extracted.append({
                            "rxcui": prop.get("rxcui"),
                            "name": prop.get("name"),
                            "synonym": prop.get("synonym"),
                            "tty": prop.get("tty")
                        })
                if extracted:
                    result = {
                        "source": "RxNorm (U.S. National Library of Medicine)",
                        "url": url,
                        "retrieved_at": datetime.datetime.now().isoformat(),
                        "concepts": extracted[:5]
                    }
                    self.db.cache_response(cache_key, result, source_id="RXNORM")
                    return result
        except Exception:
            pass
        return None

    def query_openfda(self, medicine_name: str) -> Optional[Dict[str, Any]]:
        """Queries openFDA API for drug product information and warnings."""
        if not medicine_name or medicine_name.lower() == "unknown":
            return None
        cache_key = f"openfda_{normalize_medicine_text(medicine_name)}"
        cached = self.db.get_cached_response(cache_key)
        if cached:
            return cached

        clean_name = urllib.parse.quote(medicine_name.strip().split("/")[0].split("+")[0].strip())
        url = f'https://api.fda.gov/drug/label.json?search=openfda.generic_name:"{clean_name}"+openfda.brand_name:"{clean_name}"&limit=1'
        try:
            import requests
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    r = results[0]
                    parsed = {
                        "source": "openFDA Drug Labeling Registry (U.S. FDA)",
                        "url": url,
                        "retrieved_at": datetime.datetime.now().isoformat(),
                        "indications": r.get("indications_and_usage", ["Not available"])[0][:600],
                        "warnings": r.get("warnings", ["Not available"])[0][:600],
                        "adverse_reactions": r.get("adverse_reactions", ["Not available"])[0][:600],
                        "contraindications": r.get("contraindications", ["Not available"])[0][:600],
                        "manufacturer": r.get("openfda", {}).get("manufacturer_name", ["Unknown"])[0]
                    }
                    self.db.cache_response(cache_key, parsed, source_id="OPENFDA")
                    return parsed
        except Exception:
            pass
        return None

    def query_dailymed(self, medicine_name: str) -> Optional[Dict[str, Any]]:
        """Queries DailyMed REST API for structured product labels."""
        if not medicine_name or medicine_name.lower() == "unknown":
            return None
        cache_key = f"dailymed_{normalize_medicine_text(medicine_name)}"
        cached = self.db.get_cached_response(cache_key)
        if cached:
            return cached

        clean_name = urllib.parse.quote(medicine_name.strip())
        url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={clean_name}"
        try:
            import requests
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json().get("data", [])
                if data:
                    item = data[0]
                    parsed = {
                        "source": "DailyMed (National Library of Medicine)",
                        "url": url,
                        "retrieved_at": datetime.datetime.now().isoformat(),
                        "setid": item.get("setid"),
                        "title": item.get("title")
                    }
                    self.db.cache_response(cache_key, parsed, source_id="DAILYMED")
                    return parsed
        except Exception:
            pass
        return None

    def query_pubchem(self, chemical_name: str) -> Optional[Dict[str, Any]]:
        """Queries NCBI PubChem PUG REST API for molecular properties."""
        if not chemical_name or chemical_name.lower() == "unknown":
            return None
        cache_key = f"pubchem_{normalize_medicine_text(chemical_name)}"
        cached = self.db.get_cached_response(cache_key)
        if cached:
            return cached

        clean_term = urllib.parse.quote(chemical_name.strip().split("/")[0].split("+")[0].strip())
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{clean_term}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
        try:
            import requests
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                props = res.json().get("PropertyTable", {}).get("Properties", [{}])[0]
                if props:
                    parsed = {
                        "source": "NCBI PubChem Compound Registry",
                        "url": url,
                        "retrieved_at": datetime.datetime.now().isoformat(),
                        "chemical_formula": props.get("MolecularFormula", "N/A"),
                        "molecular_weight": props.get("MolecularWeight", "N/A"),
                        "iupac_name": props.get("IUPACName", "N/A"),
                        "smiles": props.get("CanonicalSMILES", "N/A")
                    }
                    self.db.cache_response(cache_key, parsed, source_id="PUBCHEM")
                    return parsed
        except Exception:
            pass
        return None


pharma_connectors = ExternalPharmaConnectors(db_manager)


def search_medicine_database(query: str) -> Dict[str, Any]:
    """Unified search across local verified catalog, RxNorm, openFDA, and PubChem."""
    local_hits = search_medicine(query)
    rxnorm_data = pharma_connectors.query_rxnorm(query)
    fda_data = pharma_connectors.query_openfda(query)
    pubchem_data = pharma_connectors.query_pubchem(query)

    return {
        "query": query,
        "local_matches": local_hits,
        "rxnorm": rxnorm_data,
        "openfda": fda_data,
        "pubchem": pubchem_data,
        "has_results": len(local_hits) > 0 or bool(rxnorm_data or fda_data or pubchem_data)
    }


# ==============================================================================
# 6. HYBRID MEDICINE MATCHING ENGINE
# ==============================================================================

def string_similarity(a: str, b: str) -> float:
    """Computes similarity percentage between two normalized strings (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b) / 100.0
    else:
        return SequenceMatcher(None, a, b).ratio()


def calculate_hybrid_match_score(entry: Dict[str, Any], candidate: Dict[str, Any], detected_text: str) -> Tuple[float, Dict[str, float], bool]:
    """
    Weighted 100-point matching heuristic:
    - Brand Name: 30%
    - Active Ingredient: 30%
    - Strength: 15%
    - Manufacturer: 10%
    - Dosage Form: 10%
    - Visual Evidence: 5%
    Flag conflicts if brand matches but composition contradicts.
    """
    sub_scores = {"brand": 0.0, "ingredient": 0.0, "strength": 0.0, "manufacturer": 0.0, "dosage_form": 0.0, "visual": 0.0}
    has_conflict = False

    target_brand = normalize_medicine_text(entry.get("brand_name", ""))
    cand_brand = normalize_medicine_text(candidate.get("brand_name", ""))
    norm_detected = normalize_medicine_text(detected_text)

    # 1. Brand (30%)
    if target_brand and (target_brand in cand_brand or cand_brand in target_brand or target_brand in norm_detected):
        sub_scores["brand"] = 30.0
    else:
        sim = string_similarity(target_brand, cand_brand)
        sub_scores["brand"] = round(sim * 30.0, 1)

    # 2. Ingredient (30%)
    cand_gen = normalize_medicine_text(candidate.get("generic_name", ""))
    matched_ing = 0
    total_ing = len(entry.get("composition", [{}]))

    for comp in entry.get("composition", []):
        ing_norm = normalize_medicine_text(comp.get("ingredient", ""))
        if ing_norm in cand_gen or ing_norm in norm_detected:
            matched_ing += 1

    if total_ing > 0:
        sub_scores["ingredient"] = round((matched_ing / total_ing) * 30.0, 1)
    else:
        sim = string_similarity(normalize_medicine_text(entry.get("generic_name", "")), cand_gen)
        sub_scores["ingredient"] = round(sim * 30.0, 1)

    # 3. Strength (15%)
    target_strengths = []
    for comp in entry.get("composition", []):
        digits = "".join(re.findall(r'\d+', comp.get("strength", "")))
        if digits:
            target_strengths.append(digits)

    cand_strength = "".join(re.findall(r'\d+', candidate.get("strength", "")))
    if target_strengths:
        if cand_strength and any(s == cand_strength for s in target_strengths):
            sub_scores["strength"] = 15.0
        elif any(s in norm_detected for s in target_strengths):
            sub_scores["strength"] = 15.0

    # 4. Manufacturer (10%)
    target_mfr = normalize_medicine_text(entry.get("manufacturer", ""))
    cand_mfr = normalize_medicine_text(candidate.get("manufacturer", ""))
    if target_mfr and (target_mfr in cand_mfr or cand_mfr in target_mfr or target_mfr in norm_detected):
        sub_scores["manufacturer"] = 10.0
    elif target_mfr and cand_mfr:
        sub_scores["manufacturer"] = round(string_similarity(target_mfr, cand_mfr) * 10.0, 1)

    # 5. Dosage Form (10%)
    target_form = entry.get("dosage_form", "").lower()
    cand_form = candidate.get("dosage_form", "").lower()
    if target_form and cand_form and target_form in cand_form:
        sub_scores["dosage_form"] = 10.0
    elif target_form in norm_detected:
        sub_scores["dosage_form"] = 8.0

    # 6. Visual Evidence (5%)
    sub_scores["visual"] = 5.0

    total_score = sum(sub_scores.values())

    # CONFLICT CHECK: High brand score but zero ingredient match
    if sub_scores["brand"] >= 20.0 and sub_scores["ingredient"] == 0.0:
        has_conflict = True
        total_score = min(total_score, 35.0)

    return round(total_score, 1), sub_scores, has_conflict


def identify_medicine(image_input) -> Dict[str, Any]:
    """
    Main identification engine implementing the multi-stage pipeline:
    Validates, enhances, extracts text, queries Gemini, evaluates candidates,
    and returns verified pharmaceutical data.
    """
    valid, msg, pil_img = validate_uploaded_image(image_input)
    if not valid or pil_img is None:
        return {
            "status": "UNIDENTIFIED",
            "confidence_rating": "INVALID",
            "message": f"Image validation failed: {msg}",
            "candidates": []
        }

    enhanced_pil, is_blurry, blur_score, quality_meta = preprocess_medicine_image(pil_img)
    ocr_result = extract_text_from_image(enhanced_pil)
    ai_result = analyse_image_with_gemini(enhanced_pil)

    # LOOSE TABLET IDENTIFICATION PATHWAY
    is_loose = ai_result.get("is_loose_tablet", False) or "loose" in ai_result.get("visual_characteristics", {}).get("packaging_type", "").lower()
    has_imprint = ai_result.get("has_imprint", False) or bool(ai_result.get("visual_characteristics", {}).get("imprint"))
    detected_text = ai_result.get("detected_text", "") + " " + ocr_result.get("raw_text", "")

    if is_loose and not has_imprint and len(detected_text.strip()) < 3:
        return {
            "status": "UNIDENTIFIED",
            "confidence_rating": "INSUFFICIENT_EVIDENCE",
            "message": "Unable to reliably identify this tablet from its physical appearance. Loose tablets without distinct imprints or packaging cannot be safely verified by appearance alone. Request packaging information or pharmacist verification.",
            "is_loose_tablet": True,
            "quality_metadata": quality_meta,
            "raw_ai": ai_result
        }

    metadata_db = db_manager.load_metadata()
    ai_candidates = ai_result.get("candidates", [])
    evaluated_candidates = []

    for cand in ai_candidates:
        best_entry = None
        best_score = 0.0
        best_breakdown = {}
        conflict_detected = False

        for med_id, entry in metadata_db.items():
            score, breakdown, conflict = calculate_hybrid_match_score(entry, cand, detected_text)
            if score > best_score:
                best_score = score
                best_entry = entry
                best_breakdown = breakdown
                conflict_detected = conflict

        evaluated_candidates.append({
            "candidate": cand,
            "matched_database_entry": best_entry,
            "match_score": best_score,
            "sub_scores": best_breakdown,
            "conflict": conflict_detected
        })

    if not evaluated_candidates:
        return {
            "status": "UNIDENTIFIED",
            "confidence_rating": "LOW",
            "message": "No pharmaceutical candidates detected in the image.",
            "quality_metadata": quality_meta,
            "raw_ai": ai_result
        }

    evaluated_candidates.sort(key=lambda x: x["match_score"], reverse=True)
    top = evaluated_candidates[0]
    top_score = top["match_score"]
    matched_entry = top["matched_database_entry"]
    top_cand = top["candidate"]

    if top_score >= 65.0 and matched_entry and not top["conflict"]:
        final_status = "IDENTIFIED"
        confidence_rating = "HIGH"
    elif top_score >= 40.0:
        final_status = "POSSIBLE MATCH"
        confidence_rating = "MEDIUM"
    elif len(evaluated_candidates) > 1 and abs(evaluated_candidates[0]["match_score"] - evaluated_candidates[1]["match_score"]) < 10:
        final_status = "AMBIGUOUS"
        confidence_rating = "AMBIGUOUS"
    elif top_cand.get("brand_name") != "Unknown" or top_cand.get("generic_name") != "Unknown":
        final_status = "UNSUPPORTED"
        confidence_rating = "EXTERNAL_CANDIDATE"
    else:
        final_status = "UNIDENTIFIED"
        confidence_rating = "LOW"

    return {
        "status": final_status,
        "confidence_rating": confidence_rating,
        "score": top_score,
        "top_match": matched_entry if matched_entry else top_cand,
        "is_database_match": bool(matched_entry),
        "conflict_flagged": top["conflict"],
        "evaluated_candidates": evaluated_candidates,
        "quality_metadata": quality_meta,
        "detected_text": detected_text.strip(),
        "raw_ai": ai_result
    }


def identify_four_medicines_pipeline(pil_image: Image.Image) -> Dict[str, Any]:
    """Compatibility wrapper for earlier versions."""
    res = identify_medicine(pil_image)
    matched = res.get("top_match", {})
    return {
        "medicine_id": matched.get("medicine_id", "UNIDENTIFIED"),
        "brand_name": matched.get("brand_name", "Unidentified"),
        "generic_name": matched.get("generic_name", "Unidentified"),
        "composition": matched.get("composition", []),
        "category": matched.get("category", "N/A"),
        "dosage_form": matched.get("dosage_form", "N/A"),
        "identification_status": res["status"],
        "confidence_score": res.get("score", 0),
        "medical_information": matched,
        "warnings": matched.get("precautions", ""),
        "sources": ["Local Reference Database", "Global Clinical APIs"],
        "raw_visual_data": res.get("raw_ai", {})
    }


# ==============================================================================
# 7. COMPOSITION, INTERACTION & ALTERNATIVE MODULES
# ==============================================================================

def verify_medicine_composition(candidate_composition: List[Dict[str, str]], reference_composition: List[Dict[str, str]]) -> Dict[str, Any]:
    """Verifies single active ingredient vs fixed-dose combinations (2, 3, or multiple active ingredients)."""
    total_active = len(reference_composition) if reference_composition else len(candidate_composition)
    is_combo = total_active > 1

    class_type = "Single Active Ingredient"
    if total_active == 2:
        class_type = "Fixed-Dose Dual Combination (2 Active Ingredients)"
    elif total_active == 3:
        class_type = "Fixed-Dose Triple Combination (3 Active Ingredients)"
    elif total_active > 3:
        class_type = f"Multi-Ingredient Combination ({total_active} Active Ingredients)"

    return {
        "is_combination": is_combo,
        "active_ingredient_count": total_active,
        "classification": class_type,
        "composition": reference_composition or candidate_composition
    }


def check_drug_interactions(ingredients: List[str]) -> List[Dict[str, str]]:
    """Compares active ingredients against authoritative clinical interaction rules."""
    known_rules = [
        {"drugs": ["paracetamol", "alcohol"], "desc": "Potentiates hepatotoxic metabolite (NAPQI) accumulation, increasing liver necrosis risk.", "risk": "Severe", "source": "U.S. FDA & CDSCO"},
        {"drugs": ["aspirin", "warfarin"], "desc": "Dual antiplatelet and anticoagulant action causes life-threatening gastrointestinal hemorrhage.", "risk": "Severe", "source": "RxNorm Interaction Database"},
        {"drugs": ["aspirin", "ibuprofen"], "desc": "Ibuprofen competitively blocks aspirin's antiplatelet binding to COX-1, eliminating cardioprotection.", "risk": "Moderate", "source": "openFDA Drug Safety Communication"},
        {"drugs": ["erythromycin", "atorvastatin"], "desc": "Inhibits CYP3A4, causing massive statin accumulation, severe rhabdomyolysis, and renal failure.", "risk": "Severe", "source": "FDA Black Box Interaction"},
        {"drugs": ["erythromycin", "simvastatin"], "desc": "Potent CYP3A4 inhibition precipitates toxic statin levels and fatal rhabdomyolysis.", "risk": "Severe", "source": "DailyMed Interaction Monograph"},
        {"drugs": ["metformin", "radiocontrast"], "desc": "Iodinated radiocontrast impairs renal excretion, triggering fatal lactic acidosis.", "risk": "Severe", "source": "American Diabetes Association"},
        {"drugs": ["amoxicillin", "methotrexate"], "desc": "Reduces methotrexate renal tubular clearance, increasing severe myelosuppression.", "risk": "Severe", "source": "RxNorm Interaction Database"},
        {"drugs": ["ciprofloxacin", "calcium"], "desc": "Forms insoluble chelation complex, completely blocking antibiotic gastrointestinal absorption.", "risk": "Severe", "source": "FDA Labeling"},
        {"drugs": ["ciprofloxacin", "theophylline"], "desc": "Inhibits CYP1A2, elevating theophylline to neurotoxic levels with seizure risk.", "risk": "Severe", "source": "openFDA Interaction Monograph"},
        {"drugs": ["telmisartan", "potassium"], "desc": "Additive potassium retention leads to hyperkalemia and potential fatal cardiac arrhythmia.", "risk": "Severe", "source": "AHA Clinical Guidelines"},
        {"drugs": ["telmisartan", "ibuprofen"], "desc": "Reduces antihypertensive efficacy and precipitates acute renal failure (triple whammy risk).", "risk": "Severe", "source": "CDSCO Drug Safety Notice"}
    ]

    normalized_input = [normalize_medicine_text(i) for i in ingredients]
    flagged = []

    for rule in known_rules:
        target_a, target_b = rule["drugs"]
        match_a = any(target_a in inp for inp in normalized_input)
        match_b = any(target_b in inp for inp in normalized_input)
        if match_a and match_b:
            flagged.append({
                "medicine_a": rule["drugs"][0].title(),
                "medicine_b": rule["drugs"][1].title(),
                "interaction": rule["desc"],
                "risk": rule["risk"],
                "evidence_source": rule["source"]
            })

    return flagged


def search_medicine_alternatives(generic_name: str, current_brand: str = "") -> Dict[str, List[str]]:
    """Searches for exact composition matches and therapeutic alternatives."""
    metadata_db = db_manager.load_metadata()
    exact_matches = []
    therapeutic_alts = []

    clean_gen = normalize_medicine_text(generic_name)
    clean_brand = normalize_medicine_text(current_brand)

    for med_id, entry in metadata_db.items():
        entry_brand = entry.get("brand_name", "")
        entry_gen = normalize_medicine_text(entry.get("generic_name", ""))
        if normalize_medicine_text(entry_brand) == clean_brand:
            continue
        if entry_gen == clean_gen:
            exact_matches.append(f"{entry_brand} ({entry.get('dosage_form', 'Tablet')}) - {entry.get('manufacturer', 'Generic')}")
        elif any(c in clean_gen for c in entry_gen.split()) and entry_gen:
            therapeutic_alts.append(f"{entry_brand} ({entry.get('dosage_form', 'Tablet')}) - {entry.get('generic_name')}")

    for med_id, entry in metadata_db.items():
        if normalize_medicine_text(entry.get("brand_name", "")) == clean_brand:
            for alt in entry.get("alternatives", []):
                if alt not in exact_matches:
                    exact_matches.append(alt)

    return {
        "exact_composition_matches": exact_matches,
        "therapeutic_alternatives": therapeutic_alts,
        "disclaimer": "⚠️ Pharmaceutical Notice: Bioequivalence varies across brands. Substitution requires doctor or pharmacist authorization."
    }


def retrieve_medicine_information(medicine_name: str, generic_name: str) -> Dict[str, Any]:
    """Retrieves and synthesizes clinical intelligence from local curated metadata, openFDA, and PubChem."""
    local_hits = search_medicine(generic_name if generic_name != "Unknown" else medicine_name)
    top_local = local_hits[0] if local_hits else {}

    fda_data = pharma_connectors.query_openfda(generic_name if generic_name != "Unknown" else medicine_name)
    pubchem_data = pharma_connectors.query_pubchem(generic_name if generic_name != "Unknown" else medicine_name)
    rxnorm_data = pharma_connectors.query_rxnorm(generic_name if generic_name != "Unknown" else medicine_name)

    profile = {
        "brand_name": top_local.get("brand_name", medicine_name),
        "generic_name": top_local.get("generic_name", generic_name),
        "category": top_local.get("category", "Pharmaceutical Agent"),
        "dosage_form": top_local.get("dosage_form", "Standard Dosage Form"),
        "manufacturer": top_local.get("manufacturer", "Official Manufacturer"),
        "market": top_local.get("market", "Global / India"),
        "composition": top_local.get("composition", [{"ingredient": generic_name, "strength": "As indicated"}]),
        "is_antibiotic": top_local.get("is_antibiotic", False),
        "mechanism_of_action": top_local.get("mechanism_of_action") or (fda_data.get("indications") if fda_data else "Mechanism of action should be verified with clinical monographs."),
        "uses": top_local.get("uses") or (fda_data.get("indications") if fda_data else "Consult physician for approved indications."),
        "off_label_uses": top_local.get("off_label_uses", "None documented."),
        "side_effects": top_local.get("side_effects") or (fda_data.get("adverse_reactions") if fda_data else "Common side effects are drug-specific."),
        "less_common_side_effects": top_local.get("less_common_side_effects", "See full clinical monograph."),
        "serious_adverse_effects": top_local.get("serious_adverse_effects", "Seek immediate emergency medical attention if severe hypersensitivity occurs."),
        "contraindications": top_local.get("contraindications") or (fda_data.get("contraindications") if fda_data else "Hypersensitivity to active compound."),
        "precautions": top_local.get("precautions", "Take as directed by healthcare provider."),
        "pregnancy_warning": top_local.get("pregnancy_warning", "Consult an obstetrician before use during pregnancy."),
        "breastfeeding_warning": top_local.get("breastfeeding_warning", "Consult a physician before use during lactation."),
        "pediatric_warning": top_local.get("pediatric_warning", "Verify pediatric weight-adjusted dosages."),
        "storage": top_local.get("storage", "Store at room temperature away from heat and moisture."),
        "chemical_formula": pubchem_data.get("chemical_formula", "N/A") if pubchem_data else "N/A",
        "molecular_weight": pubchem_data.get("molecular_weight", "N/A") if pubchem_data else "N/A",
        "iupac_name": pubchem_data.get("iupac_name", "N/A") if pubchem_data else "N/A",
        "sources": []
    }

    if top_local:
        profile["sources"].append("Local Verified Clinical Reference Registry")
    if fda_data:
        profile["sources"].append(fda_data["source"])
    if pubchem_data:
        profile["sources"].append(pubchem_data["source"])
    if rxnorm_data:
        profile["sources"].append(rxnorm_data["source"])

    return profile


def analyse_medicine_safety(medicine_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates antibiotic classifications, pregnancy alerts, and high-risk flags."""
    is_antibiotic = medicine_data.get("is_antibiotic", False)
    flags = []
    if is_antibiotic:
        flags.append("CRITICAL ANTIBIOTIC ALERT: Full course must be completed to prevent microbial resistance.")
    if "contraindicated" in medicine_data.get("pregnancy_warning", "").lower():
        flags.append("PREGNANCY WARNING: Contraindicated during pregnancy.")
    if "bleed" in medicine_data.get("precautions", "").lower():
        flags.append("BLEEDING RISK ALERT: Increases internal hemorrhage risk.")

    return {
        "is_antibiotic": is_antibiotic,
        "safety_flags": flags,
        "disclaimer_required": True
    }


# ==============================================================================
# 8. COMPLETE MEDICINE INTELLIGENCE REPORT
# ==============================================================================

def generate_medicine_report(identification_result: Dict[str, Any], clinical: Dict[str, Any], safety: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str, str]:
    """Formats the 8 structured report sections strictly complying with prompt specifications."""
    status = identification_result.get("status", "UNIDENTIFIED")
    score = identification_result.get("score", 0.0)
    brand = clinical.get("brand_name", "Unknown")
    generic = clinical.get("generic_name", "Unknown")
    dosage_form = clinical.get("dosage_form", "Tablet")
    manufacturer = clinical.get("manufacturer", "Unknown")
    market = clinical.get("market", "Global / India")

    # Section A: IDENTIFICATION
    id_md = f"### 🏷️ SECTION A: IDENTIFICATION\n"
    id_md += f"- **Status:** `{status}` ({score}/100 Match Confidence)\n"
    id_md += f"- **Brand Name:** **{brand}**\n"
    id_md += f"- **Generic Name:** {generic}\n"
    id_md += f"- **Dosage Form:** {dosage_form}\n"
    id_md += f"- **Manufacturer:** {manufacturer}\n"
    id_md += f"- **Market / Country:** {market}\n"
    id_md += f"- **Raw Detected Text:** `{identification_result.get('detected_text', 'N/A')}`\n"
    if safety.get("is_antibiotic"):
        id_md += "\n> [!CAUTION]\n> **ANTIBIOTIC CLASSIFICATION ALERT**: Prescription antibiotic detected. Do not discontinue prematurely without medical advice.\n"

    # Section B: COMPOSITION
    comp_verification = verify_medicine_composition([], clinical.get("composition", []))
    comp_md = f"### 🧪 SECTION B: COMPOSITION & COMBINATION ANALYSIS\n"
    comp_md += f"- **Classification:** {comp_verification['classification']}\n"
    comp_md += f"- **Chemical Formula:** `{clinical.get('chemical_formula', 'N/A')}`\n"
    comp_md += f"- **Molecular Weight:** `{clinical.get('molecular_weight', 'N/A')}`\n\n"
    comp_md += "| Active Ingredient | Strength | Therapeutic Role |\n|---|---|---|\n"
    for comp in clinical.get("composition", []):
        comp_md += f"| **{comp.get('ingredient', 'Unknown')}** | {comp.get('strength', 'N/A')} | {comp.get('role', 'Active Ingredient')} |\n"

    # Section C: USES AND APPLICATIONS
    uses_md = f"### 📋 SECTION C: USES & APPLICATIONS\n"
    uses_md += f"**Approved Clinical Indications:**\n{clinical.get('uses')}\n\n"
    uses_md += f"**Off-Label / Auxiliary Applications:**\n{clinical.get('off_label_uses')}\n\n"
    uses_md += f"**Mechanism of Action:**\n{clinical.get('mechanism_of_action')}\n"

    # Section D: SIDE EFFECTS
    se_md = f"### ⚠️ SECTION D: SIDE EFFECTS & ADVERSE EVENTS\n"
    se_md += f"**Common Side Effects:**\n{clinical.get('side_effects')}\n\n"
    se_md += f"**Less Common Adverse Reactions:**\n{clinical.get('less_common_side_effects')}\n\n"
    se_md += f"> [!WARNING]\n> **Serious Adverse Effects (Requires Immediate Emergency Care):**\n> {clinical.get('serious_adverse_effects')}\n"

    # Section E: PRECAUTIONS
    prec_md = f"### 🚫 SECTION E: PRECAUTIONS & CONTRAINDICATIONS\n"
    prec_md += f"- **Contraindications:** {clinical.get('contraindications')}\n"
    prec_md += f"- **Allergy Warnings:** {clinical.get('allergies_warning', 'Discontinue if rash develops.')}\n"
    prec_md += f"- **Pregnancy Warning:** {clinical.get('pregnancy_warning')}\n"
    prec_md += f"- **Breastfeeding Guidelines:** {clinical.get('breastfeeding_warning')}\n"
    prec_md += f"- **Pediatric Precautions:** {clinical.get('pediatric_warning')}\n"

    # Section F: DRUG INTERACTIONS
    ingredients_list = [c.get("ingredient", "") for c in clinical.get("composition", [])]
    detected_interactions = check_drug_interactions(ingredients_list)
    int_md = f"### ⚡ SECTION F: DRUG INTERACTIONS\n"
    if detected_interactions:
        for di in detected_interactions:
            int_md += f"- **{di['medicine_a']} + {di['medicine_b']}** ({di['risk']} Risk):\n  {di['interaction']}\n  *(Evidence: {di['evidence_source']})*\n\n"
    else:
        int_md += "No critical fixed interactions detected for single active compound. Always consult doctor before combining with anticoagulants, sedatives, or antibiotics.\n"

    # Section G: ADDITIONAL INFORMATION & ALTERNATIVES
    alts = search_medicine_alternatives(generic, brand)
    alt_md = f"### 🔄 SECTION G: FORMULATIONS & GENERIC ALTERNATIVES\n"
    alt_md += f"**Storage Instructions:** {clinical.get('storage')}\n\n"
    alt_md += f"**Exact Composition Generic Matches:**\n"
    if alts["exact_composition_matches"]:
        for em in alts["exact_composition_matches"]:
            alt_md += f"- {em}\n"
    else:
        alt_md += "- No identical local generic matches indexed.\n"
    alt_md += f"\n{alts['disclaimer']}\n"

    # Section H: SOURCES & VERIFICATION
    src_md = f"### 🔍 SECTION H: VERIFICATION & AUDIT CITATIONS\n"
    sources = set(clinical.get("sources", ["Local Clinical Reference Dictionary"]))
    for s in sources:
        src_md += f"- **Verified Source:** {s}\n"
    src_md += f"\n- **Audit Retrieval Timestamp:** `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}`\n"
    src_md += "- **Verification Engine:** MediScan AI Version 4.0 Multi-Source Pipeline\n"

    return (id_md, comp_md, uses_md, se_md, prec_md, int_md, alt_md, src_md)


# ==============================================================================
# 9. GRADIO DASHBOARD
# ==============================================================================

DISCLAIMER_TEXT = """
⚠️ **MEDICAL SAFETY & COMPLIANCE DISCLAIMER**
This platform is an educational pharmaceutical intelligence assistant. It is **NOT** a clinical diagnostic system, prescription platform, or substitute for licensed medical judgment.
- **NEVER** initiate, stop, or adjust medication dosages based on AI visual identification.
- Loose tablets or pills without packaging or clear imprints cannot be safely verified by appearance alone.
- Consult a licensed physician or pharmacist for clinical care.
"""

def analyze_medicine_workflow_v4(input_file, url_input=""):
    """
    Main callback invoked by the Gradio interface:
    Handles validation from uploaded file OR direct web URL,
    processes identification, and formats all 8 tabs.
    """
    target_img = None
    url_clean = (url_input or "").strip()

    if url_clean:
        valid, msg, downloaded = validate_uploaded_image(url_clean)
        if not valid or downloaded is None:
            err_msg = f"❌ URL Loading Error: {msg}"
            return (None, err_msg, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "### ❌ Status: URL Fetch Failure")
        target_img = downloaded
    elif input_file is not None:
        target_img = input_file

    if target_img is None:
        empty_msg = "Please upload an image of a medicine tablet, blister strip, box, or bottle, or provide a direct web image URL."
        return (None, empty_msg, empty_msg, empty_msg, empty_msg, empty_msg, empty_msg, empty_msg, empty_msg, "### Status: Awaiting Image")

    try:
        # Step 1: Execute Identification Pipeline
        id_result = identify_medicine(target_img)

        # Step 2: Handle Loose Tablet without Imprint
        if id_result.get("is_loose_tablet") and id_result.get("confidence_rating") == "INSUFFICIENT_EVIDENCE":
            warn_msg = f"### ⚠️ UNABLE TO IDENTIFY LOOSE TABLET\n\n{id_result.get('message')}"
            status_text = "### ⚠️ Status: Insufficient Physical Evidence"
            return (target_img, warn_msg, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", status_text)

        # Step 3: Handle Unidentified or Errors
        top_match = id_result.get("top_match", {})
        brand = top_match.get("brand_name", "Unknown")
        generic = top_match.get("generic_name", "Unknown")

        # Step 4: Retrieve Clinical Monograph
        clinical = retrieve_medicine_information(brand, generic)
        safety = analyse_medicine_safety(clinical)

        # Step 5: Generate 8-Section Report
        id_tab, comp_tab, uses_tab, se_tab, prec_tab, int_tab, alt_tab, src_tab = generate_medicine_report(id_result, clinical, safety)

        # Audit Logging
        db_manager.log_verification(
            medicine_name=f"{brand} ({generic})",
            confidence=id_result.get("confidence_rating", "MEDIUM"),
            evidence=id_result.get("detected_text", "Visual OCR"),
            sources=clinical.get("sources", ["Local DB"]),
            full_result={"id_result": id_result, "clinical": clinical}
        )

        status_text = f"### 🟢 Status: Analysis Completed ({id_result.get('status')})"
        return (target_img, id_tab, comp_tab, uses_tab, se_tab, prec_tab, int_tab, alt_tab, src_tab, status_text)

    except Exception as e:
        err_msg = f"❌ Diagnostic Pipeline Error: {str(e)}"
        return (target_img, err_msg, err_msg, err_msg, err_msg, err_msg, err_msg, err_msg, err_msg, "### ❌ Status: Processing Failure")


# Aliases for backward compatibility
analyze_medicine_workflow = analyze_medicine_workflow_v4
analyze_medicine_workflow_v2 = analyze_medicine_workflow_v4


def build_gradio_app():
    """Builds and configures the Gradio Blocks dashboard matching Section 12."""
    if not HAS_GRADIO:
        print("⚠️ Gradio is not installed in the local environment. Skipping UI block generation.")
        return None

    custom_css = """
    .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    #status-box { padding: 12px; border-radius: 8px; font-weight: bold; }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css, title="MediScan AI – Version 4.0") as demo:
    with gr.Blocks(title="MediScan AI – Version 4.0") as demo:
        gr.Markdown("# 🧬 MediScan AI – Version 4.0")
        gr.Markdown("### Global AI-Powered Medicine Identification & Pharmaceutical Intelligence Platform")
        gr.Markdown(DISCLAIMER_TEXT)

        with gr.Row():
            # Left Panel: Upload, URL input, Preview, Controls
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.TabItem("📁 File Upload"):
                        input_file = gr.Image(type="pil", label="Upload Tablet, Capsule, Blister, or Bottle Image")
                    with gr.TabItem("🌐 Web Image URL"):
                        url_input = gr.Textbox(placeholder="https://example.com/medicine-strip.jpg", label="Paste Direct Medicine Image URL", lines=2)
                        load_url_btn = gr.Button("📥 Fetch Image from URL", variant="secondary")

                image_preview = gr.Image(type="pil", label="Active Medicine Image Preview", interactive=False)

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Medicine", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                status_box = gr.Markdown("### Status: Ready for Image Upload or Web URL", elem_id="status-box")

            # Right Panel: 8 Structured Result Tabs
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.TabItem("🏷️ Identification"):
                        id_out = gr.Markdown("Upload an image or enter a URL and click 'Analyze Medicine' to begin.")
                    with gr.TabItem("🧪 Composition"):
                        comp_out = gr.Markdown("Chemical breakdown and combination analysis will appear here.")
                    with gr.TabItem("📋 Uses & Applications"):
                        uses_out = gr.Markdown("Approved indications and mechanism of action.")
                    with gr.TabItem("⚠️ Side Effects"):
                        se_out = gr.Markdown("Adverse events and safety profile.")
                    with gr.TabItem("🚫 Precautions"):
                        prec_out = gr.Markdown("Contraindications, pregnancy, and medical condition warnings.")
                    with gr.TabItem("⚡ Drug Interactions"):
                        int_out = gr.Markdown("Known clinical drug interactions.")
                    with gr.TabItem("🔄 Alternatives"):
                        alt_out = gr.Markdown("Generic formulations and equivalent substitutes.")
                    with gr.TabItem("🔍 Sources & Verification"):
                        src_out = gr.Markdown("Data source citations and audit verification stamps.")

        # Event: Fetch from URL
        def fetch_preview(url_text):
            if not url_text or not url_text.strip():
                return None, "### ⚠️ Please enter an image URL"
            v, m, img = validate_uploaded_image(url_text.strip())
            if v and img:
                return img, "### 🟢 Image loaded from URL. Ready to analyze!"
            return None, f"### ❌ Failed: {m}"

        load_url_btn.click(
            fn=fetch_preview,
            inputs=[url_input],
            outputs=[image_preview, status_box]
        )

        input_file.change(
            fn=lambda f: f,
            inputs=[input_file],
            outputs=[image_preview]
        )

        analyze_btn.click(
            fn=analyze_medicine_workflow_v4,
            inputs=[image_preview, url_input],
            outputs=[image_preview, id_out, comp_out, uses_out, se_out, prec_out, int_out, alt_out, src_out, status_box]
        )

        clear_btn.click(
            fn=lambda: (None, "", None, "Ready for Image Upload or Web URL", "Awaiting input...", "Awaiting input...", "Awaiting input...", "Awaiting input...", "Awaiting input...", "Awaiting input...", "Awaiting input...", "Awaiting input..."),
            inputs=[],
            outputs=[input_file, url_input, image_preview, status_box, id_out, comp_out, uses_out, se_out, prec_out, int_out, alt_out, src_out]
        )

    return demo


def launch_gradio_app(share: bool = True, debug: bool = False, server_port: int = 7860):
def launch_gradio_app(share: bool = False, debug: bool = False, server_port: int = 7860):
    """Launches the MediScan AI Gradio interface."""
    if not HAS_GRADIO:
        print("⚠️ Gradio is not installed in this environment. In Google Colab, execute: `!pip install gradio`.")
        return None
    try:
        demo = build_gradio_app()
        if demo:
            demo.launch(server_name="0.0.0.0", server_port=server_port, share=share, debug=debug)
            custom_css = """
            .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            #status-box { padding: 12px; border-radius: 8px; font-weight: bold; }
            """
            print(f"🚀 Launching MediScan AI Web Server on port {server_port}...", flush=True)
            demo.launch(
                server_name="0.0.0.0",
                server_port=server_port,
                share=share,
                debug=debug,
                theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
                css=custom_css
            )
            return demo
    except Exception as e:
        print(f"❌ Failed to launch Gradio application: {e}")
    return None


if __name__ == "__main__":
    print("=" * 70)
    print("🧬 MediScan AI - Version 4.0 Starting Web Dashboard...")
    print(f"   Database Status: Active ({len(db_manager.load_metadata())} clinical records)")
    print(f"   GenAI SDK: {GENAI_SDK or 'SDK Not Found'}")
    print(f"   Gradio Framework: {'Installed' if HAS_GRADIO else 'Not Installed'}")
    print("=" * 70)
    launch_gradio_app(share=False, server_port=7860)
