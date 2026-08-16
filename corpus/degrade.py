"""Phase 2 corpus degradation: seeded split + degradation recipe (PLAN.md §5.2).

Selects 40/100 clean PDFs for degradation (stratified 12 FIN / 10 ENG / 10 HR / 8 LEG,
seeded on SEED=42), degrades each into a scanned-looking, image-only PDF, and writes
corpus/manifest.json recording the split and every degradation parameter used (so runs
are reproducible and auditable).

Run: python -m corpus.degrade [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import img2pdf
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "corpus" / "clean"
DEGRADED_DIR = ROOT / "corpus" / "degraded"
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"

SEED = 42
DOC_DOMAIN_PREFIXES = {"finance": "FIN", "engineering": "ENG", "hr": "HR", "legal": "LEG"}
DEGRADED_PER_DOMAIN = {"finance": 12, "engineering": 10, "hr": 10, "legal": 8}

DPI = 170
ROTATION_RANGE_DEG = (-1.5, 1.5)
NOISE_SIGMA_RANGE = (4.0, 10.0)
BRIGHTNESS_JITTER = 0.10
CONTRAST_JITTER = 0.10
JPEG_QUALITY_RANGE = (55, 75)
BLUR_PROBABILITY = 0.10
BLUR_RADIUS_RANGE = (0.5, 1.0)


def _doc_domain(doc_id: str) -> str | None:
    for domain, prefix in DOC_DOMAIN_PREFIXES.items():
        if doc_id.startswith(f"{prefix}-"):
            return domain
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_split() -> dict[str, str]:
    """Deterministically select which doc_ids get degraded. Returns {doc_id: variant}."""
    doc_ids = sorted(p.stem for p in CLEAN_DIR.glob("*.pdf"))
    by_domain: dict[str, list[str]] = {d: [] for d in DEGRADED_PER_DOMAIN}
    for doc_id in doc_ids:
        domain = _doc_domain(doc_id)
        if domain:
            by_domain[domain].append(doc_id)

    rng = random.Random(SEED)
    degraded: set[str] = set()
    for domain in sorted(DEGRADED_PER_DOMAIN):  # fixed iteration order for reproducibility
        degraded.update(rng.sample(sorted(by_domain[domain]), DEGRADED_PER_DOMAIN[domain]))

    return {doc_id: ("degraded" if doc_id in degraded else "clean") for doc_id in doc_ids}


def build_manifest_skeleton() -> dict:
    """Split + checksums only, for every doc (clean and degraded)."""
    split = select_split()
    entries = {}
    for doc_id, variant in sorted(split.items()):
        clean_pdf = CLEAN_DIR / f"{doc_id}.pdf"
        entries[doc_id] = {
            "variant": variant,
            "clean_pdf_sha256": _sha256(clean_pdf),
        }
    return entries


def _degrade_page(img: Image.Image, rng: random.Random) -> tuple[bytes, dict]:
    """Apply one page's degradation recipe (PLAN.md §5.2); return JPEG bytes + params used."""
    page_seed = rng.randrange(2**31)
    np_rng = np.random.default_rng(page_seed)

    rotation_deg = rng.uniform(*ROTATION_RANGE_DEG)
    img = img.convert("RGB").rotate(rotation_deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    noise_sigma = rng.uniform(*NOISE_SIGMA_RANGE)
    arr = np.asarray(img, dtype=np.float32) + np_rng.normal(0.0, noise_sigma, np.asarray(img).shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    brightness_delta = rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER)
    contrast_delta = rng.uniform(-CONTRAST_JITTER, CONTRAST_JITTER)
    img = ImageEnhance.Brightness(img).enhance(1 + brightness_delta)
    img = ImageEnhance.Contrast(img).enhance(1 + contrast_delta)

    blurred = rng.random() < BLUR_PROBABILITY
    blur_radius = rng.uniform(*BLUR_RADIUS_RANGE) if blurred else None
    if blurred:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    jpeg_quality = rng.randrange(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, dpi=(DPI, DPI))

    params = {
        "rotation_deg": round(rotation_deg, 4),
        "noise_sigma": round(noise_sigma, 4),
        "brightness_delta": round(brightness_delta, 4),
        "contrast_delta": round(contrast_delta, 4),
        "jpeg_quality": jpeg_quality,
        "blurred": blurred,
        "blur_radius": round(blur_radius, 4) if blurred else None,
        "page_seed": page_seed,
    }
    return buf.getvalue(), params


def degrade_doc(doc_id: str) -> dict:
    """Degrade one clean PDF into a scanned-looking, image-only PDF.

    Deterministic per doc_id (independent RNG stream keyed on SEED + doc_id), so any
    single document can be regenerated in isolation without affecting the others.
    """
    rng = random.Random(f"{SEED}:{doc_id}")
    clean_pdf = CLEAN_DIR / f"{doc_id}.pdf"
    pages = convert_from_path(str(clean_pdf), dpi=DPI)

    page_bytes: list[bytes] = []
    page_params: list[dict] = []
    for i, page in enumerate(pages):
        jpeg_bytes, params = _degrade_page(page, rng)
        params["page"] = i
        page_bytes.append(jpeg_bytes)
        page_params.append(params)

    DEGRADED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEGRADED_DIR / f"{doc_id}_scan.pdf"
    layout_fun = img2pdf.get_fixed_dpi_layout_fun((DPI, DPI))
    out_path.write_bytes(img2pdf.convert(page_bytes, layout_fun=layout_fun))

    return {"degraded_pdf_sha256": _sha256(out_path), "dpi": DPI, "pages": page_params}


def degrade_all() -> dict:
    manifest = build_manifest_skeleton()
    for doc_id in sorted(manifest):
        if manifest[doc_id]["variant"] == "degraded":
            manifest[doc_id].update(degrade_doc(doc_id))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    if not CLEAN_DIR.exists() or not any(CLEAN_DIR.glob("*.pdf")):
        print(f"No clean PDFs found in {CLEAN_DIR} — run corpus.render_pdfs first.", file=sys.stderr)
        return 1

    manifest = degrade_all()
    counts = {"clean": 0, "degraded": 0}
    by_domain_degraded: dict[str, int] = {d: 0 for d in DEGRADED_PER_DOMAIN}
    for doc_id, entry in manifest.items():
        counts[entry["variant"]] += 1
        if entry["variant"] == "degraded":
            by_domain_degraded[_doc_domain(doc_id)] += 1

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest.json: {counts['clean']} clean / {counts['degraded']} degraded")
    print("degraded per domain:", by_domain_degraded)
    print(f"scanned PDFs written to {DEGRADED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
