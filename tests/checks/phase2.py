"""Phase 2 — OCR & Degradation checks P2.01-P2.08."""

from __future__ import annotations

from tests.checks import common
from tests.checks.contract import Check, CheckResult

REQUIRED_PAGE_PARAM_KEYS = {
    "rotation_deg",
    "noise_sigma",
    "brightness_delta",
    "contrast_delta",
    "jpeg_quality",
    "blurred",
    "blur_radius",
    "page_seed",
    "page",
}


def _degraded_doc_ids(manifest: dict) -> list[str]:
    return sorted(doc_id for doc_id, entry in manifest.items() if entry["variant"] == "degraded")


def check_split_ratio(config: dict) -> CheckResult:
    cfg = config["phase2"]
    desired = f"manifest shows {cfg['clean_count']} clean + {cfg['degraded_count']} degraded"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")

    counts: dict[str, int] = {"clean": 0, "degraded": 0}
    for entry in manifest.values():
        counts[entry["variant"]] = counts.get(entry["variant"], 0) + 1
    current = f"{counts.get('clean', 0)} clean + {counts.get('degraded', 0)} degraded"

    if counts.get("clean") == cfg["clean_count"] and counts.get("degraded") == cfg["degraded_count"]:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_text_layer_destroyed(config: dict) -> CheckResult:
    desired = "all 40 degraded PDFs: PyMuPDF text extraction returns <10 characters"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")
    doc_ids = _degraded_doc_ids(manifest)
    if not doc_ids:
        return common.no_data(desired, "no degraded docs in manifest")

    import fitz  # local import: only needed by this check

    failures = []
    checked = 0
    for doc_id in doc_ids:
        pdf_path = common.DEGRADED_PDF_DIR / f"{doc_id}_scan.pdf"
        if not pdf_path.exists():
            failures.append(f"- {doc_id}: {pdf_path.name} missing")
            continue
        checked += 1
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        if len(text.strip()) >= 10:
            failures.append(f"- {doc_id}: {len(text.strip())} extractable chars (>=10)")

    current = f"{checked - len(failures)}/{len(doc_ids)} pass"
    if failures:
        return CheckResult("fail", desired, current, "\n".join(failures[:10]))
    return CheckResult("pass", desired, current)


def check_ocr_output_exists(config: dict) -> CheckResult:
    desired = "every degraded doc has a corresponding ocr_output/{doc_id}.md"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")
    doc_ids = _degraded_doc_ids(manifest)
    if not doc_ids:
        return common.no_data(desired, "no degraded docs in manifest")
    if not common.OCR_OUTPUT_DIR.exists():
        return common.no_data(desired, f"{common.OCR_OUTPUT_DIR} missing — run the Colab OCR notebook")

    missing = [d for d in doc_ids if not (common.OCR_OUTPUT_DIR / f"{d}.md").exists()]
    current = f"{len(doc_ids) - len(missing)}/{len(doc_ids)} present"
    if missing:
        details = "\n".join(f"- {d}: no ocr_output/{d}.md" for d in missing[:10])
        return CheckResult("fail", desired, current, details)
    return CheckResult("pass", desired, current)


def check_cer_wer_computed(config: dict) -> CheckResult:
    desired = "ocr_metrics.json has per-doc CER/WER for all degraded docs, plus corpus aggregate"
    manifest = common.load_json(common.MANIFEST_JSON)
    metrics = common.load_json(common.OCR_METRICS_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")
    if metrics is None:
        return common.no_data(desired, f"{common.OCR_METRICS_JSON} missing — run corpus.entity_integrity")

    doc_ids = _degraded_doc_ids(manifest)
    initial = metrics.get("initial", {})
    per_doc = initial.get("aggregate") and initial.get("per_doc")
    if not per_doc:
        return CheckResult("fail", desired, "'initial' section missing 'per_doc' or 'aggregate'")

    missing = [d for d in doc_ids if d not in per_doc or "cer" not in per_doc[d] or "wer" not in per_doc[d]]
    agg = initial["aggregate"]
    has_aggregate = "cer" in agg and "wer" in agg
    current = f"{len(doc_ids) - len(missing)}/{len(doc_ids)} docs have cer+wer, aggregate present={has_aggregate}"

    if missing or not has_aggregate:
        details = "\n".join(f"- {d}: missing cer/wer entry" for d in missing[:10])
        return CheckResult("fail", desired, current, details)
    return CheckResult("pass", desired, current)


def check_initial_vs_final(config: dict) -> CheckResult:
    desired = "ocr_metrics.json has both 'initial' and 'final' metric sets (pre/post-tuning)"
    metrics = common.load_json(common.OCR_METRICS_JSON)
    if metrics is None:
        return common.no_data(desired, f"{common.OCR_METRICS_JSON} missing")

    initial, final = metrics.get("initial"), metrics.get("final")
    ok = bool(initial and final and "aggregate" in initial and "aggregate" in final)
    current = (
        f"initial present={bool(initial)}, final present={bool(final)}, "
        f"identical={initial == final if ok else 'n/a'}"
    )
    if ok:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_entity_survival(config: dict) -> CheckResult:
    threshold = config["phase2"]["entity_survival_threshold"]
    desired = f"post-canonical-map entity survival >= {threshold:.0%}"
    metrics = common.load_json(common.OCR_METRICS_JSON)
    if metrics is None:
        return common.no_data(desired, f"{common.OCR_METRICS_JSON} missing")

    final_agg = metrics.get("final", {}).get("aggregate", {})
    survival = final_agg.get("entity_survival")
    if survival is None:
        return CheckResult("fail", desired, "'final.aggregate.entity_survival' missing")

    found, total = final_agg.get("entities_found"), final_agg.get("entities_total")
    current = f"{survival:.2%} ({found}/{total})"
    if survival >= threshold:
        return CheckResult("pass", desired, current)
    return CheckResult("fail", desired, current)


def check_canonical_map(config: dict) -> CheckResult:
    desired = "entity_canonical_map.json exists, maps corrupted->canonical, no self-mappings"
    canonical_map = common.load_json(common.CANONICAL_MAP_JSON)
    if canonical_map is None:
        return common.no_data(desired, f"{common.CANONICAL_MAP_JSON} missing")

    self_mappings = [k for k, v in canonical_map.items() if k.strip().lower() == v.strip().lower()]
    current = f"{len(canonical_map)} mapping(s), {len(self_mappings)} self-mapping(s)"
    if self_mappings:
        details = "\n".join(f"- {k!r} -> {v!r}" for k in self_mappings[:10] for v in [canonical_map[k]])
        return CheckResult("fail", desired, current, details)
    return CheckResult("pass", desired, current)


def check_reproducibility_params(config: dict) -> CheckResult:
    desired = "all 40 degraded docs have complete per-page degradation param records"
    manifest = common.load_json(common.MANIFEST_JSON)
    if manifest is None:
        return common.no_data(desired, f"{common.MANIFEST_JSON} missing")
    doc_ids = _degraded_doc_ids(manifest)
    if not doc_ids:
        return common.no_data(desired, "no degraded docs in manifest")

    incomplete = []
    for doc_id in doc_ids:
        entry = manifest[doc_id]
        pages = entry.get("pages")
        if not pages:
            incomplete.append(f"- {doc_id}: no 'pages' param records")
            continue
        for i, page in enumerate(pages):
            missing_keys = REQUIRED_PAGE_PARAM_KEYS - page.keys()
            if missing_keys:
                incomplete.append(f"- {doc_id} page {i}: missing {sorted(missing_keys)}")

    current = f"{len(doc_ids) - len({e.split(':')[0] for e in incomplete})}/{len(doc_ids)} complete"
    if incomplete:
        return CheckResult("fail", desired, current, "\n".join(incomplete[:10]))
    return CheckResult("pass", desired, current)


CHECKS: list[Check] = [
    Check("P2.01", "Split ratio", check_split_ratio),
    Check("P2.02", "Degradation destroys text layer", check_text_layer_destroyed),
    Check("P2.03", "OCR output exists", check_ocr_output_exists),
    Check("P2.04", "CER/WER computed", check_cer_wer_computed),
    Check("P2.05", "Initial vs final CER/WER", check_initial_vs_final),
    Check("P2.06", "Entity survival", check_entity_survival),
    Check("P2.07", "Canonical map populated", check_canonical_map),
    Check("P2.08", "Reproducibility", check_reproducibility_params),
]
