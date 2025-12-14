"""Helper utilities that adapt the research-only combined detector
pipeline (ai/neodata2/combined_detector.py) for online usage inside
FastAPI.

The original script was tailored for batch experiments over the entire
validation corpus. This service wraps the same hyper-parameters and zone
logic, but focuses on a single uploaded photo name. The heavy lifting
(stitching SAM masks + CAD coverage) remains untouched, we simply load
the already-exported *_masks.json artifacts and expose a lightweight
`analyze_by_filename` method the API layer can consume.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from ai.neodata2 import combined_detector as cd

MaskIndexEntry = Dict[str, object]


class CombinedDetectorService:
    """Load CAD metadata + segmentation masks once and reuse for requests."""

    def __init__(self, backend_dir: Path):
        self.backend_dir = backend_dir
        self.ai_dir = backend_dir / "ai"
        self.neodata_dir = self.ai_dir / "neodata2"
        self.seg_dir = self.ai_dir / "segmented_output_improved"
        self.validation_dir = self.ai_dir / "validation"

        self._cad_models = self._load_cad_models()
        self._default_model = next(iter(self._cad_models.keys())) if self._cad_models else None
        if not self._default_model:
            raise RuntimeError("No CAD models available for combined detector.")

        self._jpeg_mappings = self._load_classification_results()
        self._mask_index = self._build_mask_index()
        if not self._mask_index:
            raise RuntimeError(
                "Segmented_output_improved folder is empty. Run SAM_Improved_Segmentation.ipynb first."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_by_filename(self, filename: str) -> Dict[str, object]:
        """Return combined detection summary for a photo name.

        The uploaded file itself does not go through SAM here; instead we
        expect its base name (without extension) to match the already
        segmented datasets. That keeps the online path lightweight while
        still surfacing the new combined metrics in the UI.
        """

        if not filename:
            raise ValueError("Filename is required.")

        stem = Path(filename).stem
        canonical = self._normalize_stem(stem)
        entries = self._mask_index.get(canonical)
        if not entries:
            raise FileNotFoundError(
                f"Nema pripadajućih _masks.json datoteka za '{filename}'. Pokrenite SAM segmentaciju."
            )

        # Prefer entry whose base string best matches the provided stem.
        entry = self._pick_best_entry(entries, stem)
        mask_path = entry["path"]
        folder = entry["folder"]
        base = entry["base"]

        with mask_path.open("r", encoding="utf-8") as handle:
            mask_data = json.load(handle)

        jpeg_candidates = self._derive_jpeg_candidates(base, stem)
        matched_model = self._select_model(jpeg_candidates)

        detection = self._analyze_mask(mask_data)
        defects = detection["defects"]
        confidence = max((d.get("confidence", 0.0) for d in defects), default=0.99 if not defects else 0.0)
        label = "fail" if defects else "pass"
        verdict = self._build_verdict(defects)
        mask_relative = self._to_posix(mask_path, self.seg_dir)
        overlay_relative = self._overlay_relative(mask_path)

        return {
            "image": mask_path.name,
            "folder": folder,
            "original_jpeg": jpeg_candidates[0],
            "matched_model": matched_model,
            "mask_source": self._to_posix(mask_path, self.backend_dir),
            "mask_relative": mask_relative,
            "mask_overlay": overlay_relative,
            "defects": defects,
            "all_defects": detection["all_defects"],
            "detection_stats": detection["stats"],
            "zone_coverage": detection["zone_coverage"],
            "label": label,
            "confidence": round(float(confidence), 4),
            "verdict": verdict,
        }

    def refresh_index(self) -> None:
        """Allow hot reloads without restarting the API server."""

        self._mask_index = self._build_mask_index()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_cad_models(self) -> Dict[str, dict]:
        cad_path = self.neodata_dir / "cad_models.json"
        if not cad_path.exists():
            raise RuntimeError(f"CAD model definitions missing: {cad_path}")
        with cad_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_classification_results(self) -> Dict[str, str]:
        results_path = self.neodata_dir / "classification_results_v2.json"
        if not results_path.exists():
            print("[combined] classification_results_v2.json not found; using fallback CAD model for all images.")
            return {}

        with results_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        mapping: Dict[str, str] = {}
        for row in payload:
            image_name = row.get("image")
            matched_model = row.get("matched_model")
            if image_name and matched_model:
                mapping[image_name] = matched_model
        print(f"[combined] Loaded {len(mapping)} JPEG -> CAD mappings.")
        return mapping

    def _build_mask_index(self) -> Dict[str, List[MaskIndexEntry]]:
        if not self.seg_dir.exists():
            raise RuntimeError(f"Segmented output folder not found: {self.seg_dir}")

        index: Dict[str, List[MaskIndexEntry]] = {}
        for subset in ("positive", "negative"):
            subset_dir = self.seg_dir / subset
            if not subset_dir.exists():
                continue
            for mask_path in subset_dir.glob("*_masks.json"):
                base = mask_path.stem.replace("_masks", "")
                canonical = self._normalize_stem(base)
                entry = {"path": mask_path, "folder": subset, "base": base}
                index.setdefault(canonical, []).append(entry)

        print(f"[combined] Indexed {sum(len(v) for v in index.values())} mask files from segmented_output_improved.")
        return index

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------
    def _normalize_stem(self, value: str) -> str:
        cleaned = value.lower().replace("_facade", "").replace("facade", "")
        cleaned = cleaned.replace(" ", "_")
        return re.sub(r"[^a-z0-9]", "", cleaned)

    def _pick_best_entry(self, entries: List[MaskIndexEntry], stem: str) -> MaskIndexEntry:
        if len(entries) == 1:
            return entries[0]
        target = self._normalize_stem(stem)
        for entry in entries:
            if self._normalize_stem(entry["base"]) == target:
                return entry
        return entries[0]

    def _overlay_relative(self, mask_path: Path) -> Optional[str]:
        overlay_name = mask_path.name.replace("_masks.json", "_overlay.jpg")
        overlay_path = mask_path.with_name(overlay_name)
        if overlay_path.exists():
            return self._to_posix(overlay_path, self.seg_dir)
        return None

    def _to_posix(self, path: Path, base: Path) -> str:
        return str(path.relative_to(base)).replace("\\", "/")

    def _derive_jpeg_candidates(self, mask_base: str, uploaded_stem: str) -> List[str]:
        jpeg_base = mask_base.replace("_facade", "")
        candidates = [
            re.sub(r"_(\d+)$", r" \1", jpeg_base) + ".jpg",
            jpeg_base + ".jpg",
            jpeg_base.replace("_", " ") + ".jpg",
            uploaded_stem + ".jpg",
        ]
        seen = set()
        ordered: List[str] = []
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(candidate)
        return ordered

    def _select_model(self, jpeg_candidates: List[str]) -> str:
        for candidate in jpeg_candidates:
            model = self._jpeg_mappings.get(candidate)
            if model:
                return model
        return self._default_model

    def _build_verdict(self, defects: List[Dict[str, object]]) -> str:
        if not defects:
            return "Nisu pronađeni defekti prema kombiniranom detektoru."
        summary = ", ".join(
            f"{d.get('component', 'n/a')} · {d.get('type', 'unknown')}" for d in defects
        )
        return f"Otkriveno {len(defects)} defekata: {summary}."

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------
    def _analyze_mask(self, mask_data: dict) -> Dict[str, object]:
        components = mask_data.get("components", {})
        tin_masks = components.get("tin", [])
        glass_masks = components.get("glass", [])
        screw_masks = components.get("screw", [])
        hole_masks = components.get("hole", [])
        seal_masks = components.get("seal", [])

        w, h = mask_data.get("size", [1536, 688])
        zones = [
            "top-left",
            "top",
            "top-right",
            "left",
            "center",
            "right",
            "bottom-left",
            "bottom",
            "bottom-right",
        ]
        zone_tin = {zone: 0.0 for zone in zones}

        for mask in tin_masks:
            bbox = mask.get("bbox")
            if not bbox:
                continue
            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "")
            row = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "")
            if row and col:
                zone = f"{row}-{col}"
            elif row:
                zone = row
            elif col:
                zone = col
            else:
                zone = "center"
            zone_tin[zone] += mask.get("coverage_percent", 0.0)

        expected_tin_zones = ["top", "bottom", "left", "right"]
        tin_gaps = [zone for zone in expected_tin_zones if zone_tin[zone] < cd.GAP_THRESHOLD]
        well_covered = [zone for zone in expected_tin_zones if zone_tin[zone] >= cd.GOOD_COVERAGE]
        center_coverage = zone_tin.get("center", 0.0)

        defects: List[Dict[str, object]] = []
        if center_coverage > cd.CENTER_MIN_COVERAGE and well_covered:
            if len(tin_gaps) >= cd.MIN_GAPS_STRONG:
                defects.append(self._build_tin_defect(zone_tin, tin_gaps, well_covered, cd.CONFIDENCE_STRONG))
            elif len(tin_gaps) >= cd.MIN_GAPS_MEDIUM:
                defects.append(self._build_tin_defect(zone_tin, tin_gaps, well_covered, cd.CONFIDENCE_MEDIUM))
            elif len(tin_gaps) >= cd.MIN_GAPS_WEAK and len(well_covered) >= cd.MIN_COVERED_FOR_WEAK:
                defects.append(self._build_tin_defect(zone_tin, tin_gaps, well_covered, cd.CONFIDENCE_WEAK))

        high_conf = [d for d in defects if d.get("confidence", 0) >= cd.CONFIDENCE_THRESHOLD]
        stats = {
            "tin_coverage": float(sum(mask.get("coverage_percent", 0.0) for mask in tin_masks)),
            "glass_coverage": float(sum(mask.get("coverage_percent", 0.0) for mask in glass_masks)),
            "screw_count": len(screw_masks),
            "hole_count": len(hole_masks),
            "seal_coverage": float(sum(mask.get("coverage_percent", 0.0) for mask in seal_masks)),
            "tin_masks": len(tin_masks),
            "glass_masks": len(glass_masks),
        }

        return {
            "stats": stats,
            "defects": high_conf,
            "all_defects": defects,
            "zone_coverage": zone_tin,
        }

    def _build_tin_defect(
        self,
        zone_tin: Dict[str, float],
        tin_gaps: List[str],
        well_covered: List[str],
        confidence: float,
    ) -> Dict[str, object]:
        return {
            "component": "tin",
            "type": "missing",
            "confidence": float(confidence),
            "zones_missing": tin_gaps,
            "zones_present": well_covered,
            "zone_coverage": zone_tin,
            "source": "zone_analysis",
        }
