"""
Tumor Map Reconstruction and Automated Patient pN-Staging Module.
Implements post-processing, tumor area quantification, lesion categorization,
and 5-slide patient aggregation following Tauqeer et al. (Scientific Reports 2025):
- Area calculation: A = n * (pixelsize)^2 um^2 (pixelsize = 0.24 um at 40x)
- ITC:               A <= 200 um^2
- Micro-metastasis: 200 um^2 < A <= 2.0 mm^2 (2,000,000 um^2)
- Macro-metastasis: A > 2.0 mm^2
- Automated pN Stage: pN0, pN0(i+), pN1mi, pN1, pN2
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2


class TumorMapperAndStagingEvaluator:
    """
    Constructs 2D tumor probability and binary maps, calculates connected-component
    metastasis areas, classifies lesion types, and computes the patient-level pN stage.
    """

    def __init__(
        self,
        pixel_size_um: float = 0.24,
        patch_size_px: int = 256,
        itc_cutoff_um2: float = 200.0,
        micro_cutoff_um2: float = 2_000_000.0,  # 2.0 mm^2
    ):
        self.pixel_size_um = pixel_size_um
        self.patch_size_px = patch_size_px
        self.itc_cutoff_um2 = itc_cutoff_um2
        self.micro_cutoff_um2 = micro_cutoff_um2

        # Area of a single 256x256 patch at 0.24 um/pixel:
        # 256 * 256 * (0.24)^2 ~= 3774.87 um^2
        self.patch_area_um2 = (patch_size_px * pixel_size_um) ** 2

    def reconstruct_tumor_map(
        self,
        coords: np.ndarray,
        predictions: np.ndarray,
        wsi_width: int,
        wsi_height: int,
        downsample_factor: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects patch predictions back onto the 2D spatial coordinates of the WSI.
        
        Args:
            coords: (N, 2) array of Level 0 (x, y) coordinates.
            predictions: (N,) binary predictions (1=tumor, 0=normal) or tumor probabilities.
            wsi_width: Width of WSI at Level 0.
            wsi_height: Height of WSI at Level 0.
            downsample_factor: Downsample scale for the output tumor map.
            
        Returns:
            binary_map: (H_map, W_map) binary mask (uint8)
            heatmap: (H_map, W_map) continuous probability map (float32)
        """
        map_w = int(np.ceil(wsi_width / downsample_factor))
        map_h = int(np.ceil(wsi_height / downsample_factor))

        binary_map = np.zeros((map_h, map_w), dtype=np.uint8)
        heatmap = np.zeros((map_h, map_w), dtype=np.float32)

        patch_w_map = max(1, int(round(self.patch_size_px / downsample_factor)))
        patch_h_map = max(1, int(round(self.patch_size_px / downsample_factor)))

        num_patches = len(coords)
        for i in range(num_patches):
            x, y = int(coords[i, 0]), int(coords[i, 1])
            mx = int(round(x / downsample_factor))
            my = int(round(y / downsample_factor))

            score = float(predictions[i])
            is_tumor = int(score >= 0.5)

            y1, y2 = my, min(my + patch_h_map, map_h)
            x1, x2 = mx, min(mx + patch_w_map, map_w)

            if y2 > y1 and x2 > x1:
                heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], score)
                if is_tumor:
                    binary_map[y1:y2, x1:x2] = 1

        return binary_map, heatmap

    def quantify_slide_metastases(
        self,
        binary_map: np.ndarray,
        downsample_factor: int = 16,
    ) -> Dict[str, Any]:
        """
        Analyzes connected tumor regions on the tumor map and classifies them into
        ITC, micro-metastasis, and macro-metastasis based on their physical area.
        
        Args:
            binary_map: (H, W) binary mask from reconstruct_tumor_map
            downsample_factor: Resolution scale of the binary_map relative to Level 0
            
        Returns:
            lesion_summary: Dict with counts and maximum lesion area
        """
        if np.count_nonzero(binary_map) == 0:
            return {
                "is_positive": False,
                "highest_category": "negative",
                "max_area_um2": 0.0,
                "num_itc": 0,
                "num_micro": 0,
                "num_macro": 0,
                "lesion_areas_um2": [],
            }

        # Connected component analysis to isolate individual metastatic foci
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_map.astype(np.uint8), connectivity=8
        )

        pixel_area_um2 = (self.pixel_size_um * downsample_factor) ** 2

        num_itc = 0
        num_micro = 0
        num_macro = 0
        lesion_areas = []

        for lbl in range(1, num_labels):
            pixel_count = stats[lbl, cv2.CC_STAT_AREA]
            area_um2 = float(pixel_count) * pixel_area_um2
            lesion_areas.append(area_um2)

            if area_um2 <= self.itc_cutoff_um2:
                num_itc += 1
            elif area_um2 <= self.micro_cutoff_um2:
                num_micro += 1
            else:
                num_macro += 1

        max_area = max(lesion_areas) if lesion_areas else 0.0

        if num_macro > 0:
            highest_category = "macro"
        elif num_micro > 0:
            highest_category = "micro"
        elif num_itc > 0:
            highest_category = "itc"
        else:
            highest_category = "negative"

        return {
            "is_positive": highest_category != "negative",
            "highest_category": highest_category,
            "max_area_um2": max_area,
            "num_itc": num_itc,
            "num_micro": num_micro,
            "num_macro": num_macro,
            "lesion_areas_um2": lesion_areas,
        }

    @staticmethod
    def compute_patient_pn_stage(
        slide_summaries: List[Dict[str, Any]],
    ) -> str:
        """
        Aggregates lesion findings across all examined lymph node slides (typically 5)
        to assign the patient-level pathological nodal (pN) stage.
        
        Clinical Staging Criteria (CAMELYON17 / Tauqeer et al.):
        - pN0:      No metastasis in any node
        - pN0(i+):  Only isolated tumor cells (ITCs, area <= 200 um^2)
        - pN1mi:    Only micro-metastases (no macro-metastasis in any node)
        - pN1:      1–3 positive lymph nodes with at least one macro-metastasis
        - pN2:      4–9 positive lymph nodes with at least one macro-metastasis
        """
        num_nodes = len(slide_summaries)
        if num_nodes == 0:
            return "pN0"

        total_itc = sum(s.get("num_itc", 0) for s in slide_summaries)
        total_micro = sum(s.get("num_micro", 0) for s in slide_summaries)
        total_macro = sum(s.get("num_macro", 0) for s in slide_summaries)

        # Count positive nodes (slides that contain micro or macro lesions)
        positive_nodes = sum(
            1 for s in slide_summaries
            if s.get("highest_category") in ["micro", "macro"]
        )

        if total_macro == 0 and total_micro == 0 and total_itc == 0:
            return "pN0"

        if total_macro == 0 and total_micro == 0 and total_itc > 0:
            return "pN0(i+)"

        if total_macro == 0 and total_micro > 0:
            return "pN1mi"

        # At least one macro-metastasis exists:
        if 1 <= positive_nodes <= 3:
            return "pN1"
        elif positive_nodes >= 4:
            return "pN2"

        return "pN1"
