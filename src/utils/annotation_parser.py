"""
ASAP XML Annotation Parser for CAMELYON datasets.
Extracts tumor contours and determines patch-level tumor intersection.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, box
from shapely.validation import make_valid


class AnnotationParser:
    """
    Parses ASAP XML files and maps polygon annotations to spatial patch coordinates.
    """

    def __init__(self, xml_path: Optional[str | Path] = None):
        self.polygons: List[Polygon] = []
        if xml_path is not None:
            self.load_xml(xml_path)

    def load_xml(self, xml_path: str | Path) -> List[Polygon]:
        path = Path(xml_path)
        if not path.exists():
            self.polygons = []
            return self.polygons

        tree = ET.parse(str(path))
        root = tree.getroot()
        polygons = []

        for annotation in root.findall(".//Annotation"):
            coords = []
            for coord in annotation.findall(".//Coordinate"):
                x = float(coord.get("X"))
                y = float(coord.get("Y"))
                coords.append((x, y))

            if len(coords) >= 3:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                polygons.append(poly)

        self.polygons = polygons
        return self.polygons

    def is_tumor_patch(
        self,
        x: int,
        y: int,
        patch_size: int,
        min_overlap: float = 0.25,
    ) -> Tuple[int, float]:
        """
        Tests whether a patch at (x, y) overlaps with any metastatic lesion.
        
        Args:
            x: Top-left X coordinate at Level 0.
            y: Top-left Y coordinate at Level 0.
            patch_size: Patch width and height at Level 0.
            min_overlap: Minimum intersection ratio (area / patch_area) to label as tumor.
            
        Returns:
            Tuple[int, float]: (label [1 for tumor, 0 for normal], overlap_ratio)
        """
        if not self.polygons:
            return 0, 0.0

        patch_poly = box(x, y, x + patch_size, y + patch_size)
        patch_area = patch_poly.area
        max_ratio = 0.0

        for poly in self.polygons:
            if patch_poly.intersects(poly):
                intersection = patch_poly.intersection(poly)
                ratio = intersection.area / patch_area
                if ratio > max_ratio:
                    max_ratio = ratio

        label = 1 if max_ratio >= min_overlap else 0
        return label, max_ratio
