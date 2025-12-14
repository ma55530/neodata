"""
STEP File Parser for Facade Element Geometry Extraction
Extracts circles (screws, holes) from STEP files using text parsing.
"""

import os
import re
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Circle3D:
    """Represents a circular feature (screw hole, mounting hole)."""
    center: Tuple[float, float, float]
    radius: float
    normal: Tuple[float, float, float] = (0, 0, 1)
    component_type: str = ""
    
    def __post_init__(self):
        if self.radius <= 1.5:
            self.component_type = "small_hole"
        elif self.radius <= 3.0:
            self.component_type = "screw_hole"
        elif self.radius <= 6.0:
            self.component_type = "mounting_hole"
        else:
            self.component_type = "large_hole"


@dataclass 
class CADModel:
    """Parsed CAD model with geometric features."""
    filename: str
    circles: List[Circle3D] = field(default_factory=list)
    bounding_box: Tuple[float, ...] = (0,0,0,0,0,0)
    width: float = 0.0
    height: float = 0.0
    depth: float = 0.0
    
    def compute_features(self):
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounding_box
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.depth = zmax - zmin
    
    def get_signature(self) -> Dict:
        screw_holes = [c for c in self.circles if c.component_type == "screw_hole"]
        mounting_holes = [c for c in self.circles if c.component_type == "mounting_hole"]
        return {
            'screw_holes': len(screw_holes),
            'mounting_holes': len(mounting_holes),
            'total_circles': len(self.circles),
            'width_mm': round(self.width, 1),
            'height_mm': round(self.height, 1),
            'aspect_ratio': round(self.width / self.height, 2) if self.height > 0 else 0,
        }
    
    def to_dict(self) -> Dict:
        return {
            'filename': self.filename,
            'bounding_box': list(self.bounding_box),
            'dimensions': {'width': self.width, 'height': self.height, 'depth': self.depth},
            'circles': [{'center': list(c.center), 'radius': c.radius, 
                        'normal': list(c.normal), 'type': c.component_type} 
                       for c in self.circles],
            'signature': self.get_signature()
        }


def parse_step_file(filepath: str) -> CADModel:
    """Parse STEP file and extract circles."""
    model = CADModel(filename=os.path.basename(filepath))
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Parse entities
    entities = {}
    for match in re.finditer(r'#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;', content, re.DOTALL):
        entities[int(match.group(1))] = (match.group(2), match.group(3).strip())
    
    # Extract circles: CIRCLE('',#axis_ref,radius)
    seen = set()
    for eid, (etype, edata) in entities.items():
        if etype == 'CIRCLE':
            match = re.match(r"'[^']*'\s*,\s*#(\d+)\s*,\s*([\d.E+-]+)", edata)
            if match:
                axis_ref, radius = int(match.group(1)), float(match.group(2))
                center = resolve_axis_center(entities, axis_ref)
                if center:
                    key = (round(center[0], 1), round(center[1], 1), round(center[2], 1), round(radius, 1))
                    if key not in seen:
                        seen.add(key)
                        model.circles.append(Circle3D(center=center, radius=radius))
    
    # Bounding box from points
    points = []
    for eid, (etype, edata) in entities.items():
        if etype == 'CARTESIAN_POINT':
            match = re.match(r"'[^']*'\s*,\s*\(\s*([\d.E+-]+)\s*,\s*([\d.E+-]+)\s*,\s*([\d.E+-]+)", edata)
            if match:
                points.append([float(match.group(i)) for i in (1,2,3)])
    
    if points:
        pts = np.array(points)
        model.bounding_box = (pts[:,0].min(), pts[:,1].min(), pts[:,2].min(),
                              pts[:,0].max(), pts[:,1].max(), pts[:,2].max())
    
    model.compute_features()
    return model


def resolve_axis_center(entities, axis_ref):
    """Resolve AXIS2_PLACEMENT_3D to get center point."""
    if axis_ref not in entities:
        return None
    etype, edata = entities[axis_ref]
    if etype != 'AXIS2_PLACEMENT_3D':
        return None
    
    match = re.match(r"'[^']*'\s*,\s*#(\d+)", edata)
    if not match:
        return None
    
    point_ref = int(match.group(1))
    if point_ref not in entities:
        return None
    
    ptype, pdata = entities[point_ref]
    if ptype != 'CARTESIAN_POINT':
        return None
    
    match = re.match(r"'[^']*'\s*,\s*\(\s*([\d.E+-]+)\s*,\s*([\d.E+-]+)\s*,\s*([\d.E+-]+)", pdata)
    if match:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
    return None


def main():
    print("="*60)
    print("STEP File Parser")
    print("="*60)
    
    # Look for STEP files in parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    step_files = [f for f in os.listdir(parent_dir) if f.upper().endswith('.STP')]
    print(f"Looking in: {parent_dir}")
    print(f"Found {len(step_files)} STEP files\n")
    
    models = {}
    for sf in step_files:
        filepath = os.path.join(parent_dir, sf)
        print(f"Parsing: {sf}")
        model = parse_step_file(filepath)
        models[sf] = model
        
        sig = model.get_signature()
        print(f"  Dimensions: {model.width:.1f} x {model.height:.1f} x {model.depth:.1f} mm")
        print(f"  Circles: {sig['total_circles']} (screws: {sig['screw_holes']}, mounting: {sig['mounting_holes']})")
        print()
    
    # Save to JSON
    output = {name: m.to_dict() for name, m in models.items()}
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad_models.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
