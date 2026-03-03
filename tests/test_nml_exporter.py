import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scan_plan.nml_exporter import generate_nml


class TestGenerateNml:
    def test_generates_valid_xml(self, tmp_path):
        out = tmp_path / "test.nml"
        pts = np.array([[100, 200, 300], [400, 500, 600]])
        generate_nml(str(out), pts, D=50, H=80)

        tree = ET.parse(str(out))
        root = tree.getroot()
        assert root.tag == "things"
        bboxes = list(root.iter("userBoundingBox"))
        assert len(bboxes) == 2

    def test_bbox_coordinates(self, tmp_path):
        out = tmp_path / "coords.nml"
        pts = np.array([[100, 200, 300]])
        D, H = 50, 80
        generate_nml(str(out), pts, D=D, H=H)

        tree = ET.parse(str(out))
        bbox = list(tree.iter("userBoundingBox"))[0]
        # TopLeft = Center - Dim/2
        assert int(bbox.get("topLeftX")) == 100 - D // 2
        assert int(bbox.get("topLeftY")) == 200 - D // 2
        assert int(bbox.get("topLeftZ")) == 300 - H // 2
        assert int(bbox.get("width")) == D
        assert int(bbox.get("height")) == D
        assert int(bbox.get("depth")) == H

    def test_custom_color(self, tmp_path):
        out = tmp_path / "color.nml"
        pts = np.array([[0, 0, 0]])
        generate_nml(str(out), pts, D=10, H=10, color_hex="#FF0000")

        tree = ET.parse(str(out))
        bbox = list(tree.iter("userBoundingBox"))[0]
        assert float(bbox.get("color.r")) == pytest.approx(1.0)
        assert float(bbox.get("color.g")) == pytest.approx(0.0)
        assert float(bbox.get("color.b")) == pytest.approx(0.0)

    def test_empty_points(self, tmp_path):
        out = tmp_path / "empty.nml"
        pts = np.empty((0, 3))
        generate_nml(str(out), pts, D=10, H=10)

        tree = ET.parse(str(out))
        bboxes = list(tree.iter("userBoundingBox"))
        assert len(bboxes) == 0
