import xml.etree.ElementTree as ET

def generate_nml(filename, points, D, H, color_hex="#00FFFF"):
    things = ET.Element("things")
    ET.SubElement(things, "meta", name="writer", content="ScanPlanner")
    params = ET.SubElement(things, "parameters")

    h_color = color_hex.lstrip('#')
    try:
        r, g, b = tuple(int(h_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
    except ValueError:
        r, g, b = (0.0, 1.0, 1.0) 

    for i, pt in enumerate(points):
        # WebKnossos expects TopLeft (Center - Dim/2)
        tlx = int(pt[0] - D/2)
        tly = int(pt[1] - D/2)
        tlz = int(pt[2] - H/2)

        bbox = ET.SubElement(params, "userBoundingBox")
        bbox.set("id", str(i+1))
        bbox.set("name", f"Tile_{i}")
        bbox.set("isVisible", "true")
        bbox.set("color.r", str(r))
        bbox.set("color.g", str(g))
        bbox.set("color.b", str(b))
        bbox.set("color.a", "1.0")
        bbox.set("topLeftX", str(tlx))
        bbox.set("topLeftY", str(tly))
        bbox.set("topLeftZ", str(tlz))
        bbox.set("width", str(int(D)))
        bbox.set("height", str(int(D)))
        bbox.set("depth", str(int(H)))

    tree = ET.ElementTree(things)
    tree.write(filename, encoding="utf-8", xml_declaration=True)