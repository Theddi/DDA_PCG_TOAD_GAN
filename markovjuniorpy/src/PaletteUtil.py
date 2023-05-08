import xml.etree.ElementTree as ET

color_dict = {}

tree = ET.parse('palette.xml')
root = tree.getroot()

for color_elem in root.findall('color'):
    symbol = color_elem.get('symbol')
    hex = color_elem.get('value')
    color_dict[symbol] = '#' + hex

def get_color(abbrv: str):
    return color_dict.get(abbrv)
