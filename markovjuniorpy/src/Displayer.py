import PaletteUtil as pu
from PIL import Image

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

PIXEL_SIZE = 5

def save_as_png(image:str, name='output'):
    # Split the input string into rows and columns
    rows = image.split('\n')
    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)

    image = Image.new('RGB', (num_cols * PIXEL_SIZE, num_rows * PIXEL_SIZE))

    for row_idx, row in enumerate(rows):
        for col_idx, char in enumerate(row):
            
            color = pu.get_color(char)

            x = col_idx * PIXEL_SIZE
            y = row_idx * PIXEL_SIZE
            image.paste(color, (x, y, x + PIXEL_SIZE, y + PIXEL_SIZE))

    image.save(name+'.png', 'png')

def save_as_txt(image:str, name='output'):
    with open(name+'.txt', 'w') as file:
        file.write(image)