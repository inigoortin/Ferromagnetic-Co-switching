# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:34:53 2025

@author: iorti
"""

# Add numbers to the experimental setup image

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Load image
img = Image.open("montaje.jpg")
draw = ImageDraw.Draw(img)

# Font
try:
    font = ImageFont.truetype("arial.ttf", size=24)
except:
    font = ImageFont.load_default()

# Coordinates of the points
puntos = [(1000, 250), (1115, 260), (1150, 800), (1280,770), (1040,700), (825,580), (1170,600)]
for i, (x, y) in enumerate(puntos, start=1):
    r = 15
    draw.ellipse((x - r, y - r, x + r, y + r), outline='red', fill='white', width=2)

    text = str(i)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Centre the text at the point
    draw.text((x - text_width / 2, y - text_height / 2), text, font=font, fill='red')

# Save the modified image
img.save("montaje_numerado.jpg")

# Display image without white borders
plt.imshow(img)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
