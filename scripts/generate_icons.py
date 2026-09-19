"""Generate PNG icons for the StudyTube Chrome extension."""

import os
from PIL import Image, ImageDraw

def create_icon(size: int, output_path: str):
    # Create image with red background and white graduation cap / play symbol
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Red rounded rectangle background
    radius = int(size * 0.22)
    draw.rounded_rectangle(
        [(0, 0), (size - 1, size - 1)],
        radius=radius,
        fill=(255, 0, 0, 255),
    )

    # White play triangle in center
    pad_left = int(size * 0.35)
    pad_right = int(size * 0.72)
    pad_top = int(size * 0.28)
    pad_bottom = int(size * 0.72)
    mid_y = int(size * 0.5)

    triangle_points = [
        (pad_left, pad_top),
        (pad_right, mid_y),
        (pad_left, pad_bottom),
    ]
    draw.polygon(triangle_points, fill=(255, 255, 255, 255))

    img.save(output_path, "PNG")
    print(f"Saved {output_path} ({size}x{size})")

if __name__ == "__main__":
    os.makedirs("extension/icons", exist_ok=True)
    create_icon(16, "extension/icons/icon-16.png")
    create_icon(48, "extension/icons/icon-48.png")
    create_icon(128, "extension/icons/icon-128.png")
