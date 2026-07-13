"""Bounding box visualization - draws detected details on the original image.

Creates a copy of the input image with colored bounding boxes and labels
for each detected detail. Used for qualitative analysis of detection quality.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Distinct colors for up to 10 details
BBOX_COLORS = [
    (255, 0, 0),      # red
    (0, 180, 0),      # green
    (0, 100, 255),    # blue
    (255, 165, 0),    # orange
    (180, 0, 255),    # purple
    (0, 200, 200),    # cyan
    (255, 255, 0),    # yellow
    (255, 0, 150),    # pink
    (100, 255, 100),  # light green
    (255, 150, 150),  # salmon
]


def draw_bboxes(
    image_path: str,
    details: list[dict],
    output_path: str,
) -> str | None:
    """Draw bounding boxes on the image and save the result.

    Args:
        image_path: Path to the original image.
        details: List of detail dicts, each with 'name' and optional 'bbox'.
        output_path: Where to save the annotated image.

    Returns:
        The output path if successful, None if no bboxes to draw.
    """
    # Filter to details with valid bboxes
    boxed = [d for d in details if d.get("bbox") is not None]
    if not boxed:
        return None

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try to load a decent font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for i, detail in enumerate(boxed):
        color = BBOX_COLORS[i % len(BBOX_COLORS)]
        bbox = detail["bbox"]
        name = detail.get("name", f"detail_{i}")

        # Convert normalized coords to pixels
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        # Draw rectangle (3px thick)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Draw label background
        label = f"{i + 1}. {name}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Position label above the box, or below if too close to top
        label_y = y1 - text_h - 6
        if label_y < 0:
            label_y = y2 + 4

        draw.rectangle(
            [x1, label_y, x1 + text_w + 8, label_y + text_h + 4],
            fill=color,
        )
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path
