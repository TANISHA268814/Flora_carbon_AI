"""
Populates sample_images/ directory with benchmark test images for hackathon evaluation:
1. DeepForest standard sample (OSBS NEON plot)
2. Sundarbans Mangrove Sector 4B Orthomosaic
3. Tropical Evergreen High-Density Canopy
4. Temperate Pine Woodland Plot
"""

import os
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(SAMPLE_DIR, exist_ok=True)


def generate_procedural_forest(
    filename: str,
    width: int = 1024,
    height: int = 768,
    tree_count: int = 48,
    palette_type: str = "mangrove"
):
    """Generates a realistic procedural aerial forest orthomosaic."""
    # Background terrain
    if palette_type == "mangrove":
        # Muddy water channels and dark silt soil
        base_color = (25, 45, 32)
        accent_ground = (18, 30, 24)
        tree_colors = [
            (28, 95, 48), (38, 120, 62), (20, 80, 40), (45, 140, 75),
            (15, 65, 35), (55, 155, 85), (32, 105, 52)
        ]
    elif palette_type == "tropical":
        # Rich rainforest soil with vibrant emergent crowns
        base_color = (30, 42, 28)
        accent_ground = (22, 35, 20)
        tree_colors = [
            (35, 125, 50), (48, 155, 68), (25, 100, 38), (60, 175, 80),
            (20, 85, 30), (52, 145, 60), (70, 190, 95)
        ]
    else: # pine
        base_color = (38, 45, 36)
        accent_ground = (28, 32, 26)
        tree_colors = [
            (30, 75, 55), (42, 98, 70), (22, 60, 45), (50, 115, 82),
            (18, 52, 38), (36, 88, 62)
        ]

    # Create base soil image
    img = Image.new("RGB", (width, height), base_color)
    draw = ImageDraw.Draw(img)

    # Texture background noise
    np.random.seed(42 if palette_type == "mangrove" else (101 if palette_type == "tropical" else 202))
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    base_arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(base_arr)
    draw = ImageDraw.Draw(img)

    # Draw irregular waterways / paths
    if palette_type == "mangrove":
        for _ in range(3):
            points = [(0, np.random.randint(height // 4, 3 * height // 4))]
            for x in range(100, width + 100, 100):
                points.append((x, points[-1][1] + np.random.randint(-40, 40)))
            draw.line(points, fill=(12, 24, 28), width=np.random.randint(18, 36), joint="curve")

    # Generate tree crowns with multi-layered canopy foliage
    np.random.seed(77)
    for _ in range(tree_count):
        cx = np.random.randint(40, width - 40)
        cy = np.random.randint(40, height - 40)
        radius = np.random.randint(22, 55)
        
        # Shadow underneath
        shadow_offset = (int(radius * 0.25), int(radius * 0.35))
        draw.ellipse(
            [cx - radius + shadow_offset[0], cy - radius + shadow_offset[1],
             cx + radius + shadow_offset[0], cy + radius + shadow_offset[1]],
            fill=(10, 18, 12, 160)
        )
        
        # Crown base foliage lobes
        t_color = tree_colors[np.random.randint(0, len(tree_colors))]
        for _ in range(6):
            ox = np.random.randint(-radius // 3, radius // 3)
            oy = np.random.randint(-radius // 3, radius // 3)
            sub_r = int(radius * np.random.uniform(0.65, 0.95))
            draw.ellipse(
                [cx + ox - sub_r, cy + oy - sub_r, cx + ox + sub_r, cy + oy + sub_r],
                fill=t_color
            )
            
        # Highlight sunlit crown apex
        highlight_color = tuple(min(255, c + 35) for c in t_color)
        hl_r = int(radius * 0.45)
        draw.ellipse(
            [cx - int(radius*0.15) - hl_r, cy - int(radius*0.15) - hl_r,
             cx - int(radius*0.15) + hl_r, cy - int(radius*0.15) + hl_r],
            fill=highlight_color
        )

    # Slight blur to simulate aerial sensor point-spread function
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    
    out_path = os.path.join(SAMPLE_DIR, filename)
    img.save(out_path, quality=95)
    print(f"Generated sample benchmark image: {out_path}")
    return out_path


def copy_deepforest_samples():
    """Tries to find and copy built-in DeepForest sample images if available."""
    try:
        from deepforest import get_data
        for name in ["OSBS_029.png", "SOAP_061.png"]:
            try:
                src = get_data(name)
                if os.path.exists(src):
                    dest = os.path.join(SAMPLE_DIR, name)
                    shutil.copyfile(src, dest)
                    print(f"Copied DeepForest standard benchmark: {dest}")
            except Exception:
                pass
    except Exception as e:
        print(f"Could not copy deepforest sample data: {e}")


if __name__ == "__main__":
    copy_deepforest_samples()
    generate_procedural_forest("Sundarbans_Sector_4B.png", width=1024, height=768, tree_count=65, palette_type="mangrove")
    generate_procedural_forest("Amazon_Tropical_Plot_08.png", width=1024, height=768, tree_count=85, palette_type="tropical")
    generate_procedural_forest("Temperate_Pine_Canopy.png", width=1024, height=768, tree_count=45, palette_type="pine")
    print("All sample images generated in sample_images/")
