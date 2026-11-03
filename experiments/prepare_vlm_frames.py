
from pathlib import Path
import shutil

source = Path("data/Raw_Data")
destination = Path("vlm_frames")
destination.mkdir(exist_ok=True)

extensions = {".jpg", ".jpeg", ".png"}

for video in ["vidd_1", "vidd_2", "vidd_3"]:
    folder = source / video

    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        print(f"No images found in {folder}")
        continue

    first_frame = images[0]
    output = destination / f"{video}{first_frame.suffix.lower()}"

    shutil.copy2(first_frame, output)
    print(f"Copied {first_frame} -> {output}")
