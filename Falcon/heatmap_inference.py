import os
import re
import json         # ← handy if the HTML embeds JSON
from bs4 import BeautifulSoup

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
# 1. ----------  PARSE THE HTML LOG  -------------------------
# -----------------------------------------------------------

def parse_wsol_html_log(html_path):
    """
    Return a dictionary in the exact format your later code needs,
    i.e. {sample_id: {'img_path': str,
                      'predictions': [{'final_glimpse_loc_and_dim': torch.Tensor([x,y,w,h])}, …]}}
    """
    with open(html_path, "r") as f:
        soup = BeautifulSoup(f, "html.parser")

    collected = {}

    # -- Example 1: each sample is a <tr> in a table --------------------------
    for row in soup.select("table#results tbody tr"):
        # adapt selectors / indices to match your HTML structure
        sample_id = int(row.select_one("td.sample-id").text)
        img_path  = row.select_one("td.img-path").text.strip()

        # glimpse boxes stored as JSON in a <data-*> attribute?
        boxes_json = row["data-glimpses"]          # e.g. "[[x,y,w,h], …]"
        boxes = json.loads(boxes_json)

        collected[sample_id] = {
            "img_path": img_path,
            "predictions": [
                {"final_glimpse_loc_and_dim": torch.tensor(b)}
                for b in boxes
            ],
        }

    # -- Example 2: if the HTML embeds a single <script> with a big JSON blob –
    #     <script id="sample-data" type="application/json">{…}</script>
    # raw_json = soup.select_one("#sample-data").string
    # collected = json.loads(raw_json)          # already the right shape

    return collected


# ------------------------------------------------------------------
# 2. -------  CONFIG & DATA LOADING  -------------------------------
# ------------------------------------------------------------------

HTML_LOG = (
    "/home/20204130/Falcon/FALcon-main/results/imagenet/"
    "wsol_method_PSOL/trained_on_train_split/"
    "arch_vgg16_pretrained_init_normalization_none_seed_16/"
    "log_wsol_all_at_once.html"
)

collected_samples = parse_wsol_html_log(HTML_LOG)      # ← **new**

img_size   = (224, 224)
output_dir = "heatmaps_per_image"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor()])

# ------------------------------------------------------------------
# 3. -------  HEATMAPS PER IMAGE  ----------------------------------
# ------------------------------------------------------------------

for sample_id, sample in collected_samples.items():

    # --- optional: verify the image exists ------------------------
    img_path = sample["img_path"]
    if not os.path.isfile(img_path):
        print(f"[warn] Image not found: {img_path}")
        continue

    # --- build importance map -------------------------------------
    glimpse_boxes = [p["final_glimpse_loc_and_dim"] for p in sample["predictions"]]
    if not glimpse_boxes:
        continue

    importance_map = torch.zeros(img_size, dtype=torch.int)
    for box in glimpse_boxes:
        x, y, w, h = box.int().tolist()
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, img_size[1]), min(y + h, img_size[0])
        importance_map[y1:y2, x1:x2] += 1

    # --- save visualisation ---------------------------------------
    importance_np = importance_map.numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(importance_np, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Importance (Overlap Count)")
    plt.axis("off")
    plt.title(f"Sample {sample_id} Heatmap")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"sample_{sample_id}_heatmap.png"))
    plt.close()

    torch.save(importance_map,
               os.path.join(output_dir, f"sample_{sample_id}_heatmap.pt"))
