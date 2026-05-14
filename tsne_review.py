import os
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================

PATCHES_DIR = "patches"
LABELS_FILE = "labels.csv"

# =========================
# LOAD LABELS
# =========================

df = pd.read_csv(LABELS_FILE)

images = []
labels = []
filenames = []

# =========================
# LOAD IMAGES
# =========================

for _, row in df.iterrows():

    filename = row["filename"]

    # CMYK level
    level = max(row["C"], row["M"], row["Y"], row["K"])

    img_path = None

    # ищем файл
    for lvl in range(6):

        possible = os.path.join(PATCHES_DIR, str(lvl), filename)

        if os.path.exists(possible):
            img_path = possible
            break

    if img_path is None:
        continue

    try:
        img = Image.open(img_path).convert("RGB")

        # resize для скорости
        img = img.resize((32, 32))

        # embedding
        img_array = np.array(img).flatten()

        images.append(img_array)
        labels.append(level)
        filenames.append(filename)

    except:
        continue

print(f"Loaded: {len(images)} images")

# =========================
# t-SNE
# =========================

X = np.array(images)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)

X_tsne = tsne.fit_transform(X)

# =========================
# SAVE t-SNE PLOT
# =========================

plt.figure(figsize=(10, 8))

scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)

plt.colorbar(scatter)

plt.title("t-SNE Embedding Visualization")

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# SAVE PNG
plt.savefig("tsne_plot.png", dpi=300, bbox_inches="tight")

plt.show()

# =========================
# OUTLIER DETECTION
# =========================

center = np.mean(X_tsne, axis=0)

distances = np.linalg.norm(X_tsne - center, axis=1)

top_idx = np.argsort(distances)[-20:]

print("\nTOP PRIORITY REVIEW SAMPLES:\n")

priority_rows = []

for idx in top_idx:

    print(
        filenames[idx], "| Level:", labels[idx], "| Distance:", round(distances[idx], 2)
    )

    priority_rows.append([filenames[idx], labels[idx], round(distances[idx], 2)])

# =========================
# SAVE CSV
# =========================

priority_df = pd.DataFrame(priority_rows, columns=["filename", "label", "distance"])

priority_df.to_csv("priority_review_samples.csv", index=False)

print("\nSaved:")
print("✓ tsne_plot.png")
print("✓ priority_review_samples.csv")
