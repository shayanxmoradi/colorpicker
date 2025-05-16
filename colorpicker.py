import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from sklearn.mixture import GaussianMixture
from collections import Counter
import os

def get_dominant_colors_meanshift(image, n_colors=5):
    img = image.reshape((-1, 3))
    bandwidth = estimate_bandwidth(img, quantile=0.1, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(img)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_.astype(np.uint8)

    label_counts = Counter(labels)
    top_labels = label_counts.most_common(n_colors)
    return np.array([cluster_centers[label] for label, _ in top_labels])

def get_dominant_colors_dbscan(image, n_colors=5):
    img = image.reshape((-1, 3))
    db = DBSCAN(eps=10, min_samples=100).fit(img)
    labels = db.labels_
    label_counts = Counter(labels[labels >= 0])
    top_labels = label_counts.most_common(n_colors)
    dominant_colors = []
    for label, _ in top_labels:
        dominant_colors.append(np.mean(img[labels == label], axis=0))
    return np.array(dominant_colors).astype(np.uint8)

def get_dominant_colors_gmm(image, n_colors=5):
    img = image.reshape((-1, 3))
    gmm = GaussianMixture(n_components=n_colors).fit(img)
    return gmm.means_.astype(np.uint8)

def create_color_image(image, palette):
    img = image.reshape((-1, 3))
    distances = np.sqrt(((img[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2).sum(axis=2))
    labels = np.argmin(distances, axis=1)
    quantized = palette[labels]
    return quantized.reshape(image.shape).astype(np.uint8)

def visualize_and_save(original, result_img, palette, method_name):
    result_img = result_img.astype(np.uint8)
    cv2.imwrite(f"quantized_{method_name}.jpg", result_img)

    # Palette bar
    palette_img = np.zeros((100, 300, 3), dtype=np.uint8)
    step = palette_img.shape[1] // len(palette)
    for i, color in enumerate(palette):
        palette_img[:, i*step:(i+1)*step] = color

    # Display original, result, and palette
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'{method_name} Result')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(palette_img, cv2.COLOR_BGR2RGB))
    plt.title('Dominant Colors')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"output_{method_name}.png")
    print(f"✔ Saved quantized_{method_name}.jpg and output_{method_name}.png")

def run_all_algorithms(image_path, n_colors=5):
    if not os.path.isfile(image_path):
        print(f"❌ File not found: {image_path}")
        return

    original = cv2.imread(image_path)

    # Resize to limit memory use
    max_dim = 400
    if max(original.shape[:2]) > max_dim:
        scale = max_dim / max(original.shape[:2])
        original = cv2.resize(original, (0, 0), fx=scale, fy=scale)

    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    methods = {
        "MeanShift": get_dominant_colors_meanshift,
        "DBSCAN": get_dominant_colors_dbscan,
        "GMM": get_dominant_colors_gmm,
    }

    for method_name, method_func in methods.items():
        try:
            print(f"🔍 Running {method_name}...")
            palette = method_func(image_rgb, n_colors=n_colors)
            result_img = create_color_image(original, palette)
            visualize_and_save(original, result_img, palette, method_name)
        except Exception as e:
            print(f"⚠️ {method_name} failed: {e}")


run_all_algorithms("/Users/shayan/PycharmProjects/colorpicker/Screenshot 2025-05-15 at 08.16.57.png", n_colors=5)

# run_all_algorithms("/Users/shayan/PycharmProjects/colorpicker/Screenshot 2025-05-15 at 08.16.57.png")
