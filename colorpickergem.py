from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from collections import Counter
from scipy.spatial import KDTree  # For efficient nearest color lookup
import os  # For creating output directory


def get_dominant_colors_median_cut(image_path, num_colors=6, resize_factor=0.1):
    """
    Extracts dominant colors from an image using the Median Cut algorithm.
    Returns dominant colors and the (potentially resized) image used for extraction.
    """
    try:
        img = Image.open(image_path).convert('RGB')

        img_for_extraction = img
        if resize_factor < 1.0:
            new_width = int(img.width * resize_factor)
            new_height = int(img.height * resize_factor)
            img_for_extraction = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pixels = list(img_for_extraction.getdata())

        if not pixels:
            print("Warning: No pixel data found after loading and resizing.")
            return [], img

        buckets = [pixels]

        while len(buckets) < num_colors:
            next_buckets = []
            for bucket in buckets:
                if not bucket:
                    continue

                min_r, min_g, min_b = float('inf'), float('inf'), float('inf')
                max_r, max_g, max_b = float('-inf'), float('-inf'), float('-inf')

                for r, g, b in bucket:
                    min_r, max_r = min(min_r, r), max(max_r, r)
                    min_g, max_g = min(min_g, g), max(max_g, g)
                    min_b, max_b = min(min_b, b), max(max_b, b)

                range_r, range_g, range_b = max_r - min_r, max_g - min_g, max_b - min_b

                split_dim = 0
                if range_g > range_r and range_g > range_b:
                    split_dim = 1
                elif range_b > range_r and range_b > range_g:
                    split_dim = 2

                bucket.sort(key=lambda x: x[split_dim])

                median_index = len(bucket) // 2
                next_buckets.append(bucket[:median_index])
                next_buckets.append(bucket[median_index:])
            buckets = next_buckets
            if len(buckets) >= 2 ** 8:
                break

        dominant_colors = []
        for bucket in buckets:
            if not bucket:
                continue
            avg_r = sum(p[0] for p in bucket) // len(bucket)
            avg_g = sum(p[1] for p in bucket) // len(bucket)
            avg_b = sum(p[2] for p in bucket) // len(bucket)
            dominant_colors.append((avg_r, avg_g, avg_b))

        return dominant_colors[:num_colors], img  # Return original full-res image for recoloring

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return [], None
    except Exception as e:
        print(f"An error occurred in Median Cut: {e}")
        return [], None


def get_dominant_colors_sklearn(image_path, algorithm='agglomerative', num_colors=6, resize_factor=0.1):
    """
    Extracts dominant colors using scikit-learn clustering algorithms.
    Returns dominant colors and the original full-resolution image.
    """
    try:
        original_img = Image.open(image_path).convert('RGB')
        img_for_extraction = original_img

        if resize_factor < 1.0:
            new_width = int(original_img.width * resize_factor)
            new_height = int(original_img.height * resize_factor)
            img_for_extraction = original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pixels = np.array(img_for_extraction.getdata())

        if pixels.shape[0] == 0:
            print("Warning: No pixel data found after loading and resizing.")
            return [], original_img

        actual_num_colors = min(num_colors, len(np.unique(pixels, axis=0)))
        if actual_num_colors == 0:
            if len(pixels) > 0:
                return [tuple(map(int, pixels[0]))], original_img
            else:
                return [], original_img

        if actual_num_colors < 1:  # Ensure at least 1 cluster
            actual_num_colors = 1

        if actual_num_colors == 1 and algorithm == 'gmm':
            counts = Counter(map(tuple, pixels))
            return [counts.most_common(1)[0][0]], original_img if counts else [], original_img
        elif actual_num_colors < 2 and algorithm == 'gmm':  # GMM often needs n_components >= 2
            # Fallback for GMM if only 1 color is requested or available after unique check
            print(
                "Warning: GMM requires at least 2 components for typical covariance types, or 1 with specific handling. Falling back to most frequent for 1 color.")
            counts = Counter(map(tuple, pixels))
            return [counts.most_common(1)[0][0]], original_img if counts else [], original_img

        model = None
        if algorithm == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=actual_num_colors, linkage='ward')
        elif algorithm == 'gmm':
            # For GMM, n_components must be <= n_samples.
            # If actual_num_colors is 1 after adjustments, GaussianMixture might still want more samples for certain covariance_types.
            # 'diag', 'tied', 'spherical' can sometimes work with n_components=1 if n_samples >= 1.
            # 'full' usually requires n_components <= n_samples - n_features.
            if pixels.shape[0] < actual_num_colors:  # Ensure enough samples
                print(
                    f"Warning: Not enough samples ({pixels.shape[0]}) for GMM with {actual_num_colors} components. Reducing components.")
                actual_num_colors = max(1, pixels.shape[0])  # at least 1 component

            model = GaussianMixture(n_components=actual_num_colors, random_state=0, covariance_type='diag')
        else:
            raise ValueError("Unsupported algorithm. Choose 'agglomerative' or 'gmm'.")

        model.fit(pixels)

        dominant_colors = []
        if algorithm == 'agglomerative':
            labels = model.labels_
            for i in range(actual_num_colors):
                cluster_pixels = pixels[labels == i]
                if cluster_pixels.size > 0:
                    dominant_color = np.mean(cluster_pixels, axis=0).astype(int)
                    dominant_colors.append(tuple(dominant_color))
        elif algorithm == 'gmm':
            dominant_colors = [tuple(map(int, mean)) for mean in model.means_]
            if hasattr(model, 'weights_'):
                sorted_indices = np.argsort(model.weights_)[::-1]
                dominant_colors = [dominant_colors[i] for i in sorted_indices]

        # Ensure we return a list of colors even if actual_num_colors was 0 or 1.
        if not dominant_colors and len(pixels) > 0:  # If clustering failed but pixels exist
            counts = Counter(map(tuple, pixels))
            if counts:
                dominant_colors = [counts.most_common(1)[0][0]]

        return dominant_colors, original_img  # Return original full-res image for recoloring

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return [], None
    except Exception as e:
        print(f"An error occurred in {algorithm} clustering: {e}")
        return [], None


def create_recolored_image(original_image, dominant_colors):
    """
    Recolors the original_image using only the dominant_colors.
    Each pixel in the original image is mapped to the nearest color in dominant_colors.
    """
    if not dominant_colors:
        print("No dominant colors provided for recoloring.")
        return original_image  # Return original if no colors

    # Use KDTree for efficient nearest neighbor search for colors
    palette_array = np.array(dominant_colors)
    kdtree = KDTree(palette_array)

    # Get pixel data from the original image (full resolution)
    img_pixels = np.array(original_image.getdata())

    # Find the closest dominant color for each pixel
    distances, indices = kdtree.query(img_pixels, k=1)
    new_pixels = palette_array[indices]

    # Create new image
    recolored_img = Image.new(original_image.mode, original_image.size)
    recolored_img.putdata([tuple(pixel) for pixel in new_pixels])

    return recolored_img


def display_and_save_results(colors, original_img, base_filename="output", title="Dominant Colors",
                             output_dir="output_images", show_images=True, save_images=True):
    """
    Displays the extracted colors, shows/saves a color palette, and shows/saves a recolored image.
    """
    if not colors:
        print(f"{title}: No colors extracted or an error occurred.")
        return

    print(f"\n{title}:")
    for i, color in enumerate(colors):
        hex_color = '#%02x%02x%02x' % color
        print(f"  Color {i + 1}: RGB{color} - HEX {hex_color}")

    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Create and handle color palette
    palette_height = 50
    palette_width_per_color = 50
    palette = Image.new('RGB', (len(colors) * palette_width_per_color, palette_height))
    draw = ImageDraw.Draw(palette)
    for i, color in enumerate(colors):
        box = (i * palette_width_per_color, 0, (i + 1) * palette_width_per_color, palette_height)
        draw.rectangle(box, fill=color)

    if show_images:
        try:
            palette.show(title=f"{title} Palette")
        except Exception as e:
            print(f"Could not display color palette image: {e}")
    if save_images:
        palette_filename = os.path.join(output_dir, f"{base_filename}_palette.png")
        try:
            palette.save(palette_filename)
            print(f"Saved color palette to {palette_filename}")
        except Exception as e:
            print(f"Could not save color palette: {e}")

    # Create and handle recolored image
    if original_img:
        recolored_image = create_recolored_image(original_img, colors)
        if show_images:
            try:
                recolored_image.show(title=f"{title} Recolored Image")
            except Exception as e:
                print(f"Could not display recolored image: {e}")
        if save_images:
            recolored_filename = os.path.join(output_dir, f"{base_filename}_recolored.png")
            try:
                recolored_image.save(recolored_filename)
                print(f"Saved recolored image to {recolored_filename}")
            except Exception as e:
                print(f"Could not save recolored image: {e}")
    else:
        print("Original image not available for recoloring.")


if __name__ == "__main__":
    image_file = "/Users/shayan/PycharmProjects/colorpicker/Screenshot 2025-05-15 at 08.16.57.png"  # <--- CHANGE THIS TO YOUR IMAGE FILE
    num_dominant_colors = 6
    output_directory = "color_extraction_results"  # Directory to save images

    # --- Ensure the input image exists ---
    if not os.path.exists(image_file):
        print(f"ERROR: Input image '{image_file}' not found. Please check the path.")
    else:
        # --- Test Median Cut ---
        print("--- Using Median Cut Algorithm ---")
        median_cut_colors, mc_original_img = get_dominant_colors_median_cut(
            image_file,
            num_colors=num_dominant_colors,
            resize_factor=0.05  # Resize for faster extraction, but recolor full original
        )
        if mc_original_img:  # Proceed if image was loaded
            display_and_save_results(
                median_cut_colors,
                mc_original_img,
                base_filename=f"median_cut_{num_dominant_colors}colors",
                title=f"Median Cut ({num_dominant_colors} colors)",
                output_dir=output_directory,
                show_images=True,  # Set to False if you don't want pop-ups
                save_images=True
            )

        # --- Test Agglomerative Clustering ---
        print("\n--- Using Agglomerative Clustering ---")
        agg_colors, agg_original_img = get_dominant_colors_sklearn(
            image_file,
            algorithm='agglomerative',
            num_colors=num_dominant_colors,
            resize_factor=0.05  # Smaller factor for faster extraction
        )
        if agg_original_img:
            display_and_save_results(
                agg_colors,
                agg_original_img,
                base_filename=f"agglomerative_{num_dominant_colors}colors",
                title=f"Agglomerative Clustering ({num_dominant_colors} colors)",
                output_dir=output_directory,
                show_images=True,
                save_images=True
            )

        # --- Test Gaussian Mixture Model ---
        print("\n--- Using Gaussian Mixture Model ---")
        gmm_colors, gmm_original_img = get_dominant_colors_sklearn(
            image_file,
            algorithm='gmm',
            num_colors=num_dominant_colors,
            resize_factor=0.1  # GMM can often handle a bit more data
        )
        if gmm_original_img:
            display_and_save_results(
                gmm_colors,
                gmm_original_img,
                base_filename=f"gmm_{num_dominant_colors}colors",
                title=f"Gaussian Mixture Model ({num_dominant_colors} colors)",
                output_dir=output_directory,
                show_images=True,
                save_images=True
            )