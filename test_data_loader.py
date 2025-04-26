import numpy as np
import cv2
import os
import sys

# import matplotlib.pyplot as plt # Uncomment for visualization

# Add the directory containing data_loader.py to the Python path
# Adjust the path if test_data_loader.py is not in the same directory as TRAIN.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

from data_loader import load_coco_data, _read_and_process_single_sample, read_batch

# --- Test Configuration ---
# IMPORTANT: Replace with paths to a *small* subset of your actual data
# or create dummy COCO JSON and dummy image files for testing.
DUMMY_ANNOTATION_FILE = r"../output/train.json"  # Example path
DUMMY_IMAGE_DIR = r"../images/"  # Example path

# Use configurations similar to TRAIN.py or modify for testing
TARGET_IMAGE_SIZE = 512  # Smaller size for faster testing maybe?
AUGMENTATIONS = {
    "enabled": True,
    "prob": 1.0,  # Apply augmentations every time for testing
    "scale": {"min": 0.8, "max": 1.2},
    "translate": {"max_x": 0.1, "max_y": 0.1},
    "rotate": {"max_deg": 5},
    "noise": {"enabled": True, "std_dev": 15},
}
NUM_TEST_BATCHES = 5
SAVE_SAMPLE_IMAGE = True  # Set to True to save a visualized sample
SAMPLE_OUTPUT_DIR = "./test_dataloader_output"
# --- End Test Configuration ---


def test_data_loader():
    print("--- Starting Data Loader Test ---")

    # 1. Test loading data
    print(f"Loading test annotations from: {DUMMY_ANNOTATION_FILE}")
    print(f"Loading test images from: {DUMMY_IMAGE_DIR}")
    try:
        data = load_coco_data(DUMMY_ANNOTATION_FILE, DUMMY_IMAGE_DIR)
        assert len(data) > 0, "load_coco_data returned an empty list. Check paths and annotations."
        print(f"Successfully loaded {len(data)} image entries.")
    except Exception as e:
        print(f"Error during load_coco_data: {e}")
        print("Ensure DUMMY_ANNOTATION_FILE and DUMMY_IMAGE_DIR point to valid data.")
        return

    # 2. Test reading single samples
    print(f"\nAttempting to read {NUM_TEST_BATCHES} single samples (using _read_and_process_single_sample)...")
    successful_samples = 0
    first_sample_data = None

    for i in range(NUM_TEST_BATCHES):
        print(f"-- Reading sample {i+1}/{NUM_TEST_BATCHES} --")
        # Call the single-sample function directly
        sample_data = _read_and_process_single_sample(data, TARGET_IMAGE_SIZE, AUGMENTATIONS)

        if sample_data[0] is not None:
            img, masks, points, labels = sample_data
            print(f"  Sample {i+1} loaded successfully.")
            successful_samples += 1

            # Basic Assertions (similar to before, but shapes might vary in N dimension)
            assert isinstance(img, np.ndarray), f"Image is not a numpy array (type: {type(img)})"
            assert img.shape == (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3), f"Image shape incorrect: {img.shape}"
            assert img.dtype == np.uint8, f"Image dtype incorrect: {img.dtype}"

            # Check if masks/points/labels were actually returned (could be empty if no valid annotations)
            if masks is None or points is None or labels is None:
                print(f"  Warning: Sample {i+1} has None for masks/points/labels.")
                continue  # Skip detailed checks if no annotations processed

            if masks.shape[0] == 0:
                print(f"  Warning: Sample {i+1} has 0 masks.")
                # Allow continuing if points/labels are also 0, otherwise it's an error
                assert points.shape[0] == 0, "Points found for 0 masks"
                assert labels.shape[0] == 0, "Labels found for 0 masks"
                continue  # Skip further checks

            assert isinstance(masks, np.ndarray), f"Masks is not a numpy array (type: {type(masks)})"
            assert masks.ndim == 3, f"Masks ndim incorrect: {masks.ndim}"
            assert masks.shape[1:] == (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), f"Masks shape incorrect: {masks.shape}"
            assert masks.dtype == np.uint8, f"Masks dtype incorrect: {masks.dtype}"
            assert np.all((masks == 0) | (masks == 1)), "Masks contain values other than 0 or 1"

            assert isinstance(points, np.ndarray), f"Points is not a numpy array (type: {type(points)})"
            assert points.ndim == 3, f"Points ndim incorrect: {points.ndim}"
            assert (
                points.shape[0] == masks.shape[0]
            ), f"Number of points ({points.shape[0]}) != number of masks ({masks.shape[0]})"
            # Shape is now (N, 1, 2) where N is number of masks
            assert points.shape[1:] == (1, 2), f"Points inner shape incorrect: {points.shape}"
            assert (
                np.all(points >= 0)
                and np.all(points[:, :, 0] < TARGET_IMAGE_SIZE)
                and np.all(points[:, :, 1] < TARGET_IMAGE_SIZE)
            ), "Points coordinates out of bounds"

            assert isinstance(labels, np.ndarray), f"Labels is not a numpy array (type: {type(labels)})"
            # Shape is now (N, 1)
            assert labels.shape == (points.shape[0], 1), f"Labels shape incorrect: {labels.shape}"
            assert np.all(labels == 1), "Labels contain values other than 1"

            print(f"  Assertions passed for sample {i+1}. Masks: {masks.shape[0]}, Points: {points.shape[0]}")

            if first_sample_data is None:
                first_sample_data = sample_data  # Save data from the first successful sample
        else:
            print(f"  Sample {i+1} failed to load (returned None). Check warnings above.")

    print(f"\nSuccessfully loaded {successful_samples}/{NUM_TEST_BATCHES} single samples.")
    assert successful_samples > 0, "Failed to load any single samples. Check data and loader logic."

    # 3. Test reading batches using the new read_batch function
    print(f"\nAttempting to read {NUM_TEST_BATCHES} batches (using read_batch)...")
    successful_batches = 0
    first_batch_data = None
    BATCH_SIZE_TEST = 4  # Define a batch size for testing

    for i in range(NUM_TEST_BATCHES):
        print(f"-- Reading batch {i+1}/{NUM_TEST_BATCHES} (Size: {BATCH_SIZE_TEST}) --")
        # Call the new batch function
        batch_data = read_batch(data, BATCH_SIZE_TEST, TARGET_IMAGE_SIZE, AUGMENTATIONS)

        if batch_data[0] is not None:
            batch_imgs, batch_masks, batch_points, batch_labels = batch_data
            print(f"  Batch {i+1} loaded successfully.")
            successful_batches += 1

            # Batch Assertions
            assert isinstance(batch_imgs, list), f"Batch images is not a list (type: {type(batch_imgs)})"
            assert len(batch_imgs) == BATCH_SIZE_TEST, f"Incorrect number of images in batch: {len(batch_imgs)}"
            for img_idx, img in enumerate(batch_imgs):
                assert isinstance(img, np.ndarray), f"Image {img_idx} is not a numpy array"
                assert img.shape == (
                    TARGET_IMAGE_SIZE,
                    TARGET_IMAGE_SIZE,
                    3,
                ), f"Image {img_idx} shape incorrect: {img.shape}"
                assert img.dtype == np.uint8, f"Image {img_idx} dtype incorrect: {img.dtype}"

            assert isinstance(batch_masks, np.ndarray), f"Batch masks is not a numpy array (type: {type(batch_masks)})"
            assert batch_masks.shape == (
                BATCH_SIZE_TEST,
                TARGET_IMAGE_SIZE,
                TARGET_IMAGE_SIZE,
            ), f"Batch masks shape incorrect: {batch_masks.shape}"
            assert batch_masks.dtype == np.uint8, f"Batch masks dtype incorrect: {batch_masks.dtype}"
            assert np.all((batch_masks == 0) | (batch_masks == 1)), "Batch masks contain values other than 0 or 1"

            assert isinstance(
                batch_points, np.ndarray
            ), f"Batch points is not a numpy array (type: {type(batch_points)})"
            assert batch_points.shape == (BATCH_SIZE_TEST, 1, 2), f"Batch points shape incorrect: {batch_points.shape}"
            assert (
                np.all(batch_points >= 0)
                and np.all(batch_points[:, :, 0] < TARGET_IMAGE_SIZE)
                and np.all(batch_points[:, :, 1] < TARGET_IMAGE_SIZE)
            ), "Batch points coordinates out of bounds"

            assert isinstance(
                batch_labels, np.ndarray
            ), f"Batch labels is not a numpy array (type: {type(batch_labels)})"
            assert batch_labels.shape == (BATCH_SIZE_TEST, 1), f"Batch labels shape incorrect: {batch_labels.shape}"
            assert np.all(batch_labels == 1), "Batch labels contain values other than 1"

            print(f"  Assertions passed for batch {i+1}.")

            if first_batch_data is None:
                first_batch_data = batch_data  # Save for visualization
        else:
            print(f"  Batch {i+1} failed to load (returned None). Check warnings above.")

    print(f"\nSuccessfully loaded {successful_batches}/{NUM_TEST_BATCHES} batches.")
    assert successful_batches > 0, "Failed to load any batches using read_batch. Check logic."

    # 4. Optional: Visualize the first successful batch
    if SAVE_SAMPLE_IMAGE and first_batch_data is not None:
        print("\nVisualizing first successful batch...")
        batch_imgs, batch_masks, batch_points, _ = first_batch_data

        # Visualize the *first image* in the batch
        img_to_show = batch_imgs[0]
        mask_to_show = batch_masks[0:1]  # Slice to keep dim (1, H, W)
        point_to_show = batch_points[0:1]  # Slice to keep dim (1, 1, 2)

        os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(SAMPLE_OUTPUT_DIR, "sample_batch_visualization.png")

        try:
            import matplotlib.pyplot as plt  # Local import for visualization

            plt.figure(figsize=(10, 10))
            plt.imshow(img_to_show)
            # Overlay mask (only one mask per image in batch now)
            mask_image = np.ma.masked_where(mask_to_show[0] == 0, mask_to_show[0])
            plt.imshow(mask_image, cmap=plt.cm.viridis, alpha=0.5, vmin=0, vmax=1)
            # Plot the corresponding point
            point_coords = point_to_show[0, 0]  # Shape is (1, 1, 2) -> grab inner (1, 2)
            plt.scatter(
                point_coords[0], point_coords[1], color="red", marker="*", s=100, edgecolor="white", linewidth=1.5
            )

            plt.title(f"Sample Batch Visualization (Image 0 from Batch)")
            plt.axis("off")
            plt.savefig(output_path)
            print(f"Saved visualization to {output_path}")
            plt.close()
        except ImportError:
            print("matplotlib not found. Skipping visualization.")
            print("Install it via: pip install matplotlib")
        except Exception as e:
            print(f"Error during visualization: {e}")

    print("--- Data Loader Test Finished --- ")


if __name__ == "__main__":
    test_data_loader()
