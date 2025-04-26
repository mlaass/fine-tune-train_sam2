import os
import json
import random
import cv2
import numpy as np


def load_coco_data(annotation_file, image_dir):
    """Loads COCO annotations and links them to image paths."""
    if not os.path.isfile(annotation_file):
        print(f"Error: Annotation file not found at {annotation_file}")
        exit()
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        exit()

    with open(annotation_file, "r") as f:
        coco_data = json.load(f)

    image_id_to_info = {img["id"]: img for img in coco_data["images"]}
    image_id_to_annotations = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    processed_data = []
    missing_images = 0
    for img_id, img_info in image_id_to_info.items():
        image_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(image_path):
            # print(f"Warning: Image file not found for {img_info['file_name']}, skipping.")
            missing_images += 1
            continue

        if img_id in image_id_to_annotations:
            # Ensure annotations list is not empty for this image
            if image_id_to_annotations[img_id]:
                processed_data.append(
                    {
                        "image_id": img_id,
                        "image_path": image_path,
                        "height": img_info["height"],
                        "width": img_info["width"],
                        "annotations": image_id_to_annotations[img_id],
                    }
                )
            # else: # Optionally handle images with annotations list but empty
            # print(f"Warning: Empty annotations list for image {img_info['file_name']}.")
        # else: # Optionally handle images with no annotations key
        # print(f"Warning: No annotations found for image {img_info['file_name']}.")

    if missing_images > 0:
        print(f"Warning: Skipped {missing_images} entries due to missing image files.")

    if not processed_data:
        print(f"Error: No valid image-annotation pairs found. Check paths and annotation file format.")
        exit()

    print(f"Loaded {len(processed_data)} images with annotations.")
    return processed_data


def apply_augmentations(image, masks, config):
    """Applies configured augmentations (scale, rotate, translate, noise) to image and masks."""
    if not config["enabled"] or random.random() > config["prob"]:
        return image, masks

    h, w = image.shape[:2]
    # Ensure masks is a numpy array before checking shape
    if not isinstance(masks, np.ndarray) or masks.ndim < 2:
        print(f"Warning: Invalid masks shape for augmentation: {type(masks)}. Skipping augmentation.")
        return image, masks

    # Handle case where masks might be empty after loading
    if masks.shape[0] == 0:
        print("Warning: Empty masks array passed to apply_augmentations. Skipping augmentation.")
        return image, masks

    n = masks.shape[0]  # Now safe to access shape[0]
    cx, cy = w // 2, h // 2

    # Generate random parameters
    scale = random.uniform(config["scale"]["min"], config["scale"]["max"])
    angle = random.uniform(-config["rotate"]["max_deg"], config["rotate"]["max_deg"])
    trans_x = random.uniform(-config["translate"]["max_x"], config["translate"]["max_x"]) * w
    trans_y = random.uniform(-config["translate"]["max_y"], config["translate"]["max_y"]) * h

    # Combined transformation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += trans_x
    M[1, 2] += trans_y

    # Apply transformation to image
    # Using BORDER_CONSTANT with black padding
    aug_img = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    # Apply the same transformation to all masks
    aug_masks = np.zeros_like(masks)
    for i in range(n):
        aug_masks[i] = cv2.warpAffine(
            masks[i],
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,  # Crucial for masks
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    final_img = aug_img
    # Add noise if enabled
    if config.get("noise", {}).get("enabled", False):  # Check if noise config exists and is enabled
        std_dev = config["noise"].get("std_dev", 0)
        if std_dev > 0:
            noise = np.random.normal(0, std_dev, aug_img.shape)
            noisy_img = aug_img + noise
            final_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return final_img, aug_masks


def _read_and_process_single_sample(data, target_image_size, augmentations_config):
    """Prepares a single sample (image, masks, points) from loaded COCO data.
    Internal function used by the new read_batch.
    """
    if not data:
        print("Error: Data list is empty in _read_and_process_single_sample.")
        return None, None, None, None

    # -- Select Image and Annotations --
    entry = random.choice(data)  # Choose a random entry
    image_path = entry["image_path"]
    annotations = entry["annotations"]
    # Original height/width might be needed for mask decoding if not present in image file itself
    # Let's keep them just in case, but prioritize loaded image dimensions
    original_height = entry["height"]
    original_width = entry["width"]

    # -- Load Image --
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}, skipping batch.")
        return None, None, None, None
    img = img[..., ::-1]  # BGR to RGB

    # Use actual loaded image dimensions
    height, width = img.shape[:2]

    # -- Load Masks from COCO Annotations --
    masks = []
    valid_annotations = 0
    for ann in annotations:
        segmentation = ann["segmentation"]
        # Check if segmentation is polygon format (list of lists)
        if isinstance(segmentation, list) and len(segmentation) > 0:
            # Ensure polygons are not empty
            if not any(segmentation):
                # print(f"Warning: Empty polygon list for ann {ann['id']} in {image_path}.")
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            try:
                polygons_drawn = 0
                for seg_poly in segmentation:
                    # Check if polygon list has enough points (at least 3 pairs for a triangle)
                    if len(seg_poly) >= 6:
                        poly = np.array(seg_poly, dtype=np.int32).reshape(-1, 2)
                        # Check if reshape was successful and gives pairs
                        if poly.ndim == 2 and poly.shape[1] == 2:
                            cv2.fillPoly(mask, [poly], 1)
                            polygons_drawn += 1
                        # else:
                        #     print(f"Warning: Invalid polygon coordinate structure for ann {ann['id']} in {image_path}. Polygon: {seg_poly}")
                    # else:
                    #    print(f"Warning: Insufficient points in polygon for ann {ann['id']} in {image_path}. Points: {len(seg_poly)}")

                # Only add mask if at least one valid polygon was drawn
                if polygons_drawn > 0:
                    masks.append(mask)
                    valid_annotations += 1
                # else:
                #    print(f"Warning: No valid polygons drawn for ann {ann['id']} in {image_path}.")

            except Exception as e:
                print(f"Warning: Failed to decode polygon segmentation for ann {ann['id']} in {image_path}: {e}")
            # TODO: Add handling for RLE format if necessary
            # elif isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation:
            # mask = decode_rle(segmentation) # Requires pycocotools or custom implementation
            # masks.append(mask)
            # valid_annotations += 1
            # else: # Allow skipping non-list segmentations silently for now
            # print(f"Warning: Unsupported segmentation format for ann {ann['id']} in {image_path}. Type: {type(segmentation)}")
            pass

    if not masks:
        # print(f"Warning: No valid masks could be decoded for {image_path}, skipping batch.")
        # Instead of skipping the whole batch, maybe continue if we want images without masks?
        # For SAM fine-tuning, we need masks, so skipping is appropriate here.
        return None, None, None, None

    masks = np.array(masks)  # Convert list of masks to a NumPy array (N, H, W)

    # -- Apply Augmentations --
    augmented_img, augmented_masks = apply_augmentations(img, masks, augmentations_config)

    # -- Resize Image and Masks --
    # Calculate scaling factor to fit the augmented image into target_image_size
    # while preserving aspect ratio.
    current_h, current_w = augmented_img.shape[:2]
    scale_h = target_image_size / current_h
    scale_w = target_image_size / current_w
    scale_factor = min(scale_h, scale_w)

    new_h, new_w = int(current_h * scale_factor), int(current_w * scale_factor)

    # Resize image
    final_img = cv2.resize(augmented_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Resize masks
    # Input masks are (N, H, W), resize each one
    final_masks = []
    if augmented_masks.shape[0] > 0:
        # Ensure augmented_masks is a numpy array before iterating
        if isinstance(augmented_masks, np.ndarray) and augmented_masks.ndim == 3:
            for i in range(augmented_masks.shape[0]):
                resized_mask = cv2.resize(augmented_masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                final_masks.append(resized_mask)
            final_masks = np.array(final_masks)
        else:
            print(
                f"Warning: augmented_masks has unexpected shape or type ({type(augmented_masks)}, {getattr(augmented_masks, 'shape', 'N/A')}) after augmentation. Skipping mask resize."
            )
            final_masks = np.empty((0, new_h, new_w), dtype=np.uint8)  # Ensure it's an empty array

    else:  # Handle case where there might be no masks initially (shouldn't happen with current logic)
        final_masks = np.empty((0, new_h, new_w), dtype=np.uint8)

    # Optional: Pad image and masks to target_image_size x target_image_size if needed
    # (Current SAM2 predictor might handle non-square inputs, check its requirements)
    # If padding is needed:
    pad_h = target_image_size - new_h
    pad_w = target_image_size - new_w
    # Pad image (right/bottom padding)
    final_img = cv2.copyMakeBorder(final_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # Pad masks
    if final_masks.shape[0] > 0:
        # Ensure final_masks is a numpy array before padding
        if isinstance(final_masks, np.ndarray) and final_masks.ndim == 3:
            final_masks = np.pad(final_masks, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        else:
            print(
                f"Warning: final_masks has unexpected shape or type ({type(final_masks)}, {getattr(final_masks, 'shape', 'N/A')}) before padding. Skipping padding."
            )
            # Reset to empty array with target shape if padding failed
            final_masks = np.empty((0, target_image_size, target_image_size), dtype=np.uint8)

    # -- Generate Points from Final Masks --
    if final_masks.shape[0] == 0:
        # print(f"Warning: No masks available after processing for {image_path}, skipping.")
        return None, None, None, None

    points = []
    valid_mask_indices = []  # Keep track of masks from which points could be generated
    for i, mask in enumerate(final_masks):
        coords = np.argwhere(mask > 0)  # get all coordinates in mask
        if len(coords) > 0:
            yx = coords[np.random.randint(len(coords))]  # choose random point/coordinate
            # SAM expects points in (x, y) format
            points.append([[yx[1], yx[0]]])
            valid_mask_indices.append(i)
        # else: # Handle case where a mask might become empty after transforms
        # print(f"Warning: Mask {i} for {image_path} is empty after processing.")

    if not points:
        # print(f"Warning: Could not generate points for any mask in {image_path}, skipping.")
        return None, None, None, None

    # Filter masks to only include those for which points were generated
    # Ensure indices are valid before slicing
    if valid_mask_indices and max(valid_mask_indices) < final_masks.shape[0]:
        final_masks = final_masks[valid_mask_indices]
    elif valid_mask_indices:  # Check if list is not empty but indices might be wrong
        print(
            f"Warning: Inconsistent state - valid_mask_indices {valid_mask_indices} out of bounds for final_masks shape {final_masks.shape}. Returning no masks."
        )
        final_masks = np.empty((0, target_image_size, target_image_size), dtype=np.uint8)
        points = []  # Also clear points as they correspond to invalid masks
        return None, None, None, None  # Skip batch
    else:  # valid_mask_indices is empty, already handled by 'if not points' check
        pass

    # Return image, masks, points, and labels (always 1 for SAM points)
    # Ensure shapes are correct before returning
    if not isinstance(final_img, np.ndarray) or final_img.shape != (target_image_size, target_image_size, 3):
        print(
            f"Error: Final image has incorrect shape {final_img.shape}. Expected ({target_image_size}, {target_image_size}, 3). Skipping batch."
        )
        return None, None, None, None
    if (
        not isinstance(final_masks, np.ndarray)
        or final_masks.ndim != 3
        or final_masks.shape[1:] != (target_image_size, target_image_size)
    ):
        print(
            f"Error: Final masks have incorrect shape {final_masks.shape}. Expected (N, {target_image_size}, {target_image_size}). Skipping batch."
        )
        return None, None, None, None
    if (
        not isinstance(points, np.ndarray)
        or points.ndim != 3
        or points.shape[0] != final_masks.shape[0]
        or points.shape[1:] != (1, 2)
    ):
        # Check if points is a list and convert if necessary and possible
        if isinstance(points, list):
            try:
                points = np.array(points)
                if points.ndim != 3 or points.shape[0] != final_masks.shape[0] or points.shape[1:] != (1, 2):
                    raise ValueError("Converted points array shape mismatch")
            except Exception as e:
                print(
                    f"Error: Final points have incorrect shape or type ({type(points)}, {getattr(points, 'shape', 'N/A')}). Expected ({final_masks.shape[0]}, 1, 2). Error: {e}. Skipping batch."
                )
                return None, None, None, None
        else:  # It's not a list and not the correct numpy array
            print(
                f"Error: Final points have incorrect shape or type ({type(points)}, {getattr(points, 'shape', 'N/A')}). Expected ({final_masks.shape[0]}, 1, 2). Skipping batch."
            )
            return None, None, None, None

    # Generate labels: should be (num_points, 1) -> (N, 1)
    labels = np.ones([len(points), 1], dtype=np.int32)  # Use int32 as labels are usually ints

    return final_img, final_masks, points, labels


def read_batch(data, batch_size, target_image_size, augmentations_config):
    """Reads and processes a batch of samples."""
    batch_images = []
    batch_masks = []
    batch_points = []
    batch_labels = []

    attempts = 0
    max_attempts = batch_size * 2  # Allow some failures

    while len(batch_images) < batch_size and attempts < max_attempts:
        attempts += 1
        img, masks, points, labels = _read_and_process_single_sample(data, target_image_size, augmentations_config)

        if img is not None and masks is not None and points is not None and labels is not None:
            # Ensure the sample read is valid before adding
            if masks.shape[0] > 0 and points.shape[0] > 0 and masks.shape[0] == points.shape[0]:
                # We need to select ONE mask/point pair for the batch item,
                # consistent with the original multi-batch script logic.
                selected_idx = random.randrange(masks.shape[0])
                batch_images.append(img)
                batch_masks.append(masks[selected_idx])  # Select one mask
                batch_points.append(points[selected_idx])  # Select corresponding point
                batch_labels.append(labels[selected_idx])  # Select corresponding label
            # else: # Optionally log if a sample was valid but had inconsistent mask/point counts
            # print(f"Warning: Skipping sample due to inconsistent masks/points ({masks.shape[0]} vs {points.shape[0]})")

        # else: # Optionally log skipped samples
        # print(f"Warning: _read_and_process_single_sample returned None. Attempt {attempts}/{max_attempts}")

    if len(batch_images) < batch_size:
        print(
            f"Warning: Could only collect {len(batch_images)} samples after {max_attempts} attempts. Returning None for batch."
        )
        return None, None, None, None

    # Stack masks, points, and labels into batch tensors
    try:
        final_masks = np.stack(batch_masks, axis=0)  # (B, H, W)
        final_points = np.stack(batch_points, axis=0)  # Should be (B, 1, 2)
        final_labels = np.stack(batch_labels, axis=0)  # (B, 1)

        # Validate final shapes before returning
        if final_masks.shape != (batch_size, target_image_size, target_image_size):
            raise ValueError(f"Final batch masks shape mismatch: {final_masks.shape}")
        if final_points.shape != (batch_size, 1, 2):
            raise ValueError(f"Final batch points shape mismatch: {final_points.shape}")
        if final_labels.shape != (batch_size, 1):
            raise ValueError(f"Final batch labels shape mismatch: {final_labels.shape}")

    except ValueError as e:
        print(f"Error stacking batch data: {e}")
        return None, None, None, None

    # Images remain a list as expected by predictor.set_image_batch
    return batch_images, final_masks, final_points, final_labels
