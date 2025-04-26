#!/usr/bin/env python
# coding: utf-8

# Script to evaluate the fine-tuned SAM 2 model on the validation set.

import numpy as np
import torch
import cv2
import os
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from data_loader import load_coco_data  # Reuse data loading function

# We'll need a function similar to read_batch but adapted for validation
# Let's define it within this script or refine data_loader later if needed.

# --- Configuration ---
# Dataset paths
VALIDATION_ANNOTATION_FILE = r"../output/val.json"  # COCO format validation annotations
IMAGE_DIR = r"../images/"  # Path to the directory containing all images (train+val)
CATEGORY_MAP_FILE = r"../output/category_mapping.json"  # To map category IDs to names

# Model paths and definition (Should match the model trained in TRAIN.py)
# Choose the *same* model config used for training:
MODEL_CFG = "sam2_hiera_t.yaml"
# MODEL_CFG = "sam2_hiera_s.yaml"
# MODEL_CFG = "sam2_hiera_b+.yaml"
# MODEL_CFG = "sam2_hiera_l.yaml"

# This MUST point to the actual checkpoint file used for training
# (needed by build_sam2 even if we load fine-tuned weights later)
BASE_SAM2_CHECKPOINT_NAME = "sam2_hiera_tiny.pt"
# BASE_SAM2_CHECKPOINT_NAME = "sam2_hiera_small.pt"
# BASE_SAM2_CHECKPOINT_NAME = "sam2_hiera_base_plus.pt"
# BASE_SAM2_CHECKPOINT_NAME = "sam2_hiera_large.pt"
BASE_SAM2_CHECKPOINT_PATH = f"./checkpoints/{BASE_SAM2_CHECKPOINT_NAME}"

# Path to the *fine-tuned* model weights saved by TRAIN.py
FINETUNED_MODEL_DIR = "./finetuned_sam2_models"  # Directory where trained models are saved
FINETUNED_MODEL_NAME = (
    f"{os.path.splitext(BASE_SAM2_CHECKPOINT_NAME)[0]}_finetuned_decoder.pth"  # Construct name based on base checkpoint
)
FINETUNED_MODEL_PATH = os.path.join(FINETUNED_MODEL_DIR, FINETUNED_MODEL_NAME)

# Evaluation Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_IMAGE_SIZE = 1024  # Should match the training image size
IOU_THRESHOLD = 0.5  # Threshold for predicted mask binarization when calculating TP/FP/FN/TN and IoU
USE_FINETUNED_WEIGHTS = True  # Set to False to run evaluation with base model only

# --- End Configuration ---

print(f"--- Validation Configuration ---")
print(f"Validation Set: {VALIDATION_ANNOTATION_FILE}")
print(f"Image Directory: {IMAGE_DIR}")
print(f"Category Map: {CATEGORY_MAP_FILE}")
print(f"Model Config: {MODEL_CFG}")
print(f"Base Checkpoint: {BASE_SAM2_CHECKPOINT_PATH}")
if USE_FINETUNED_WEIGHTS:
    print(f"Fine-tuned Weights: {FINETUNED_MODEL_PATH}")
else:
    print(f"Fine-tuned Weights: SKIPPED (using base model only)")
print(f"Device: {DEVICE}")
print(f"Target Image Size: {TARGET_IMAGE_SIZE}")
print(f"IoU Threshold: {IOU_THRESHOLD}")
print(f"-----------------------------")

# --- Helper Functions ---


def preprocess_validation_data(image_path, annotations, target_size):
    """Loads image, annotations, preprocesses them for the model."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}, skipping.")
        return None, None, None

    img_rgb = img[..., ::-1]  # BGR to RGB
    original_h, original_w = img_rgb.shape[:2]

    # Decode Ground Truth Masks
    gt_masks = []
    valid_annotations = []
    for ann in annotations:
        segmentation = ann["segmentation"]
        category_id = ann["category_id"]
        if isinstance(segmentation, list) and len(segmentation) > 0:
            mask = np.zeros((original_h, original_w), dtype=np.uint8)
            polygons_drawn = 0
            try:
                for seg_poly in segmentation:
                    if len(seg_poly) >= 6:
                        poly = np.array(seg_poly, dtype=np.int32).reshape(-1, 2)
                        if poly.ndim == 2 and poly.shape[1] == 2:
                            cv2.fillPoly(mask, [poly], 1)
                            polygons_drawn += 1
                if polygons_drawn > 0:
                    gt_masks.append(mask)
                    valid_annotations.append(ann)  # Keep track of corresponding annotation
            except Exception as e:
                print(f"Warning: Failed to decode polygon for ann {ann['id']} in {image_path}: {e}")
                continue
        # TODO: Add RLE support if needed

    if not gt_masks:
        # print(f"Warning: No valid ground truth masks found for {image_path}.")
        return None, None, None  # Skip if no masks

    gt_masks = np.array(gt_masks)  # (N, H, W)

    # Resize image and masks consistently with training
    scale_h = target_size / original_h
    scale_w = target_size / original_w
    scale_factor = min(scale_h, scale_w)
    new_h, new_w = int(original_h * scale_factor), int(original_w * scale_factor)

    # Resize image
    resized_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Resize masks
    resized_gt_masks = []
    for mask in gt_masks:
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        resized_gt_masks.append(resized_mask)
    resized_gt_masks = np.array(resized_gt_masks)  # (N, new_H, new_W)

    # Pad image and masks to target_size x target_size
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    final_img = cv2.copyMakeBorder(resized_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if resized_gt_masks.shape[0] > 0:
        final_gt_masks = np.pad(resized_gt_masks, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    else:
        final_gt_masks = np.empty((0, target_size, target_size), dtype=np.uint8)

    # Ensure final shapes are correct
    if final_img.shape != (target_size, target_size, 3):
        print(f"Error: Final image has incorrect shape {final_img.shape} for {image_path}. Skipping.")
        return None, None, None
    if final_gt_masks.ndim != 3 or final_gt_masks.shape[1:] != (target_size, target_size):
        print(f"Error: Final masks have incorrect shape {final_gt_masks.shape} for {image_path}. Skipping.")
        return None, None, None

    return final_img, final_gt_masks, valid_annotations


def generate_point_prompt(mask):
    """Generates a single point prompt from the center of the mask."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None  # Cannot generate point if mask is empty
    # Use centroid for stability
    center_y, center_x = coords.mean(axis=0)
    # Ensure point is within bounds
    point = np.array([[int(center_x), int(center_y)]])
    # Add batch dim and point dim: shape (1, 1, 2)
    return point.reshape(1, 1, 2)


def calculate_metrics(pred_mask_binary, gt_mask_binary):
    """Calculates IoU and pixel-level TP, FP, FN, TN for a single prediction-GT pair."""
    if gt_mask_binary.shape != pred_mask_binary.shape:
        raise ValueError(f"Shape mismatch: GT {gt_mask_binary.shape}, Pred {pred_mask_binary.shape}")

    # Pixel counts
    pix_tp = np.sum((pred_mask_binary == 1) & (gt_mask_binary == 1))
    pix_fp = np.sum((pred_mask_binary == 1) & (gt_mask_binary == 0))
    pix_fn = np.sum((pred_mask_binary == 0) & (gt_mask_binary == 1))
    pix_tn = np.sum((pred_mask_binary == 0) & (gt_mask_binary == 0))

    # IoU Calculation
    intersection = pix_tp
    union = pix_tp + pix_fp + pix_fn
    iou = intersection / (union + 1e-6)  # Add epsilon for stability

    return iou, pix_tp, pix_fp, pix_fn, pix_tn


# --- Main Evaluation Logic ---


def main():
    print("Starting validation...")

    # Load Category Mapping
    try:
        with open(CATEGORY_MAP_FILE, "r") as f:
            # Convert keys back to integers if needed, though COCO uses int IDs
            category_id_to_name = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded {len(category_id_to_name)} categories from {CATEGORY_MAP_FILE}")
    except Exception as e:
        print(f"Error loading category map '{CATEGORY_MAP_FILE}': {e}")
        print("Cannot proceed without category map.")
        return

    # Load validation data
    print(f"Loading validation annotations from: {VALIDATION_ANNOTATION_FILE}")
    validation_data = load_coco_data(VALIDATION_ANNOTATION_FILE, IMAGE_DIR)
    if not validation_data:
        print("Failed to load validation data. Exiting.")
        return
    print(f"Loaded {len(validation_data)} validation image entries.")

    # Load model
    print(f"Loading base model: {MODEL_CFG} with checkpoint: {BASE_SAM2_CHECKPOINT_PATH}")
    if not os.path.exists(BASE_SAM2_CHECKPOINT_PATH):
        print(f"Error: Base checkpoint not found at {BASE_SAM2_CHECKPOINT_PATH}")
        return
    try:
        sam2_model = build_sam2(MODEL_CFG, BASE_SAM2_CHECKPOINT_PATH, device=DEVICE)
    except Exception as e:
        print(f"Error loading base model or checkpoint: {e}")
        return

    # --- Load fine-tuned weights (Optional) ---
    if USE_FINETUNED_WEIGHTS:
        print(f"Attempting to load fine-tuned weights from: {FINETUNED_MODEL_PATH}")
        if not os.path.exists(FINETUNED_MODEL_PATH):
            print(f"Error: Fine-tuned weights not found at {FINETUNED_MODEL_PATH}")
            print("Ensure TRAIN.py has run successfully and saved the model, or set USE_FINETUNED_WEIGHTS to False.")
            return
        try:
            # Load the state dict - ensure it matches the keys in sam2_model
            loaded_state_dict = torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE)
            # Get the current model's state dict
            current_model_state_dict = sam2_model.state_dict()
            # Create a new state dict matching the current model structure
            new_state_dict = current_model_state_dict.copy()

            # Filter and update weights
            updated_keys = 0
            skipped_keys = 0
            mismatched_keys = 0
            print("Comparing loaded weights with model structure...")
            for k, v in loaded_state_dict.items():
                if k in new_state_dict:
                    if new_state_dict[k].shape == v.shape:
                        new_state_dict[k] = v
                        updated_keys += 1
                    else:
                        print(f"  Skipping {k}: Shape mismatch (Model: {new_state_dict[k].shape}, File: {v.shape})")
                        mismatched_keys += 1
                else:
                    # print(f"  Skipping {k}: Key not found in current model.") # Less verbose
                    skipped_keys += 1

            print(
                f"Weight loading summary: Updated {updated_keys}, Skipped (not found) {skipped_keys}, Skipped (shape mismatch) {mismatched_keys}"
            )

            if updated_keys == 0:
                print("Warning: No weights were updated from the fine-tuned file. Check path and structure.")
                # Continue with base weights
            else:
                # Load the potentially filtered state dict
                sam2_model.load_state_dict(new_state_dict, strict=False)  # Use strict=False as we might only load parts
                print("Successfully loaded fine-tuned weights.")
        except Exception as e:
            print(f"Error loading fine-tuned weights: {e}")
            return
    else:
        print("Skipping fine-tuned weight loading as requested.")
    # --- End Fine-tuned weights loading ---

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.eval()  # Set model to evaluation mode

    # --- Precision Settings (Mimic example script) ---
    # Use autocast for mixed precision to potentially speed up and match expected dtypes
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using autocast with dtype: {autocast_dtype}")
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # Enable TF32 for Ampere GPUs for potentially better performance
        print("Enabling TF32 for matmul and cuDNN.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # --- End Precision Settings ---

    # --- Evaluation Loop ---
    # Store object-level, pixel-level counts, and list of IoUs per category
    category_metrics = defaultdict(
        lambda: {
            "iou_list": [],
            "count": 0,
            "obj_tp": 0,
            "obj_fn": 0,
            "pix_tp": 0,
            "pix_fp": 0,
            "pix_fn": 0,
            "pix_tn": 0,
        }
    )
    total_instances = 0
    processed_images = 0
    skipped_images = 0
    start_time = time.time()

    print("Starting evaluation loop...")
    # Apply autocast context manager around the inference loop
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
        # Wrap validation_data with tqdm for a progress bar
        for i, entry in enumerate(tqdm(validation_data, desc="Evaluating images")):
            image_path = entry["image_path"]
            annotations = entry["annotations"]

            # Preprocess image and ground truth masks
            processed_img, gt_masks, valid_annotations = preprocess_validation_data(
                image_path, annotations, TARGET_IMAGE_SIZE
            )

            if processed_img is None or gt_masks is None or valid_annotations is None or gt_masks.shape[0] == 0:
                skipped_images += 1
                continue  # Skip if preprocessing failed or no valid masks

            processed_images += 1

            # Set image for the predictor (encodes the image)
            try:
                predictor.set_image(processed_img)
            except Exception as e:
                print(f"Error setting image {image_path} in predictor: {e}. Skipping image.")
                skipped_images += 1
                continue

            # Process each ground truth instance (mask) for this image
            for gt_mask, ann in zip(gt_masks, valid_annotations):
                total_instances += 1
                category_id = ann["category_id"]

                # Generate point prompt from the GT mask
                point_prompt = generate_point_prompt(gt_mask)
                if point_prompt is None:
                    # print(f"Warning: Could not generate prompt for mask in {image_path}, ann {ann['id']}")
                    continue  # Skip this instance if no point can be generated

                # Predict mask using the point prompt
                try:
                    # When multimask_output=False, the first output is the mask tensor directly
                    pred_masks, scores_pred, _ = predictor.predict(
                        point_coords=point_prompt,
                        point_labels=np.array([[1]]),  # Foreground point
                        multimask_output=False,  # Get single best mask
                    )  # Masks shape (B, M, H, W), Scores shape (B, M) B=1, M=1 here

                    if pred_masks is None or pred_masks.shape[0] == 0:
                        print(f"Warning: Predictor returned no masks for {image_path}, ann {ann['id']}")
                        continue

                    # Process prediction: get mask, binarize
                    # Using the first (and only) predicted mask.
                    # Assuming pred_masks shape is (B, H, W) with B=1 here.
                    pred_mask_numpy = pred_masks[0]  # Get the (H, W) mask
                    pred_mask_binary = (pred_mask_numpy > IOU_THRESHOLD).astype(np.uint8)

                    # Ensure GT mask is binary for comparison
                    gt_mask_binary = (gt_mask > 0).astype(np.uint8)

                    # Calculate metrics for this instance
                    iou, pix_tp, pix_fp, pix_fn, pix_tn = calculate_metrics(pred_mask_binary, gt_mask_binary)

                    # Accumulate metrics per category
                    metrics = category_metrics[category_id]
                    metrics["count"] += 1
                    # Store individual IoU instead of summing immediately
                    metrics["iou_list"].append(iou)

                    # Accumulate pixel-level counts
                    metrics["pix_tp"] += pix_tp
                    metrics["pix_fp"] += pix_fp
                    metrics["pix_fn"] += pix_fn
                    metrics["pix_tn"] += pix_tn

                    # Increment object TP or FN based on IoU threshold
                    if iou > IOU_THRESHOLD:
                        metrics["obj_tp"] += 1
                    else:
                        metrics["obj_fn"] += 1

                except Exception as e:
                    print(f"Error during prediction or metric calculation for {image_path}, ann {ann['id']}: {e}")
                    # Decide whether to continue or break depending on error severity

    end_time = time.time()
    print(f"Evaluation finished in {end_time - start_time:.2f} seconds.")
    print(f"Processed {processed_images} images, skipped {skipped_images}.")
    print(f"Evaluated {total_instances} instances across all images.")

    # --- Calculate and Display Results ---
    print("--- Evaluation Results ---")

    # Calculate overall counts first
    overall_count = sum(m["count"] for m in category_metrics.values())
    if overall_count == 0:
        print("Warning: No instances were evaluated. Cannot calculate metrics.")
        # Optionally handle this case, e.g., by exiting or setting metrics to 0
        macro_iou = 0.0
        micro_iou = 0.0
        overall_pix_precision = 0.0
        overall_pix_recall = 0.0
    else:
        # Calculate overall pixel-level metrics
        overall_pix_tp = sum(m["pix_tp"] for m in category_metrics.values())
        overall_pix_fp = sum(m["pix_fp"] for m in category_metrics.values())
        overall_pix_fn = sum(m["pix_fn"] for m in category_metrics.values())
        # TN is usually huge and not used in standard overall metrics like micro IoU/Precision/Recall
        # overall_pix_tn = sum(m["pix_tn"] for m in category_metrics.values())

        micro_iou = overall_pix_tp / (overall_pix_tp + overall_pix_fp + overall_pix_fn + 1e-6)
        overall_pix_precision = overall_pix_tp / (overall_pix_tp + overall_pix_fp + 1e-6)
        overall_pix_recall = overall_pix_tp / (overall_pix_tp + overall_pix_fn + 1e-6)

    # --- Per-Category Metrics Calculation ---
    # This loop calculates metrics per category and stores them in category_results
    # It needs to run BEFORE we calculate macro_iou

    category_results = []
    sorted_cat_ids = sorted(category_metrics.keys())

    for cat_id in sorted_cat_ids:
        metrics = category_metrics[cat_id]
        cat_name = category_id_to_name.get(cat_id, "Unknown")
        count = metrics["count"]
        iou_list = metrics["iou_list"]

        if count == 0:
            # Handle categories with zero instances
            mean_iou, median_iou, min_iou, max_iou = 0.0, 0.0, 0.0, 0.0
            obj_recall = 0.0
            obj_tp = 0
            obj_fn = 0
            pix_precision = 0.0
            pix_recall = 0.0
            pix_tp, pix_fp, pix_fn, pix_tn = 0, 0, 0, 0  # Ensure these are defined
        else:
            iou_array = np.array(iou_list)
            mean_iou = np.mean(iou_array)
            median_iou = np.median(iou_array)
            min_iou = np.min(iou_array)
            max_iou = np.max(iou_array)

            # Object metrics
            obj_tp = metrics["obj_tp"]
            obj_fn = metrics["obj_fn"]
            obj_recall = obj_tp / (count + 1e-6)

            # Pixel metrics
            pix_tp = metrics["pix_tp"]
            pix_fp = metrics["pix_fp"]
            pix_fn = metrics["pix_fn"]
            pix_tn = metrics["pix_tn"]
            pix_precision = pix_tp / (pix_tp + pix_fp + 1e-6)
            pix_recall = pix_tp / (pix_tp + pix_fn + 1e-6)

        # Append combined metrics including IoU stats for JSON saving
        category_results.append(
            {
                "id": cat_id,
                "name": cat_name,
                "count": count,
                # Object metrics
                "obj_tp": obj_tp,
                "obj_fn": obj_fn,
                "obj_recall": obj_recall,
                # IoU Stats
                "mean_iou": mean_iou,
                "median_iou": median_iou,
                "min_iou": min_iou,
                "max_iou": max_iou,
                # Pixel metrics
                "pix_tp": int(pix_tp),
                "pix_fp": int(pix_fp),
                "pix_fn": int(pix_fn),
                "pix_tn": int(pix_tn),
                "pix_precision": pix_precision,
                "pix_recall": pix_recall,
            }
        )

    # --- Now calculate Macro IoU using the results ---
    if not category_results:
        macro_iou = 0.0  # Handle case where no categories were processed
    else:
        valid_mean_ious = [r["mean_iou"] for r in category_results if r["count"] > 0]
        if not valid_mean_ious:
            macro_iou = 0.0  # Handle case where categories exist but have 0 instances
        else:
            macro_iou = sum(valid_mean_ious) / len(valid_mean_ious)

    # --- Display Overall Metrics ---
    print("--- Overall Metrics ---")
    print(f"Total Instances Evaluated: {overall_count}")
    print(f"Macro-Average IoU (Instance Level): {macro_iou:.4f}")
    print(f"Micro-Average IoU (Pixel Level): {micro_iou:.4f}")
    print(f"Overall Pixel Precision: {overall_pix_precision:.4f}")
    print(f"Overall Pixel Recall: {overall_pix_recall:.4f}")

    # --- Display Per-Category Metrics ---
    print("--- Per-Category Pixel Metrics ---")
    print(
        f"{'ID':<5} {'Category':<30} {'Pix TP':<12} {'Pix FP':<12} {'Pix FN':<12} {'Pix TN':<12} {'Precision':<10} {'Recall':<10}"
    )
    print("-" * 110)
    # Use the populated category_results for printing
    for result in category_results:
        cat_id = result["id"]
        cat_name = result["name"]
        # Use already calculated values from the result dictionary
        print(
            f"{cat_id:<5} {cat_name[:30]:<30} {result['pix_tp']:<12} {result['pix_fp']:<12} {result['pix_fn']:<12} {result['pix_tn']:<12} {result['pix_precision']:<10.4f} {result['pix_recall']:<10.4f}"
        )
    print("-" * 110)

    print("--- Per-Category Object Metrics ---")
    print(f"{'ID':<5} {'Category':<30} {'Instances':<10} {'Recall':<10} {'Mean IoU':<10}")
    print("-" * 75)  # Adjusted separator length
    # Use the populated category_results for printing
    for result in category_results:
        cat_id = result["id"]
        cat_name = result["name"]
        # Use already calculated values from the result dictionary
        print(
            f"{cat_id:<5} {cat_name[:30]:<30} {result['count']:<10} {result['obj_recall']:<10.4f} {result['mean_iou']:<10.4f}"
        )
    print("-" * 75)  # Adjusted separator length

    # --- Save Detailed Results to JSON ---
    if category_results:
        # Determine filename suffix based on whether fine-tuned weights were used
        eval_type_suffix = "finetuned" if USE_FINETUNED_WEIGHTS else "base"
        base_output_name = os.path.splitext(FINETUNED_MODEL_NAME)[0]

        results_filename = os.path.join(
            FINETUNED_MODEL_DIR, f"{base_output_name}_validation_metrics_{eval_type_suffix}.json"
        )
        try:
            with open(results_filename, "w") as f:
                json.dump(category_results, f, indent=4)
            print(f"Saved detailed validation metrics to: {results_filename}")
        except Exception as e:
            print(f"Error saving validation metrics JSON: {e}")

    # --- Plotting Results ---
    if category_results:
        # Sort by IoU for better visualization (already sorted by ID for table output, re-sort for plot)
        category_results.sort(key=lambda x: x["mean_iou"], reverse=True)

        cat_names = [r["name"][:25] for r in category_results]  # Truncate long names
        cat_ious = [r["mean_iou"] for r in category_results]

        fig, ax = plt.subplots(figsize=(12, max(6, len(cat_names) * 0.4)))  # Adjust height based on num categories
        sns.barplot(x=cat_ious, y=cat_names, palette="viridis", ax=ax)

        # Add vertical grid lines at specified intervals
        ax.set_xticks(np.arange(0, 1.1, 0.2))  # Set tick positions from 0 to 1.0 with 0.2 step
        ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.5)
        ax.set_axisbelow(True)  # Ensure grid lines are drawn behind the bars

        ax.set_xlabel("Mean Instance IoU")
        ax.set_ylabel("Category")
        # Update plot title based on evaluation type
        plot_title = (
            f"Mean Instance IoU per Category (Validation Set) Model: {FINETUNED_MODEL_NAME} ({eval_type_suffix})"
        )
        ax.set_title(plot_title)
        ax.set_xlim(0, 1)
        plt.tight_layout()

        # Update plot filename
        plot_filename = os.path.join(FINETUNED_MODEL_DIR, f"{base_output_name}_validation_iou_{eval_type_suffix}.png")
        try:
            plt.savefig(plot_filename)
            print(f"Saved per-category IoU plot to: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        # plt.show() # Optionally display plot interactively

    else:
        print("No category results to plot.")


if __name__ == "__main__":
    main()
