# Train/Fine-Tune SAM 2 on the LabPics 1 dataset

# This script use a single image batch, if you want to train with multi image per batch check this script:
# https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/main/TRAIN_multi_image_batch.py

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints

import numpy as np
import torch
import cv2
import os
import random  # Added for seeding
import time  # Import time module

# import json # Removed, now handled in data_loader
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import the new data loader functions
from data_loader import load_coco_data, _read_and_process_single_sample

# --- Configuration ---
# Dataset paths
# Note: The script expects a specific structure within DATA_DIR: Simple/Train/Image/ and Simple/Train/Instance/
IMAGE_DIR = r"../images/"  # Path to the directory containing all images
ANNOTATION_FILE = r"../output/train.json"  # Path to COCO annotations file

# Model paths and definition
# Choose one model config and corresponding checkpoint:
MODEL_CFG = "sam2_hiera_t.yaml"
SAM2_CHECKPOINT = "sam2_hiera_tiny.pt"

# MODEL_CFG = "sam2_hiera_s.yaml"
# SAM2_CHECKPOINT = "sam2_hiera_small.pt"

# MODEL_CFG = "sam2_hiera_b+.yaml"
# SAM2_CHECKPOINT = "sam2_hiera_base_plus.pt"

# MODEL_CFG = "sam2_hiera_l.yaml"
# SAM2_CHECKPOINT = "sam2_hiera_large.pt"

SAM2_CHECKPOINT_PATH = f"./checkpoints/{SAM2_CHECKPOINT}"

OUTPUT_DIR = "./finetuned_sam2_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Using a more descriptive name based on the checkpoint
FINETUNED_MODEL_NAME = f"{os.path.splitext(SAM2_CHECKPOINT)[0]}_finetuned_decoder.pth"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, FINETUNED_MODEL_NAME)

# Training Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LEARNING_RATE = 1e-5  # Fine-tuning often uses smaller LRs
WEIGHT_DECAY = 4e-5  # Using original script's value
NUM_ITERATIONS = 1000
RANDOM_SEED = 42
SAVE_INTERVAL = 500  # How often to save the model
TARGET_IMAGE_SIZE = 1024  # Target size to resize images/masks to
DISPLAY_INTERVAL = 50  # How often to print status updates

# Augmentation Configuration
AUGMENTATIONS = {
    "enabled": True,  # Master switch for augmentations
    "prob": 0.75,  # Probability of applying any augmentation to a sample
    "scale": {"min": 0.8, "max": 1.2},  # Scale range (relative)
    "translate": {"max_x": 0.1, "max_y": 0.1},  # Max translation (fraction of width/height)
    "rotate": {"max_deg": 5},  # Max rotation degrees
    "noise": {"enabled": True, "std_dev": 10},  # Gaussian noise standard deviation
}

# Decide which parts of the model to train. Only training decoder/prompts is faster.
TRAIN_MASK_DECODER = True
TRAIN_PROMPT_ENCODER = True
TRAIN_IMAGE_ENCODER = False  # Requires significant GPU memory and removing no_grad calls in SAM2 code

# --- End Configuration ---

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Load data using the imported function
print(f"Loading annotations from: {ANNOTATION_FILE}")
data = load_coco_data(ANNOTATION_FILE, IMAGE_DIR)

# Load model

print(f"Loading model: {MODEL_CFG} with checkpoint: {SAM2_CHECKPOINT_PATH}")
try:
    sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT_PATH, device=DEVICE)  # load model using config
except Exception as e:
    print(f"Error loading model or checkpoint: {e}")
    print(
        f"Ensure MODEL_CFG='{MODEL_CFG}' and SAM2_CHECKPOINT='{SAM2_CHECKPOINT_PATH}' are correct and the checkpoint file exists."
    )
    exit()

predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

# --- Enable Training Components ---
# Decide which parts of the model to train. Only training decoder/prompts is faster.
# TRAIN_MASK_DECODER = True # Moved to config section
# TRAIN_PROMPT_ENCODER = True # Moved to config section
# TRAIN_IMAGE_ENCODER = False # Requires significant GPU memory and removing no_grad calls in SAM2 code # Moved to config section

print(
    f"Training - Mask Decoder: {TRAIN_MASK_DECODER}, Prompt Encoder: {TRAIN_PROMPT_ENCODER}, Image Encoder: {TRAIN_IMAGE_ENCODER}"
)

if TRAIN_MASK_DECODER:
    predictor.model.sam_mask_decoder.train(True)
else:
    predictor.model.sam_mask_decoder.train(False)

if TRAIN_PROMPT_ENCODER:
    predictor.model.sam_prompt_encoder.train(True)
else:
    predictor.model.sam_prompt_encoder.train(False)

if TRAIN_IMAGE_ENCODER:
    predictor.model.image_encoder.train(True)
    print(
        "Warning: Training the image encoder requires significant GPU memory and might require code changes in SAM2 to remove 'no_grad' blocks."
    )
else:
    predictor.model.image_encoder.train(False)
# --- End Enable Training Components ---


optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, predictor.model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)  # Use config LR and WD, filter un-trainable params
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))  # mixed precision, enable only for CUDA


# Training loop
print(f"Starting training for {NUM_ITERATIONS} iterations...")
mean_iou = 0  # Initialize mean_iou
start_time = time.time()  # Record start time
last_display_time = start_time
last_display_itr = 0

for itr in range(NUM_ITERATIONS):  # Use config NUM_ITERATIONS
    # Call the imported _read_and_process_single_sample function, passing necessary configs
    # This function now returns potentially multiple masks/points for a single image
    image, all_masks, all_points, all_labels = _read_and_process_single_sample(data, TARGET_IMAGE_SIZE, AUGMENTATIONS)

    if image is None:  # _read_and_process_single_sample now returns None on failure/skip
        print(f"Skipping iteration {itr} due to data loading issue.")
        continue

    # --- Select ONE mask/point for training (original TRAIN.py logic) ---
    # The _read_and_process_single_sample now returns all masks/points found
    # We need to randomly select one pair for this training script.
    if all_masks is None or all_masks.shape[0] == 0:
        print(f"Warning: Skipping iteration {itr} due to empty or None masks after processing.")
        continue  # ignore if no valid masks were found

    num_available_masks = all_masks.shape[0]
    selected_idx = random.randrange(num_available_masks)

    mask = all_masks[selected_idx : selected_idx + 1]  # Keep dimension (1, H, W)
    input_point = all_points[selected_idx : selected_idx + 1]  # Keep dimension (1, 1, 2)
    input_label = all_labels[selected_idx : selected_idx + 1]  # Keep dimension (1, 1)
    # --- End Mask/Point Selection ---

    if mask.shape[0] == 0:
        print(f"Warning: Skipping iteration {itr} due to empty mask after selection.")
        continue  # ignore empty batches (shouldn't happen with check above)

    try:  # Add try-except for CUDA errors etc.
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):  # cast to mix precision based on device
            predictor.set_image(image)  # apply SAM image encoder to the image

            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=None,
                masks=None,
            )

            # mask decoder

            batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )  # Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            # Ensure mask is on the correct device and is float32
            gt_mask = torch.tensor(mask.astype(np.float32)).to(DEVICE)
            prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
            # Add stability term epsilon
            epsilon = 1e-5
            seg_loss = (
                -gt_mask * torch.log(prd_mask + epsilon) - (1 - gt_mask) * torch.log(1 - prd_mask + epsilon)
            ).mean()  # binary cross entropy loss with epsilon

            # Score loss calculation (intersection over union) IOU

            iou_pred_thresh = 0.5
            prd_mask_binary = prd_mask > iou_pred_thresh
            inter = (gt_mask * prd_mask_binary).sum(dim=(1, 2))  # Sum over H, W
            union = gt_mask.sum(dim=(1, 2)) + prd_mask_binary.sum(dim=(1, 2)) - inter
            iou = torch.mean(inter / (union + epsilon))  # Calculate mean IOU over batch dimension (even if 1)

            # Ensure prd_scores shape matches expected usage if needed, default is [num_masks, num_classes (usually 1)]
            # Assuming prd_scores[:, 0] is the relevant score for the single class case
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()  # Use calculated mean IOU

            loss = seg_loss + score_loss * 0.05  # mix losses

        # apply back propogation

        predictor.model.zero_grad()  # empty gradient
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update()  # Mix precision

        current_iou = iou.cpu().detach().item()  # Get scalar IOU value
        mean_iou = mean_iou * 0.99 + 0.01 * current_iou

        if itr % SAVE_INTERVAL == 0 and itr > 0:  # Save using config interval, skip first iteration
            # Save only the trainable parts if desired (usually smaller files)
            # state_dict_to_save = {k: v for k, v in predictor.model.state_dict().items() if v.requires_grad}
            state_dict_to_save = predictor.model.state_dict()  # Saving full state dict for now
            torch.save(state_dict_to_save, OUTPUT_MODEL_PATH)
            print(f"--- Iteration {itr}: Model saved to {OUTPUT_MODEL_PATH} ---")

        # Display results periodically
        if itr % DISPLAY_INTERVAL == 0:
            current_time = time.time()
            elapsed_time_total = current_time - start_time
            elapsed_time_interval = current_time - last_display_time
            iterations_in_interval = itr - last_display_itr

            # Calculate ETR based on the last interval
            if iterations_in_interval > 0:
                time_per_iter = elapsed_time_interval / iterations_in_interval
                remaining_iters = NUM_ITERATIONS - itr
                etr_seconds = remaining_iters * time_per_iter

                # Format ETR into H:M:S
                etr_h = int(etr_seconds // 3600)
                etr_m = int((etr_seconds % 3600) // 60)
                etr_s = int(etr_seconds % 60)
                etr_str = f"{etr_h:02d}:{etr_m:02d}:{etr_s:02d}"
            else:
                etr_str = "N/A"  # Not enough info yet

            # Format total elapsed time
            elapsed_h = int(elapsed_time_total // 3600)
            elapsed_m = int((elapsed_time_total % 3600) // 60)
            elapsed_s = int(elapsed_time_total % 60)
            elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"

            print(
                f"Iter {itr}/{NUM_ITERATIONS} | Loss: {loss.item():.4f} (Seg: {seg_loss.item():.4f}, Score: {score_loss.item():.4f}) | "
                f"Inst IOU: {current_iou:.4f} | Mean IOU: {mean_iou:.4f} | Elapsed: {elapsed_str} | ETR: {etr_str}"
            )
            last_display_time = current_time
            last_display_itr = itr

    except Exception as e:
        print(f"Error during training iteration {itr}: {e}")
        # Optional: break or continue depending on error type
        if isinstance(e, torch.cuda.OutOfMemoryError):
            print("CUDA Out of Memory. Try reducing image size or model complexity if training encoder.")
            # Consider stopping or skipping
        # break # Stop training on error

print(f"--- Training Finished ---")
# Final save
# state_dict_to_save = {k: v for k, v in predictor.model.state_dict().items() if v.requires_grad}
state_dict_to_save = predictor.model.state_dict()  # Saving full state dict
torch.save(state_dict_to_save, OUTPUT_MODEL_PATH)
print(f"Final model saved to {OUTPUT_MODEL_PATH}")
