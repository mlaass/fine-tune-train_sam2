# Train/Fine Tune SAM 2 with Batched Input
# This script uses multiple images per batch, leveraging the refactored data_loader.

import numpy as np
import torch
import cv2
import os
import random
import time

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import the data loader functions
from data_loader import load_coco_data, read_batch  # Use the new read_batch

# --- Configuration ---
# Dataset paths
IMAGE_DIR = r"../images/"  # Path to the directory containing all images
ANNOTATION_FILE = r"../output/train.json"  # Path to COCO annotations file

# Model paths and definition
# Choose one model config and corresponding checkpoint:
# MODEL_CFG = "sam2_hiera_t.yaml"
# SAM2_CHECKPOINT = "sam2_hiera_tiny.pt"

MODEL_CFG = "sam2_hiera_s.yaml"
SAM2_CHECKPOINT = "sam2_hiera_small.pt"

# MODEL_CFG = "sam2_hiera_b+.yaml"
# SAM2_CHECKPOINT = "sam2_hiera_base_plus.pt"

# MODEL_CFG = "sam2_hiera_l.yaml"
# SAM2_CHECKPOINT = "sam2_hiera_large.pt"

SAM2_CHECKPOINT_PATH = f"./checkpoints/{SAM2_CHECKPOINT}"

OUTPUT_DIR = "./finetuned_sam2_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINETUNED_MODEL_NAME = f"{os.path.splitext(SAM2_CHECKPOINT)[0]}_finetuned_decoder_batched.pth"  # Added _batched suffix
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, FINETUNED_MODEL_NAME)

# Training Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 8  # Configurable batch size
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 4e-5
NUM_ITERATIONS = int(1000 / BATCH_SIZE)  # Keep original iteration count(100000) or adjust as needed
RANDOM_SEED = 42
SAVE_INTERVAL = int(1000 / BATCH_SIZE)  # Keep original save interval(1000) or adjust
TARGET_IMAGE_SIZE = 1024
DISPLAY_INTERVAL = int(50 / BATCH_SIZE)  # How often to print status updates

# Augmentation Configuration (same as TRAIN.py)
AUGMENTATIONS = {
    "enabled": True,
    "prob": 0.75,
    "scale": {"min": 0.8, "max": 1.2},
    "translate": {"max_x": 0.1, "max_y": 0.1},
    "rotate": {"max_deg": 5},
    "noise": {"enabled": True, "std_dev": 10},
}

# Decide which parts of the model to train.
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
if not data:
    print("Exiting: Failed to load data.")
    exit()

# Load model
print(f"Loading model: {MODEL_CFG} with checkpoint: {SAM2_CHECKPOINT_PATH}")
try:
    sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT_PATH, device=DEVICE)
except Exception as e:
    print(f"Error loading model or checkpoint: {e}")
    print(f"Ensure MODEL_CFG='{MODEL_CFG}' and SAM2_CHECKPOINT='{SAM2_CHECKPOINT_PATH}' are correct.")
    exit()

predictor = SAM2ImagePredictor(sam2_model)

# --- Enable Training Components ---
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
    print("Warning: Training image encoder enabled. Ensure no_grad blocks are handled if necessary.")
else:
    predictor.model.image_encoder.train(False)
# --- End Enable Training Components ---

# Setup optimizer and scaler
optimizer = torch.optim.AdamW(
    params=filter(lambda p: p.requires_grad, predictor.model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# Training loop
print(f"Starting training for {NUM_ITERATIONS} iterations with batch size {BATCH_SIZE}...")
mean_iou = 0
start_time = time.time()
last_display_time = start_time
last_display_itr = 0

for itr in range(NUM_ITERATIONS):
    # Load data batch using the new read_batch function
    batch_data = read_batch(data, BATCH_SIZE, TARGET_IMAGE_SIZE, AUGMENTATIONS)

    if batch_data[0] is None:  # Check if batch loading failed
        print(f"Skipping iteration {itr} due to data loading issue (failed to create full batch).")
        continue

    images, masks, input_points, input_labels = batch_data

    # Ensure masks are not empty for the batch (should be guaranteed by read_batch now)
    if masks.shape[0] != BATCH_SIZE:
        print(f"Warning: Skipping iteration {itr} due to incomplete batch ({masks.shape[0]}/{BATCH_SIZE}).")
        continue

    try:
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            # Set the batch of images
            predictor.set_image_batch(images)  # Expects a list of numpy arrays

            # Prepare prompts for the batch
            # Input points shape: (B, 1, 2), Input labels shape: (B, 1)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
            )
            # unnorm_coords shape should be (B, 1, 2), labels shape (B, 1)

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),  # Pass tuples (coords, labels)
                boxes=None,
                masks=None,
            )
            # sparse_embeddings shape likely (B, num_prompts, embed_dim)

            # Mask decoder for the batch
            # Need to handle features correctly for batch processing
            # Assuming predictor._features caches features for the batch correctly after set_image_batch
            if (
                predictor._features is None
                or "high_res_feats" not in predictor._features
                or "image_embed" not in predictor._features
            ):
                print(f"Error: Features not found after set_image_batch in iteration {itr}. Skipping.")
                continue

            # Extract features for the batch - Check shapes carefully
            # image_embed might be (B, embed_dim, H, W) or list? Check SAM2 code/predictor.
            # Let's assume image_embed is stacked: (B, embed_dim, H, W)
            # high_res_feats might be a list of lists/tensors? Assume list (levels) of tensors (B, C, H, W)
            image_embeddings_batch = predictor._features["image_embed"]  # Assuming shape (B, E, H, W)
            # Assuming high_res_feats is a list (levels) of features (B, C, H, W)
            # We need the last level for each item in the batch? No, the structure might be different.
            # Let's use the structure from the original script: list of last features from each level.
            # If set_image_batch prepares features correctly per image, maybe we don't need to change this part?
            # Let's assume predictor internal state handles batch features correctly.
            # The original multi-image batch script did this:
            # high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]] -> This looks wrong for batch > 1
            # Let's trust predictor._features contains batched features correctly structured if set_image_batch works as expected.
            # high_res_features needs to be List[Tensor(B, C, H, W)] or similar? Revisit if errors occur.

            # Check if predictor handles feature batching internally. If not, manual stacking might be needed.
            # For now, assume predictor handles it.

            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=image_embeddings_batch,  # Pass batched embeddings
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),  # PE should be consistent
                sparse_prompt_embeddings=sparse_embeddings,  # Batched sparse embeddings
                dense_prompt_embeddings=dense_embeddings,  # Batched dense embeddings (if any)
                multimask_output=True,  # Keep True as per original
                repeat_image=False,  # Should be False if image_embeddings are already batched
                high_res_features=predictor._features.get(
                    "high_res_feats"
                ),  # Pass potentially batched high-res features
            )
            # low_res_masks shape (B*num_masks_out, 1, H, W), prd_scores (B*num_masks_out, num_classes)
            # Updated based on debug: low_res_masks shape is (B, N, H_low, W_low)
            # prd_scores shape is likely (B, N, num_classes)

            # Postprocess masks - _transforms might need adapting if it doesn't handle batch dim implicitly
            # Assuming _transforms and _orig_hw are updated by set_image_batch for the whole batch? Unlikely.
            # Loop through batch to postprocess masks individually using original sizes
            num_masks_out = low_res_masks.shape[1]  # Correct: Get N from (B, N, H, W)
            processed_masks_list = []
            target_size = (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE)

            for i in range(BATCH_SIZE):
                # Slice low_res_masks for image i -> (N, H_low, W_low)
                masks_i = low_res_masks[i]
                orig_hw = predictor._orig_hw[i]  # Get original H, W for image i

                # Process masks for image i (upscales to original size)
                # Need to add channel dim: (N, H_low, W_low) -> (N, 1, H_low, W_low)
                processed_mask_i = predictor._transforms.postprocess_masks(
                    masks_i.unsqueeze(1), orig_hw  # Original size for image i
                )
                # processed_mask_i shape: (N, 1, H_orig, W_orig)

                # Resize masks to the TARGET_IMAGE_SIZE to ensure uniformity before concat
                if processed_mask_i.shape[-2:] != target_size:
                    processed_mask_i = torch.nn.functional.interpolate(
                        processed_mask_i,  # Shape (N, 1, H_orig, W_orig)
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                # Now shape is (N, 1, TARGET_H, TARGET_W)

                processed_masks_list.append(processed_mask_i)

            # Concatenate the processed masks back together
            # Concatenate along the *batch* dimension. List contains B tensors of shape (N, 1, H, W)
            # We want the final shape to be separable by batch and mask index
            # Stacking might be easier: Creates (B, N, 1, H, W)
            prd_masks = torch.stack(processed_masks_list, dim=0)
            # prd_masks shape is now (B, N, 1, H_target, W_target)

            # Reshape prd_scores from (B * N, num_classes) ? or (B, N, num_classes)? Assume (B, N, num_classes)
            prd_scores = prd_scores.view(
                BATCH_SIZE, num_masks_out, -1
            )  # Ensure this matches actual decoder output shape

            # Select the first mask output for loss calculation (consistent with original)
            # Reshape low_res_masks and prd_scores to (B, num_masks_out, ...)
            # Use the correct num_masks_out (calculated above, likely 3)
            # prd_masks is already (B, N, 1, H, W) - no view needed here.

            # Select the primary mask/score (index 0)
            prd_masks_primary = prd_masks[:, 0, 0, :, :]  # Shape (B, H_target, W_target)
            prd_scores_primary = prd_scores[:, 0, 0]  # (B,) Assume single class score

            # --- Loss Calculation ---
            # Ensure gt_mask is on the correct device and is float32
            gt_mask = torch.tensor(masks.astype(np.float32)).to(DEVICE)  # Shape (B, H_target, W_target)

            # Ensure predicted mask has same size as GT mask (after padding/resizing in dataloader)
            if prd_masks_primary.shape[-2:] != gt_mask.shape[-2:]:
                # Resize predicted masks to match GT mask size (target_image_size)
                prd_masks_primary = torch.nn.functional.interpolate(
                    prd_masks_primary.unsqueeze(1),  # Add channel dim -> (B, 1, H, W)
                    size=gt_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(
                    1
                )  # Remove channel dim -> (B, H, W)

            prd_prob = torch.sigmoid(prd_masks_primary)  # Turn logit map to probability map (B, H, W)
            epsilon = 1e-5
            # Calculate loss per image, then mean over batch
            seg_loss_per_image = (
                -gt_mask * torch.log(prd_prob + epsilon) - (1 - gt_mask) * torch.log(1 - prd_prob + epsilon)
            ).mean(
                dim=(1, 2)
            )  # Mean over H, W
            seg_loss = seg_loss_per_image.mean()  # Mean over B

            # IOU Calculation (Score Loss)
            iou_pred_thresh = 0.5
            prd_mask_binary = prd_prob > iou_pred_thresh  # (B, H, W)
            inter = (gt_mask * prd_mask_binary).sum(dim=(1, 2))  # Sum over H, W -> (B,)
            union = gt_mask.sum(dim=(1, 2)) + prd_mask_binary.sum(dim=(1, 2)) - inter  # (B,)
            iou_per_image = inter / (union + epsilon)  # (B,)
            iou = iou_per_image.mean()  # Mean IOU over batch

            # Score loss (compare predicted score with calculated IOU)
            score_loss = torch.abs(prd_scores_primary - iou_per_image).mean()  # Mean over B

            loss = seg_loss + score_loss * 0.05  # mix losses

        # Backpropagation
        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- Logging and Saving ---
        current_iou_scalar = iou.cpu().detach().item()
        mean_iou = mean_iou * 0.99 + 0.01 * current_iou_scalar  # Use scalar value for tracking mean

        if itr % SAVE_INTERVAL == 0 and itr > 0:
            state_dict_to_save = predictor.model.state_dict()
            torch.save(state_dict_to_save, OUTPUT_MODEL_PATH)
            print(f"--- Iteration {itr}: Model saved to {OUTPUT_MODEL_PATH} ---")

        if itr % DISPLAY_INTERVAL == 0:
            current_time = time.time()
            elapsed_time_total = current_time - start_time
            elapsed_time_interval = current_time - last_display_time
            iterations_in_interval = itr - last_display_itr

            if iterations_in_interval > 0:
                time_per_iter = elapsed_time_interval / iterations_in_interval
                remaining_iters = NUM_ITERATIONS - itr
                etr_seconds = remaining_iters * time_per_iter
                etr_h, etr_m, etr_s = int(etr_seconds // 3600), int((etr_seconds % 3600) // 60), int(etr_seconds % 60)
                etr_str = f"{etr_h:02d}:{etr_m:02d}:{etr_s:02d}"
            else:
                etr_str = "N/A"

            elapsed_h, elapsed_m, elapsed_s = (
                int(elapsed_time_total // 3600),
                int((elapsed_time_total % 3600) // 60),
                int(elapsed_time_total % 60),
            )
            elapsed_str = f"{elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}"

            print(
                f"Iter {itr}/{NUM_ITERATIONS} | Loss: {loss.item():.4f} (Seg: {seg_loss.item():.4f}, Score: {score_loss.item():.4f}) | "
                f"Batch IOU: {current_iou_scalar:.4f} | Mean IOU: {mean_iou:.4f} | Elapsed: {elapsed_str} | ETR: {etr_str}"
            )
            last_display_time = current_time
            last_display_itr = itr

    except Exception as e:
        print(f"Error during training iteration {itr}: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback
        if isinstance(e, torch.cuda.OutOfMemoryError):
            print("CUDA Out of Memory. Try reducing BATCH_SIZE or TARGET_IMAGE_SIZE.")
            # Consider stopping or trying to recover (e.g., clear cache)
        # break # Option to stop training on error

print(f"--- Training Finished ---")
# Final save
state_dict_to_save = predictor.model.state_dict()
torch.save(state_dict_to_save, OUTPUT_MODEL_PATH)
print(f"Final model saved to {OUTPUT_MODEL_PATH}")
