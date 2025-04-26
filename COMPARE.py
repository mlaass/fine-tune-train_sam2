#!/usr/bin/env python
# coding: utf-8

# Script to compare validation metrics between base and fine-tuned models.

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- Configuration ---
# Directory where VALIDATE.py saved the results
RESULTS_DIR = "./finetuned_sam2_models"

# Base name used in VALIDATE.py (adjust if you changed it there)
# Example: If your fine-tuned model name in VALIDATE.py was 'sam2_hiera_tiny_finetuned_decoder.pth'
# the base name here would be 'sam2_hiera_tiny_finetuned_decoder'
BASE_OUTPUT_NAME = "sam2_hiera_tiny_finetuned_decoder"  # MODIFY THIS IF NEEDED

# Construct the full paths to the JSON files
BASE_METRICS_FILE = os.path.join(RESULTS_DIR, f"{BASE_OUTPUT_NAME}_validation_metrics_base.json")
FINETUNED_METRICS_FILE = os.path.join(RESULTS_DIR, f"{BASE_OUTPUT_NAME}_validation_metrics_finetuned.json")

# Output plot filenames
COMPARISON_IOU_PLOT_FILE = os.path.join(RESULTS_DIR, f"{BASE_OUTPUT_NAME}_comparison_mean_iou.png")
COMPARISON_RECALL_PLOT_FILE = os.path.join(RESULTS_DIR, f"{BASE_OUTPUT_NAME}_comparison_obj_recall.png")
# --- End Configuration ---


def load_metrics(filepath):
    """Loads metrics from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Metrics file not found at {filepath}")
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(data)
        # Use category ID as index for robust merging
        if "id" in df.columns:
            df.set_index("id", inplace=True)
        else:
            print(f"Warning: 'id' column not found in {filepath}. Cannot set index.")
        return df
    except Exception as e:
        print(f"Error loading or parsing {filepath}: {e}")
        return None


def plot_comparison(df_merged, metric_base, metric_finetuned, title, ylabel, output_filename):
    """Generates and saves a side-by-side bar plot comparing a metric."""
    if metric_base not in df_merged.columns or metric_finetuned not in df_merged.columns:
        print(
            f"Error: Required columns ('{metric_base}', '{metric_finetuned}') not found in merged data for plot: {title}"
        )
        return

    # Sort by the fine-tuned metric for potentially better visualization
    df_plot = df_merged.sort_values(by=metric_finetuned, ascending=False)

    # Get category names for labels (using the index which is the category ID)
    # We need the original name mapping if available, otherwise just use IDs
    # Assuming 'name_base' or 'name_finetuned' exists after merge
    cat_names = df_plot.get("name_finetuned", df_plot.get("name_base", df_plot.index.astype(str)))
    cat_names = [name[:25] for name in cat_names]  # Truncate long names

    base_values = df_plot[metric_base]
    finetuned_values = df_plot[metric_finetuned]

    x = np.arange(len(cat_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, max(8, len(cat_names) * 0.3)))  # Adjust size
    rects1 = ax.bar(x - width / 2, base_values, width, label="Base Model", color="skyblue")
    rects2 = ax.bar(x + width / 2, finetuned_values, width, label="Fine-tuned Model", color="lightcoral")

    # Add some text for labels, title and axes ticks
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=90, ha="center")  # Rotate labels
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0

    fig.tight_layout()

    try:
        plt.savefig(output_filename)
        print(f"Saved comparison plot to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}")
    plt.close(fig)  # Close the figure to free memory


def main():
    print("--- Starting Comparison --- ")
    print(f"Loading base metrics from: {BASE_METRICS_FILE}")
    df_base = load_metrics(BASE_METRICS_FILE)

    print(f"Loading fine-tuned metrics from: {FINETUNED_METRICS_FILE}")
    df_finetuned = load_metrics(FINETUNED_METRICS_FILE)

    if df_base is None or df_finetuned is None:
        print("Cannot proceed without both metrics files. Exiting.")
        return

    # Merge the dataframes based on category ID (index)
    # Use outer join to keep categories present in only one file
    df_merged = pd.merge(
        df_base, df_finetuned, left_index=True, right_index=True, how="outer", suffixes=("_base", "_finetuned")
    )

    # Fill NaN values that result from the outer join (e.g., a category only in base)
    # Fill metrics with 0, keep names if one exists
    for col in df_merged.columns:
        if pd.api.types.is_numeric_dtype(df_merged[col]):
            df_merged[col].fillna(0, inplace=True)
        # Consolidate name column if necessary (prefer finetuned name if exists)
        if "name_finetuned" in df_merged.columns and "name_base" in df_merged.columns:
            df_merged["name"] = df_merged["name_finetuned"].fillna(df_merged["name_base"])
        elif "name_finetuned" in df_merged.columns:
            df_merged["name"] = df_merged["name_finetuned"]
        elif "name_base" in df_merged.columns:
            df_merged["name"] = df_merged["name_base"]

    print(f"Merged data for {len(df_merged)} categories.")

    # --- Generate Plots ---

    # Plot 1: Mean IoU Comparison
    plot_comparison(
        df_merged=df_merged,
        metric_base="mean_iou_base",
        metric_finetuned="mean_iou_finetuned",
        title="Mean Instance IoU Comparison (Base vs. Fine-tuned)",
        ylabel="Mean IoU",
        output_filename=COMPARISON_IOU_PLOT_FILE,
    )

    # Plot 2: Object Recall Comparison
    plot_comparison(
        df_merged=df_merged,
        metric_base="obj_recall_base",
        metric_finetuned="obj_recall_finetuned",
        title="Object Recall Comparison (Base vs. Fine-tuned)",
        ylabel="Object Recall",
        output_filename=COMPARISON_RECALL_PLOT_FILE,
    )

    print("--- Comparison Finished --- ")


if __name__ == "__main__":
    main()
