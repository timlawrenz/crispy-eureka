# Crispy Eureka: Classifier-Guided Generation for ComfyUI

This repository provides a set of custom nodes for ComfyUI that allow you to guide the image generation process using one or more pre-trained classifiers. By analyzing the image at each step of the diffusion process, these nodes can steer the generation towards desired attributes (e.g., "a photo of a cat") and away from undesired ones (e.g., "blurry" or "black and white").

This technique uses gradient-based guidance, a powerful method for controlling the output of diffusion models beyond simple text prompts.

## Functionality

This suite includes two main nodes and a behind-the-scenes hijack to enable the functionality:

*   **Classifier Loader**: This node loads your `.pkl` classifier models. You can specify "positive" classifiers for attributes you want to encourage and "negative" classifiers for attributes you want to avoid. It aggregates these into a single `CLASSIFIER` object.
*   **Classifier Guidance**: This node takes the `CLASSIFIER` object and plugs into the sampler. It performs the core logic of calculating the guidance and applying it to the latent image during sampling.
*   **KSamplerAdvanced Hijack**: On startup, a small patch is applied to ComfyUI's `KSamplerAdvanced` node, adding a `latent_hook` input. This allows the `Classifier Guidance` node to intervene in the sampling process.

## Usage Guide

### 1. Prerequisites: Install Classifiers

You need one or more classifier models. These should be scikit-learn compatible models (like `LogisticRegression` or `SVC`) that have been trained to predict an attribute from a CLIP image embedding and saved as a `.pkl` file.

Place your classifier files in the following directory:
`ComfyUI/models/classifiers/`

(If this directory does not exist, the node will create it for you).

### 2. ComfyUI Workflow

1.  **Add Loader Node**: Add a `Classifier Loader` node to your workflow (found under `latent/guidance`).
2.  **Configure Loader**:
    *   In the `positive_classifiers` text box, list the filenames of classifiers for attributes you want to **see** in the image (e.g., `cat.pkl`, `happy.pkl`). Put one filename per line.
    *   In the `negative_classifiers` text box, list the filenames for attributes you want to **avoid** (e.g., `blurry.pkl`, `ugly.pkl`).
3.  **Add Guidance Node**: Add a `Classifier Guidance` node to your workflow.
4.  **Connect Nodes**:
    *   Connect the `CLASSIFIER` output of the `Classifier Loader` to the `classifier` input of the `Classifier Guidance` node.
    *   Connect your `MODEL`, `CLIP`, and `VAE` to the `Classifier Guidance` node.
    *   Connect the `LATENT_HOOK` output of the `Classifier Guidance` node to the `latent_hook` input on a `KSamplerAdvanced` node.
    *   Connect all other standard inputs (`model`, `positive`, `negative`, `latent_image`) to your `KSamplerAdvanced` node as usual.

### 3. Configure Guidance Parameters

On the `Classifier Guidance` node, you can set:

*   `guidance_scale`: Controls the strength of the classifier's influence. Higher values have a stronger effect but can cause artifacts. (Default: 1.0)
*   `start_step` / `end_step`: Defines the range of sampling steps during which guidance is active. Applying guidance only in the middle steps is often effective.

## How It Works (Technical Details)

The guidance process runs inside the sampler for each step specified:

1.  **Predict Clean Image**: The node takes the current noisy latent (`x_t`) and asks the diffusion model to predict the final, clean image (`pred_x0`).
2.  **Get CLIP Embedding**: This `pred_x0` is decoded into an image, which is then fed through a CLIP model to get a semantic embedding.
3.  **Calculate Loss**: The embedding is passed to all loaded classifiers. The node calculates a single "loss" value representing how far the image is from the desired attributes.
4.  **Compute Gradient**: Using `torch.autograd`, the node computes the gradient of the loss with respect to the noisy latent (`x_t`). This gradient points in the direction that will most effectively satisfy the classifiers.
5.  **Apply Guidance**: The gradient is scaled by `guidance_scale` and subtracted from the latent `x_t`, nudging the sampling process in the desired direction for the next step.

## Acknowledgments

The method for hooking into the KSamplerAdvanced node is inspired by the technique used in the [ComfyUI-Advanced-Latent-Control](https://github.com/kuschanow/ComfyUI-Advanced-Latent-Control) repository.
