"""
Phase 3 -- Grad-CAM heatmaps for the trained OA classifier.

Loads the checkpoint saved by train.py, runs inference on one or more
X-ray images, and saves a heatmap overlay showing which regions of the
image drove the prediction. Implemented manually via forward/backward
hooks on the last conv block of ResNet18 (no extra dependency needed
beyond torch/torchvision/matplotlib/pillow).

Usage:
    # Single image
    python grad_cam.py --checkpoint ./checkpoints/oa_resnet18_binary.pt \\
        --image /path/to/xray.png --out_dir ./gradcam_out

    # Whole folder of images
    python grad_cam.py --checkpoint ./checkpoints/oa_resnet18_binary.pt \\
        --image_dir /path/to/folder --out_dir ./gradcam_out
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Low-confidence threshold -- below this, we tell the user to defer to a clinician
# rather than presenting a falsely confident answer.
UNCERTAIN_THRESHOLD = 0.65

EXPLANATIONS = {
    "healthy_mild": (
        "No strong indicators of significant joint space narrowing or "
        "osteophyte formation were highlighted by the model."
    ),
    "at_risk": (
        "The model flagged regions consistent with possible joint space "
        "narrowing and/or osteophyte formation. This should be reviewed "
        "by a clinician -- this tool does not provide a diagnosis."
    ),
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class GradCAM:
    """Manual Grad-CAM via hooks on a target conv layer (ResNet18's layer4)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, F.softmax(output, dim=1).detach().cpu().numpy()[0]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def overlay_heatmap(original_img: Image.Image, cam: np.ndarray, alpha=0.45):
    """Resize CAM to original image size and blend with a jet colormap."""
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(original_img.size, Image.BILINEAR)
    ) / 255.0

    heatmap = cm.jet(cam_resized)[:, :, :3]  # drop alpha channel from colormap
    original_rgb = np.array(original_img.convert("RGB")) / 255.0

    overlaid = (1 - alpha) * original_rgb + alpha * heatmap
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid


def process_image(image_path, model, gradcam, class_names, device, out_dir):
    original_img = Image.open(image_path).convert("RGB")
    transform = build_eval_transform()
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(False)  # gradients needed w.r.t. activations, not input

    # Forward pass to get prediction first (no grad needed for this part)
    with torch.no_grad():
        logits = model(input_tensor)
        pred_idx = logits.argmax(dim=1).item()

    # Re-run forward WITH grad enabled for Grad-CAM (backward() needs a live graph)
    input_tensor.requires_grad_(True)
    cam, probs = gradcam.generate(input_tensor, pred_idx)

    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    explanation = EXPLANATIONS.get(pred_class, "")
    if confidence < UNCERTAIN_THRESHOLD:
        explanation = (
            f"Model confidence is low ({confidence:.0%}). Treat this result as "
            "inconclusive and recommend clinician review rather than relying on "
            "the predicted label. " + explanation
        )

    overlaid = overlay_heatmap(original_img, cam)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_img.convert("L"), cmap="gray")
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(overlaid)
    axes[1].set_title(f"Prediction: {pred_class} ({confidence:.0%})")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"{Path(image_path).stem}_gradcam.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

    print(f"\n{Path(image_path).name}")
    print(f"  Prediction: {pred_class}  (confidence: {confidence:.1%})")
    print(f"  Explanation: {explanation}")
    print(f"  Saved: {out_path}")

    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for OA X-ray predictions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, help="Path to a folder of images")
    parser.add_argument("--out_dir", type=str, default="./gradcam_out")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        raise SystemExit("Provide either --image or --image_dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    model, class_names = load_model(args.checkpoint, device)
    print(f"Loaded model. Classes: {class_names}")

    gradcam = GradCAM(model, target_layer=model.layer4)

    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths.extend(sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in exts))

    if not image_paths:
        raise SystemExit("No images found to process.")

    for img_path in image_paths:
        process_image(img_path, model, gradcam, class_names, device, out_dir)

    print(f"\nDone. {len(image_paths)} image(s) processed. Overlays saved to: {out_dir}")


if __name__ == "__main__":
    main()
