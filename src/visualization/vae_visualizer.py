import torch
from matplotlib import pyplot as plt
from src.visualization.base_visualizer import BaseVisualizer


class VAEVisualizer(BaseVisualizer):
    def visualize_reconstructions(self, num_samples=8):
        """
        Visualize original and reconstructed images.
        """
        self.model.eval()
        sample_imgs = []
        reconstructions = []

        # Collect some samples and their reconstructions
        for imgs, _ in self.val_loader:
            imgs = imgs.to(self.device)
            with torch.no_grad():
                x_hat, _, _ = self.model(imgs)
            sample_imgs = imgs[:num_samples].cpu()
            reconstructions = x_hat[:num_samples].cpu()
            break  # Only visualize the first batch

        # Ensure tensors have valid shapes for visualization
        sample_imgs = sample_imgs.clamp(0, 1)  # Clamp values to [0, 1] range
        reconstructions = reconstructions.clamp(0, 1)

        # Plot
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        for i in range(num_samples):
            # Original images
            img = sample_imgs[i]
            if img.dim() == 4:  # Ensure correct dimensionality
                img = img.squeeze(0)
            axes[0, i].imshow(img.permute(1, 2, 0).numpy())  # Convert to HWC
            axes[0, i].axis('off')

            # Reconstructed images
            rec = reconstructions[i]
            if rec.dim() == 4:  # Ensure correct dimensionality
                rec = rec.squeeze(0)
            axes[1, i].imshow(rec.permute(1, 2, 0).numpy())  # Convert to HWC
            axes[1, i].axis('off')

        plt.suptitle("Original Images (Top) vs Reconstructed Images (Bottom)", fontsize=16)
        plt.show()
