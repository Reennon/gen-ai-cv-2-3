import matplotlib.pyplot as plt
import torch

class BaseVisualizer:
    def __init__(self, model, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device

    def visualize_reconstructions(self, num_samples=8):
        """
        Visualize original and reconstructed images.
        """
        self.model.eval()
        # Get a batch of validation data
        sample_imgs, _ = next(iter(self.val_loader))
        sample_imgs = sample_imgs[:num_samples].to(self.device)

        # Reconstruct images
        with torch.no_grad():
            reconstructions = self.model(sample_imgs)

        # Plot original and reconstructed images
        fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        for i in range(num_samples):
            axs[0, i].imshow(sample_imgs[i].cpu().permute(1, 2, 0))
            axs[0, i].axis('off')
            axs[1, i].imshow(reconstructions[i].cpu().permute(1, 2, 0))
            axs[1, i].axis('off')
        plt.suptitle("Original Images (Top) vs Reconstructed Images (Bottom)", fontsize=16)
        plt.show()