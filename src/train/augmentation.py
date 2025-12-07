# src/train/augmentation.py
"""
Data augmentation for writer retrieval training.

Reference: icdar23 uses morphological erosion/dilation with p=0.3 each.
This simulates pen thickness variations in historical documents.
"""
import random
import torch
import torch.nn.functional as F


def get_random_kernel(size: int = 3) -> torch.Tensor:
    """Generate a random morphological kernel."""
    k = torch.rand(size, size).round()
    k[size // 2, size // 2] = 1  # Ensure center is always 1
    return k


class Erosion:
    """Morphological erosion - simulates thinner pen strokes."""
    
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply erosion to image.
        
        Args:
            img: [C, H, W] or [H, W] tensor
        Returns:
            Eroded image with same shape as input
        """
        original_shape = img.shape
        original_dim = img.dim()
        
        if img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif img.dim() == 3:
            img = img.unsqueeze(0)  # [1, C, H, W]
        
        kernel = get_random_kernel(self.kernel_size)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(img.device)
        
        # Erosion is min over neighborhood
        # Use -max(-x) trick with max_pool
        padding = self.kernel_size // 2
        
        # For binary/grayscale images, erosion = min filter
        # Approximate with dilation of inverted image
        inverted = 1.0 - img
        dilated = F.max_pool2d(inverted, self.kernel_size, stride=1, padding=padding)
        result = 1.0 - dilated
        
        # Restore original dimensions
        if original_dim == 2:
            result = result.squeeze(0).squeeze(0)
        elif original_dim == 3:
            result = result.squeeze(0)
        
        return result


class Dilation:
    """Morphological dilation - simulates thicker pen strokes."""
    
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply dilation to image.
        
        Args:
            img: [C, H, W] or [H, W] tensor
        Returns:
            Dilated image with same shape as input
        """
        original_shape = img.shape
        original_dim = img.dim()
        
        if img.dim() == 2:
            img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif img.dim() == 3:
            img = img.unsqueeze(0)  # [1, C, H, W]
        
        padding = self.kernel_size // 2
        
        # Dilation is max over neighborhood
        result = F.max_pool2d(img, self.kernel_size, stride=1, padding=padding)
        
        # Restore original dimensions
        if original_dim == 2:
            result = result.squeeze(0).squeeze(0)
        elif original_dim == 3:
            result = result.squeeze(0)
        
        return result


class MorphologicalAugmentation:
    """
    Apply random morphological augmentation.
    
    Randomly applies erosion and/or dilation to simulate
    pen thickness variations in handwritten documents.
    """
    
    def __init__(self, erosion_prob: float = 0.3, dilation_prob: float = 0.3,
                 kernel_size: int = 3):
        self.erosion_prob = erosion_prob
        self.dilation_prob = dilation_prob
        self.erosion = Erosion(kernel_size)
        self.dilation = Dilation(kernel_size)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random morphological transformations."""
        if random.random() < self.erosion_prob:
            img = self.erosion(img)
        if random.random() < self.dilation_prob:
            img = self.dilation(img)
        return img


class RandomNoise:
    """Add random Gaussian noise to simulate scanner artifacts."""
    
    def __init__(self, std: float = 0.05, prob: float = 0.2):
        self.std = std
        self.prob = prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            noise = torch.randn_like(img) * self.std
            img = torch.clamp(img + noise, 0, 1)
        return img


class RandomContrast:
    """Adjust contrast randomly to simulate different ink densities."""
    
    def __init__(self, factor_range=(0.8, 1.2), prob: float = 0.3):
        self.factor_range = factor_range
        self.prob = prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            factor = random.uniform(*self.factor_range)
            mean = img.mean()
            img = (img - mean) * factor + mean
            img = torch.clamp(img, 0, 1)
        return img


class RandomBrightness:
    """Adjust brightness randomly to simulate different paper qualities."""
    
    def __init__(self, delta_range=(-0.1, 0.1), prob: float = 0.3):
        self.delta_range = delta_range
        self.prob = prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            delta = random.uniform(*self.delta_range)
            img = img + delta
            img = torch.clamp(img, 0, 1)
        return img


class RandomAffine:
    """Apply small random affine transformations."""
    
    def __init__(self, degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), prob=0.3):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.prob = prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            import torchvision.transforms.functional as TF
            
            # Random parameters
            angle = random.uniform(-self.degrees, self.degrees)
            dx = random.uniform(-self.translate[0], self.translate[0]) * img.shape[-1]
            dy = random.uniform(-self.translate[1], self.translate[1]) * img.shape[-2]
            scale = random.uniform(*self.scale)
            
            # Apply affine transformation
            img = TF.affine(img, angle=angle, translate=[dx, dy], scale=scale, shear=0)
        
        return img


class RandomCutout:
    """Randomly mask out rectangular regions (regularization)."""
    
    def __init__(self, n_holes=1, hole_size=8, prob=0.2):
        self.n_holes = n_holes
        self.hole_size = hole_size
        self.prob = prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            h, w = img.shape[-2:]
            for _ in range(self.n_holes):
                y = random.randint(0, h - self.hole_size)
                x = random.randint(0, w - self.hole_size)
                img[..., y:y+self.hole_size, x:x+self.hole_size] = 0
        return img


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


def get_train_augmentation(
    erosion_prob: float = 0.3, 
    dilation_prob: float = 0.3,
    noise_prob: float = 0.1,
    contrast_prob: float = 0.2,
    brightness_prob: float = 0.2,
    affine_prob: float = 0.2,
    cutout_prob: float = 0.1,
    strong: bool = False
) -> Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        erosion_prob: Probability of erosion
        dilation_prob: Probability of dilation
        noise_prob: Probability of adding noise
        contrast_prob: Probability of contrast adjustment
        brightness_prob: Probability of brightness adjustment
        affine_prob: Probability of affine transformation
        cutout_prob: Probability of cutout
        strong: If True, use stronger augmentation probabilities
    
    Returns:
        Compose of augmentation transforms
    """
    if strong:
        # Stronger augmentation for small datasets
        erosion_prob = 0.4
        dilation_prob = 0.4
        noise_prob = 0.2
        contrast_prob = 0.3
        brightness_prob = 0.3
        affine_prob = 0.3
        cutout_prob = 0.2
    
    return Compose([
        MorphologicalAugmentation(erosion_prob, dilation_prob),
        RandomContrast(prob=contrast_prob),
        RandomBrightness(prob=brightness_prob),
        RandomAffine(prob=affine_prob),
        RandomNoise(std=0.03, prob=noise_prob),
        RandomCutout(prob=cutout_prob),
    ])
