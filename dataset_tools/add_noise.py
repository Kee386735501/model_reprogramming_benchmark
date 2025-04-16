import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset

class NoisyDataset(Dataset):
    def __init__(self, base_dataset: Dataset, noise_type: str = "gaussian", noise_level: float = 0.1):
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = np.array(img)

        if self.noise_level > 0:
            if self.noise_type == 'gaussian':
                noise = np.random.normal(0, self.noise_level * 255, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)

            elif self.noise_type == 'salt_pepper':
                prob = self.noise_level
                mask = np.random.choice([0, 255, -1], size=img.shape, p=[prob/2, prob/2, 1 - prob])
                img = np.where(mask == -1, img, mask).astype(np.uint8)

            elif self.noise_type == 'blur':
                img = Image.fromarray(img)
                img = img.filter(ImageFilter.GaussianBlur(radius=self.noise_level * 10))
                img = np.array(img)

        img = Image.fromarray(img)
        return self.transform(img), label


def get_noisy_dataset(base_dataset: Dataset, noise_type: str = "gaussian", noise_level: float = 0.1) -> Dataset:
    return NoisyDataset(base_dataset, noise_type=noise_type, noise_level=noise_level)
