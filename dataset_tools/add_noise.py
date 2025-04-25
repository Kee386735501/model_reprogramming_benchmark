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

        # ✅ 强制确保图像是 RGB 模式的 PIL.Image
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        img = img.convert('RGB')

        img = np.array(img)  # HWC, uint8

        # ✅ 添加噪声
        if self.noise_level > 0:
            if self.noise_type == 'gaussian':
                noise = np.random.normal(0, self.noise_level * 255, img.shape)
                img = img + noise
                img = np.clip(img, 0, 255).astype(np.uint8)

            elif self.noise_type == 'salt_pepper':
                prob = self.noise_level
                mask = np.random.choice([0, 255, -1], size=img.shape, p=[prob / 2, prob / 2, 1 - prob])
                img = np.where(mask == -1, img, mask).astype(np.uint8)

            elif self.noise_type == 'blur':
                img = Image.fromarray(img)
                img = img.filter(ImageFilter.GaussianBlur(radius=self.noise_level * 10))
                img = np.array(img).astype(np.uint8)

        # ✅ 再次确认图像为 RGB 3 通道格式
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[0] == 1 and img.shape[1] == 1:
            img = np.transpose(img, (2, 0, 1))  # (1,1,224) → (224,1,1)
            img = np.transpose(img, (1, 2, 0))  # → (1,1,224) → (224,224,1) → repeat to RGB
            img = np.repeat(img, 3, axis=2)

        img = img.astype(np.uint8)
        img = Image.fromarray(img)

        return self.transform(img), label





def get_noisy_dataset(base_dataset: Dataset, noise_type: str = "gaussian", noise_level: float = 0.1) -> Dataset:
    return NoisyDataset(base_dataset, noise_type=noise_type, noise_level=noise_level)
