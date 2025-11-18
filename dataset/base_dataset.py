import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class DifferenceAFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_folder: str, kernel_size: int = 3, transform=None,
                 one_channel_image: bool = False, normalize_output: bool = False):
        self._image_folder = image_folder

        self._df = df

        self._kernel_size = kernel_size

        self._transform = transform

        self._one_channel_image = one_channel_image

        self._normalize_output = normalize_output
        if self._normalize_output:
            self._max_defocus_z_position = self.get_max_defocus()

    def get_max_defocus(self) -> float:
        """
        We reckon here that the 0 position is centered in the interval [min_pos; max_pos]
        :return:
        """
        return max(np.abs(self._df['z2_diff_focus'].min()), self._df['z2_diff_focus'].max())

    @classmethod
    def from_excel(cls, excel_filepath: str, image_folder: str, kernel_size: int = 3, transform=None,
                   one_channel_image: bool = False, normalize_output: bool = False):
        return cls(df=pd.read_excel(excel_filepath), image_folder=image_folder, kernel_size=kernel_size,
                   transform=transform, one_channel_image=one_channel_image, normalize_output=normalize_output)

    @classmethod
    def from_csv(cls, excel_filepath: str, image_folder: str, kernel_size: int = 3, transform=None,
                   one_channel_image: bool = False, normalize_output: bool = False):
        return cls(df=pd.read_csv(excel_filepath), image_folder=image_folder, kernel_size=kernel_size,
                   transform=transform, one_channel_image=one_channel_image, normalize_output=normalize_output)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx):
        item = self._df.iloc[idx]
        z1_image_name = item['z1_image']
        z2_image_name = item['z2_image']
        xy_position = item['xy_position']

        image_z1_path = os.path.join(self._image_folder, xy_position, z1_image_name)
        image_z2_path = os.path.join(self._image_folder, xy_position, z2_image_name)

        image_z1 = cv2.imread(image_z1_path)
        image_z2 = cv2.imread(image_z2_path)

        blurred_image_z1 = cv2.medianBlur(image_z1, self._kernel_size)
        blurred_image_z2 = cv2.medianBlur(image_z2, self._kernel_size)

        # difference_image = cv2.subtract(blurred_image_z2, blurred_image_z1)
        difference_image = blurred_image_z2 - blurred_image_z1

        blurred_difference_image = np.float32(cv2.medianBlur(difference_image, self._kernel_size))

        y = float(item['z2_diff_focus'])
        if self._normalize_output:
            y /= self._max_defocus_z_position

        norm_image = self.normalize_standard_channelwise(img_np=blurred_difference_image)

        if self._one_channel_image:
            norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)

        if self._transform is None:
            transform = transforms.ToTensor()

            # Convert the image to PyTorch tensor
            tensor_image = transform(norm_image)

        else:
            transformed = self._transform(image=norm_image)
            tensor_image = transformed["image"]

        return {'X': tensor_image, 'y': y, 'std': torch.tensor(self.luminance_weighted_std(img_np=image_z1))}

    @staticmethod
    def luminance_weighted_std(img_np: np.ndarray) -> float:
        std_image = 0
        # [0.114, 0.587, 0.299] -> BGR luminance weights
        for c, coeff in zip(range(img_np.shape[2]), [0.114, 0.587, 0.299]):
            channel = img_np[..., c]
            std = channel.std()
            std_image += std * coeff

        return std_image

    @staticmethod
    def transform_to_grayscale(img_np: np.ndarray) -> np.ndarray:

        grayscale_image = np.zeros(img_np.shape[:2])
        luminance_weights = [0.114, 0.587, 0.299]

        for index, weight in enumerate(luminance_weights):
            grayscale_image += weight * img_np[..., index]

        return grayscale_image

    @staticmethod
    def normalize_standard_channelwise(img_np):
        norm = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            channel = img_np[..., c]
            mean, std = channel.mean(), channel.std()
            norm[..., c] = (channel - mean) / std
        return norm
