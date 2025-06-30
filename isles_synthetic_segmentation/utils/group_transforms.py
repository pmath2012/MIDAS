import cv2
import torch
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_transform(image, mask, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    mask = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)

    return image, mask

class Normalize(object):
    """Normalize the image in a sample to a given intensity.

    Args:
        output_intensity (tuple or int):
    """

    def __init__(self, output_intensity=1):
        assert isinstance(output_intensity, (int, tuple))
        self.output_intensity = output_intensity

    def __call__(self, sample):
        image, masks = sample['image'], sample['mask']

        img = cv2.normalize(image, dst=np.zeros_like(image), alpha=0, beta=self.output_intensity, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        msk = cv2.normalize(masks, dst=np.zeros_like(masks), alpha=0, beta=self.output_intensity, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return {'image': img, 'mask': msk.astype(int)}


class AugmentTransform(object):
    """Apply augmentation transforms to a sample.

    Args:
        v_flip (bool): Whether to apply vertical flip
        h_flip (bool): Whether to apply horizontal flip
        elastic_transform (bool): Whether to apply elastic transform
        p (float): Probability of applying each transform
    """

    def __init__(self, v_flip=False, h_flip=False, elastic_transform=False, p=0.5):
        self.p = p
        self.v_flip = v_flip
        self.h_flip = h_flip
        self.elastic_transform = elastic_transform

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if self.v_flip:
            if torch.rand(1) < self.p:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
        if self.h_flip:
            if torch.rand(1) < self.p:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
        # Elastic transform disabled - keeping only flipping

        return {'image': image, 'mask': mask}