import logging
from typing import Any

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class MaskPostProcessor:
    def __init__(self, 
        image: Image.Image,
        min_area_ratio: float,
        mask_iou_thresh: float,
        # taken from from https://pytorch.org/vision/main/transforms.html
        fill_value=np.array([0.485, 0.456, 0.406]),
        crop_size=(256, 256),
        margin=10
    ):
        self.image = image
        self.min_area_ratio = min_area_ratio
        self.mask_iou_thresh = mask_iou_thresh
        self.fill_value = (fill_value * 255).astype(np.uint8)
        self.crop_size = crop_size
        self.margin = margin

        self.im_array = np.asarray(image)
        
    def __call__(self, outputs: list[dict]) -> dict[str, Any]:
        im_area = self.image.size[0] * self.image.size[1]

        prev_masks = []
        patches, boxes = [], []
        for ann in filter(lambda ann: ann['area'] >= im_area * self.min_area_ratio, outputs):
            for prev_mask in prev_masks:
                iou = np.sum(np.logical_and(prev_mask, ann['segmentation'])) / np.sum(np.logical_or(prev_mask, ann['segmentation']))
                if iou > self.mask_iou_thresh:
                    break
            else:
                # crop from https://github.com/amiratag/ACE/blob/35d0228693adb747c4d1eee2359561c8f31ccfe8/ace.py#L256
                mask_expanded = ann['segmentation'][:, :, None]
                patch = (mask_expanded * self.im_array + (1 - mask_expanded) * self.fill_value).astype(np.uint8)
                ones = np.where(ann['segmentation'] == 1)
                h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
                patch = Image.fromarray(patch[h1:h2, w1:w2], mode='RGB').resize(self.crop_size, resample=Image.Resampling.BICUBIC)
                patches.append(patch)
                boxes.append(ann['bbox'])  # x, y, w, h
                prev_masks.append(ann['segmentation'])

        return {
            "patches": patches,
            "boxes": boxes,
        }