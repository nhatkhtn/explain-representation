from PIL import Image

class MaskPostProcessor:
    def __init__(self, image: Image.Image, min_area_ratio: float, margin=10):
        self.image = image
        self.min_area_ratio = min_area_ratio
        self.margin = margin

    def __call__(self, outputs: list[dict]):
        im_area = self.image.size[0] * self.image.size[1]
        margin = self.margin
        image = self.image
        crops = []
        for mask in filter(lambda mask: mask['area'] >= im_area * self.min_area_ratio, outputs):
            x, y, w, h = mask['bbox']
            left = max(0, x - margin)
            top = max(0, y - margin)
            right = min(image.width, x + w + margin)
            bottom = min(image.height, y + h + margin)
            if left > right or top > bottom:
                logger.warning(f"Invalid crop: {left=}, {right=}, {top=}, {bottom=}")
                continue
            crop_image = image.crop((left, top, right, bottom))
            crops.append(crop_image)

        return crops