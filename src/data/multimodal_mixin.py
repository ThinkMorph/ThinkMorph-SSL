import os

from PIL import Image


class MultiModalDataLoadingMixin:
    """
    Mixin for loading multimodal data.
    """

    def load_image(self, image_path: str, data_folder=None) -> Image.Image:
        """
        Load an image from a local file path.

        Args:
            image_path: Path to the image file
            data_folder: Optional folder path to prepend

        Returns:
            PIL Image object
        """
        if data_folder is not None:
            image_path = os.path.join(data_folder, image_path)

        return Image.open(image_path)
