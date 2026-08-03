import numpy as np
from PIL import Image

from dsga.augmentation import ChestXStyleTransform


def test_chestx_transform_returns_three_channel_image():
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (48, 48, 3), dtype=np.uint8))
    transformed = ChestXStyleTransform(blur_kernel=5)(image)
    array = np.asarray(transformed)
    assert array.shape == (48, 48, 3)
    assert np.array_equal(array[..., 0], array[..., 1])
    assert np.array_equal(array[..., 1], array[..., 2])

