import torch

from dsga.models import ResNet10


def test_resnet10_exposes_paper_feature_shapes():
    model = ResNet10(num_classes=5).eval()
    with torch.no_grad():
        maps = model.feature_maps(torch.randn(2, 3, 224, 224))
        embeddings = model.embeddings(torch.randn(2, 3, 224, 224))
    assert [tuple(x.shape[1:]) for x in maps] == [
        (64, 56, 56),
        (128, 28, 28),
        (256, 14, 14),
        (512, 7, 7),
    ]
    assert [x.shape for x in embeddings] == [(2, 64), (2, 128), (2, 256), (2, 512)]

