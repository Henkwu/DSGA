import torch

from dsga.prototypical import multilevel_episode


def test_multilevel_episode_prefers_matching_prototypes():
    support = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    query = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    labels = torch.tensor([0, 1])
    logits, loss = multilevel_episode(
        [support] * 4, labels, [query] * 4, labels, [0.1, 0.27, 0.189, 0.441], 0.1
    )
    assert logits.argmax(1).tolist() == [0, 1]
    assert loss is not None and loss.item() >= 0

