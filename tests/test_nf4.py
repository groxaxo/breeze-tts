from torch import nn

from breeze_infer.nf4 import apply_nf4


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone_model = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, 128),
        )
        self.depth_decoder = nn.Sequential(nn.Linear(128, 128))
        self.codec_model = nn.Sequential(nn.Linear(128, 128))
        self.embed_text_tokens = nn.Embedding(32, 128)
        self.lm_head = nn.Linear(128, 32)


def test_apply_nf4_skips_decoder_codec_embeddings(monkeypatch) -> None:
    class FakeParams4bit(nn.Parameter):
        def __new__(cls, data=None, requires_grad=False, **kwargs):
            return nn.Parameter.__new__(cls, data, requires_grad=requires_grad)

    class FakeLinear4bit(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias)

    import sys
    import types

    fake_bnb = types.ModuleType("bitsandbytes")
    fake_nn = types.ModuleType("bitsandbytes.nn")
    fake_nn.Linear4bit = FakeLinear4bit
    fake_nn.Params4bit = FakeParams4bit
    sys.modules["bitsandbytes"] = fake_bnb
    sys.modules["bitsandbytes.nn"] = fake_nn

    model = _Toy()
    stats = apply_nf4(model)
    assert stats.replaced == 2
    assert isinstance(model.backbone_model[0], FakeLinear4bit)
    assert type(model.depth_decoder[0]) is nn.Linear
    assert type(model.codec_model[0]) is nn.Linear
    assert type(model.lm_head) is nn.Linear
