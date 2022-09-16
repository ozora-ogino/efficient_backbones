# Efficient Backbones

Efficient backbones is the project which aims to provide efficient SOTA vision backbones based on PyTorch.

Currently following models are supported;

- [Resnet](https://arxiv.org/abs/1512.03385)
- [EfficientNetv2](https://arxiv.org/pdf/2104.00298.pdf)
- [Xception](https://arxiv.org/pdf/1610.02357.pdfs)
- [Ghostnet](https://arxiv.org/abs/1911.11907)
- [G-Ghost RegNet](https://arxiv.org/abs/2201.03297)
- [LambdaNet](https://arxiv.org/abs/2102.08602)
- [HarDNet](https://arxiv.org/abs/1909.00948ß)

NOTE: All models implemented in this project doesn't have head.

## Installation

```bash
pip install git+https://github.com/ozora-ogino/efficient_backbones
```

## Usage

All model have `out_channels` attribute in order to access the number of out channels of backbone in the same way.

```python
from efficient_backbones import build_backbone

resnet_backbone = build_backbone("resnet18")
out_channels = resnet_backbone.out_channels
```