from .resnet import resnet18, resnet34, resnet50, resnet101
from .ghostnet import ghostnet
from .g_ghost_regnet import (
    regnetx_002,
    regnetx_004,
    regnetx_006,
    regnetx_008,
    regnetx_016,
    regnetx_032,
)
from .xception import Xception
from .efficientnet_v2 import effnetv2_s, effnetv2_m, effnetv2_l, effnetv2_xl
from .hardnet import hardnet_39, hardnet_68, hardnet_85
from .lambdanet import lambda_resnet18, lambda_resnet50


def build_backbone(backbone: str):
    """Build backbone by given backbone name."""
    backbones = {
        # ResNet
        "resnet18": resnet18(),
        "resnet34": resnet34(),
        "resnet50": resnet50(),
        "resnet101": resnet101(),
        # Xception
        "xception": Xception(),
        # EfficientNet
        "efficientnetv2_s": effnetv2_s(),
        "efficientnetv2_m": effnetv2_m(),
        "efficientnetv2_l": effnetv2_l(),
        "efficientnetv2_xl": effnetv2_xl(),
        # GhostNet
        "ghostnet": ghostnet(),
        "regnetx_002": regnetx_002(),
        "regnetx_004": regnetx_004(),
        "regnetx_006": regnetx_006(),
        "regnetx_008": regnetx_008(),
        "regnetx_016": regnetx_016(),
        "regnetx_032": regnetx_032(),
        # HarDNet
        "hardnet39": hardnet_39(),
        "hardnet68": hardnet_68(),
        "hardnet85": hardnet_85(),
        # LambdaResNet
        "lambda_resnet18": lambda_resnet18(),
        "lambda_resnet50": lambda_resnet50(),
    }
    return backbones[backbone]
