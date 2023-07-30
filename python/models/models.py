from models.mobilenetmodified import MobileNetModified
from models.resnetmodified import ResNetModified

models = {
    "MobileNet": {
        "model": MobileNetModified,
        "batch_size": 32,
        "epochs": 50
    },
    "ResNet": {
        "model": ResNetModified,
        "batch_size": 128,
        "epochs": 30
    }
}
