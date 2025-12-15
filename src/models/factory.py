from .cnn import CNN
from .hybrid_cnn_transformer import HybridCNNTransformer

def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == "cnn":
        return CNN(n_classes=num_classes)
    elif name == "hybridcnntransformer":
        return HybridCNNTransformer(n_classes=num_classes)
    else:
        raise ValueError(f"Modèle inconnu : {name}")
