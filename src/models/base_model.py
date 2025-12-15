import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """Classe de base commune à tous les modèles."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        raise NotImplementedError("La méthode forward() doit être redéfinie dans le modèle enfant.")

    def save(self, path: str):
        """Sauvegarde les poids."""
        torch.save(self.state_dict(), path)
        print(f"[INFO] Modèle sauvegardé → {path}")

    def load(self, path: str, device="cpu"):
        """Charge les poids."""
        self.load_state_dict(torch.load(path, map_location=device))
        print(f"[INFO] Poids chargés depuis {path}")
