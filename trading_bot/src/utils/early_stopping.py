# trading_bot/src/utils/early_stopping.py

import numpy as np
import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = False, path: str = 'checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')  # Usar float('inf') para claridad
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module):
        """
        Checks if validation loss has improved. If so, saves the model. Otherwise, increments counter.
        When counter reaches patience, sets early_stop to True.
        """
        if model is None:
            raise ValueError("[EarlyStopping] Modelo no puede ser None para guardar checkpoint.")

        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # Guardar mejor estado del modelo
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"EarlyStopping: Mejora en loss a {val_loss:.4f}, guardando checkpoint.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: Sin mejora ({self.counter}/{self.patience}).")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping: Paciencia agotada, deteniendo entrenamiento.")