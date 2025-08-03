# trading_bot/src/news_only_trainer.py

import os
import pandas as pd
import torch
import numpy as np
import joblib
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.metrics import classification_report, f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, AutoModel
from datetime import datetime, date

from trading_bot.config import TradingConfig
from trading_bot.src.utils.early_stopping import EarlyStopping
from trading_bot.src.data.db_manager import load_all_data, load_all_news
from trading_bot.src.utils.yaml_logger import guardar_resultado_yaml

class NewsOnlyTrainer:
    def __init__(
        self,
        learning_rate: float = 1e-4,
        batch_size: int    = 32,
        hidden_size: int   = 128,
        dropout: float     = 0.1,
        optuna_mode: bool  = False
    ):
        self.cfg            = TradingConfig()
        self.learning_rate  = learning_rate
        self.batch_size     = batch_size
        self.hidden_size    = hidden_size
        self.dropout        = dropout
        self.optuna_mode    = optuna_mode

        self.tokenizer     = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.news_encoder  = AutoModel.from_pretrained("distilbert-base-uncased")
        self.model_path    = self.cfg.models_dir / "news_only_model.pt"
        self.thresh_path   = self.cfg.models_dir / "news_best_thresh.pkl"  # Para guardar el mejor threshold
        self.news_emb_size = self.news_encoder.config.hidden_size + 1  # +1 para sentiment

    def prepare_data(self):
        df = load_all_data()
        df = df[df["symbol"].isin(self.cfg.symbols)].copy()
        df["date"] = pd.to_datetime(df["date"])

        # Construir etiquetas diarias
        labels = self._build_labels(df)
        X_list, y_list = [], []

        for symbol in df["symbol"].unique():
            sub = df[df["symbol"] == symbol]
            for dt in sub["date"].dt.date.unique():
                emb    = self.embed_news(symbol, dt)
                target = labels.get(symbol, {}).get(dt)
                if target is not None:
                    X_list.append(emb)
                    y_list.append(target)

        if not X_list:
            return None, None

        X = np.array(X_list)
        y = np.array(y_list)
        print("[NewsOnlyTrainer] Distribución original de clases:", Counter(y))

        # SMOTE en CPU
        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X, y)
        print("[NewsOnlyTrainer] Tras SMOTE:", Counter(y_res))

        # Sampler balanceado para DataLoader
        counts   = Counter(y_res)
        weights  = [1.0/counts[int(label)] for label in y_res]
        sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        # Tensores
        X_t = torch.tensor(X_res, dtype=torch.float32)
        y_t = torch.tensor(y_res, dtype=torch.float32).unsqueeze(1)
        ds  = TensorDataset(X_t, y_t)

        split = int(len(ds) * self.cfg.validation_split)
        train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-split, split])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  sampler=sampler)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size)

        return train_loader, val_loader

    def embed_news(self, sym: str, dt: date) -> np.ndarray:
        """Embed noticias reales del día; integra sentiment."""
        dt_str = dt.strftime('%Y-%m-%d')
        news_df = load_all_news(symbol=sym, start=dt_str, end=dt_str)
        if news_df.empty:
            txt = "No news available for this day."
            sentiment = 0.0
        else:
            txt = ' '.join(news_df['title'].fillna('') + ' ' + news_df['description'].fillna(''))
            sentiment = news_df['sentiment'].mean() if 'sentiment' in news_df and not news_df['sentiment'].isnull().all() else 0.0

        inputs  = self.tokenizer(txt, return_tensors="pt",
                                  truncation=True, padding="max_length", max_length=512)
        out     = self.news_encoder(**inputs)
        emb = out.last_hidden_state.mean(dim=1).detach().squeeze(0).numpy()  # Mean pooling
        return np.concatenate([emb, [sentiment]])

    def _build_labels(self, df: pd.DataFrame) -> dict:
        df_daily = df.groupby(["symbol", df["date"].dt.date])["close"].last()
        df_next  = df_daily.groupby(level=0).shift(-1).rename("next_close")
        merged   = df_daily.to_frame("close").join(df_next)
        merged["target"] = (merged["next_close"] > merged["close"]).astype(int)

        targets = {}
        for (sym, dt), row in merged.iterrows():
            targets.setdefault(sym, {})[dt] = int(row["target"])
        return targets

    def train(self) -> dict | None:
        loaders = self.prepare_data()
        if loaders[0] is None:
            print("[NewsOnlyTrainer] No hay datos de noticias.")
            return None
        train_loader, val_loader = loaders

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Manejo de device

        model = nn.Sequential(
            nn.Linear(self.news_emb_size, self.hidden_size),  # Ajustado a 768 +1 para sentiment
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        stopper   = EarlyStopping(patience=self.cfg.early_stopping_patience,
                                  verbose=True, path=str(self.model_path))

        for epoch in range(self.cfg.num_epochs):
            model.train()
            loss_tr = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                l   = criterion(out, yb)
                l.backward()
                optimizer.step()
                loss_tr += l.item()
            print(f"[NewsOnlyTrainer] Epoch {epoch+1}: Train loss={loss_tr/len(train_loader):.4f}")

            # validación
            model.eval()
            probs, labs = [], []
            loss_va = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    out = model(Xb)
                    loss_va += criterion(out, yb).item()
                    probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
                    labs.extend(yb.cpu().numpy().flatten().tolist())
            print(f"[NewsOnlyTrainer] Val loss={loss_va/len(val_loader):.4f}")
            stopper(loss_va/len(val_loader), model)
            if stopper.early_stop:
                print("[NewsOnlyTrainer] Early stopping.")
                break

        # cargar mejor peso
        model.load_state_dict(torch.load(self.model_path, map_location=device))

        # Threshold tuning (para consistencia con multi)
        best_thresh, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 17):
            preds_t = [1 if p > t else 0 for p in probs]
            f1      = f1_score(labs, preds_t)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        print(f"[Threshold tuning] Mejor umbral: {best_thresh:.2f}, F1-macro: {best_f1:.4f}")
        joblib.dump(best_thresh, self.thresh_path)  # Guardar el mejor threshold

        final_preds = [1 if p > best_thresh else 0 for p in probs]
        report_dict = classification_report(labs, final_preds, output_dict=True, zero_division=0)
        report_dict["f1_score"] = report_dict["weighted avg"]["f1-score"]
        print(classification_report(labs, final_preds, zero_division=0))

        guardar_resultado_yaml({
            "accuracy":  report_dict["accuracy"],
            "f1_score":  report_dict["f1_score"],
            "per_class": {
                "0": report_dict["0"],
                "1": report_dict["1"]
            },
            "params": {
                "learning_rate": self.learning_rate,
                "batch_size":    self.batch_size,
                "hidden_size":   self.hidden_size,
                "dropout":       self.dropout
            }
        })
        return report_dict

    # ==============================================
    # PREDICCIÓN PARA EL DÍA SIGUIENTE
    # ==============================================
    def predict_next_day(self):
        """
        Carga el modelo entrenado, embedea las noticias del último día y devuelve:
          - prob: probabilidad de que suba mañana
          - etiqueta: "SUBE" si prob>threshold, sino "BAJA"
          - sl: Stop Loss sugerido (basado en último close y multiplicador fijo, ya que no hay indicadores)
          - tp: Take Profit sugerido
          - entry_sugerido: Precio para mayor % de éxito (aproximado, ej. último close)
          - prob_mayor: Probabilidad mayor si entra en ese precio
        """
        import torch
        import joblib

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cargar threshold
        try:
            best_thresh = joblib.load(self.thresh_path)
        except FileNotFoundError:
            best_thresh = 0.5  # Fallback

        # Traer último día de datos para símbolo principal
        df = load_all_data("1d")
        simbolo = self.cfg.symbols[0]
        df_symbol = df[df["symbol"] == simbolo].sort_values("date").copy()
        if df_symbol.empty:
            raise RuntimeError("No hay datos para predecir.")

        fecha_ult = df_symbol["date"].dt.date.iloc[-1]
        last_close = df_symbol["close"].iloc[-1]

        # Embed noticias reales
        emb = self.embed_news(simbolo, fecha_ult)
        X_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)

        # Cargar modelo
        model = nn.Sequential(
            nn.Linear(self.news_emb_size, self.hidden_size),  # Ajustado a +1
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        ).to(device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()

        # Inferencia
        with torch.no_grad():
            logit = model(X_tensor)
            prob  = torch.sigmoid(logit).cpu().item()

        # Etiqueta
        etiqueta = "SUBE" if prob > best_thresh else "BAJA"

        # SL/TP simple (sin ATR, usa multiplicador fijo sobre close; ajusta según necesidades)
        atr_aprox = last_close * 0.01  # Aprox 1% volatilidad; idealmente cargar ATR de precios
        if etiqueta == "SUBE":
            sl = last_close - atr_aprox * 1.5
            tp = last_close + atr_aprox * 2.0
            entry_sugerido = last_close * 0.99  # Ej. 1% abajo para mejor entry
            prob_mayor = prob + 0.05
        else:
            sl = last_close + atr_aprox * 1.5
            tp = last_close - atr_aprox * 2.0
            entry_sugerido = last_close * 1.01  # 1% arriba para short
            prob_mayor = (1 - prob) + 0.05

        return prob, etiqueta, sl, tp, entry_sugerido, prob_mayor