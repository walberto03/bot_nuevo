# trading_bot/src/multi_timescale_trainer.py

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from collections import Counter
from imblearn.over_sampling import SMOTE
import joblib
from transformers import AutoTokenizer, AutoModel

from trading_bot.config import TradingConfig
from trading_bot.src.pattern_analyzer import PatternAnalyzer
from trading_bot.src.data.db_manager import load_all_data
from trading_bot.src.utils.early_stopping import EarlyStopping
from trading_bot.src.utils.losses import FocalLoss
from trading_bot.src.multi_timescale_trainer import MultiModalAttentionModel



class MultiTimescaleTrainer:
    def __init__(self,
                 learning_rate=None,
                 batch_size=None,
                 sequence_length=None,
                 hidden_size=None,
                 num_layers=None,
                 attention_heads=None,
                 attention_feedforward_dim=None,
                 dropout=None,
                 pos_weight_factor=None,
                 early_stopping_patience=None,
                 classification_threshold=None,
                 weight_decay=None,
                 focal_gamma=None,
                 optuna_mode=False):

        self.cfg = TradingConfig()
        self.pattern = PatternAnalyzer()
        self.optuna_mode = optuna_mode

        # Hiperparámetros
        self.learning_rate   = learning_rate   or self.cfg.learning_rate
        self.batch_size      = batch_size      or self.cfg.batch_size
        self.sequence_length = sequence_length or self.cfg.sequence_length
        self.hidden_size     = hidden_size     or self.cfg.hidden_size
        self.num_layers      = num_layers      or self.cfg.num_layers

        # Attention
        self.attention_heads = attention_heads or getattr(self.cfg, "attention_heads", 8)
        self.attention_feedforward_dim = attention_feedforward_dim or getattr(self.cfg, "attention_ff_dim", 256)

        # Nuevos
        self.dropout                  = dropout or getattr(self.cfg, "dropout", 0.1)
        self.pos_weight_factor        = pos_weight_factor or getattr(self.cfg, "pos_weight_factor", 1.0)
        self.early_stopping_patience  = early_stopping_patience or self.cfg.early_stopping_patience
        self.classification_threshold = classification_threshold or getattr(self.cfg, "classification_threshold", 0.5)
        self.weight_decay             = weight_decay or getattr(self.cfg, "weight_decay", 0.0)
        self.focal_gamma              = focal_gamma or getattr(self.cfg, "focal_gamma", 2.0)

        self.input_size = len([
            "rsi","macd","signal","bb_up","bb_dn",
            f"sma_{self.cfg.sma_short}", f"sma_{self.cfg.sma_long}", "atr",
            "ema_short", "ema_long", "vwap", "fib_382", "fib_618"
        ])

        self.model_path  = self.cfg.models_dir / "multi_lstm_attention_model.pt"
        self.scaler_path = self.cfg.models_dir / "multi_scaler.pkl"

        self.tokenizer     = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.news_encoder  = AutoModel.from_pretrained("distilbert-base-uncased")

    def prepare_data(self):
        df = load_all_data()
        df = df[df['symbol'].isin(self.cfg.symbols)].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol','date'])

        labels = self._build_daily_labels(df)

        Xs, Ns, ys = [], [], []
        for sym in df['symbol'].unique():
            sub = self.pattern.calculate_technical_indicators(
                df[df['symbol']==sym]
            ).dropna().rename(columns=str.lower)
            for i in range(len(sub)-self.sequence_length):
                win = sub.iloc[i:i+self.sequence_length]
                dt  = win['date'].dt.date.iloc[-1]
                if dt in labels.get(sym, {}):
                    Xs.append(win[[
                        "rsi","macd","signal","bb_up","bb_dn",
                        f"sma_{self.cfg.sma_short}", f"sma_{self.cfg.sma_long}", "atr",
                        "ema_short", "ema_long", "vwap", "fib_382", "fib_618"
                    ]].values)
                    Ns.append(self.embed_text(f"News about {sym} on {dt}"))
                    ys.append(labels[sym][dt])

        X = np.array(Xs)
        N = np.array(Ns)
        y = np.array(ys)

        print("[Clase Balance] Original:", Counter(y))
        if getattr(self.cfg, "oversample", True):
            sm = SMOTE()
            X_flat, y_sm = sm.fit_resample(X.reshape(len(X),-1), y)
            N_sm = resample(N, n_samples=len(y_sm))
            X = X_flat.reshape(-1, self.sequence_length, self.input_size)
            y = y_sm
            N = N_sm
            print("[Clase Balance] SMOTE:", Counter(y))

        scaler = MinMaxScaler().fit(X.reshape(-1,self.input_size))
        joblib.dump(scaler, self.scaler_path)
        X = scaler.transform(X.reshape(-1,self.input_size)).reshape(X.shape)

        X_t = [torch.tensor(x,dtype=torch.float32) for x in X]
        N_t = torch.tensor(N,dtype=torch.float32)
        y_t = torch.tensor(y,dtype=torch.float32).unsqueeze(1)
        data = list(zip(X_t, N_t, y_t))

        split = int(len(data) * self.cfg.validation_split)
        train, val = data[:-split], data[-split:]

        # Sampler balanceado
        labs = [int(b.item()) for _, _, b in train]
        counts = Counter(labs)
        wts = [1.0 / counts[l] for l in labs]
        sampler = WeightedRandomSampler(wts, num_samples=len(wts), replacement=True)

        train_loader = DataLoader(train,
                                  batch_size=self.batch_size,
                                  sampler=sampler,
                                  collate_fn=self.collate_fn)
        val_loader   = DataLoader(val,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  collate_fn=self.collate_fn)
        return train_loader, val_loader

    def collate_fn(self, batch):
        X, N, y = zip(*batch)
        return (torch.stack(X), torch.stack(N)), torch.stack(y)

    def embed_text(self, txt):
        toks = self.tokenizer(txt,
                              return_tensors="pt",
                              truncation=True,
                              padding=True,
                              max_length=64)
        out = self.news_encoder(**toks)
        return out.last_hidden_state[:,0,:].detach().squeeze(0).numpy()

    def _build_daily_labels(self, df):
        df = df.copy()
        df.columns = df.columns.str.lower()
        df['date'] = pd.to_datetime(df['date'])
        daily = df.groupby(['symbol', df['date'].dt.date])['close'].last()
        nxt   = daily.groupby(level=0).shift(-1)
        m     = daily.to_frame('close').join(nxt.to_frame('next_close'))
        m['target'] = (m['next_close'] > m['close']).astype(int)

        targs = {}
        for (symbol, fecha), row in m.iterrows():
            targs.setdefault(symbol, {})[fecha] = int(row['target'])
        return targs

    def train(self):
        print("[MultiTimescaleTrainer] Entrenando…")
        tr, va = self.prepare_data()
        if tr is None:
            print("[MultiTimescaleTrainer] No hay datos.")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalAttentionModel(
            price_input_size=self.input_size,
            news_emb_size    =self.news_encoder.config.hidden_size,
            hidden_size      =self.hidden_size,
            num_layers       =self.num_layers,
            attention_heads  =self.attention_heads,
            attention_ff_dim =self.attention_feedforward_dim,
            dropout          =self.dropout
        ).to(device)

        # Pos weight dinámico
        labels_list = []
        for (_, _), yb in tr:
            labels_list += yb.flatten().tolist()
        n0, n1 = labels_list.count(0), labels_list.count(1)
        pw      = (n0 / max(1, n1)) * self.pos_weight_factor

        criterion = FocalLoss(
            gamma     =self.focal_gamma,
            alpha     =1.0,
            reduction ="mean",
            pos_weight=torch.tensor([pw], device=device)
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            verbose =True,
            path    =str(self.model_path)
        )

        for epoch in range(self.cfg.num_epochs):
            model.train()
            loss_tr = 0
            for (Xp, Xn), yb in tr:
                Xp, Xn, yb = Xp.to(device), Xn.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xp, Xn)
                l   = criterion(out, yb)
                l.backward()
                optimizer.step()
                loss_tr += l.item()
            loss_tr /= len(tr)

            model.eval()
            loss_val = 0
            with torch.no_grad():
                for (Xp, Xn), yb in va:
                    Xp, Xn, yb = Xp.to(device), Xn.to(device), yb.to(device)
                    loss_val += criterion(model(Xp, Xn), yb).item()
            loss_val /= len(va)

            print(f"Epoch {epoch+1}/{self.cfg.num_epochs} — "
                  f"Train: {loss_tr:.4f} — Val: {loss_val:.4f}")
            stopper(loss_val, model)
            if stopper.early_stop:
                break

        model.load_state_dict(torch.load(self.model_path))
        print("[MultiTimescaleTrainer] ✅ Modelo guardado.")

        # Threshold tuning
        all_probs, all_labels = [], []
        with torch.no_grad():
            for (Xp, Xn), yb in va:
                Xp, Xn = Xp.to(device), Xn.to(device)
                p = torch.sigmoid(model(Xp, Xn)).cpu().numpy().flatten()
                all_probs.extend(p.tolist())
                all_labels.extend(yb.cpu().numpy().flatten().tolist())

        best_thresh, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 17):
            preds_t = [1 if p > t else 0 for p in all_probs]
            f1      = f1_score(all_labels, preds_t)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        print(f"[Threshold tuning] Mejor umbral: {best_thresh:.2f}, F1-macro: {best_f1:.4f}")
        final_preds = [1 if p > best_thresh else 0 for p in all_probs]
        print("[MultiTimescaleTrainer] Reporte final:")
        print(classification_report(all_labels, final_preds, zero_division=0))

        return {
            "accuracy": float((np.array(final_preds) == np.array(all_labels)).mean()),
            "f1_score": float(best_f1)
        }

    # ==============================================
    # PREDICCIÓN PARA EL DÍA SIGUIENTE
    # ==============================================
    def predict_next_day(self):
        """
        Carga el scaler y el modelo entrenado, toma las últimas `sequence_length`
        barras diarias (técnicos + noticias) y devuelve:
          - prob: probabilidad de que suba mañana
          - etiqueta: "SUBE" si prob>threshold, sino "BAJA"
        """
        import joblib
        import torch
        from trading_bot.src.data.db_manager import load_all_data
        from trading_bot.src.multi_timescale_trainer import MultiModalAttentionModel

        # 1) Cargamos el scaler
        scaler = joblib.load(self.scaler_path)

        # 2) Traemos la serie diaria para el primer símbolo
        df = load_all_data("1d")
        simbolo = self.cfg.symbols[0]
        df_symbol = df[df["symbol"] == simbolo].sort_values("date").copy()
        if len(df_symbol) < self.sequence_length:
            raise RuntimeError(f"No hay suficientes datos para predecir ({self.sequence_length} barras).")

        # 3) Última ventana y cálculo de indicadores
        ventana = df_symbol.iloc[-self.sequence_length:].copy()
        ventana = self.pattern.calculate_technical_indicators(ventana)\
                                 .dropna()\
                                 .rename(columns=str.lower)
        if len(ventana) < self.sequence_length:
            raise RuntimeError("Tras indicadores quedan menos filas de las necesarias.")

        # 4) Tensor de precios
        X_vals = ventana[[
            "rsi","macd","signal","bb_up","bb_dn",
            f"sma_{self.cfg.sma_short}", f"sma_{self.cfg.sma_long}", "atr",
            "ema_short", "ema_long", "vwap", "fib_382", "fib_618"
        ]].values
        X_norm = scaler.transform(X_vals.reshape(-1, self.input_size))\
                         .reshape(self.sequence_length, self.input_size)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)

        # 5) Tensor de noticias
        fecha_ult = ventana["date"].dt.date.iloc[-1].strftime("%Y-%m-%d")
        texto = f"News about {simbolo} on {fecha_ult}"
        emb = self.embed_text(texto)
        N_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        # 6) Reconstrucción y carga del modelo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalAttentionModel(
            price_input_size=self.input_size,
            news_emb_size    =self.news_encoder.config.hidden_size,
            hidden_size      =self.hidden_size,
            num_layers       =self.num_layers,
            attention_heads  =self.attention_heads,
            attention_ff_dim =self.attention_feedforward_dim,
            dropout          =self.dropout
        ).to(device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()

        # 7) Inferencia
        X_tensor = X_tensor.to(device)
        N_tensor = N_tensor.to(device)
        with torch.no_grad():
            logit = model(X_tensor, N_tensor)
            prob  = torch.sigmoid(logit).cpu().item()

        # 8) Etiqueta según threshold
        etiqueta = "SUBE" if prob > self.classification_threshold else "BAJA"

        return prob, etiqueta


class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + self.drop(a))
        f    = self.ffn(x)
        return self.norm2(x + self.drop(f))


class MultiModalAttentionModel(nn.Module):
    def __init__(self,
                 price_input_size,
                 news_emb_size,
                 hidden_size=128,
                 num_layers=2,
                 attention_heads=4,
                 attention_ff_dim=128,
                 dropout=0.1):
        super().__init__()
        if hidden_size % attention_heads:
            attention_heads = max(
                h for h in range(1, hidden_size+1)
                if hidden_size % h == 0
            )
        self.price_enc = nn.LSTM(price_input_size, hidden_size,
                                 num_layers=num_layers, batch_first=True)
        self.news_proj = nn.Linear(news_emb_size, hidden_size)
        self.transf    = CustomTransformerBlock(
            hidden_size, attention_heads, attention_ff_dim, dropout
        )
        self.fc        = nn.Linear(hidden_size, 1)

    def forward(self, price_seq, news_emb):
        pe, _   = self.price_enc(price_seq)
        npj     = self.news_proj(news_emb).unsqueeze(1)
        x       = torch.cat([pe, npj], dim=1)
        x       = self.transf(x)
        return self.fc(x[:, -1, :])