# trading_bot/config.py

import os
from datetime import datetime as _dt
from pathlib import Path
import yaml

# ================================  Carga YAML
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR.parent / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Falta config.yaml en: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ================================  Directorios
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"
DB_PATH    = DATA_DIR / "historical.db"
for d in (DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ================================  API Keys
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY") or cfg["api"].get("alpha_vantage_api_key")
FMP_API_KEY           = os.getenv("FMP_API_KEY")           or cfg["api"].get("fmp_api_key")
FINNHUB_API_KEY       = os.getenv("FINNHUB_API_KEY")       or cfg["api"].get("finnhub_api_key")
TWELVE_DATA_API_KEY   = os.getenv("TWELVE_DATA_API_KEY")   or cfg["api"].get("twelve_data_api_key")

# ================================  Símbolos y Fechas
SYMBOLS    = cfg.get("symbols", [])
TIMEFRAMES = cfg.get("resolutions", ["1h", "4h", "1d"])

# convierte a datetime
START_DATE = _dt.fromisoformat(cfg["date"]["start_cache"])
END_DATE   = _dt.utcnow()

# ================================  Parámetros indicadores
p = cfg.get("parameters", {})
SMA_SHORT      = p.get("sma_short", 50)  # Default si no en YAML
SMA_LONG       = p.get("sma_long", 200)
RSI_WINDOW     = p.get("rsi_window", 14)
MACD_FAST      = p.get("macd_fast", 12)
MACD_SLOW      = p.get("macd_slow", 26)
MACD_SIGNAL    = p.get("macd_signal", 9)
BOLLINGER_WIN  = p.get("bollinger_window", 20)
ATR_WINDOW     = p.get("atr_window", 14)
MIN_WIN_RATE   = p.get("target_accuracy", 0.6)
RISK_REWARD    = p.get("risk_reward", 2.0)

# ================================  Parámetros de SL/TP y Riesgo
# porcentaje de riesgo (stop loss) por operación (si no está en YAML, usa 1% por defecto)
STOP_LOSS_RATIO    = p.get("stop_loss_ratio", 0.01)
# take profit = stop_loss_pct * risk_reward
TAKE_PROFIT_RATIO  = STOP_LOSS_RATIO * RISK_REWARD
# tamaño de posición por defecto (unidades de base)
DEFAULT_POSITION_SIZE = cfg.get("risk", {}).get("default_position_size", 1000)

# ================================  Parámetros de entrenamiento
t = cfg.get("training", {})
NUM_EPOCHS                = t.get("num_epochs", 20)
BATCH_SIZE                = t.get("batch_size", 32)
SEQUENCE_LENGTH           = t.get("sequence_length", 24)
LEARNING_RATE             = t.get("learning_rate", 0.001)
BIDIRECTIONAL             = t.get("bidirectional", False)
HIDDEN_SIZE               = t.get("hidden_size", 50)
NUM_LAYERS                = t.get("num_layers", 2)
EARLY_STOPPING_PATIENCE   = t.get("early_stopping_patience", 5)
OVERSAMPLE                = t.get("oversample", False)
UNDERSAMPLE               = t.get("undersample", False)
VALIDATION_SPLIT          = t.get("validation_split", 0.2)

# ================================  Rutas y logs
MODEL_SAVE_PATH = MODELS_DIR / "trading_model.pkl"
LOG_FILE        = LOGS_DIR   / "bot.log"

# ================================  Clase de Config
class TradingConfig:
    def __init__(self):
        # Proveedores
        self.price_provider   = cfg["api"]["price_provider"]
        self.news_provider    = cfg["api"]["news_provider"]
        # API Keys
        self.alpha_vantage    = ALPHA_VANTAGE_API_KEY
        self.fmp              = FMP_API_KEY
        self.finnhub          = FINNHUB_API_KEY
        self.news_api_key     = cfg["api"].get("newsapi_key")
        self.twitter_bearer   = cfg["api"].get("twitter_bearer")
        self.alphasense_key   = cfg["api"].get("alpha_sense_key")
        self.twelve_data      = TWELVE_DATA_API_KEY

        # Fechas y símbolos
        self.symbols          = SYMBOLS
        self.timeframes       = TIMEFRAMES
        self.start_date       = START_DATE
        self.end_date         = END_DATE

        # Indicadores
        self.sma_short        = SMA_SHORT
        self.sma_long         = SMA_LONG
        self.rsi_window       = RSI_WINDOW
        self.macd_fast        = MACD_FAST
        self.macd_slow        = MACD_SLOW
        self.macd_signal      = MACD_SIGNAL
        self.bollinger_window = BOLLINGER_WIN
        self.atr_window       = ATR_WINDOW

        # Parámetros de rendimiento y riesgo
        self.min_win_rate        = MIN_WIN_RATE
        self.risk_reward         = RISK_REWARD
        self.stop_loss_ratio     = STOP_LOSS_RATIO
        self.take_profit_ratio   = TAKE_PROFIT_RATIO
        self.default_position_size = DEFAULT_POSITION_SIZE

        # Hiperparámetros de entrenamiento
        self.num_epochs              = NUM_EPOCHS
        self.batch_size              = BATCH_SIZE
        self.sequence_length         = SEQUENCE_LENGTH
        self.learning_rate           = LEARNING_RATE
        self.bidirectional           = BIDIRECTIONAL
        self.hidden_size             = HIDDEN_SIZE
        self.num_layers              = NUM_LAYERS
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        self.oversample              = OVERSAMPLE
        self.undersample             = UNDERSAMPLE
        self.validation_split        = VALIDATION_SPLIT

        # Rutas
        self.base_dir         = BASE_DIR
        self.data_dir         = DATA_DIR
        self.cache_dir        = CACHE_DIR
        self.models_dir       = MODELS_DIR
        self.logs_dir         = LOGS_DIR
        self.db_path          = DB_PATH
        self.model_save_path  = MODEL_SAVE_PATH
        self.log_file         = LOG_FILE