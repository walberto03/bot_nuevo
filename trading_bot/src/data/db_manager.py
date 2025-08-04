# trading_bot/src/data/db_manager.py

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from trading_bot.config import TradingConfig

def init_db():
    """
    Crea las tablas 'prices' y 'news' si no existen.
    """
    db_path = Path(TradingConfig().db_path)
    conn    = sqlite3.connect(db_path)
    cursor  = conn.cursor()

    # --- Tabla de precios ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            resolution TEXT,
            PRIMARY KEY (date, symbol, source, resolution)
        )
    ''')

    # --- Tabla de noticias ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (
            date TEXT,
            symbol TEXT,
            title TEXT,
            description TEXT,
            url TEXT,
            sentiment REAL,
            source TEXT,
            PRIMARY KEY (date, symbol, title)
        )
    ''')

    conn.commit()
    conn.close()


def save_prices(symbol: str, df: pd.DataFrame, source: str = "alpha", resolution: str = "1d"):
    """
    Inserta datos de precios en la tabla 'prices'.
    Espera un DataFrame con índice datetime y columnas Open/High/Low/Close/Volume.
    """
    db_path = Path(TradingConfig().db_path)
    conn    = sqlite3.connect(db_path)
    data    = df.copy().reset_index()
    data.rename(columns={
        'index': 'datetime',
        'Open':  'open',
        'High':  'high',
        'Low':   'low',
        'Close': 'close',
        'Volume':'volume'
    }, inplace=True)
    data['symbol']     = symbol
    data['source']     = source
    data['resolution'] = resolution
    data['date']       = pd.to_datetime(data['datetime']).dt.strftime('%Y-%m-%d')

    try:
        data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source', 'resolution']] \
            .to_sql('prices', conn, if_exists='append', index=False)
    except Exception as e:
        print(f"[save_prices] Error al guardar: {e}")

    conn.commit()
    conn.close()


def load_all_data(timeframe: str = None) -> pd.DataFrame:
    """
    Carga todos los precios desde la tabla 'prices'.
    Si se pasa un timeframe (por ejemplo "1d" o "4h"), filtra por esa resolución.
    """
    db_path = Path(TradingConfig().db_path)
    conn    = sqlite3.connect(db_path)
    query   = "SELECT * FROM prices"
    if timeframe:
        query += f" WHERE resolution = '{timeframe}'"
    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = np.nan

    df.rename(columns={
        'open':   'Open',
        'high':   'High',
        'low':    'Low',
        'close':  'Close',
        'volume': 'Volume'
    }, inplace=True)

    df.columns = [
        col.capitalize() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] else col
        for col in df.columns
    ]
    return df


def load_grouped_data(timeframe: str = None) -> dict[str, pd.DataFrame]:
    """
    Devuelve un diccionario {symbol: DataFrame_ordenado_por_fecha} usando load_all_data().
    """
    df      = load_all_data(timeframe)
    grouped = {}
    for symbol, group in df.groupby("symbol"):
        grouped[symbol] = group.sort_values("date")
    return grouped


# -----------------------------
# Ahora: save_news corregido
# -----------------------------

def save_news(symbol: str, date: str, articles: list[dict], sentiment_score: float, source: str = "newsapi"):
    """
    Guarda en la tabla 'news' cada artículo para un día concreto.

    Parámetros:
      - symbol:          e.g. "EURUSD"
      - date:            e.g. "2022-01-01"  (string en formato 'YYYY-MM-DD')
      - articles:        lista de dicts, cada dict con keys 'title', 'description' y opcionalmente 'url'
      - sentiment_score: float con el promedio de sentimiento de ese conjunto
      - source:          identificador de origen de noticias (por defecto "newsapi")

    Crea primero la tabla 'news' si no existe, y luego hace INSERT IGNORE por
    PRIMARY KEY (date, symbol, title) para no duplicar.
    """
    db_path = Path(TradingConfig().db_path)
    conn    = sqlite3.connect(db_path)
    cursor  = conn.cursor()

    # Asegurarnos de que la tabla 'news' existe (en caso de que init_db() no se haya llamado):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (
            date TEXT,
            symbol TEXT,
            title TEXT,
            description TEXT,
            url TEXT,
            sentiment REAL,
            source TEXT,
            PRIMARY KEY (date, symbol, title)
        )
    ''')
    conn.commit()

    # Construir lista de tuplas a insertar:
    # Cada artículo puede venir con keys 'title', 'description', 'url'.
    rows_to_insert = []
    for art in articles:
        title       = art.get("title", "")[:200]        # limitar longitud
        description = art.get("description", "")[:500]
        url         = art.get("url", "")
        rows_to_insert.append((date, symbol, title, description, url, sentiment_score, source))

    # Insertar todas las filas (si ya existiera una noticia con misma PK, se ignora):
    for r in rows_to_insert:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO news
                    (date, symbol, title, description, url, sentiment, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', r)
        except Exception as e:
            print(f"[save_news] Error al insertar noticia '{r[2]}': {e}")

    conn.commit()
    conn.close()
    print(f"[save_news] Guardadas {len(rows_to_insert)} filas de noticias para {symbol} (source={source})")


def load_all_news(symbol: str = None, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Carga todas las noticias de la tabla 'news'.
    - Si se pasa 'symbol', filtra por ese símbolo.
    - Si se pasan 'start' y 'end' (fechas 'YYYY-MM-DD'), filtra por rango de fecha.
    """
    db_path = Path(TradingConfig().db_path)
    conn    = sqlite3.connect(db_path)

    query = "SELECT * FROM news"
    conds = []
    if symbol:
        conds.append(f"symbol = '{symbol}'")
    if start:
        conds.append(f"date >= '{start}'")
    if end:
        conds.append(f"date <= '{end}'")
    if conds:
        query += " WHERE " + " AND ".join(conds)

    df = pd.read_sql(query, conn, parse_dates=['date'])
    conn.close()
    return df
# trading_bot/src/data/db_manager.py (añade al final)
def save_trade_result(symbol, date, prediction, real, prob, sl, tp, outcome):
    db_path = Path(TradingConfig().db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_results (
            date TEXT,
            symbol TEXT,
            prediction TEXT,
            real TEXT,
            probability REAL,
            sl REAL,
            tp REAL,
            outcome INTEGER,
            PRIMARY KEY (date, symbol)
        )
    ''')
    cursor.execute('''
        INSERT OR REPLACE INTO trade_results (date, symbol, prediction, real, probability, sl, tp, outcome)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (date.strftime('%Y-%m-%d'), symbol, prediction, real, prob, sl, tp, outcome))
    conn.commit()
    conn.close()
    print(f"[TradeResult] Guardado resultado para {symbol} en {date}")