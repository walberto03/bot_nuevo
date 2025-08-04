import sqlite3, json
import pandas as pd
from datetime import datetime

DB = "bot_cache.db"

class PriceCache:
    def __init__(self):
        self.conn = sqlite3.connect(DB)
        self.conn.execute("""
          CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT, date TEXT, open REAL, high REAL,
            low REAL, close REAL, volume REAL,
            PRIMARY KEY(symbol, date)
          )""")

    def get_last_date(self, symbol):
        cur = self.conn.execute(
            "SELECT MAX(date) FROM prices WHERE symbol=?", (symbol,))
        r = cur.fetchone()[0]
        return datetime.fromisoformat(r) if r else None

    def append_prices(self, symbol, df: pd.DataFrame):
        df = df.reset_index().rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df["symbol"] = symbol
        df[["symbol","date","open","high","low","close","volume"]].to_sql(
            "prices", self.conn, if_exists="append", index=False)

    def get_prices(self, symbol, start, end):
        q = """
          SELECT date,open,high,low,close,volume
          FROM prices WHERE symbol=? AND date BETWEEN ? AND ?
          """
        return pd.read_sql(q, self.conn, params=(symbol, start, end),
                           parse_dates=["date"])

class NewsCache:
    def __init__(self):
        self.conn = sqlite3.connect(DB)
        self.conn.execute("""
          CREATE TABLE IF NOT EXISTS news (
            id TEXT PRIMARY KEY, date TEXT, title TEXT,
            content TEXT, data JSON
          )""")

    def get_last_date(self):
        cur = self.conn.execute("SELECT MAX(date) FROM news")
        r = cur.fetchone()[0]
        return datetime.fromisoformat(r) if r else None

    def append_news(self, articles: list):
        to_insert = []
        for art in articles:
            to_insert.append((
                art["url"],
                art["publishedAt"][:10],
                art["title"],
                art.get("content",""),
                json.dumps(art)
            ))
        self.conn.executemany(
            "INSERT OR IGNORE INTO news VALUES (?,?,?,?,?)", to_insert)
        self.conn.commit()

    def get_news(self, date):
        cur = self.conn.execute(
            "SELECT data FROM news WHERE date=?", (date,))
        return [json.loads(r[0]) for r in cur]
