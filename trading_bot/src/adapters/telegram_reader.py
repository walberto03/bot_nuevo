```python
# bot_nuevo/trading_bot/src/adapters/telegram_reader.py
import re
import logging
import os
from telegram.ext import Updater, MessageHandler, Filters
import MetaTrader5 as mt5
from trading_bot.src.data.db_manager import save_trade_result
from trading_bot.config import cfg

logging.basicConfig(level=logging.INFO)

class TelegramReader:
    def __init__(self):
        self.token = cfg["notifications"].get("telegram_token") or os.getenv("TELEGRAM_TOKEN")
        self.chat_id = cfg["notifications"].get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID")
        self.updater = Updater(self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.signals = []
        self.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.handle_message))

    def start_polling(self):
        logging.info("Iniciando polling de Telegram...")
        self.updater.start_polling()
        self.updater.idle()

    def handle_message(self, update, context):
        chat_id = update.message.chat_id
        logging.info(f"Chat ID: {chat_id}")
        message = update.message.text
        signal = self.parse_signal(message)
        if signal:
            self.signals.append(signal)
            logging.info(f"Señal parseada: {signal}")
            self.execute_trade(signal)

    def parse_signal(self, message):
        # Formato: #XAUUSD BUY/SELL @price1-price2 SL price TP price TP price TP price TP open
        pattern = r"#XAUUSD (BUY|SELL) @(\d+)-(\d+) SL (\d+) TP (\d+) TP (\d+) TP (\d+) TP (open)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            action = match.group(1)
            entry_low = float(match.group(2))
            entry_high = float(match.group(3))
            entry = (entry_low + entry_high) / 2
            sl = float(match.group(4))
            tp1 = float(match.group(5))
            tp2 = float(match.group(6))
            tp3 = float(match.group(7))
            tp_open = match.group(8) == "open"
            return {"action": action, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "tp_open": tp_open}
        return None

    def get_signals_sentiment(self):
        buys = sum(1 for s in self.signals if s['action'] == 'BUY')
        sells = sum(1 for s in self.signals if s['action'] == 'SELL')
        return (buys - sells) / max(1, buys + sells)

    def execute_trade(self, signal):
        mt5_login = os.getenv("MT5_LOGIN")
        mt5_password = os.getenv("MT5_PASSWORD")
        mt5_server = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
        if not mt5.initialize(login=int(mt5_login) if mt5_login else 0, password=mt5_password, server=mt5_server):
            logging.error("MT