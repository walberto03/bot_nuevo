import re
from telegram.ext import Updater, MessageHandler, Filters
from telegram import Bot
import MetaTrader5 as mt5
from trading_bot.config import cfg

class TelegramReader:
    def __init__(self):
        self.token = cfg["notifications"].get("telegram_token")
        self.chat_id = cfg["notifications"].get("telegram_chat_id")
        self.bot = Bot(token=self.token)
        self.signals = []

    def parse_signal(self, message):
        pattern = r"(Buy|Sell) (Gold|XAUUSD) at (\d+\.?\d*) SL (\d+\.?\d*) TP (\d+\.?\d*)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            action = match.group(1)
            asset = match.group(2)
            price = float(match.group(3))
            sl = float(match.group(4))
            tp = float(match.group(5))
            return {"action": action, "asset": asset, "price": price, "sl": sl, "tp": tp}
        return None

    def handle_message(self, update, context):
        message = update.message.text
        signal = self.parse_signal(message)
        if signal:
            self.signals.append(signal)
            print(f"Señal parseada: {signal}")
            self.execute_trade(signal)

    def execute_trade(self, signal):
        if not mt5.initialize():
            print("MT5 no inicializado")
            return
        symbol = "XAUUSD" if signal["asset"] == "Gold" else signal["asset"]
        lot = 0.1
        price = mt5.symbol_info_tick(symbol).ask if signal["action"] == "Buy" else mt5.symbol_info_tick(symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if signal["action"] == "Buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": signal["sl"],
            "tp": signal["tp"],
            "magic": 123456,
            "comment": "Signal from Telegram",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Error en trade: {result.comment}")
        else:
            print(f"Trade demo ejecutado: {signal}")

    def start_polling(self):
        updater = Updater(self.token, use_context=True)
        updater.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.handle_message))
        updater.start_polling()
        updater.idle()

    def get_signals_sentiment(self):
        if not self.signals:
            return 0.0
        buys = sum(1 for s in self.signals if s["action"] == "Buy")
        sells = len(self.signals) - buys
        return (buys - sells) / max(len(self.signals), 1)
