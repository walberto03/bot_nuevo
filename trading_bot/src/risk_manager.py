from trading_bot.config import TradingConfig

class RiskManager:
    """
    Gestiona el tamaño de la posición, stop loss y take profit
    según la configuración de riesgo.
    """
    def __init__(self, config: TradingConfig):
        self.cfg = config
        # Relación riesgo:beneficio
        self.risk_reward = config.risk_reward
        # Exposición máxima del capital por operación (porcentaje)
        self.max_exposure = getattr(config, 'max_exposure', 0.05)
        # Proporción de stop loss (ej. 0.01 = 1%)
        self.stop_loss_ratio = config.stop_loss_ratio
        # Proporción de take profit calculada desde stop_loss_ratio y risk_reward
        self.take_profit_ratio = config.take_profit_ratio
        # Tamaño de posición por defecto (unidades de la divisa base)
        self.default_position_size = config.default_position_size

    def compute_stop_loss_price(self, entry_price: float) -> float:
        """Calcula el precio de stop loss dado el precio de entrada."""
        return entry_price * (1 - self.stop_loss_ratio)

    def compute_take_profit_price(self, entry_price: float) -> float:
        """Calcula el precio de take profit dado el precio de entrada."""
        return entry_price * (1 + self.take_profit_ratio)

    def compute_position_size(self, account_balance: float, entry_price: float = None) -> float:
        """
        Calcula el tamaño de la posición en base al balance de cuenta y
        al stop loss. Si existe default_position_size, se usa directamente.
        """
        # Si se definió un tamaño fijo en configuración, usarlo
        if self.default_position_size:
            return self.default_position_size

        if entry_price is None:
            raise ValueError("Debe indicar entry_price para cálcular el tamaño de posición.")

        # Riesgo máximo en valor absoluto
        risk_amount = account_balance * self.max_exposure
        # Pérdida por unidad si se activa stop loss
        loss_per_unit = entry_price * self.stop_loss_ratio
        # Unidades a comprar/vender para no exceder el riesgo máximo
        size = risk_amount / loss_per_unit
        return size

    def summary(self, entry_price: float, account_balance: float) -> dict:
        """Retorna un resumen de parámetros de la orden: size, SL, TP."""
        size = self.compute_position_size(account_balance, entry_price)
        stop_price = self.compute_stop_loss_price(entry_price)
        tp_price = self.compute_take_profit_price(entry_price)
        return {
            'entry_price': entry_price,
            'position_size': size,
            'stop_loss_price': stop_price,
            'take_profit_price': tp_price
        }
