# trading_bot/src/utils/metric_watcher.py

from trading_bot.src.hyperparameter_search import run_hyperparameter_search

class MetricWatcher:
    def __init__(self, tolerance=0.05, min_accuracy=0.60, min_f1_score=0.55):
        self.tolerance = tolerance  # 5% = 0.05
        self.last_accuracy = None
        self.last_f1_score = None
        self.min_f1_score = min_f1_score
        self.min_accuracy = min_accuracy

    def should_trigger(self, current_metrics: dict) -> bool:
        """
        Evalúa si el accuracy o el f1_score bajaron más de lo permitido o están por debajo de mínimos.
        """
        if "accuracy" not in current_metrics or "f1_score" not in current_metrics:
            print("[MetricWatcher] ⚠️ Métricas incompletas en el dict proporcionado.")
            return False

        current_accuracy = current_metrics["accuracy"]
        current_f1 = current_metrics["f1_score"]

        # Check mínimos absolutos
        if current_accuracy < self.min_accuracy or current_f1 < self.min_f1_score: 
            print(f"[MetricWatcher] Métricas por debajo del mínimo. Accuracy={current_accuracy:.2f}, F1={current_f1:.2f}")
            return True

        if self.last_accuracy is None or self.last_f1_score is None:
            # Primera vez: solo guardar y no disparar Optuna
            self.last_accuracy = current_accuracy
            self.last_f1_score = current_f1
            return False

        # Calcular % de caída
        accuracy_drop = (self.last_accuracy - current_accuracy) / max(self.last_accuracy, 1e-8)
        f1_drop = (self.last_f1_score - current_f1) / max(self.last_f1_score, 1e-8)

        print(f"[MetricWatcher] Comparando métricas... Accuracy Drop: {accuracy_drop*100:.2f}%, F1 Drop: {f1_drop*100:.2f}%")

        # Actualizar para próxima vez
        self.last_accuracy = current_accuracy
        self.last_f1_score = current_f1

        return accuracy_drop > self.tolerance or f1_drop > self.tolerance