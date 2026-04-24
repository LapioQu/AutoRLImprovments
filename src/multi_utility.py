"""
Модуль багатокритеріальної утиліти з ентропійною регуляризацією.

Реалізує функцію корисності:
U = w1·perf − w2·var − w3·compute − w4·switch

з автоматичною калібруванням ваг через ентропійну регуляризацію.

Гіпотеза H2: Багатокритеріальна утиліта знижує дисперсію винагороди ≥20%.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class UtilityConfig:
    """Конфігурація багатокритеріальної утиліти."""
    w_performance: float = 1.0      # w1: вага продуктивності
    w_variance: float = 0.5         # w2: вага дисперсії (штраф)
    w_compute: float = 0.1          # w3: вага обчислювальних витрат
    w_switch: float = 0.3           # w4: вага штрафу за перемикання
    entropy_regularization: float = 0.1  # Коефіцієнт ентропійної регуляризації
    normalize: bool = True          # Чи нормалізувати метрики


@dataclass
class UtilityResult:
    """Результат обчислення утиліти."""
    utility: float                  # Загальна утиліта
    components: Dict[str, float]    # Компоненти утиліти
    weights: Dict[str, float]       # Використані ваги
    is_normalized: bool             # Чи була нормалізація


class MultiCriteriaUtility:
    """
    Багатокритеріальна функція утиліти для оцінки стратегій.
    
    Математична формула:
    U(s) = w₁·P(s) - w₂·V(s) - w₃·C(s) - w₄·S(s) + λ·H(w)
    
    де:
    - P(s): продуктивність стратегії (середня винагорода)
    - V(s): дисперсія винагород (ризик)
    - C(s): обчислювальні витрати (час/крок)
    - S(s): штраф за часті перемикання
    - H(w): ентропія ваг для регуляризації
    - λ: коефіцієнт регуляризації
    
    Гіпотеза H2: Така утиліта знижує дисперсію кумулятивної винагороди.
    """
    
    def __init__(self, config: Optional[UtilityConfig] = None):
        """
        Ініціалізація багатокритеріальної утиліти.
        
        Args:
            config: Конфігурація утиліти.
        """
        self.config = config or UtilityConfig()
        self.normalization_stats: Dict[str, Tuple[float, float]] = {}
        self.n_updates = 0
        
    def update_normalization_stats(self, metrics: Dict[str, float]) -> None:
        """
        Оновлення статистики для нормалізації метрик.
        
        Використовує онлайн-алгоритм Вельфа для обчислення
        середнього та дисперсії в реальному часі.
        
        Args:
            metrics: Словник з поточними метриками.
        """
        self.n_updates += 1
        
        for key, value in metrics.items():
            if key not in self.normalization_stats:
                self.normalization_stats[key] = (value, 0.0, value, value)  # mean, M2, min, max
            else:
                mean, m2, min_val, max_val = self.normalization_stats[key]
                
                # Оновлення середнього (Welford's algorithm)
                delta = value - mean
                mean += delta / self.n_updates
                delta2 = value - mean
                m2 += delta * delta2
                
                # Оновлення мін/макс для robust нормалізації
                min_val = min(min_val, value)
                max_val = max(max_val, value)
                
                self.normalization_stats[key] = (mean, m2, min_val, max_val)
    
    def _normalize_value(self, value: float, key: str) -> float:
        """
        Нормалізація значення на основі накопиченої статистики.
        
        Використовує robust scaling: (x - min) / (max - min)
        Якщо статистики ще немає, повертає значення як є.
        
        Args:
            value: Значення для нормалізації.
            key: Ключ метрики.
            
        Returns:
            Нормалізоване значення в діапазоні [0, 1].
        """
        if not self.config.normalize or key not in self.normalization_stats:
            return value
        
        mean, m2, min_val, max_val = self.normalization_stats[key]
        
        # Уникаємо ділення на нуль
        range_val = max_val - min_val
        if range_val < 1e-8:
            return 0.5  # Центральне значення якщо всі значення однакові
        
        return (value - min_val) / range_val
    
    def compute_utility(self, 
                       performance: float,
                       reward_variance: float,
                       compute_cost: float,
                       switch_penalty: float,
                       strategy_id: Optional[str] = None) -> UtilityResult:
        """
        Обчислення багатокритеріальної утиліти.
        
        Args:
            performance: Продуктивність стратегії (середня винагорода).
            reward_variance: Дисперсія винагород.
            compute_cost: Обчислювальні витрати (сек/крок).
            switch_penalty: Штраф за перемикання (0 або 1).
            strategy_id: Опціональний ідентифікатор стратегії.
            
        Returns:
            Результат обчислення утиліти.
        """
        # Нормалізація метрик
        if self.config.normalize:
            perf_norm = self._normalize_value(performance, 'performance')
            var_norm = self._normalize_value(reward_variance, 'variance')
            comp_norm = self._normalize_value(compute_cost, 'compute')
            switch_norm = self._normalize_value(switch_penalty, 'switch')
        else:
            perf_norm = performance
            var_norm = reward_variance
            comp_norm = compute_cost
            switch_norm = switch_penalty
        
        # Обчислення компонентів утиліти
        utility_components = {
            'performance': self.config.w_performance * perf_norm,
            'variance_penalty': -self.config.w_variance * var_norm,
            'compute_penalty': -self.config.w_compute * comp_norm,
            'switch_penalty': -self.config.w_switch * switch_norm
        }
        
        # Додавання ентропійної регуляризації
        entropy_reg = self._compute_entropy_regularization()
        utility_components['entropy_regularization'] = entropy_reg
        
        # Загальна утиліта
        total_utility = sum(utility_components.values())
        
        return UtilityResult(
            utility=total_utility,
            components=utility_components,
            weights={
                'w_performance': self.config.w_performance,
                'w_variance': self.config.w_variance,
                'w_compute': self.config.w_compute,
                'w_switch': self.config.w_switch
            },
            is_normalized=self.config.normalize
        )
    
    def _compute_entropy_regularization(self) -> float:
        """
        Обчислення ентропійної регуляризації ваг.
        
        Ентропія сприяє більш рівномірному розподілу ваг,
        запобігаючи домінуванню однієї метрики.
        
        H(w) = -Σ w_i · log(w_i)
        
        Returns:
            Значення ентропійної регуляризації.
        """
        weights = np.array([
            self.config.w_performance,
            self.config.w_variance,
            self.config.w_compute,
            self.config.w_switch
        ])
        
        # Нормалізація ваг до ймовірнісного розподілу
        weight_sum = np.sum(weights)
        if weight_sum < 1e-8:
            return 0.0
        
        prob_weights = weights / weight_sum
        
        # Уникаємо log(0)
        prob_weights = np.clip(prob_weights, 1e-10, 1.0)
        
        # Обчислення ентропії
        entropy = -np.sum(prob_weights * np.log(prob_weights))
        
        # Максимальна ентропія для нормалізації
        max_entropy = np.log(len(weights))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return self.config.entropy_regularization * normalized_entropy
    
    def calibrate_weights(self, 
                         historical_data: List[Dict[str, float]],
                         target_variance_reduction: float = 0.2) -> Dict[str, float]:
        """
        Автоматична калібрування ваг через оптимізацію.
        
        Використовує історичні дані для підбору ваг, що мінімізують
        дисперсію утиліти при збереженні продуктивності.
        
        Args:
            historical_data: Список словників з метриками.
            target_variance_reduction: Цільове зниження дисперсії.
            
        Returns:
            Оптимальні ваги.
        """
        if len(historical_data) < 10:
            logger.warning("Замало даних для калібрування ваг")
            return self._get_current_weights()
        
        def objective(weight_vector: np.ndarray) -> float:
            """Цільова функція: мінімізація дисперсії утиліти."""
            # Нормалізація ваг
            w = np.abs(weight_vector)
            w = w / np.sum(w) if np.sum(w) > 0 else np.ones(4) / 4
            
            # Тимчасове оновлення ваг
            old_weights = (
                self.config.w_performance,
                self.config.w_variance,
                self.config.w_compute,
                self.config.w_switch
            )
            
            self.config.w_performance = w[0]
            self.config.w_variance = w[1]
            self.config.w_compute = w[2]
            self.config.w_switch = w[3]
            
            # Обчислення утиліт для всіх даних
            utilities = []
            for data in historical_data:
                result = self.compute_utility(
                    performance=data.get('performance', 0.0),
                    reward_variance=data.get('reward_variance', 0.0),
                    compute_cost=data.get('compute_cost', 0.0),
                    switch_penalty=data.get('switch_penalty', 0.0)
                )
                utilities.append(result.utility)
            
            # Повернення старих ваг
            (self.config.w_performance, self.config.w_variance, 
             self.config.w_compute, self.config.w_switch) = old_weights
            
            # Мінімізація дисперсії при максимізації середнього
            variance = np.var(utilities)
            mean_utility = np.mean(utilities)
            
            return variance - 0.1 * mean_utility  # Композитна ціль
        
        # Початкове наближення
        x0 = np.array([
            self.config.w_performance,
            self.config.w_variance,
            self.config.w_compute,
            self.config.w_switch
        ])
        
        # Оптимізація
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                            bounds=[(0.01, 2.0)] * 4)
            
            optimal_weights = np.abs(result.x)
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            calibrated = {
                'w_performance': float(optimal_weights[0]),
                'w_variance': float(optimal_weights[1]),
                'w_compute': float(optimal_weights[2]),
                'w_switch': float(optimal_weights[3])
            }
            
            logger.info(f"Калібровано ваги: {calibrated}")
            return calibrated
            
        except Exception as e:
            logger.warning(f"Помилка калібрування ваг: {e}")
            return self._get_current_weights()
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Отримання поточних ваг."""
        return {
            'w_performance': self.config.w_performance,
            'w_variance': self.config.w_variance,
            'w_compute': self.config.w_compute,
            'w_switch': self.config.w_switch
        }
    
    def reset(self) -> None:
        """Скидання статистики нормалізації."""
        self.normalization_stats = {}
        self.n_updates = 0


def create_utility_from_config(config_dict: Dict) -> MultiCriteriaUtility:
    """
    Фабрична функція для створення утиліти з конфігурації.
    
    Args:
        config_dict: Словник з параметрами утиліти.
        
    Returns:
        Екземпляр MultiCriteriaUtility.
    """
    config = UtilityConfig(
        w_performance=config_dict.get('w_performance', 1.0),
        w_variance=config_dict.get('w_variance', 0.5),
        w_compute=config_dict.get('w_compute', 0.1),
        w_switch=config_dict.get('w_switch', 0.3),
        entropy_regularization=config_dict.get('entropy_regularization', 0.1),
        normalize=config_dict.get('normalize', True)
    )
    
    return MultiCriteriaUtility(config)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    utility = MultiCriteriaUtility()
    
    # Симуляція даних
    print("=== Тест багатокритеріальної утиліти ===\n")
    
    for i in range(20):
        # Генерація випадкових метрик
        perf = np.random.uniform(0.3, 0.8)
        var = np.random.uniform(0.01, 0.2)
        compute = np.random.uniform(0.001, 0.01)
        switch = 1 if np.random.random() < 0.3 else 0
        
        # Оновлення статистики нормалізації
        metrics = {
            'performance': perf,
            'variance': var,
            'compute': compute,
            'switch': switch
        }
        utility.update_normalization_stats(metrics)
        
        # Обчислення утиліти
        result = utility.compute_utility(
            performance=perf,
            reward_variance=var,
            compute_cost=compute,
            switch_penalty=switch
        )
        
        print(f"Крок {i+1}: U={result.utility:.4f} "
              f"(perf={result.components['performance']:.3f}, "
              f"var={result.components['variance_penalty']:.3f})")
    
    # Тест калібрування
    print("\n=== Калібрування ваг ===")
    historical = [
        {
            'performance': np.random.uniform(0.3, 0.8),
            'reward_variance': np.random.uniform(0.01, 0.2),
            'compute_cost': np.random.uniform(0.001, 0.01),
            'switch_penalty': np.random.choice([0, 1])
        }
        for _ in range(50)
    ]
    
    optimal_weights = utility.calibrate_weights(historical)
    print(f"Оптимальні ваги: {optimal_weights}")
