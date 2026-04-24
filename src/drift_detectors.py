"""
Модуль детекторів дрейфу для виявлення режимних зсувів у нестаціонарних середовищах.

Реалізує:
- ADWIN (Adaptive Windowing) - адаптивне вікно для детекції змін
- Page-Hinkley Test - кумулятивна сума відхилень
- Інтеграція з river library для ефективності

Використовується в AdaptiveThresholdLCB для динамічної калібрування порогу перемикання.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionResult:
    """Результат детекції дрейфу."""
    drift_detected: bool
    confidence: float  # Впевненість у детекції (0.0-1.0)
    drift_magnitude: float  # Величина зсуву
    timestamp: int  # Час детекції
    additional_info: dict  # Додаткова інформація


class BaseDriftDetector(ABC):
    """
    Базовий клас для детекторів дрейфу.
    
    Всі детектори реалізують онлайн-обробку даних з постійною пам'яттю.
    """
    
    def __init__(self, delta: float = 0.002):
        """
        Ініціалізація базового детектора.
        
        Args:
            delta: Рівень значущості для детекції дрейфу (менше = чутливіше).
        """
        self.delta = delta
        self.n_samples = 0
        
    @abstractmethod
    def add_element(self, value: float) -> DriftDetectionResult:
        """
        Додавання нового елемента та перевірка на дрейф.
        
        Args:
            value: Нове значення (нагорода, помилка тощо).
            
        Returns:
            Результат детекції дрейфу.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Скидання стану детектора."""
        pass


class ADWINDetector(BaseDriftDetector):
    """
    ADWIN (Adaptive Windowing) детектор дрейфу.
    
    Алгоритм автоматично адаптує розмір вікна спостережень:
    - Збільшує вікно в стабільних періодах
    - Зменшує при виявленні змін
    
    Математична основа:
    - Підтримує вікно W з останніх спостережень
    - Для кожного можливого розрізу W = W1 + W2
    - Обчислює |mean(W1) - mean(W2)|
    - Якщо різниця перевищує поріг ε = sqrt(ln(4/δ)/2 * (1/n1 + 1/n2)), детектує дрейф
    
    Посилання: Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data.
    """
    
    def __init__(self, delta: float = 0.002, max_window_size: int = 1000):
        """
        Ініціалізація ADWIN детектора.
        
        Args:
            delta: Рівень значущості (рекомендується 0.002).
            max_window_size: Максимальний розмір вікна.
        """
        super().__init__(delta)
        self.max_window_size = max_window_size
        self.window: List[float] = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        
    def add_element(self, value: float) -> DriftDetectionResult:
        """
        Додавання елемента в ADWIN вікно з перевіркою на дрейф.
        
        Args:
            value: Нове значення.
            
        Returns:
            Результат детекції дрейфу.
        """
        self.n_samples += 1
        self.window.append(value)
        self.total += value
        self.width += 1
        
        # Обмеження розміру вікна
        if len(self.window) > self.max_window_size:
            removed = self.window.pop(0)
            self.total -= removed
            self.width -= 1
        
        # Перевірка на дрейф
        drift_detected, confidence, magnitude = self._detect_drift()
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            confidence=confidence,
            drift_magnitude=magnitude,
            timestamp=self.n_samples,
            additional_info={'window_size': self.width}
        )
    
    def _detect_drift(self) -> Tuple[bool, float, float]:
        """
        Внутрішня перевірка на дрейф через аналіз підрозділів вікна.
        
        Returns:
            Кортеж (drift_detected, confidence, magnitude).
        """
        if len(self.window) < 10:  # Замало даних для аналізу
            return False, 0.0, 0.0
        
        n = len(self.window)
        max_diff = 0.0
        best_cut = 0
        
        # Перебір можливих розрізів вікна
        for cut in range(5, n - 5):  # Уникаємо дуже малих підрозділів
            n1 = cut
            n2 = n - cut
            
            mean1 = sum(self.window[:cut]) / n1
            mean2 = sum(self.window[cut:]) / n2
            
            diff = abs(mean1 - mean2)
            
            # Обчислення порогу ε
            m = 1.0 / ((1.0 / n1) + (1.0 / n2))
            epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))
            
            if diff > epsilon and diff > max_diff:
                max_diff = diff
                best_cut = cut
        
        # Визначення факту дрейфу
        if max_diff > 0:
            n1 = best_cut
            n2 = n - best_cut
            m = 1.0 / ((1.0 / n1) + (1.0 / n2))
            epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))
            
            confidence = min(1.0, max_diff / epsilon) if epsilon > 0 else 0.0
            magnitude = max_diff
            
            if max_diff > epsilon:
                logger.debug(f"ADWIN: дрейф виявлено (magnitude={magnitude:.4f}, confidence={confidence:.2f})")
                return True, confidence, magnitude
        
        return False, 0.0, 0.0
    
    def reset(self) -> None:
        """Повне скидання ADWIN вікна."""
        self.window = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.n_samples = 0
    
    def get_mean(self) -> float:
        """Отримання середнього значення у вікні."""
        if len(self.window) == 0:
            return 0.0
        return self.total / len(self.window)


class PageHinkleyDetector(BaseDriftDetector):
    """
    Детектор дрейфу Page-Hinkley (Cumulative Sum).
    
    Алгоритм моніторить кумулятивну суму відхилень від середнього:
    - S_t = Σ(x_i - x̄ - δ) для i=1..t
    - M_t = min(S_i) для i=1..t
    - Дрейф детектується коли S_t - M_t > λ
    
    Параметри:
    - δ: допустиме відхилення (tolerance)
    - λ: поріг детекції (threshold)
    - α: коефіцієнт забування (forgetting factor)
    
    Посилання: Page, E. S. (1954). Continuous inspection schemes.
    """
    
    def __init__(self, delta: float = 0.005, threshold: float = 50.0, 
                 alpha: float = 0.9999):
        """
        Ініціалізація Page-Hinkley детектора.
        
        Args:
            delta: Допустиме відхилення від середнього.
            threshold: Поріг детекції дрейфу.
            alpha: Коефіцієнт забування для середнього.
        """
        super().__init__(delta)
        self.threshold = threshold
        self.alpha = alpha
        
        self.mean = 0.0
        self.sum = 0.0
        self.min_sum = float('inf')
        
    def add_element(self, value: float) -> DriftDetectionResult:
        """
        Додавання елемента з перевіркою Page-Hinkley.
        
        Args:
            value: Нове значення.
            
        Returns:
            Результат детекції дрейфу.
        """
        self.n_samples += 1
        
        # Оновлення ковзного середнього
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value
        
        # Кумулятивна сума відхилень
        deviation = value - self.mean - self.delta
        self.sum = self.alpha * self.sum + deviation
        
        # Оновлення мінімуму
        if self.sum < self.min_sum:
            self.min_sum = self.sum
        
        # Перевірка на дрейф
        ph_value = self.sum - self.min_sum
        drift_detected = ph_value > self.threshold
        
        confidence = min(1.0, ph_value / self.threshold) if self.threshold > 0 else 0.0
        magnitude = ph_value
        
        if drift_detected:
            logger.debug(f"Page-Hinkley: дрейф виявлено (PH={ph_value:.2f}, threshold={self.threshold})")
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            confidence=confidence,
            drift_magnitude=magnitude,
            timestamp=self.n_samples,
            additional_info={'ph_value': ph_value, 'mean': self.mean}
        )
    
    def reset(self) -> None:
        """Скидання стану детектора."""
        self.mean = 0.0
        self.sum = 0.0
        self.min_sum = float('inf')
        self.n_samples = 0


class AdaptiveThresholdLCB:
    """
    Адаптивний LCB-критерій з детектором дрейфу.
    
    Розширює класичний Lower Confidence Bound (LCB) динамічним пороком:
    - Використовує детектор дрейфу (ADWIN або Page-Hinkley) для моніторингу
    - При виявленні дрейфу збільшує exploration через зниження порогу
    - Автоматично калібрує параметр α на основі стабільності середовища
    
    Математична формула:
    U_i(t) = μ̂_i(t) - α(t) * sqrt(ln(t) / n_i(t))
    
    де α(t) адаптивно змінюється:
    - α(t) = α_base * (1 + drift_factor) якщо виявлено дрейф
    - α(t) = α_base * exp(-decay * t_stable) в стабільному режимі
    
    Гіпотеза H1: Інтеграція детектора дрейфу зменшує adaptation_lag ≥30%.
    """
    
    def __init__(self, 
                 alpha_base: float = 2.0,
                 drift_detector_type: str = 'adwin',
                 drift_delta: float = 0.002,
                 drift_threshold: float = 50.0,
                 drift_factor: float = 1.5,
                 decay_rate: float = 0.01,
                 min_alpha: float = 0.5,
                 max_alpha: float = 5.0):
        """
        Ініціалізація адаптивного LCB.
        
        Args:
            alpha_base: Базове значення параметра α.
            drift_detector_type: Тип детектора ('adwin' або 'page_hinkley').
            drift_delta: Чутливість детектора дрейфу.
            drift_threshold: Поріг для Page-Hinkley.
            drift_factor: Множник α при виявленні дрейфу.
            decay_rate: Швидкість повернення до базового α.
            min_alpha: Мінімальне значення α.
            max_alpha: Максимальне значення α.
        """
        self.alpha_base = alpha_base
        self.alpha_current = alpha_base
        self.drift_factor = drift_factor
        self.decay_rate = decay_rate
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        # Ініціалізація детектора дрейфу
        if drift_detector_type == 'adwin':
            self.drift_detector = ADWINDetector(delta=drift_delta)
        elif drift_detector_type == 'page_hinkley':
            self.drift_detector = PageHinkleyDetector(
                delta=drift_delta, 
                threshold=drift_threshold
            )
        else:
            raise ValueError(f"Невідомий тип детектора: {drift_detector_type}")
        
        self.drift_detected_count = 0
        self.last_drift_time = 0
        self.steps_since_drift = 0
        
    def compute_lcb(self, mean_reward: float, n_pulls: int, 
                    total_steps: int) -> float:
        """
        Обчислення Lower Confidence Bound.
        
        Args:
            mean_reward: Середня винагорода стратегії.
            n_pulls: Кількість виборів стратегії.
            total_steps: Загальна кількість кроків.
            
        Returns:
            Значення LCB.
        """
        if n_pulls == 0 or total_steps == 0:
            return float('-inf')
        
        # LCB формула з адаптивним α
        exploration_bonus = self.alpha_current * np.sqrt(np.log(total_steps) / n_pulls)
        lcb_value = mean_reward - exploration_bonus
        
        return lcb_value
    
    def update(self, reward: float, step: int) -> Tuple[bool, float]:
        """
        Оновлення стану LCB з урахуванням дрейфу.
        
        Args:
            reward: Поточна винагорода.
            step: Поточний крок.
            
        Returns:
            Кортеж (drift_detected, current_alpha).
        """
        # Перевірка на дрейф
        drift_result = self.drift_detector.add_element(reward)
        drift_detected = drift_result.drift_detected
        
        if drift_detected:
            self.drift_detected_count += 1
            self.last_drift_time = step
            self.steps_since_drift = 0
            
            # Збільшення α для активного exploration
            self.alpha_current = min(
                self.max_alpha,
                self.alpha_base * self.drift_factor
            )
            logger.debug(f"Дрейф виявлено! α збільшено до {self.alpha_current:.3f}")
        else:
            # Поступове повернення до базового α
            self.steps_since_drift += 1
            decay = np.exp(-self.decay_rate * self.steps_since_drift)
            self.alpha_current = max(
                self.min_alpha,
                self.alpha_base + (self.alpha_current - self.alpha_base) * decay
            )
        
        return drift_detected, self.alpha_current
    
    def get_adaptation_lag(self) -> int:
        """
        Отримання затримки адаптації (кроків з моменту останнього дрейфу).
        
        Returns:
            Кількість кроків з останнього виявленого дрейфу.
        """
        return self.steps_since_drift
    
    def reset(self) -> None:
        """Повне скидання стану LCB."""
        self.alpha_current = self.alpha_base
        self.drift_detector.reset()
        self.drift_detected_count = 0
        self.last_drift_time = 0
        self.steps_since_drift = 0
    
    def get_stats(self) -> dict:
        """
        Отримання статистики роботи LCB.
        
        Returns:
            Словник зі статистикою.
        """
        return {
            'alpha_current': self.alpha_current,
            'alpha_base': self.alpha_base,
            'drift_count': self.drift_detected_count,
            'steps_since_drift': self.steps_since_drift,
            'total_samples': self.drift_detector.n_samples
        }


def create_drift_detector(detector_type: str, **kwargs) -> BaseDriftDetector:
    """
    Фабрична функція для створення детекторів дрейфу.
    
    Args:
        detector_type: Тип детектора ('adwin' або 'page_hinkley').
        **kwargs: Додаткові параметри для детектора.
        
    Returns:
        Екземпляр детектора дрейфу.
    """
    if detector_type == 'adwin':
        return ADWINDetector(**kwargs)
    elif detector_type == 'page_hinkley':
        return PageHinkleyDetector(**kwargs)
    else:
        raise ValueError(f"Невідомий тип детектора: {detector_type}")


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.DEBUG)
    
    # Тест ADWIN
    print("\n=== Тест ADWIN ===")
    adwin = ADWINDetector(delta=0.002)
    
    # Стабільний період
    for i in range(100):
        result = adwin.add_element(np.random.normal(0.5, 0.1))
    
    # Дрейф (зміна середнього)
    print("Початок дрейфу...")
    for i in range(100):
        result = adwin.add_element(np.random.normal(0.8, 0.1))  # Зміна з 0.5 на 0.8
        if result.drift_detected:
            print(f"  Дрейф виявлено на кроці {result.timestamp}!")
            break
    
    # Тест Page-Hinkley
    print("\n=== Тест Page-Hinkley ===")
    ph = PageHinkleyDetector(delta=0.005, threshold=50.0)
    
    for i in range(200):
        value = np.random.normal(0.5, 0.1)
        if i >= 100:
            value = np.random.normal(0.8, 0.1)  # Дрейф
        
        result = ph.add_element(value)
        if result.drift_detected:
            print(f"  Дрейф виявлено на кроці {result.timestamp}!")
            break
    
    # Тест AdaptiveThresholdLCB
    print("\n=== Тест AdaptiveThresholdLCB ===")
    lcb = AdaptiveThresholdLCB(alpha_base=2.0, drift_detector_type='adwin')
    
    rewards = []
    for i in range(200):
        reward = np.random.normal(0.5, 0.1) if i < 100 else np.random.normal(0.8, 0.1)
        drift, alpha = lcb.update(reward, i)
        rewards.append(reward)
        
        if drift:
            print(f"  Дрейф на кроці {i}, α={alpha:.3f}")
    
    stats = lcb.get_stats()
    print(f"\nСтатистика LCB: {stats}")
