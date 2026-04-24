"""
Модуль мета-ознак та прунінгу портфеля стратегій.

Реалізує:
- Видобуток мета-ознак (reward_mean/var, action_entropy, gradient_norm, observation_var)
- Фільтрація портфеля стратегій перед викликом метаконтролера
- Оцінка різноманітності стратегій

Гіпотеза H3: Мета-прунінг скорочує час прийняття рішення ≥40%.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetaFeatures:
    """Мета-ознаки стратегії."""
    strategy_id: str
    reward_mean: float = 0.0
    reward_variance: float = 0.0
    reward_std: float = 0.0
    action_entropy: float = 0.0
    gradient_norm: float = 0.0
    observation_variance: float = 0.0
    n_observations: int = 0
    performance_trend: float = 0.0  # Ковзний тренд продуктивності
    stability_score: float = 0.0    # Оцінка стабільності
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник."""
        return {
            'strategy_id': self.strategy_id,
            'reward_mean': self.reward_mean,
            'reward_variance': self.reward_variance,
            'reward_std': self.reward_std,
            'action_entropy': self.action_entropy,
            'gradient_norm': self.gradient_norm,
            'observation_variance': self.observation_variance,
            'n_observations': self.n_observations,
            'performance_trend': self.performance_trend,
            'stability_score': self.stability_score
        }


@dataclass
class PruningResult:
    """Результат прунінгу портфеля."""
    original_size: int
    pruned_size: int
    retained_strategies: List[str]
    pruning_ratio: float
    time_saved_ms: float
    expected_utility_loss: float


class MetaFeaturesCalculator:
    """
    Калькулятор мета-ознак для стратегій.
    
    Обчислює набір ознак для кожної стратегії в портфелі:
    1. reward_mean/var/std - статистика винагород
    2. action_entropy - ентропія розподілу дій
    3. gradient_norm - норма градієнтів (якщо доступно)
    4. observation_variance - дисперсія спостережень
    5. performance_trend - тренд зміни продуктивності
    6. stability_score - загальна оцінка стабільності
    """
    
    def __init__(self, window_size: int = 50):
        """
        Ініціалізація калькулятора.
        
        Args:
            window_size: Розмір вікна для ковзних статистик.
        """
        self.window_size = window_size
        self.reward_history: Dict[str, deque] = {}
        self.action_history: Dict[str, deque] = {}
        self.observation_history: Dict[str, deque] = {}
        self.gradient_history: Dict[str, deque] = {}
        
    def update(self, strategy_id: str, 
               reward: float, 
               action: Optional[int] = None,
               observation: Optional[np.ndarray] = None,
               gradient: Optional[np.ndarray] = None) -> None:
        """
        Оновлення історії для стратегії.
        
        Args:
            strategy_id: Ідентифікатор стратегії.
            reward: Отримана винагорода.
            action: Виконана дія.
            observation: Спостереження з середовища.
            gradient: Градієнт (якщо доступно).
        """
        # Ініціалізація черг якщо потрібно
        if strategy_id not in self.reward_history:
            self.reward_history[strategy_id] = deque(maxlen=self.window_size)
            self.action_history[strategy_id] = deque(maxlen=self.window_size)
            self.observation_history[strategy_id] = deque(maxlen=self.window_size)
            self.gradient_history[strategy_id] = deque(maxlen=self.window_size)
        
        # Додавання даних
        self.reward_history[strategy_id].append(reward)
        
        if action is not None:
            self.action_history[strategy_id].append(action)
        
        if observation is not None:
            obs_flat = observation.flatten() if hasattr(observation, 'flatten') else observation
            self.observation_history[strategy_id].append(obs_flat)
        
        if gradient is not None:
            self.gradient_history[strategy_id].append(gradient)
    
    def compute_features(self, strategy_id: str) -> StrategyMetaFeatures:
        """
        Обчислення мета-ознак для стратегії.
        
        Args:
            strategy_id: Ідентифікатор стратегії.
            
        Returns:
            Об'єкт StrategyMetaFeatures.
        """
        features = StrategyMetaFeatures(strategy_id=strategy_id)
        
        # Статистика винагород
        if strategy_id in self.reward_history and len(self.reward_history[strategy_id]) > 0:
            rewards = list(self.reward_history[strategy_id])
            features.reward_mean = float(np.mean(rewards))
            features.reward_variance = float(np.var(rewards))
            features.reward_std = float(np.std(rewards))
            features.n_observations = len(rewards)
            
            # Обчислення тренду (лінійна регресія на останніх 20 точках)
            if len(rewards) >= 5:
                recent = rewards[-min(20, len(rewards)):]
                x = np.arange(len(recent))
                slope = np.polyfit(x, recent, 1)[0]
                features.performance_trend = float(slope)
            
            # Оцінка стабільності (обернено пропорційна до коефіцієнта варіації)
            if features.reward_mean != 0:
                cv = features.reward_std / abs(features.reward_mean)
                features.stability_score = float(1.0 / (1.0 + cv))
            else:
                features.stability_score = 0.5
        
        # Ентропія дій
        if strategy_id in self.action_history and len(self.action_history[strategy_id]) > 0:
            actions = list(self.action_history[strategy_id])
            unique, counts = np.unique(actions, return_counts=True)
            probs = counts / len(actions)
            probs = np.clip(probs, 1e-10, 1.0)
            features.action_entropy = float(-np.sum(probs * np.log(probs)))
        
        # Норма градієнтів
        if strategy_id in self.gradient_history and len(self.gradient_history[strategy_id]) > 0:
            gradients = list(self.gradient_history[strategy_id])
            grad_norms = [np.linalg.norm(g) for g in gradients]
            features.gradient_norm = float(np.mean(grad_norms))
        
        # Дисперсія спостережень
        if strategy_id in self.observation_history and len(self.observation_history[strategy_id]) > 0:
            observations = list(self.observation_history[strategy_id])
            if len(observations[0]) > 0:
                obs_array = np.array(observations)
                features.observation_variance = float(np.mean(np.var(obs_array, axis=0)))
        
        return features
    
    def get_all_features(self, strategy_ids: List[str]) -> Dict[str, StrategyMetaFeatures]:
        """
        Обчислення ознак для всіх стратегій.
        
        Args:
            strategy_ids: Список ідентифікаторів стратегій.
            
        Returns:
            Словник {strategy_id: features}.
        """
        return {sid: self.compute_features(sid) for sid in strategy_ids}
    
    def reset(self, strategy_id: Optional[str] = None) -> None:
        """
        Скидання історії.
        
        Args:
            strategy_id: Якщо вказано, скидає тільки цю стратегію.
        """
        if strategy_id is None:
            self.reward_history.clear()
            self.action_history.clear()
            self.observation_history.clear()
            self.gradient_history.clear()
        else:
            for history in [self.reward_history, self.action_history, 
                          self.observation_history, self.gradient_history]:
                if strategy_id in history:
                    del history[strategy_id]


class PortfolioPruner:
    """
    Прунер портфеля стратегій на основі мета-ознак.
    
    Реалізує кілька стратегій відбору:
    1. Performance-based: залишає топ-K за продуктивністю
    2. Diversity-based: залишає найбільш різноманітні стратегії
    3. Hybrid: комбінація продуктивності та різноманітності
    4. Stability-based: залишає найстабільніші стратегії
    
    Гіпотеза H3: Такий прунінг скорочує час прийняття рішення ≥40%.
    """
    
    def __init__(self, 
                 min_strategies: int = 2,
                 max_strategies: int = 5,
                 pruning_strategy: str = 'hybrid',
                 diversity_weight: float = 0.3,
                 performance_weight: float = 0.7):
        """
        Ініціалізація прунера.
        
        Args:
            min_strategies: Мінімальна кількість стратегій після прунінгу.
            max_strategies: Максимальна кількість стратегій.
            pruning_strategy: Стратегія прунінгу ('performance', 'diversity', 'hybrid', 'stability').
            diversity_weight: Вага різноманітності в hybrid стратегії.
            performance_weight: Вага продуктивності в hybrid стратегії.
        """
        self.min_strategies = min_strategies
        self.max_strategies = max_strategies
        self.pruning_strategy = pruning_strategy
        self.diversity_weight = diversity_weight
        self.performance_weight = performance_weight
        
        self.feature_calculator = MetaFeaturesCalculator()
        self.pruning_history: List[PruningResult] = []
        
    def prune(self, 
              strategy_ids: List[str],
              current_step: int = 0) -> PruningResult:
        """
        Виконання прунінгу портфеля.
        
        Args:
            strategy_ids: Поточний список стратегій.
            current_step: Поточний крок експерименту.
            
        Returns:
            Результат прунінгу.
        """
        import time
        start_time = time.time()
        
        original_size = len(strategy_ids)
        
        # Обчислення мета-ознак для всіх стратегій
        features = self.feature_calculator.get_all_features(strategy_ids)
        
        # Визначення кількості стратегій для збереження
        target_size = self._compute_target_size(original_size, current_step)
        
        if original_size <= target_size:
            # Прунінг не потрібен
            return PruningResult(
                original_size=original_size,
                pruned_size=original_size,
                retained_strategies=strategy_ids.copy(),
                pruning_ratio=0.0,
                time_saved_ms=0.0,
                expected_utility_loss=0.0
            )
        
        # Застосування стратегії прунінгу
        if self.pruning_strategy == 'performance':
            retained = self._prune_by_performance(features, target_size)
        elif self.pruning_strategy == 'diversity':
            retained = self._prune_by_diversity(features, target_size)
        elif self.pruning_strategy == 'stability':
            retained = self._prune_by_stability(features, target_size)
        else:  # hybrid
            retained = self._prune_hybrid(features, target_size)
        
        # Обчислення метрик
        time_saved = (time.time() - start_time) * 1000  # мс
        pruning_ratio = (original_size - len(retained)) / original_size
        utility_loss = self._estimate_utility_loss(features, strategy_ids, retained)
        
        result = PruningResult(
            original_size=original_size,
            pruned_size=len(retained),
            retained_strategies=retained,
            pruning_ratio=pruning_ratio,
            time_saved_ms=time_saved,
            expected_utility_loss=utility_loss
        )
        
        self.pruning_history.append(result)
        logger.debug(f"Прунінг: {original_size} → {len(retained)} стратегій "
                    f"(економія {time_saved:.2f}мс)")
        
        return result
    
    def _compute_target_size(self, current_size: int, step: int) -> int:
        """
        Обчислення цільового розміру портфеля.
        
        Адаптивно змінюється залежно від етапу експерименту:
        - На початку: більше стратегій для exploration
        - Пізніше: менше стратегій для швидкості
        
        Args:
            current_size: Поточний розмір.
            step: Поточний крок.
            
        Returns:
            Цільовий розмір.
        """
        # Базове правило: залишити 40-70% від початкового розміру
        base_ratio = 0.5
        
        # Адаптація за кроком
        if step < 100:
            ratio = 0.8  # Більше на початку
        elif step < 500:
            ratio = 0.6
        else:
            ratio = base_ratio
        
        target = max(
            self.min_strategies,
            min(self.max_strategies, int(current_size * ratio))
        )
        
        return target
    
    def _prune_by_performance(self, 
                             features: Dict[str, StrategyMetaFeatures],
                             target_size: int) -> List[str]:
        """Відбір топ-K стратегій за продуктивністю."""
        sorted_strategies = sorted(
            features.keys(),
            key=lambda sid: features[sid].reward_mean,
            reverse=True
        )
        return sorted_strategies[:target_size]
    
    def _prune_by_diversity(self,
                           features: Dict[str, StrategyMetaFeatures],
                           target_size: int) -> List[str]:
        """
        Відбір найбільш різноманітних стратегій.
        
        Використовує жадібний алгоритм максимізації мінімальної відстані.
        """
        if len(features) <= target_size:
            return list(features.keys())
        
        # Обчислення попарних відстаней за ознаками
        strategy_ids = list(features.keys())
        
        def distance(sid1: str, sid2: str) -> float:
            """Відстань між стратегіями за ознаками."""
            f1, f2 = features[sid1], features[sid2]
            
            # Нормалізовані відмінності
            diff_mean = abs(f1.reward_mean - f2.reward_mean) / (abs(f1.reward_mean) + 1e-8)
            diff_var = abs(f1.reward_variance - f2.reward_variance) / (abs(f1.reward_variance) + 1e-8)
            diff_entropy = abs(f1.action_entropy - f2.action_entropy) / (abs(f1.action_entropy) + 1e-8)
            
            return diff_mean + diff_var + diff_entropy
        
        # Жадібний вибір
        selected = [strategy_ids[0]]  # Починаємо з першої
        
        while len(selected) < target_size:
            best_candidate = None
            best_min_dist = -1
            
            for candidate in strategy_ids:
                if candidate in selected:
                    continue
                
                # Мінімальна відстань до вже вибраних
                min_dist = min(distance(candidate, s) for s in selected)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
            else:
                break
        
        return selected
    
    def _prune_by_stability(self,
                           features: Dict[str, StrategyMetaFeatures],
                           target_size: int) -> List[str]:
        """Відбір найстабільніших стратегій."""
        sorted_strategies = sorted(
            features.keys(),
            key=lambda sid: features[sid].stability_score,
            reverse=True
        )
        return sorted_strategies[:target_size]
    
    def _prune_hybrid(self,
                     features: Dict[str, StrategyMetaFeatures],
                     target_size: int) -> List[str]:
        """
        Гібридний відбір: продуктивність + різноманітність.
        
        Score = w_perf * normalized_perf + w_div * diversity_contribution
        """
        if len(features) <= target_size:
            return list(features.keys())
        
        strategy_ids = list(features.keys())
        
        # Нормалізація продуктивності
        perf_values = [features[sid].reward_mean for sid in strategy_ids]
        perf_min, perf_max = min(perf_values), max(perf_values)
        perf_range = perf_max - perf_min if perf_max > perf_min else 1.0
        
        normalized_perf = {
            sid: (features[sid].reward_mean - perf_min) / perf_range
            for sid in strategy_ids
        }
        
        # Обчислення внеску в різноманітність
        def diversity_contribution(sid: str, selected: List[str]) -> float:
            if not selected:
                return 1.0
            
            f_sid = features[sid]
            avg_diff = 0.0
            
            for sel_sid in selected:
                f_sel = features[sel_sid]
                diff = (
                    abs(f_sid.reward_mean - f_sel.reward_mean) +
                    abs(f_sid.action_entropy - f_sel.action_entropy)
                )
                avg_diff += diff
            
            return avg_diff / len(selected)
        
        # Жадібний гібридний вибір
        selected = []
        remaining = strategy_ids.copy()
        
        while len(selected) < target_size and remaining:
            best_score = -float('inf')
            best_candidate = None
            
            for candidate in remaining:
                perf_score = normalized_perf[candidate]
                div_score = diversity_contribution(candidate, selected) / 2.0
                
                hybrid_score = (
                    self.performance_weight * perf_score +
                    self.diversity_weight * div_score
                )
                
                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _estimate_utility_loss(self,
                              features: Dict[str, StrategyMetaFeatures],
                              original: List[str],
                              retained: List[str]) -> float:
        """
        Оцінка очікуваної втрати утиліти через прунінг.
        
        Args:
            features: Мета-ознаки стратегій.
            original: Оригінальний список.
            retained: Список після прунінгу.
            
        Returns:
            Оцінка втрати утиліти (0.0-1.0).
        """
        if not original or not retained:
            return 0.0
        
        removed = set(original) - set(retained)
        if not removed:
            return 0.0
        
        # Середня продуктивність видалених стратегій
        removed_perf = [features[sid].reward_mean for sid in removed if sid in features]
        original_perf = [features[sid].reward_mean for sid in original if sid in features]
        
        if not removed_perf or not original_perf:
            return 0.0
        
        avg_removed = np.mean(removed_perf)
        avg_original = np.mean(original_perf)
        
        # Нормалізована втрата
        if abs(avg_original) < 1e-8:
            return 0.0
        
        loss = max(0.0, (avg_original - avg_removed) / abs(avg_original))
        return min(1.0, loss)
    
    def update_features(self, strategy_id: str, **kwargs) -> None:
        """
        Оновлення мета-ознак стратегії.
        
        Args:
            strategy_id: Ідентифікатор стратегії.
            **kwargs: Параметри для оновлення (reward, action, observation, gradient).
        """
        self.feature_calculator.update(strategy_id, **kwargs)
    
    def reset(self) -> None:
        """Повне скидання прунера."""
        self.feature_calculator.reset()
        self.pruning_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Отримання статистики прунінгу.
        
        Returns:
            Словник зі статистикою.
        """
        if not self.pruning_history:
            return {'total_prunings': 0}
        
        total_time_saved = sum(r.time_saved_ms for r in self.pruning_history)
        avg_pruning_ratio = np.mean([r.pruning_ratio for r in self.pruning_history])
        avg_utility_loss = np.mean([r.expected_utility_loss for r in self.pruning_history])
        
        return {
            'total_prunings': len(self.pruning_history),
            'avg_pruning_ratio': avg_pruning_ratio,
            'avg_time_saved_ms': total_time_saved / len(self.pruning_history),
            'total_time_saved_ms': total_time_saved,
            'avg_utility_loss': avg_utility_loss
        }


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("=== Тест MetaFeaturesCalculator ===\n")
    
    calculator = MetaFeaturesCalculator(window_size=50)
    
    # Симуляція даних для 3 стратегій
    strategies = ['UCB1', 'EXP3', 'LinUCB']
    
    for step in range(100):
        for strategy in strategies:
            reward = np.random.normal(0.5 + 0.1 * (step / 100), 0.1)
            action = np.random.randint(0, 4)
            observation = np.random.randn(5)
            
            calculator.update(strategy, reward=reward, action=action, 
                            observation=observation)
    
    # Обчислення ознак
    for strategy in strategies:
        features = calculator.compute_features(strategy)
        print(f"{strategy}:")
        print(f"  Mean: {features.reward_mean:.3f}, Var: {features.reward_variance:.3f}")
        print(f"  Entropy: {features.action_entropy:.3f}, Stability: {features.stability_score:.3f}")
    
    print("\n=== Тест PortfolioPruner ===\n")
    
    pruner = PortfolioPruner(
        min_strategies=2,
        max_strategies=3,
        pruning_strategy='hybrid'
    )
    
    # Перенесення даних з калькулятора
    pruner.feature_calculator = calculator
    
    result = pruner.prune(strategies, current_step=100)
    
    print(f"Оригінальний розмір: {result.original_size}")
    print(f"Після прунінгу: {result.pruned_size}")
    print(f"Залишено: {result.retained_strategies}")
    print(f"Економія часу: {result.time_saved_ms:.2f}мс")
    print(f"Очікувана втрата утиліти: {result.expected_utility_loss:.3f}")
    
    stats = pruner.get_stats()
    print(f"\nСтатистика: {stats}")
