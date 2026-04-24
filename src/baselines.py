"""
Модуль базових стратегій для порівняння.

Реалізує:
- UCB1: Upper Confidence Bound
- EXP3: Exponential-weight algorithm for Exploration and Exploitation
- LinUCB: Linear UCB для контекстних задач
- Fixed-Policy: Статична політика
- Naive-Mean: Наївне середнє з правилом перемикання
- RandomPolicy: Випадковий вибір (базова лінія)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Базовий клас для всіх стратегій."""
    
    def __init__(self, strategy_id: str, n_actions: int = 4):
        """
        Ініціалізація стратегії.
        
        Args:
            strategy_id: Унікальний ідентифікатор.
            n_actions: Кількість доступних дій.
        """
        self.strategy_id = strategy_id
        self.n_actions = n_actions
        self.total_pulls = 0
        self.total_reward = 0.0
        
    @abstractmethod
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        """Вибір дії."""
        pass
    
    @abstractmethod
    def update(self, action: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        """Оновлення стратегії після отримання винагороди."""
        pass
    
    def reset(self) -> None:
        """Скидання стану стратегії."""
        self.total_pulls = 0
        self.total_reward = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Отримання статистики."""
        return {
            'strategy_id': self.strategy_id,
            'total_pulls': self.total_pulls,
            'total_reward': self.total_reward,
            'mean_reward': self.total_reward / max(1, self.total_pulls)
        }


class UCB1(BaseStrategy):
    """
    UCB1 (Upper Confidence Bound) алгоритм.
    
    Формула вибору:
    a_t = argmax_a [Q(a) + c * sqrt(ln(t) / N(a))]
    
    де:
    - Q(a): середня винагорода дії a
    - N(a): кількість виборів дії a
    - c: параметр exploration (за замовчуванням 2.0)
    """
    
    def __init__(self, n_actions: int = 4, c: float = 2.0):
        super().__init__('UCB1', n_actions)
        self.c = c
        self.action_counts = np.zeros(n_actions)
        self.action_rewards = np.zeros(n_actions)
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        # Якщо є дії, які ще не обиралися
        if np.any(self.action_counts == 0):
            return int(np.argmax(self.action_counts == 0))
        
        # Обчислення UCB для всіх дій
        ucb_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            q_value = self.action_rewards[a] / self.action_counts[a]
            exploration_bonus = self.c * np.sqrt(np.log(self.total_pulls) / self.action_counts[a])
            ucb_values[a] = q_value + exploration_bonus
        
        return int(np.argmax(ucb_values))
    
    def update(self, action: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        self.action_counts[action] += 1
        self.action_rewards[action] += reward
        self.total_pulls += 1
        self.total_reward += reward


class EXP3(BaseStrategy):
    """
    EXP3 (Exponential-weight algorithm for Exploration and Exploitation).
    
    Використовує експоненціальні ваги для балансу exploration/exploitation.
    Добре працює в нестаціонарних середовищах.
    
    Параметри:
    - gamma: ймовірність випадкового вибору (exploration rate)
    - eta: параметр навчання
    """
    
    def __init__(self, n_actions: int = 4, gamma: float = 0.1, eta: float = 0.5):
        super().__init__('EXP3', n_actions)
        self.gamma = gamma
        self.eta = eta
        self.weights = np.ones(n_actions)
        self.probs = np.ones(n_actions) / n_actions
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        # Оновлення ймовірностей
        self.probs = (1 - self.gamma) * self.weights / np.sum(self.weights) + \
                     self.gamma / self.n_actions
        
        # Вибір дії за розподілом
        return int(np.random.choice(self.n_actions, p=self.probs))
    
    def update(self, action: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        # Нормалізація винагороди до [0, 1]
        reward_norm = np.clip(reward, 0, 1)
        
        # Оновлення ваг
        estimated_reward = reward_norm / self.probs[action]
        self.weights[action] *= np.exp(self.eta * estimated_reward)
        
        self.total_pulls += 1
        self.total_reward += reward


class LinUCB(BaseStrategy):
    """
    LinUCB для контекстних багаторуких бандитів.
    
    Використовує лінійну модель для оцінки винагород:
    r_a = θ_a^T x + noise
    
    LCB формула:
    UCB_a = θ_a^T x + α * sqrt(x^T A_a^{-1} x)
    """
    
    def __init__(self, n_actions: int = 4, context_dim: int = 5, alpha: float = 1.0):
        super().__init__('LinUCB', n_actions)
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Ініціалізація матриць для кожної дії
        self.A = [np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]
        self.theta = [np.zeros(context_dim) for _ in range(n_actions)]
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        if context is None:
            context = np.ones(self.context_dim)
        
        # Обчислення UCB для всіх дій
        ucb_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            self.theta[a] = A_inv @ self.b[a]
            
            exploitation = self.theta[a] @ context
            exploration = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_values[a] = exploitation + exploration
        
        return int(np.argmax(ucb_values))
    
    def update(self, action: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        if context is None:
            context = np.ones(self.context_dim)
        
        # Оновлення матриць
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
        
        self.total_pulls += 1
        self.total_reward += reward


class FixedPolicy(BaseStrategy):
    """
    Фіксована політика (статична стратегія).
    
    Завжди обирає одну й ту саму дію або слідує заданому розподілу.
    Використовується як базова лінія для порівняння.
    """
    
    def __init__(self, n_actions: int = 4, fixed_action: int = 0, 
                 distribution: Optional[np.ndarray] = None):
        super().__init__('Fixed-Policy', n_actions)
        
        if distribution is not None:
            self.distribution = distribution / np.sum(distribution)
            self.fixed_action = None
        else:
            self.fixed_action = fixed_action
            self.distribution = None
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        if self.distribution is not None:
            return int(np.random.choice(self.n_actions, p=self.distribution))
        else:
            return self.fixed_action
    
    def update(self, action: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        self.total_pulls += 1
        self.total_reward += reward


class NaiveMean(BaseStrategy):
    """
    Наївна стратегія з переключенням при покращенні.
    
    Проста евристика:
    - Обирає дію з найвищим середнім
    - Перемикається якщо інша дія має ΔU > threshold
    """
    
    def __init__(self, n_actions: int = 4, switch_threshold: float = 0.05):
        super().__init__('Naive-Mean', n_actions)
        self.switch_threshold = switch_threshold
        self.action_counts = np.zeros(n_actions)
        self.action_rewards = np.zeros(n_actions)
        self.current_action = 0
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        # Спочатку пробуємо всі дії хоча б по разу
        if np.any(self.action_counts == 0):
            return int(np.argmin(self.action_counts))
        
        # Обчислення середніх
        means = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if self.action_counts[a] > 0:
                means[a] = self.action_rewards[a] / self.action_counts[a]
        
        # Перевірка на перемикання
        best_action = int(np.argmax(means))
        current_mean = means[self.current_action] if self.action_counts[self.current_action] > 0 else 0
        
        if means[best_action] - current_mean > self.switch_threshold:
            self.current_action = best_action
        
        return self.current_action
    
    def update(self, action: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        self.action_counts[action] += 1
        self.action_rewards[action] += reward
        self.total_pulls += 1
        self.total_reward += reward


class RandomPolicy(BaseStrategy):
    """Випадкова політика (повний baseline)."""
    
    def __init__(self, n_actions: int = 4):
        super().__init__('Random-Policy', n_actions)
        
    def select_action(self, context: Optional[np.ndarray] = None) -> int:
        return np.random.randint(self.n_actions)
    
    def update(self, action: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        self.total_pulls += 1
        self.total_reward += reward


def create_strategy(strategy_name: str, n_actions: int = 4, **kwargs) -> BaseStrategy:
    """
    Фабрична функція для створення стратегій.
    
    Args:
        strategy_name: Назва стратегії.
        n_actions: Кількість дій.
        **kwargs: Додаткові параметри.
        
    Returns:
        Екземпляр стратегії.
    """
    strategies = {
        'ucb1': lambda: UCB1(n_actions, kwargs.get('c', 2.0)),
        'exp3': lambda: EXP3(n_actions, kwargs.get('gamma', 0.1), kwargs.get('eta', 0.5)),
        'linucb': lambda: LinUCB(n_actions, kwargs.get('context_dim', 5), kwargs.get('alpha', 1.0)),
        'fixed': lambda: FixedPolicy(n_actions, kwargs.get('fixed_action', 0)),
        'naive': lambda: NaiveMean(n_actions, kwargs.get('switch_threshold', 0.05)),
        'random': lambda: RandomPolicy(n_actions)
    }
    
    if strategy_name.lower() not in strategies:
        raise ValueError(f"Невідома стратегія: {strategy_name}")
    
    return strategies[strategy_name.lower()]()


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("=== Тест стратегій ===\n")
    
    n_actions = 4
    n_steps = 1000
    
    strategies = [
        UCB1(n_actions),
        EXP3(n_actions),
        LinUCB(n_actions, context_dim=5),
        FixedPolicy(n_actions, fixed_action=1),
        NaiveMean(n_actions),
        RandomPolicy(n_actions)
    ]
    
    results = {}
    
    for strategy in strategies:
        total_reward = 0.0
        
        for step in range(n_steps):
            # Генерація контексту для LinUCB
            context = np.random.randn(5) if isinstance(strategy, LinUCB) else None
            
            # Вибір дії
            action = strategy.select_action(context)
            
            # Симуляція винагороди (дія 1 найкраща)
            reward = np.random.normal(0.5 if action == 1 else 0.3, 0.1)
            
            # Оновлення
            strategy.update(action, reward, context)
            total_reward += reward
        
        results[strategy.strategy_id] = total_reward / n_steps
        print(f"{strategy.strategy_id}: середня винагорода = {results[strategy.strategy_id]:.4f}")
    
    print("\nНайкраща стратегія:", max(results, key=results.get))
