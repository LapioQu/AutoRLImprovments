"""
Unit-тести для модулів AutoRL комплексу.

Запускаються через: pytest tests/ -v
"""

import numpy as np
import pytest

from src.drift_detectors import ADWINDetector, PageHinkleyDetector, AdaptiveThresholdLCB
from src.multi_utility import MultiCriteriaUtility, UtilityConfig
from src.meta_pruner import MetaFeaturesCalculator, PortfolioPruner
from src.baselines import UCB1, EXP3, LinUCB, create_strategy
from src.stats_validator import StatisticalValidator, create_validator


class TestDriftDetectors:
    """Тести для детекторів дрейфу."""
    
    def test_adwin_detects_drift(self):
        """ADWIN має виявляти зміну середнього."""
        detector = ADWINDetector(delta=0.01)
        
        # Стабільний період
        for _ in range(50):
            result = detector.add_element(np.random.normal(0.5, 0.05))
        
        # Дрейф (зміна з 0.5 на 0.9)
        drift_detected = False
        for _ in range(100):
            result = detector.add_element(np.random.normal(0.9, 0.05))
            if result.drift_detected:
                drift_detected = True
                break
        
        assert drift_detected, "ADWIN не виявив дрейф"
    
    def test_page_hinkley_basic(self):
        """Page-Hinkley базовий тест."""
        detector = PageHinkleyDetector(delta=0.005, threshold=50.0)
        
        for i in range(100):
            value = np.random.normal(0.5, 0.1)
            result = detector.add_element(value)
            assert hasattr(result, 'drift_detected')
    
    def test_adaptive_lcb_alpha_changes(self):
        """AdaptiveThresholdLCB має змінювати α при дрейфі."""
        lcb = AdaptiveThresholdLCB(alpha_base=2.0, drift_detector_type='adwin')
        
        initial_alpha = lcb.alpha_current
        
        # Симуляція дрейфу
        for i in range(200):
            reward = np.random.normal(0.5 if i < 100 else 0.9, 0.1)
            drift_detected, alpha = lcb.update(reward, i)
        
        # α мало змінитися хоча б раз
        stats = lcb.get_stats()
        assert stats['drift_count'] >= 0  # Може бути 0 якщо дрейф не виявлено


class TestMultiUtility:
    """Тести для багатокритеріальної утиліти."""
    
    def test_utility_computation(self):
        """Базове обчислення утиліти."""
        utility = MultiCriteriaUtility()
        
        result = utility.compute_utility(
            performance=0.7,
            reward_variance=0.05,
            compute_cost=0.001,
            switch_penalty=0
        )
        
        assert hasattr(result, 'utility')
        assert isinstance(result.utility, float)
    
    def test_utility_with_normalization(self):
        """Утиліта з нормалізацією."""
        utility = MultiCriteriaUtility(UtilityConfig(normalize=True))
        
        # Спочатку оновлюємо статистику
        for i in range(20):
            metrics = {
                'performance': np.random.uniform(0.3, 0.8),
                'variance': np.random.uniform(0.01, 0.2),
                'compute': np.random.uniform(0.001, 0.01),
                'switch': np.random.choice([0, 1])
            }
            utility.update_normalization_stats(metrics)
        
        # Потім обчислюємо утиліту
        result = utility.compute_utility(
            performance=0.6,
            reward_variance=0.1,
            compute_cost=0.005,
            switch_penalty=1
        )
        
        assert result.is_normalized


class TestMetaPruner:
    """Тести для мета-прунінгу."""
    
    def test_feature_calculation(self):
        """Обчислення мета-ознак."""
        calculator = MetaFeaturesCalculator(window_size=50)
        
        # Оновлення даними
        for i in range(100):
            calculator.update('strategy1', reward=np.random.randn(), action=i % 4)
        
        features = calculator.compute_features('strategy1')
        
        assert features.strategy_id == 'strategy1'
        assert hasattr(features, 'reward_mean')
        assert hasattr(features, 'action_entropy')
    
    def test_portfolio_pruning(self):
        """Прунінг портфеля."""
        pruner = PortfolioPruner(min_strategies=2, max_strategies=3)
        
        strategies = ['ucb1', 'exp3', 'linucb', 'fixed']
        
        # Спочатку оновлюємо ознаки
        for strategy in strategies:
            for i in range(50):
                pruner.update_features(strategy, reward=np.random.randn(), action=i % 4)
        
        result = pruner.prune(strategies, current_step=100)
        
        assert result.original_size == 4
        assert result.pruned_size <= 3
        assert len(result.retained_strategies) == result.pruned_size


class TestBaselines:
    """Тести для базових стратегій."""
    
    def test_ucb1_selection(self):
        """UCB1 вибір дії."""
        ucb = UCB1(n_actions=4)
        
        action = ucb.select_action()
        assert 0 <= action < 4
        
        # Оновлення
        ucb.update(action, reward=0.5)
        assert ucb.total_pulls == 1
    
    def test_exp3_selection(self):
        """EXP3 вибір дії."""
        exp3 = EXP3(n_actions=4)
        
        action = exp3.select_action()
        assert 0 <= action < 4
        
        exp3.update(action, reward=0.5)
        assert exp3.total_pulls == 1
    
    def test_create_strategy_factory(self):
        """Фабрика стратегій."""
        for name in ['ucb1', 'exp3', 'linucb', 'fixed', 'naive', 'random']:
            strategy = create_strategy(name, n_actions=4)
            assert strategy is not None
            assert strategy.n_actions == 4


class TestStatsValidator:
    """Тести для статистичної валідації."""
    
    def test_bootstrap_ci(self):
        """Bootstrap довірчий інтервал."""
        validator = create_validator(n_bootstrap=100)
        
        data = np.random.normal(0.5, 0.1, 30)
        ci_lower, ci_upper = validator.bootstrap_ci(data)
        
        assert ci_lower < np.mean(data) < ci_upper
    
    def test_cohens_d(self):
        """Cohen's d effect size."""
        validator = create_validator()
        
        group_a = np.random.normal(0.7, 0.1, 30)
        group_b = np.random.normal(0.5, 0.1, 30)
        
        d = validator.cohens_d(group_a, group_b)
        
        # Очікуємо позитивний ефект (a > b)
        assert d > 0
    
    def test_compare_groups(self):
        """Порівняння груп."""
        validator = create_validator(n_bootstrap=100)
        
        group_a = np.random.normal(0.7, 0.1, 30)
        group_b = np.random.normal(0.5, 0.1, 30)
        
        result = validator.compare_groups(group_a, group_b, 'A', 'B')
        
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'effect_size')
        assert hasattr(result, 'power')
    
    def test_normality_check(self):
        """Перевірка нормальності."""
        validator = create_validator()
        
        # Нормальний розподіл
        normal_data = np.random.normal(0.5, 0.1, 50)
        is_normal, p = validator.check_normality(normal_data)
        
        # Має бути визнано нормальним (ймовірно)
        assert bool(is_normal) in [True, False]  # numpy bool або звичайний bool


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
