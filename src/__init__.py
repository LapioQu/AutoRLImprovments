# src/__init__.py

"""
AutoRL Research Complex - Пакет для досліджень динамічного перемикання стратегій.

Цей пакет містить модулі для:
- Завантаження бенчмарк-даних (ASSISTments, MiniGrid, OpenML)
- Детекції дрейфу (ADWIN, Page-Hinkley)
- Багатокритеріальної утиліти з ентропійною регуляризацією
- Мета-прунінгу портфеля стратегій
- Базових алгоритмів (UCB1, EXP3, LinUCB)
- Статистичної валідації результатів

Використовується в магістерських дослідженнях з AutoRL та адаптивних систем.
"""

__version__ = '1.0.0'
__author__ = 'Research Team'

from src.data_loader import get_benchmark, BenchmarkDataManager
from src.drift_detectors import AdaptiveThresholdLCB, ADWINDetector, PageHinkleyDetector
from src.multi_utility import MultiCriteriaUtility, create_utility_from_config
from src.meta_pruner import PortfolioPruner, MetaFeaturesCalculator
from src.baselines import create_strategy, UCB1, EXP3, LinUCB
from src.stats_validator import StatisticalValidator, create_validator

__all__ = [
    'get_benchmark',
    'BenchmarkDataManager',
    'AdaptiveThresholdLCB',
    'ADWINDetector',
    'PageHinkleyDetector',
    'MultiCriteriaUtility',
    'create_utility_from_config',
    'PortfolioPruner',
    'MetaFeaturesCalculator',
    'create_strategy',
    'UCB1',
    'EXP3',
    'LinUCB',
    'StatisticalValidator',
    'create_validator'
]
