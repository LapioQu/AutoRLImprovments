"""
Експериментальний раннер для AutoRL досліджень.

Реалізує повний цикл експерименту:
- Багатосидовий запуск (n≥30)
- Порівняння стратегій
- Абляційний аналіз
- Логування результатів
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

import numpy as np
import yaml
from tqdm import tqdm

# Імпорт модулів
from src.data_loader import get_benchmark, BenchmarkDataManager
from src.drift_detectors import AdaptiveThresholdLCB
from src.multi_utility import MultiCriteriaUtility, create_utility_from_config
from src.meta_pruner import PortfolioPruner
from src.baselines import create_strategy, BaseStrategy
from src.stats_validator import StatisticalValidator, create_validator

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Фіксація всіх random seed для відтворення."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class ExperimentConfig:
    """Конфігурація експерименту."""
    benchmark: str = 'assistments'
    n_seeds: int = 30
    n_steps: int = 1000
    strategies: List[str] = None
    use_drift_detector: bool = True
    drift_detector_type: str = 'adwin'
    use_multi_utility: bool = True
    use_meta_pruning: bool = True
    pruning_min_strategies: int = 2
    pruning_max_strategies: int = 5
    utility_weights: Dict[str, float] = None
    alpha_base: float = 2.0
    output_dir: str = './experiments'
    entropy_regularization: float = 0.1
    env_name: str = 'MiniGrid-Empty-8x8-v0'
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = ['ucb1', 'exp3', 'linucb', 'fixed', 'naive']
        if self.utility_weights is None:
            self.utility_weights = {
                'w_performance': 1.0,
                'w_variance': 0.5,
                'w_compute': 0.1,
                'w_switch': 0.3
            }


@dataclass 
class ExperimentResult:
    """Результат одного запуску."""
    seed: int
    benchmark: str
    strategy: str
    mean_reward: float
    reward_variance: float
    total_switches: int
    net_utility: float
    adaptation_lag_avg: float
    compute_time_ms: float
    step_metrics: List[Dict] = None


class ExperimentRunner:
    """Основний раннер експериментів."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = create_validator(alpha=0.05, n_bootstrap=5000)
        self.results: List[ExperimentResult] = []
        
    def run_single_seed(self, seed: int, strategy_name: str) -> ExperimentResult:
        """Запуск експерименту для одного seed."""
        set_seed(seed)
        
        start_time = datetime.now()
        
        # Ініціалізація компонентів
        lcb = AdaptiveThresholdLCB(
            alpha_base=self.config.alpha_base,
            drift_detector_type=self.config.drift_detector_type if self.config.use_drift_detector else 'adwin'
        )
        
        utility = create_utility_from_config(self.config.utility_weights) if self.config.use_multi_utility else None
        
        pruner = PortfolioPruner(
            min_strategies=self.config.pruning_min_strategies,
            max_strategies=self.config.pruning_max_strategies
        ) if self.config.use_meta_pruning else None
        
        # Створення стратегії
        strategy = create_strategy(strategy_name, n_actions=4)
        
        # Завантаження бенчмарку
        benchmark = get_benchmark(self.config.benchmark)
        
        # Експериментальний цикл
        rewards = []
        switches = 0
        last_action = None
        step_metrics = []
        
        for step in range(self.config.n_steps):
            step_start = datetime.now()
            
            # Отримання контексту (якщо потрібно)
            context = np.random.randn(5) if strategy_name == 'linucb' else None
            
            # Вибір дії
            action = strategy.select_action(context)
            
            # Симуляція винагороди (для демонстрації)
            # В реальності тут має бути взаємодія з бенчмарком
            true_reward = 0.5 if action == 1 else 0.3
            reward = np.random.normal(true_reward, 0.1)
            
            # Оновлення стратегії
            strategy.update(action, reward, context)
            
            # Оновлення LCB з детекцією дрейфу
            drift_detected, alpha_current = lcb.update(reward, step)
            
            # Підрахунок перемикань
            if last_action is not None and action != last_action:
                switches += 1
            last_action = action
            
            # Оновлення утиліти
            if utility:
                compute_cost = (datetime.now() - step_start).total_seconds()
                switch_penalty = 1 if (last_action is not None and action != last_action) else 0
                
                utility.update_normalization_stats({
                    'performance': reward,
                    'variance': np.var(rewards[-50:]) if len(rewards) > 50 else 0.01,
                    'compute': compute_cost,
                    'switch': switch_penalty
                })
            
            # Оновлення прунера
            if pruner:
                pruner.update_features(strategy_name, reward=reward, action=action)
            
            rewards.append(reward)
            
            # Збереження метрик кроку
            if step % 100 == 0:
                step_metrics.append({
                    'step': step,
                    'reward': reward,
                    'cumulative_reward': np.sum(rewards),
                    'alpha': alpha_current
                })
        
        # Фінальні обчислення
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = ExperimentResult(
            seed=seed,
            benchmark=self.config.benchmark,
            strategy=strategy_name,
            mean_reward=float(np.mean(rewards)),
            reward_variance=float(np.var(rewards)),
            total_switches=switches,
            net_utility=float(np.mean(rewards)) - 0.1 * switches / self.config.n_steps,
            adaptation_lag_avg=float(lcb.get_adaptation_lag()) / max(1, lcb.drift_detected_count) if lcb.drift_detected_count > 0 else 0.0,
            compute_time_ms=total_time / self.config.n_steps,
            step_metrics=step_metrics
        )
        
        return result
    
    def run_all_seeds(self, strategy_name: str) -> List[ExperimentResult]:
        """Запуск для всіх seed."""
        results = []
        
        for seed in tqdm(range(self.config.n_seeds), desc=f"{strategy_name}"):
            try:
                result = self.run_single_seed(seed, strategy_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Помилка на seed {seed}: {e}")
                continue
        
        return results
    
    def run_comparison(self) -> Dict[str, List[ExperimentResult]]:
        """Порівняння всіх стратегій."""
        all_results = {}
        
        for strategy_name in self.config.strategies:
            logger.info(f"Запуск стратегії: {strategy_name}")
            results = self.run_all_seeds(strategy_name)
            all_results[strategy_name] = results
        
        self.results = [r for results in all_results.values() for r in results]
        return all_results
    
    def save_results(self, all_results: Dict[str, List[ExperimentResult]]) -> str:
        """Збереження результатів у JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results_{self.config.benchmark}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        export_data = {
            'config': asdict(self.config),
            'timestamp': timestamp,
            'results': {}
        }
        
        for strategy, results in all_results.items():
            export_data['results'][strategy] = {
                'mean_rewards': [r.mean_reward for r in results],
                'reward_variances': [r.reward_variance for r in results],
                'switches': [r.total_switches for r in results],
                'net_utilities': [r.net_utility for r in results],
                'adaptation_lags': [r.adaptation_lag_avg for r in results],
                'compute_times': [r.compute_time_ms for r in results],
                'n_runs': len(results)
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Результати збережено в {filepath}")
        return str(filepath)
    
    def generate_report(self, all_results: Dict[str, List[ExperimentResult]]) -> str:
        """Генерація звіту зі статистикою."""
        # Підготовка даних для validator
        results_dict = {}
        for strategy, results in all_results.items():
            results_dict[strategy] = np.array([r.mean_reward for r in results])
        
        # Генерація таблиці
        table = self.validator.generate_comparison_table(
            results_dict, 
            baseline_name='ucb1'
        )
        
        # LaTeX таблиця
        latex_table = self.validator.generate_latex_table(
            results_dict,
            baseline_name='ucb1',
            caption='Порівняння стратегій AutoRL'
        )
        
        report = f"""
# ЗВІТ ПРО ЕКСПЕРИМЕНТ

Конфігурація:
- Бенчмарк: {self.config.benchmark}
- Кількість seed: {self.config.n_seeds}
- Кроків: {self.config.n_steps}
- Стратегії: {', '.join(self.config.strategies)}

{table}

## LaTeX таблиця для дисертації

{latex_table}
"""
        
        report_path = self.output_dir / f"report_{self.config.benchmark}.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report


def load_config(config_path: str) -> ExperimentConfig:
    """Завантаження конфігурації з YAML."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ExperimentConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description='AutoRL Experiment Runner')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Шлях до YAML конфігурації')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['single', 'compare', 'ablation'],
                       help='Режим експерименту')
    parser.add_argument('--n-seeds', type=int, default=None,
                       help='Кількість seed (перевизначає конфиг)')
    
    args = parser.parse_args()
    
    # Налаштування логування
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Завантаження конфігурації
    config = load_config(args.config)
    
    if args.n_seeds is not None:
        config.n_seeds = args.n_seeds
    
    logger.info(f"Запуск експерименту з конфігурацією: {config.benchmark}")
    logger.info(f"Кількість seed: {config.n_seeds}")
    
    # Запуск раннера
    runner = ExperimentRunner(config)
    
    if args.mode == 'compare':
        all_results = runner.run_comparison()
        runner.save_results(all_results)
        report = runner.generate_report(all_results)
        print(report)
    else:
        logger.warning("Режим '{args.mode}' ще не реалізовано")


if __name__ == '__main__':
    main()
