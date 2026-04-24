"""
Модуль статистичної валідації результатів експериментів.

Реалізує:
- Bootstrap 95% довірчі інтервали
- Парний тест Wilcoxon signed-rank
- t-test Стьюдента з перевіркою нормальності
- Effect size (Cohen's d)
- Power analysis
- Генерація LaTeX/CSV таблиць

Відповідає вимогам до статистичної значущості в наукових роботах.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, ttest_rel, wilcoxon

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Результат статистичного тесту."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # Чи значущий при alpha=0.05
    effect_size: Optional[float] = None  # Cohen's d
    ci_lower: Optional[float] = None  # Нижня межа CI
    ci_upper: Optional[float] = None  # Верхня межа CI
    power: Optional[float] = None  # Статистична потужність
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник."""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'significant': self.significant,
            'effect_size': self.effect_size,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'power': self.power,
            **self.additional_info
        }


class StatisticalValidator:
    """
    Комплексний validator для статистичної перевірки результатів.
    
    Реалізує повний протокол статистичної валідації:
    1. Перевірка нормальності розподілів (Shapiro-Wilk)
    2. Перевірка гомогенності дисперсій (Levene)
    3. Вибір відповідного тесту (t-test або Wilcoxon)
    4. Обчислення effect size (Cohen's d)
    5. Bootstrap довірчі інтервали
    6. Power analysis
    """
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 5000, seed: int = 42):
        """
        Ініціалізація validator'а.
        
        Args:
            alpha: Рівень значущості (за замовчуванням 0.05).
            n_bootstrap: Кількість bootstrap ітерацій.
            seed: Random seed для відтворення.
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        np.random.seed(seed)
        
    def check_normality(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        Перевірка нормальності розподілу (Shapiro-Wilk).
        
        Args:
            data: Масив даних.
            
        Returns:
            Кортеж (is_normal, p_value).
        """
        if len(data) < 3:
            return True, 1.0  # Замало даних для перевірки
        
        statistic, p_value = shapiro(data)
        is_normal = p_value > self.alpha
        
        logger.debug(f"Shapiro-Wilk: stat={statistic:.4f}, p={p_value:.4f}, "
                    f"normal={is_normal}")
        
        return is_normal, p_value
    
    def check_homogeneity(self, data_a: np.ndarray, 
                         data_b: np.ndarray) -> Tuple[bool, float]:
        """
        Перевірка гомогенності дисперсій (Levene's test).
        
        Args:
            data_a: Перша група даних.
            data_b: Друга група даних.
            
        Returns:
            Кортеж (is_homogeneous, p_value).
        """
        statistic, p_value = levene(data_a, data_b)
        is_homogeneous = p_value > self.alpha
        
        logger.debug(f"Levene: stat={statistic:.4f}, p={p_value:.4f}, "
                    f"homogeneous={is_homogeneous}")
        
        return is_homogeneous, p_value
    
    def bootstrap_ci(self, data: np.ndarray, 
                    confidence: float = 0.95,
                    statistic_func: callable = np.mean) -> Tuple[float, float]:
        """
        Bootstrap довірчий інтервал.
        
        Args:
            data: Масив даних.
            confidence: Рівень довіри (0.95 для 95% CI).
            statistic_func: Функція статистики (mean, median тощо).
            
        Returns:
            Кортеж (ci_lower, ci_upper).
        """
        n = len(data)
        if n < 2:
            return float('-inf'), float('inf')
        
        np.random.seed(self.seed)
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper
    
    def cohens_d(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """
        Обчислення Cohen's d (effect size).
        
        Інтерпретація:
        - 0.2: малий ефект
        - 0.5: середній ефект
        - 0.8: великий ефект
        
        Args:
            data_a: Перша група.
            data_b: Друга група.
            
        Returns:
            Значення Cohen's d.
        """
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        
        var_a = np.var(data_a, ddof=1)
        var_b = np.var(data_b, ddof=1)
        
        n_a = len(data_a)
        n_b = len(data_b)
        
        # pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std < 1e-10:
            return 0.0
        
        return (mean_a - mean_b) / pooled_std
    
    def power_analysis(self, effect_size: float, n: int, 
                      alpha: Optional[float] = None) -> float:
        """
        Аналіз статистичної потужності.
        
        Args:
            effect_size: Очікуваний розмір ефекту (Cohen's d).
            n: Розмір вибірки.
            alpha: Рівень значущості.
            
        Returns:
            Статистична потужність (0-1).
        """
        if alpha is None:
            alpha = self.alpha
        
        # Наближений розрахунок потужності для t-test
        # Використовуємо non-central t-distribution
        df = 2 * n - 2
        ncp = effect_size * np.sqrt(n / 2)  # non-centrality parameter
        
        # Критичне значення t
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        
        # Потужність
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        
        return float(power)
    
    def compare_groups(self, 
                      group_a: np.ndarray,
                      group_b: np.ndarray,
                      group_a_name: str = 'A',
                      group_b_name: str = 'B',
                      paired: bool = False) -> StatisticalTestResult:
        """
        Повне порівняння двох груп з автоматичним вибором тесту.
        
        Протокол:
        1. Перевірка нормальності обох груп
        2. Якщо обидві нормальні → t-test (paired або unpaired)
        3. Якщо ні → Wilcoxon signed-rank (paired) або Mann-Whitney U
        4. Обчислення effect size
        5. Bootstrap CI для різниці середніх
        6. Power analysis
        
        Args:
            group_a: Дані першої групи.
            group_b: Дані другої групи.
            group_a_name: Назва першої групи.
            group_b_name: Назва другої групи.
            paired: Чи є групи парними (залежними).
            
        Returns:
            Результат статистичного тесту.
        """
        # Перевірка на пусті дані
        if len(group_a) == 0 or len(group_b) == 0:
            return StatisticalTestResult(
                test_name='empty_data',
                statistic=0.0,
                p_value=1.0,
                significant=False,
                additional_info={'error': 'Empty data'}
            )
        
        # Перевірка нормальності
        normal_a, p_norm_a = self.check_normality(group_a)
        normal_b, p_norm_b = self.check_normality(group_b)
        both_normal = normal_a and normal_b
        
        # Вибір тесту
        if both_normal:
            if paired:
                statistic, p_value = ttest_rel(group_a, group_b)
                test_name = 'paired_t_test'
            else:
                # Перевірка гомогенності для unpaired t-test
                homo, p_homo = self.check_homogeneity(group_a, group_b)
                if homo:
                    statistic, p_value = ttest_ind(group_a, group_b, equal_var=True)
                    test_name = 'independent_t_test'
                else:
                    statistic, p_value = ttest_ind(group_a, group_b, equal_var=False)
                    test_name = 'welch_t_test'
        else:
            if paired and len(group_a) == len(group_b):
                statistic, p_value = wilcoxon(group_a, group_b)
                test_name = 'wilcoxon_signed_rank'
            else:
                statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
                test_name = 'mann_whitney_u'
        
        # Effect size
        effect_size = self.cohens_d(group_a, group_b)
        
        # Bootstrap CI для різниці середніх
        diff_means = np.mean(group_a) - np.mean(group_b)
        bootstrap_diffs = []
        np.random.seed(self.seed)
        
        n_a, n_b = len(group_a), len(group_b)
        for _ in range(self.n_bootstrap):
            sample_a = np.random.choice(group_a, size=n_a, replace=True)
            sample_b = np.random.choice(group_b, size=n_b, replace=True)
            bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))
        
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Power analysis
        power = self.power_analysis(abs(effect_size), min(n_a, n_b))
        
        return StatisticalTestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            effect_size=effect_size,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            power=power,
            additional_info={
                'group_a_name': group_a_name,
                'group_b_name': group_b_name,
                'group_a_mean': float(np.mean(group_a)),
                'group_b_mean': float(np.mean(group_b)),
                'group_a_std': float(np.std(group_a)),
                'group_b_std': float(np.std(group_b)),
                'group_a_n': n_a,
                'group_b_n': n_b,
                'both_normal': both_normal,
                'diff_mean': diff_means
            }
        )
    
    def generate_comparison_table(self, 
                                 results: Dict[str, np.ndarray],
                                 baseline_name: str = 'baseline') -> str:
        """
        Генерація таблиці порівняння всіх методів з baseline.
        
        Args:
            results: Словник {method_name: array_of_metrics}.
            baseline_name: Назва baseline методу.
            
        Returns:
            Форматована таблиця (Markdown/LaTeX стиль).
        """
        lines = []
        lines.append("=" * 100)
        lines.append("СТАТИСТИЧНЕ ПОРІВНЯННЯ МЕТОДІВ")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"{'Метод':<20} {'Mean±Std':<20} {'95% CI':<25} {'p-value':<12} "
                    f"{'Cohen d':<12} {'Power':<10} {'Значущість':<15}")
        lines.append("-" * 100)
        
        baseline_data = results.get(baseline_name)
        
        for method_name, data in results.items():
            mean_val = np.mean(data)
            std_val = np.std(data)
            ci_lower, ci_upper = self.bootstrap_ci(data)
            
            if method_name == baseline_name:
                line = f"{method_name:<20} {mean_val:.4f}±{std_val:.4f}     " \
                      f"[{ci_lower:.4f}, {ci_upper:.4f}]     {'-':<12} {'-':<12} {'-':<10} {'baseline':<15}"
            else:
                result = self.compare_groups(data, baseline_data, method_name, baseline_name)
                sig_marker = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else ""
                
                significance = "Так" if result.significant else "Ні"
                if result.p_value < 0.001:
                    significance += " ***"
                elif result.p_value < 0.01:
                    significance += " **"
                elif result.p_value < 0.05:
                    significance += " *"
                
                line = f"{method_name:<20} {mean_val:.4f}±{std_val:.4f}     " \
                      f"[{ci_lower:.4f}, {ci_upper:.4f}]     {result.p_value:<12.6f} " \
                      f"{result.effect_size:<12.4f} {result.power:<10.4f} {significance:<15}"
            
            lines.append(line)
        
        lines.append("-" * 100)
        lines.append("* p<0.05, ** p<0.01, *** p<0.001")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_latex_table(self,
                            results: Dict[str, np.ndarray],
                            baseline_name: str = 'baseline',
                            caption: str = 'Порівняння методів') -> str:
        """
        Генерація LaTeX таблиці для дисертації.
        
        Args:
            results: Словник з результатами.
            baseline_name: Назва baseline.
            caption: Підпис таблиці.
            
        Returns:
            LaTeX код таблиці.
        """
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{" + caption + "}")
        lines.append("\\label{tab:comparison}")
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Метод} & \\textbf{Mean} & \\textbf{Std} & "
                    "\\textbf{95\\% CI} & \\textbf{p-value} & "
                    "\\textbf{Cohen's d} & \\textbf{Значущість} \\\\")
        lines.append("\\midrule")
        
        baseline_data = results.get(baseline_name)
        
        for method_name, data in sorted(results.items()):
            mean_val = np.mean(data)
            std_val = np.std(data)
            ci_lower, ci_upper = self.bootstrap_ci(data)
            
            if method_name == baseline_name:
                line = f"\\textbf{{{method_name}}} & {mean_val:.4f} & {std_val:.4f} & " \
                      f"[{ci_lower:.3f}, {ci_upper:.3f}] & -- & -- & baseline \\\\"
            else:
                result = self.compare_groups(data, baseline_data, method_name, baseline_name)
                sig_marker = "\\textbf{***}" if result.p_value < 0.001 else "\\textbf{**}" if result.p_value < 0.01 else "\\textbf{*}" if result.p_value < 0.05 else ""
                
                line = f"{method_name} & {mean_val:.4f} & {std_val:.4f} & " \
                      f"[{ci_lower:.3f}, {ci_upper:.3f}] & {result.p_value:.4f}{sig_marker} & " \
                      f"{result.effect_size:.3f} & {'Так' if result.significant else 'Ні'} \\\\"
            
            lines.append(line)
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def export_to_csv(self, 
                     results: Dict[str, np.ndarray],
                     filename: str) -> None:
        """
        Експорт результатів у CSV формат.
        
        Args:
            results: Словник з результатами.
            filename: Ім'я файлу.
        """
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Mean', 'Std', 'CI_Lower', 'CI_Upper', 
                           'Min', 'Max', 'N'])
            
            for method_name, data in results.items():
                ci_lower, ci_upper = self.bootstrap_ci(data)
                writer.writerow([
                    method_name,
                    f"{np.mean(data):.6f}",
                    f"{np.std(data):.6f}",
                    f"{ci_lower:.6f}",
                    f"{ci_upper:.6f}",
                    f"{np.min(data):.6f}",
                    f"{np.max(data):.6f}",
                    len(data)
                ])
        
        logger.info(f"Результати експортовано в {filename}")


def create_validator(alpha: float = 0.05, n_bootstrap: int = 5000) -> StatisticalValidator:
    """
    Фабрична функція для створення validator'а.
    
    Args:
        alpha: Рівень значущості.
        n_bootstrap: Кількість bootstrap ітерацій.
        
    Returns:
        Екземпляр StatisticalValidator.
    """
    return StatisticalValidator(alpha=alpha, n_bootstrap=n_bootstrap)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("=== Тест StatisticalValidator ===\n")
    
    validator = StatisticalValidator(alpha=0.05, n_bootstrap=1000)
    
    # Генерація тестових даних
    np.random.seed(42)
    method_a = np.random.normal(0.75, 0.1, 30)  # Запропонований метод
    method_b = np.random.normal(0.65, 0.12, 30)  # Baseline
    method_c = np.random.normal(0.70, 0.15, 30)  # Інший метод
    
    # Перевірка нормальності
    print("Перевірка нормальності:")
    for name, data in [('Method A', method_a), ('Method B', method_b)]:
        is_normal, p = validator.check_normality(data)
        print(f"  {name}: normal={is_normal}, p={p:.4f}")
    
    # Порівняння груп
    print("\nПорівняння Method A vs Method B:")
    result = validator.compare_groups(method_a, method_b, 'Method A', 'Method B')
    print(f"  Тест: {result.test_name}")
    print(f"  p-value: {result.p_value:.6f}")
    print(f"  Significant: {result.significant}")
    print(f"  Effect size (Cohen's d): {result.effect_size:.4f}")
    print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"  Power: {result.power:.4f}")
    
    # Генерація таблиці
    print("\n" + "="*80)
    results = {
        'Proposed': method_a,
        'Baseline': method_b,
        'Alternative': method_c
    }
    
    table = validator.generate_comparison_table(results, baseline_name='Baseline')
    print(table)
    
    # LaTeX таблиця
    print("\nLaTeX таблиця:")
    latex_table = validator.generate_latex_table(results, baseline_name='Baseline')
    print(latex_table[:500] + "...")
