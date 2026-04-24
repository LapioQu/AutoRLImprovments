# AutoRL Research Complex - Дослідницький комплекс для AutoRL

## 📋 Опис

Повністю відтворюваний дослідницький комплекс для перевірки гіпотез щодо динамічного перемикання стратегій самонавчання в нестаціонарних середовищах. Розроблено відповідно до вимог NeurIPS/ICML reproducibility checklists та стандартів ВАК/ДСТУ для магістерських робіт.

## 🎯 Наукові гіпотези

### H1: Детектор дрейфу
Інтеграція легкого детектора режимного зсуву (ADWIN/Page-Hinkley) у контур LCB-критерію зменшує адаптаційну затримку мінімум на 30% порівняно зі статичним LCB.

### H2: Багатокритеріальна утиліта
Багатокритеріальна утиліта U = w₁·perf − w₂·var − w₃·compute − w₄·switch з ентропійною регуляризацією знижує дисперсію кумулятивної винагороди ≥20%.

### H3: Мета-прунінг
Мета-ознаковий прунінг портфеля стратегій перед викликом метаконтролера скорочує час прийняття рішення ≥40%.

## 🏗 Архітектура проекту

```
workspace/
├── src/
│   ├── data_loader.py       # Завантаження ASSISTments, MiniGrid, OpenML
│   ├── drift_detectors.py   # ADWIN, Page-Hinkley, AdaptiveThresholdLCB
│   ├── multi_utility.py     # Багатокритеріальна утиліта
│   ├── meta_pruner.py       # Мета-ознаки та прунінг
│   ├── baselines.py         # UCB1, EXP3, LinUCB, Fixed-Policy, Naive-Mean
│   ├── runner.py            # Експериментальний раннер
│   └── stats_validator.py   # Статистична валідація
├── configs/
│   ├── assistments_config.yaml
│   └── minigrid_config.yaml
├── experiments/             # Результати експериментів
├── notebooks/               # Jupyter ноутбуки для аналізу
├── tests/                   # Unit-тести
├── requirements.txt
├── README.md
└── VALIDATION_CHECKLIST.md
```

## 🚀 Швидкий старт

### Встановлення залежностей

```bash
pip install -r requirements.txt
```

### Запуск експерименту

```bash
# Повний експеримент (n=30 seed)
PYTHONPATH=/workspace python -m src.runner --config configs/assistments_config.yaml --mode compare

# Швидкий тест (n=5 seed)
PYTHONPATH=/workspace python -m src.runner --config configs/assistments_config.yaml --n-seeds 5

# Для MiniGrid
PYTHONPATH=/workspace python -m src.runner --config configs/minigrid_config.yaml --n-seeds 5
```

## 📊 Формат результатів

Результати зберігаються у `experiments/results_{benchmark}_{timestamp}.json`:

```json
{
  "config": {...},
  "timestamp": "20260101_120000",
  "results": {
    "ucb1": {
      "mean_rewards": [0.45, 0.42, ...],
      "n_runs": 30
    },
    ...
  }
}
```

## 📈 Статистична валідація

Комплекс автоматично генерує:

1. **Bootstrap 95% CI** - довірчі інтервали для всіх метрик
2. **p-value** - парний Wilcoxon/t-test для кожного порівняння
3. **Cohen's d** - розмір ефекту
4. **Power Analysis** - статистична потужність

Приклад таблиці результатів:

```
Метод                Mean±Std           95% CI                  p-value      Cohen d      Power      Значущість
----------------------------------------------------------------------------------------------------
ucb1                 0.4123±0.0234      [0.4012, 0.4234]        baseline     baseline     baseline   baseline
exp3                 0.3876±0.0312      [0.3745, 0.4007]        0.023456     0.8234       0.9123     Так *
linucb               0.4567±0.0198      [0.4456, 0.4678]        0.001234     1.2345       0.9876     Так ***
```

## 🔬 Абляційний аналіз

Для перевірки внеску кожного компонента:

```bash
# Без детектора дрейфу
PYTHONPATH=/workspace python -c "
from src.runner import ExperimentConfig, ExperimentRunner
config = ExperimentConfig(use_drift_detector=False, n_seeds=10)
runner = ExperimentRunner(config)
results = runner.run_comparison()
"

# Без багатокритеріальної утиліти
config = ExperimentConfig(use_multi_utility=False, n_seeds=10)

# Без мета-прунінгу
config = ExperimentConfig(use_meta_pruning=False, n_seeds=10)
```

## 📝 Інтерпретація результатів

### Як читати статистичні таблиці

- **p < 0.05**: статистично значуща різниця (95% впевненості)
- **p < 0.01**: високо значуща різниця (99% впевненості)
- **p < 0.001**: дуже високо значуща різниця
- **Cohen's d > 0.8**: великий практичний ефект
- **Power > 0.8**: достатня статистична потужність

### Коли метод НЕ працює

1. **Дуже стабільні середовища** - детектор дрейфу може давати хибні спрацювання
2. **Мала кількість даних** (<100 кроків) - недостатньо для надійної оцінки
3. **Високий шум** - може маскувати реальні зміни
4. **Чутливість до параметрів**:
   - α (alpha_base): занадто високе → повільна адаптація
   - c_switch: занадто низьке → часті перемикання

### Загрози валідності

**Внутрішня валідність:**
- Випадковість ініціалізації (контролюється фіксацією seed)
- Порядок презентації стратегій (рандомізується)

**Зовнішня валідність:**
- Обмеженість бенчмарків (ASSISTments, MiniGrid)
- Специфіка параметрів середовища

## 📚 Використання в магістерській роботі

### Приклад цитування коду

```bibtex
@software{autorl_complex2026,
  author = {Your Name},
  title = {AutoRL Research Complex for Dynamic Strategy Switching},
  year = {2026},
  url = {https://github.com/your-repo/autorl-complex}
}
```

### Оформлення розділу з експериментами

1. Описати конфігурацію експерименту (Section X.Y)
2. Навести таблицю з статистичними результатами
3. Проінтерпретувати p-value та effect size
4. Обговорити обмеження методу

## ✅ Перевірочний список

Див. [`VALIDATION_CHECKLIST.md`](VALIDATION_CHECKLIST.md) для повного списку критеріїв:

- [ ] n ≥ 30 незалежних запусків
- [ ] p < 0.05 для основних порівнянь
- [ ] 95% bootstrap CI наведено
- [ ] Effect size (Cohen's d) обчислено
- [ ] Power analysis проведено
- [ ] Абляційний аналіз виконано
- [ ] Реальні дані використано (не синтетика)

## 🤝 Внесок

Цей комплекс розроблено для магістерської дисертації. Будь-які покращення вітаються.

## 📄 Ліцензія

MIT License - вільне використання для наукових цілей.

## 📧 Контакти

Для питань щодо використання комплексу звертайтесь до автора.
