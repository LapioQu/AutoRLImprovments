# Professional ML Benchmark Suite

Цей репозиторій розгортає реальне середовище тестування систем машинного навчання на офіційних бенчмарках, а не на синтетичних або саморобних сценаріях.

## Офіційні джерела

- `OpenML-CC18`: офіційний benchmark suite OpenML для класифікації.
- `UCI Adult`: офіційний датасет UCI (`id=2`).
- `UCI Bank Marketing`: офіційний датасет UCI (`id=222`).
- `UCI Covertype`: офіційний датасет UCI (`id=31`).
- `ASSISTments Skill Builder`: офіційний CSV зі Skill Builder log-ами.
- `Elec2`: реальний потоковий benchmark з ринку електроенергії NSW, доступний через `river`.

## Реальні системи, що тестуються

Batch-системи:
- `logreg`
- `random_forest`
- `hist_gb`

Stream-системи:
- `river_logreg`
- `river_nb`
- `river_hoeffding_tree`

## Встановлення

```powershell
python -m pip install -r requirements.txt
```

## Швидка перевірка

```powershell
python -m src.runner --config configs/benchmark_suite.yaml
```

## Що буде згенеровано

- `data/`: кеш офіційних наборів даних
- `results/benchmark_results.csv`: усі результати
- `results/benchmark_summary.csv`: агрегований summary
- `results/benchmark_summary.json`: машинозчитуваний звіт

## Примітка щодо професійності середовища

Середовище навмисно побудоване на зовнішніх, загальноприйнятих бенчмарках:

- OpenML дає стандартизовані набори задач і широко використовується для відтворюваного benchmark-тестування.
- UCI є класичним репозиторієм реальних табличних задач.
- River надає справжні потокові набори, включаючи `Elec2`, який є канонічним benchmark-ом для online/drift evaluation.
- ASSISTments репрезентує реальні освітні події з послідовними відповідями студентів.
