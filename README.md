# Прогноз отказов оборудования

**Яньшин А.** — автор доработок этого форка.

Форк группового учебного проекта по дисциплине «Автоматизация машинного обучения».
Исходная история разработки сохранена в Git; текущая версия посвящена рефакторингу,
воспроизводимому окружению и проверкам качества кода.

[Репозиторий форка](https://github.com/Deferon/auto_ML-prediction_machines_failures_fork) ·
[Исходный групповой проект](https://github.com/svscrip/auto_ML-prediction_machines_failures)

## Выполненные задачи

| Пункт | Реализация |
| --- | --- |
| 2. Рефакторинг | Валидация CSV, аннотации типов, логирование, единые преобразования train/predict, конфигурация путей, CLI и регрессионные тесты |
| 3. Инструменты | Poetry, lock-файл, pre-commit, Ruff (линтер, импорты, форматирование), Mypy, pytest и GitHub Actions |
| 4. Виртуальное окружение и Git | Проектное окружение `.venv`, общая конфигурация `poetry.toml`, зависимости в `pyproject.toml` и `poetry.lock`; окружение исключено через `.gitignore` |

## Быстрый старт

Нужны Python **3.11 или 3.12**, Git и Poetry **2.2.1**. Рекомендуемая версия Python
задана в `.python-version`. Poetry устанавливается отдельно от окружения проекта,
например через [pipx](https://pipx.pypa.io/stable/installation/):

```bash
pipx install poetry==2.2.1
git clone https://github.com/Deferon/auto_ML-prediction_machines_failures_fork.git
cd auto_ML-prediction_machines_failures_fork
poetry env use python
poetry sync --with dev
poetry run pre-commit install
poetry run ml-train --smoke
poetry run ml-predict
```

Если команда `python` указывает на другую версию, передайте `poetry env use` полный
путь к Python 3.11/3.12. Проверка выбранного интерпретатора: `poetry env info`.
Активация окружения не требуется: все команды запускаются через `poetry run`.

В Git хранятся **описание и lock-файл окружения**, а не его бинарные файлы.
После клонирования `poetry sync --with dev` создаёт локальную `.venv` с версиями из
lock-файла. `poetry.toml` включает создание окружения в проекте и не содержит секретов.
Зависимости для эксплуатации: `poetry sync --only main`.

## Данные и модель

Источник: [Kaggle Playground Series S3E17](https://www.kaggle.com/competitions/playground-series-s3e17).
Данные для воспроизведения находятся в `keis7-main/train.csv` и `keis7-main/test.csv`.

Модель CatBoost оценивает вероятность `Machine failure`. Используются температуры,
скорость вращения, момент, износ, тип оборудования и инженерные признаки из телеметрии:
разность температур, мощность, расчётный воздушный поток, тепловая мощность и КПД.

**Флаги TWF/HDF/PWF/OSF/RNF и накопленные отказы исключены из входов модели.**
Они описывают уже известный исход. Строки больше не фильтруются по сочетанию цели
и флагов. Преобразования не меняют порядок и число строк и не зависят от соседей
в партии. Поэтому одиночный запрос получает те же признаки и уровень риска.

Обучение использует стратифицированное разбиение 80/20, seed 42, балансировку
классов и early stopping по AUC. Smoke-выборка также стратифицирована.
Набор признаков имеет версию 2. Старую модель необходимо переобучить:
инференс отклоняет модель с несовместимыми именами признаков.

`failure_probability` — оценка модели, обученной с весами классов; калибровка
вероятностей на данных конкретного оборудования ещё требуется.
Риск задаётся фиксированными демонстрационными порогами:

| Оценка модели | Риск |
| --- | --- |
| < 0.45 | Низкий |
| ≥ 0.45 и < 0.50 | Средний |
| ≥ 0.50 | Высокий |

Пороги находятся в `src/config.py`. Перед эксплуатацией их следует подобрать по
стоимости пропущенного отказа и ложного срабатывания на независимой выборке.
Случайный hold-out учебного датасета не заменяет временную или групповую валидацию
на реальной истории оборудования.

## Команды

```bash
poetry run ml-train
poetry run ml-train --smoke
poetry run ml-train --data keis7-main/train.csv --sample-size 5000 --output-dir artifacts
poetry run ml-predict --model artifacts/model.cbm --data keis7-main/test.csv --output-dir artifacts
poetry run ml-predict --no-drift
poetry run python -m src.train --help
poetry run python -m src.predict --help
```

Прежние команды `python -m src.train` и `python -m src.predict` доступны в окружении.
`--reference-data` у инференса задаёт размеченный CSV для сравнения распределений;
`--no-drift` отключает чтение обучающих данных и ставит статус `not_evaluated`.

| Переменная окружения | По умолчанию |
| --- | --- |
| `ML_DATA_DIR` | `keis7-main/` относительно проекта |
| `ML_ARTIFACTS_DIR` | `artifacts/` относительно проекта |
| `MLFLOW_TRACKING_URI` | SQLite в `artifacts/mlflow.db` |
| `MLFLOW_EXPERIMENT_NAME` | `machine_failure_prediction` |

Переменные задаются до запуска процесса. Параметры CLI `--data`, `--output-dir`,
`--tracking-uri` имеют приоритет для соответствующего запуска.
Для установки wheel вне репозитория укажите абсолютные пути через переменные.
`--output-dir` меняет выходные файлы, но не адрес базы MLflow.

## Артефакты и мониторинг

Обучение сохраняет `model.cbm`, `model_metadata.json`, `metrics.json`,
`data_quality.json`, `monitoring_summary.json` и графики в `artifacts/plots/`.
В MLflow передаются параметры, метрики, модель, пример входа и артефакты.

Инференс создаёт `predictions.csv`, `maintenance_recommendations.csv` и
`inference_monitoring.json`. Последний содержит время, пропускную способность,
CPU/RAM, качество входа и PSI. Интервалы PSI строятся по эталонной выборке,
отдельно учитывают выход за её границы, корректно обрабатывают постоянные признаки
и нулевые частоты. Порог предупреждения — 0.1, критического дрейфа — 0.25.

MLflow повторно использует эксперимент и **не удаляет базу** при несовпадении путей.
При переносе проекта перенесите хранилище артефактов или создайте новый эксперимент.
Для HTTP tracking-сервера расположением артефактов управляет сервер.

Windows:

```powershell
.\scripts\start-local.ps1
.\scripts\start-mlflow.ps1
```

Сервер MLflow доступен локально: [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Проверки качества

```bash
poetry check --lock --strict
poetry run pre-commit run --all-files
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy
poetry run pytest -q
poetry run pytest -q -m "not slow"
poetry run pytest -q -m slow
poetry build
```

Полный pytest включает сквозной тест: обучение на 5000 строках, сохранение и повторная
загрузка модели, инференс в новую папку, одинаковый прогноз отдельно и в партии.
Тесты используют временные каталоги и отдельный MLflow store.

pre-commit запускает Ruff, Mypy, проверку lock-файла, YAML/TOML, конфликтов слияния,
пробелов и окончания файлов. Инструменты берутся из того же Poetry lock.
Ноутбуки и архивные исследования исключены из линтинга рабочего кода.

GitHub Actions проверяет стиль, типы и сборку пакета, запускает тесты на
Python 3.11/3.12 под Linux и Windows, затем собирает Docker и выполняет train/predict.

## Docker

```bash
docker compose build
docker compose up predict
docker compose up mlflow
```

`predict` стартует после успешного окончания `train`. Общий именованный том
`ml-artifacts` хранит модель и MLflow. Локальные `artifacts/` и данные Docker
разделены; `docker compose down` сохраняет том.

Образ использует зависимости из `poetry.lock`, отдельный этап сборки и пользователя
без root. UI опубликован только на `127.0.0.1:5000`.

## Структура

```text
src/                 конфигурация, ETL, train, predict, MLflow, мониторинг
tests/               модульные и интеграционные тесты
keis7-main/           воспроизводимые входные CSV
artifacts/           локальные результаты (исключены из Git)
reference-material/  архив исходного группового исследования и задания
docs/                материалы презентации
scripts/             команды запуска для PowerShell
pyproject.toml       пакет, зависимости, настройки качества
poetry.lock          точные версии и хеши зависимостей
poetry.toml          проектное виртуальное окружение
.pre-commit-config.yaml
.github/workflows/ci.yml
```

## Происхождение материалов

`artifacts/example_*.json`, прежние изображения в `docs/images/` и PPTX содержат
**исторические результаты исходного группового проекта**, включая модель с флагами
отказов. Они не являются метриками версии 0.2.0. Актуальные показатели получаются
после запуска и записываются в `artifacts/metrics.json`.

Архивные ноутбуки служат справочным материалом и не входят в устанавливаемый пакет.
Автор доработок и оформления форка: **Яньшин А.**

Настройка инструментов опирается на документацию
[Poetry](https://python-poetry.org/docs/configuration/#virtualenvsin-project),
[pre-commit](https://pre-commit.com/#repository-local-hooks) и
[Ruff](https://docs.astral.sh/ruff/configuration/).
