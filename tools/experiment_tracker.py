"""
Модуль для работы с ClearML: инициализация экспериментов, логирование данных,
загрузка и сохранение артефактов.
Поддерживает конфигурацию через Hydra или прямое задание параметров.
"""

import os
import logging
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from clearml import Task, Dataset, Model, OutputModel, Logger, TaskTypes

TASK_TYPES_CLEARML = {
    "training": TaskTypes.training,
    "testing": TaskTypes.testing,
    "inference": TaskTypes.inference,
    "data_processing": TaskTypes.data_processing,
    "optimizer": TaskTypes.optimizer,
    "custom": TaskTypes.custom,
}

try:
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logging.warning("ClearML не установлен. Установите с помощью: pip install clearml")


class ExperimentTracker:
    """
    Обертка для работы с ClearML для отслеживания экспериментов, логирования и работы с данными.
    Поддерживает конфигурацию через Hydra или прямую параметризацию.
    """

    def __init__(
        self,
        project_name: str = "Avito ML Cup 2025",
        task_name: Optional[str] = None,
        task_type: str = "training",
        config: Optional[Union[Dict[str, Any], DictConfig]] = None,
        use_clearml: bool = True,
        tags: Optional[List[str]] = None,
        reuse_last_task_id: bool = False,
        output_uri: Optional[str] = None,
    ):
        """
        Инициализация трекера экспериментов.

        Args:
            project_name: Название проекта в ClearML
            task_name: Название задачи/эксперимента. Если None, будет сгенерировано автоматически
            task_type: Тип задачи ('training', 'testing', 'inference', 'data_processing')
            config: Конфигурация эксперимента (словарь или DictConfig из Hydra)
            use_clearml: Флаг использования ClearML (можно отключить для локальных тестов)
            tags: Теги для задачи
            reuse_last_task_id: Переиспользовать последний Task ID (для продолжения эксперимента)
            output_uri: URI для сохранения моделей и артефактов (опционально)
        """
        self.config = config
        self.use_clearml = use_clearml and CLEARML_AVAILABLE
        self.task = None
        self.logger = None

        if self.use_clearml:
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                task_type=TASK_TYPES_CLEARML.get(task_type, TaskTypes.custom),
                reuse_last_task_id=reuse_last_task_id,
                output_uri=output_uri,
                tags=tags or [],
            )

            # Логируем конфигурацию
            if config:
                self.task.connect(config)

            self.logger = self.task.get_logger()
            logging.info(f"ClearML задача инициализирована: {self.task.id}")
        else:
            logging.info("ClearML отключен. Запуск в локальном режиме.")

    def log_scalar(
        self, title: str, series: str, value: float, iteration: int | None = None
    ) -> None:
        """
        Логирование скалярных значений (метрик).

        Args:
            title: Название графика
            series: Название серии данных
            value: Значение для логирования
            iteration: Итерация (шаг)
        """
        if self.use_clearml and self.logger:
            self.logger.report_scalar(
                title=title, series=series, value=value, iteration=iteration
            )
        logging.info(f"{title}/{series}: {value} (iteration={iteration})")

    def log_text(self, title: str, text: str, iteration: int | None = None) -> None:
        """
        Логирование текстовых данных.

        Args:
            title: Название текстового блока
            text: Текст для логирования
            iteration: Итерация (шаг)
        """
        if self.use_clearml and self.logger:
            self.logger.report_text(text, title=title, iteration=iteration)
        logging.info(f"{title}: {text}")

    def log_table(
        self,
        title: str,
        series: str,
        table: pd.DataFrame,
        iteration: int | None = None,
    ) -> None:
        """
        Логирование таблиц.

        Args:
            title: Название таблицы
            table: DataFrame для логирования
            iteration: Итерация (шаг)
        """
        if self.use_clearml and self.logger:
            self.logger.report_table(
                title=title, series=series, iteration=iteration, table_plot=table
            )
        logging.info(f"Logged table {title} with shape {table.shape}")

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Логирование гиперпараметров.

        Args:
            params: Словарь с гиперпараметрами
        """
        if self.use_clearml and self.task:
            self.task.connect(params)
        logging.info(f"Hyperparameters: {params}")

    def log_artifact(
        self, name: str, artifact: Any, upload_uri: str | None = None
    ) -> None:
        """
        Сохранение артефакта.

        Args:
            name: Название артефакта
            artifact: Объект для сохранения
            upload_uri: URI для загрузки артефакта (опционально)
        """
        if self.use_clearml and self.task:
            self.task.upload_artifact(
                name=name, artifact=artifact, upload_uri=upload_uri
            )

    def log_file(self, name: str, file_path: Union[str, Path]) -> None:
        """
        Сохранение файла как артефакта.

        Args:
            name: Название артефакта
            file_path: Путь к файлу
        """
        if self.use_clearml and self.task:
            self.task.upload_artifact(name=name, artifact_object=Path(file_path))

    def get_dataset(
        self,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        dataset_project: str | None = None,
        dataset_tags: List[str] | None = None,
    ) -> Optional[str]:
        """
        Получение датасета из ClearML.

        Args:
            dataset_id: ID датасета
            dataset_name: Название датасета
            dataset_project: Проект датасета
            dataset_tags: Теги датасета

        Returns:
            Путь к локальной копии датасета или None при ошибке
        """
        if not self.use_clearml:
            logging.warning("ClearML отключен, невозможно получить датасет")
            return None

        try:
            if dataset_id:
                dataset = Dataset.get(dataset_id=dataset_id)
            # else:
            #     datasets = Dataset.list_datasets(
            #         dataset_name=dataset_name,
            #         dataset_project=dataset_project,
            #         tags=dataset_tags,
            #     )
            #     if not datasets:
            #         logging.error(f"Датасет не найден: {dataset_name}")
            #         return None
            #     dataset = datasets[0]

            # Скачиваем датасет локально
            local_path = dataset.get_local_copy()  # type: ignore
            logging.info(f"Датасет скачан в: {local_path}")
            return local_path
        except Exception as e:
            logging.error(f"Ошибка при получении датасета: {e}")
            return None

    def save_model(
        self,
        model: Any,
        name: str,
        model_path: str,
        tags: List[str] | None = None,
        framework: str | None = None,
    ) -> Optional[str]:
        """
        Сохранение модели в ClearML.

        Args:
            model: Объект модели
            name: Название модели
            tags: Теги модели
            framework: Фреймворк модели (pytorch, tensorflow, etc.)
            model_path: Путь для сохранения (опционально)

        Returns:
            ID сохраненной модели или None при ошибке
        """
        try:
            output_model = OutputModel(task=self.task, name=name, tags=tags)
            # Общий случай для других моделей

            output_model.update_weights(weights_filename=model_path)

            return output_model.id
        except Exception as e:
            logging.error(f"Ошибка при сохранении модели: {e}")
            return None

    # def load_model(
    #     self,
    #     model_id: str | None = None,
    #     name: str | None = None,
    #     tags: List[str] | None = None,
    # ) -> Any:
    #     """
    #     Загрузка модели из ClearML.

    #     Args:
    #         model_id: ID модели
    #         name: Название модели
    #         tags: Теги модели

    #     Returns:
    #         Загруженная модель или None при ошибке
    #     """
    #     if not self.use_clearml:
    #         logging.warning("ClearML отключен, невозможно загрузить модель")
    #         return None

    #     try:
    #         if model_id:
    #             model = Model(model_id=model_id)
    #         else:
    #             models = Model.query_models(
    #                 model_name=name, tags=tags, only_published=True
    #             )
    #             if not models:
    #                 logging.error(f"Модель не найдена: {name}")
    #                 return None
    #             model = models[0]

    #         # Получаем локальную копию модели
    #         local_weights_path = model.get_local_weights()

    #         # Здесь требуется дополнительная логика для загрузки модели
    #         # в зависимости от фреймворка
    #         logging.info(f"Модель скачана в: {local_weights_path}")

    #         return local_weights_path
    #     except Exception as e:
    #         logging.error(f"Ошибка при загрузке модели: {e}")
    #         return None

    def close(self) -> None:
        """
        Завершение эксперимента и закрытие ClearML задачи.
        """
        if self.use_clearml and self.task:
            logging.info(f"Завершение ClearML задачи: {self.task.id}")
            self.task.close()


def create_experiment_from_hydra(cfg: DictConfig) -> ExperimentTracker:
    """
    Создание трекера экспериментов из конфигурации Hydra.

    Args:
        cfg: Конфигурация Hydra

    Returns:
        Инициализированный трекер экспериментов
    """
    # Извлекаем параметры для ClearML из конфигурации
    tracking_cfg = cfg.get("tracking", {})

    return ExperimentTracker(
        project_name=tracking_cfg.get("project_name", "Avito ML Cup 2025"),
        task_name=tracking_cfg.get("task_name"),
        task_type=tracking_cfg.get("task_type", "training"),
        config=cfg,
        use_clearml=tracking_cfg.get("use_clearml", True),
        tags=tracking_cfg.get("tags"),
        reuse_last_task_id=tracking_cfg.get("reuse_last_task_id", False),
        output_uri=tracking_cfg.get("output_uri"),
    )


# Пример использования в скрипте с Hydra
"""
import hydra
from omegaconf import DictConfig
from tools.experiment_tracker import create_experiment_from_hydra

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Инициализируем трекер экспериментов
    tracker = create_experiment_from_hydra(cfg)
    
    # Логируем метрики
    tracker.log_scalar("Training Metrics", "loss", 0.123, iteration=1)
    
    # Закрываем эксперимент
    tracker.close()

if __name__ == "__main__":
    main()
"""

# Пример использования в Jupyter Notebook
"""
from tools.experiment_tracker import ExperimentTracker

# Инициализируем трекер напрямую
tracker = ExperimentTracker(
    project_name="Avito ML Cup 2025 - Notebook",
    task_name="Exploratory Data Analysis",
    task_type="data_processing",
    config={"data_path": "data/train.csv", "sample_size": 1000},
    tags=["notebook", "eda"]
)

# Логируем данные
tracker.log_text("Data Summary", "Анализ данных для задачи классификации")

# Загружаем датасет
dataset_path = tracker.get_dataset(dataset_name="avito_train_data")

# Закрываем эксперимент
tracker.close()
"""
