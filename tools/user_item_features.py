import polars as pl
import numpy as np
import json
from datetime import timedelta, datetime
from typing import Dict, List, Any


class FeatureUserItem:
    """
    Класс для извлечения и обработки признаков пользователей и объявлений (кандидатов)
    для моделей рекомендаций и ранжирования.
    """

    def __init__(self, config=None):
        """
        Инициализирует генератор признаков для пользователей и кандидатов

        Args:
            config (dict or omegaconf.DictConfig, optional): Конфигурация генератора признаков
        """
        # Хранение основных датафреймов
        self.clickstream_df: pl.DataFrame | None = None
        self.cat_features_df: pl.DataFrame | None = None
        self.text_features_df: pl.DataFrame | None = None
        self.events_df: pl.DataFrame | None = None

        # Параметры по умолчанию
        self.user_history_days = 30  # Период истории пользователя
        self.normalize_features = True  # Нормализовать числовые признаки
        self.use_cache = True  # Использовать кэш для ускорения
        self.clean_params_expand = True  # Разворачивать параметры объявлений
        self.max_params_per_item = (
            20  # Максимальное количество параметров объявления для обработки
        )

        # Кэш для хранения промежуточных результатов
        self._cache = {}

        # Применяем конфигурацию, если она указана
        if config:
            self._apply_config(config)

    def _apply_config(self, config):
        """Применяет параметры из конфигурации"""
        if hasattr(config, "user_history_days"):
            self.user_history_days = config.user_history_days
        if hasattr(config, "normalize_features"):
            self.normalize_features = config.normalize_features
        if hasattr(config, "use_cache"):
            self.use_cache = config.use_cache
        if hasattr(config, "clean_params_expand"):
            self.clean_params_expand = config.clean_params_expand
        if hasattr(config, "max_params_per_item"):
            self.max_params_per_item = config.max_params_per_item

    def set_data(
        self,
        clickstream_df: pl.DataFrame | None = None,
        cat_features_df: pl.DataFrame | None = None,
        text_features_df: pl.DataFrame | None = None,
        events_df: pl.DataFrame | None = None,
    ):
        """
        Устанавливает данные для работы

        Args:
            clickstream_df (pl.DataFrame, optional): DataFrame с кликстримом
            cat_features_df (pl.DataFrame, optional): DataFrame с категориальными признаками объявлений
            text_features_df (pl.DataFrame, optional): DataFrame с текстовыми признаками объявлений
            events_df (pl.DataFrame, optional): DataFrame с информацией о типах событий

        Returns:
            self: Возвращает self для цепочки вызовов
        """
        if clickstream_df is not None:
            self.clickstream_df = clickstream_df
        if cat_features_df is not None:
            self.cat_features_df = cat_features_df
        if text_features_df is not None:
            self.text_features_df = text_features_df
        if events_df is not None:
            self.events_df = events_df

        # Сбрасываем кэш при обновлении данных
        self._cache = {}
        return self

    def _parse_clean_params(self, clean_params_str: str) -> Dict[str, Any]:
        """
        Парсит json строку параметров объявления в словарь

        Args:
            clean_params_str (str): Строка с параметрами объявления в формате JSON

        Returns:
            Dict[str, Any]: Словарь параметров объявления
        """
        if not clean_params_str or clean_params_str == "[]":
            return {}

        try:
            params = json.loads(clean_params_str)
            result = {}

            # Ограничиваем количество параметров для обработки
            for i, param in enumerate(params):
                if i >= self.max_params_per_item:
                    break

                attr_id = param.get("attr")
                value_id = param.get("value")

                if attr_id is not None and value_id is not None:
                    key = f"attr_{attr_id}"
                    result[key] = value_id

            return result
        except Exception:
            # В случае ошибки возвращаем пустой словарь
            return {}

    def extract_item_features(self, nodes: List[int]) -> pl.DataFrame:
        """
        Извлекает признаки для заданных объявлений (групп товаров)

        Args:
            nodes (List[int]): Список ID групп товаров (node)

        Returns:
            pl.DataFrame: DataFrame с признаками объявлений
        """
        if self.cat_features_df is None:
            raise ValueError(
                "Не установлены данные с признаками объявлений (cat_features_df)"
            )

        # Идентификатор кэша
        cache_key = f"item_features_{hash(tuple(sorted(nodes)))}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key].clone()

        # Фильтруем только нужные объявления
        cat_features_filtered = self.cat_features_df.filter(pl.col("node").is_in(nodes))

        # Базовые признаки объявлений
        item_features = cat_features_filtered.select(["node", "location", "category"])

        # Добавляем текстовые признаки, если доступны
        if self.text_features_df is not None:
            # Соединяем с текстовыми признаками
            text_features_filtered = self.text_features_df.filter(
                pl.col("item").is_in(cat_features_filtered["item"])
            )

            # Можно добавить обработку текстовых векторов
            # Например, агрегация по node (если несколько item имеют один node)
            if not text_features_filtered.is_empty():
                # Здесь можно добавить дополнительную обработку текстовых векторов
                # Например, вычисление средних векторов для node
                pass

        # Обработка параметров объявлений
        if self.clean_params_expand and "clean_params" in cat_features_filtered.columns:
            # Разворачиваем параметры в отдельные столбцы
            # Предполагаем, что clean_params - это строка в формате JSON

            # Создаем функцию для обработки строки параметров
            expanded_params = cat_features_filtered.select(
                ["node", "clean_params"]
            ).with_columns(
                pl.col("clean_params").map_elements(self._parse_clean_params)
            )

            # Функция для преобразования словаря параметров в полярные колонки
            # Этот код требует расширения для работы с реальными данными
            # В боевом режиме нужна более сложная логика обработки параметров

        # Кэшируем результаты
        if self.use_cache:
            self._cache[cache_key] = item_features

        return item_features

    def extract_user_features(self, users: List[int]) -> pl.DataFrame:
        """
        Извлекает признаки для заданных пользователей

        Args:
            users (List[int]): Список ID пользователей (cookie)

        Returns:
            pl.DataFrame: DataFrame с признаками пользователей
        """
        if self.clickstream_df is None:
            raise ValueError("Не установлены данные кликстрима (clickstream_df)")

        # Идентификатор кэша
        cache_key = f"user_features_{hash(tuple(sorted(users)))}"
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key].clone()

        # Определяем временное окно для истории пользователя
        max_date = self.clickstream_df["event_date"].max()
        cutoff_date = max_date - timedelta(days=self.user_history_days)

        # Фильтруем кликстрим для нужных пользователей и временного окна
        user_data = self.clickstream_df.filter(
            (pl.col("cookie").is_in(users)) & (pl.col("event_date") > cutoff_date)
        )

        # Базовые признаки пользователей:
        # 1. Количество просмотров за период
        # 2. Количество уникальных категорий
        # 3. Средняя частота просмотров в день
        # 4. Доля контактных событий

        # Статистика по количеству просмотров
        view_stats = user_data.group_by("cookie").agg(
            pl.count().alias("total_views"),
            pl.n_unique("node").alias("unique_nodes"),
            pl.n_unique("category").alias("unique_categories")
            if "category" in user_data.columns
            else pl.lit(None).alias("unique_categories"),
            pl.n_unique("platform").alias("unique_platforms")
            if "platform" in user_data.columns
            else pl.lit(None).alias("unique_platforms"),
            pl.n_unique("surface").alias("unique_surfaces")
            if "surface" in user_data.columns
            else pl.lit(None).alias("unique_surfaces"),
        )

        # Если доступна информация о контактных событиях
        if self.events_df is not None and "event" in user_data.columns:
            # Присоединяем информацию о событиях
            user_events = user_data.join(self.events_df, on="event", how="left")

            # Считаем статистику по контактным событиям
            contact_stats = user_events.group_by("cookie").agg(
                pl.sum("is_contact").alias("contact_events"),
                (pl.sum("is_contact") / pl.count()).alias("contact_ratio"),
            )

            # Объединяем статистики
            user_features = view_stats.join(
                contact_stats, on="cookie", how="left"
            ).with_columns(
                pl.col("contact_events").fill_null(0),
                pl.col("contact_ratio").fill_null(0),
            )
        else:
            user_features = view_stats

        # Можно добавить дополнительные признаки
        # Например, временные признаки, распределение по категориям и т.д.

        # Кэшируем результаты
        if self.use_cache:
            self._cache[cache_key] = user_features

        return user_features

    def extract_user_item_features(self, user_item_pairs: pl.DataFrame) -> pl.DataFrame:
        """
        Извлекает признаки взаимодействия пользователь-объявление для пар (пользователь, объявление)

        Args:
            user_item_pairs (pl.DataFrame): DataFrame с парами (cookie, node)

        Returns:
            pl.DataFrame: DataFrame с признаками взаимодействия
        """
        if self.clickstream_df is None:
            raise ValueError("Не установлены данные кликстрима (clickstream_df)")

        # Получаем уникальные значения пользователей и объявлений
        unique_users = user_item_pairs["cookie"].unique().to_list()
        unique_nodes = user_item_pairs["node"].unique().to_list()

        # Идентификатор кэша
        cache_key = f"user_item_features_{hash((tuple(sorted(unique_users)), tuple(sorted(unique_nodes))))}"
        if self.use_cache and cache_key in self._cache:
            # Присоединяем кэшированный результат к исходным парам
            return user_item_pairs.join(
                self._cache[cache_key], on=["cookie", "node"], how="left"
            )

        # Определяем временное окно для истории
        max_date = self.clickstream_df["event_date"].max()
        cutoff_date = max_date - timedelta(days=self.user_history_days)

        # Фильтруем кликстрим для нужных пользователей, объявлений и временного окна
        interaction_data = self.clickstream_df.filter(
            (pl.col("cookie").is_in(unique_users))
            & (pl.col("node").is_in(unique_nodes))
            & (pl.col("event_date") > cutoff_date)
        )

        # Базовые признаки взаимодействия:
        # 1. Количество просмотров пользователем данного объявления
        # 2. Время с последнего просмотра
        # 3. Наличие контактного события

        # Статистика взаимодействий
        interaction_stats = interaction_data.group_by(["cookie", "node"]).agg(
            pl.count().alias("interaction_count"),
            pl.max("event_date").alias("last_interaction_date"),
            pl.min("event_date").alias("first_interaction_date"),
        )

        # Вычисляем время с последнего взаимодействия
        interaction_stats = interaction_stats.with_columns(
            (
                (max_date - pl.col("last_interaction_date")).dt.total_seconds() / 3600
            ).alias("hours_since_last_interaction"),
            (
                (
                    pl.col("last_interaction_date") - pl.col("first_interaction_date")
                ).dt.total_seconds()
                / 3600
            ).alias("interaction_duration_hours"),
        )

        # Если доступна информация о контактных событиях
        if self.events_df is not None and "event" in interaction_data.columns:
            # Присоединяем информацию о событиях
            interaction_events = interaction_data.join(
                self.events_df, on="event", how="left"
            )

            # Считаем статистику по контактным событиям
            contact_stats = interaction_events.group_by(["cookie", "node"]).agg(
                pl.sum("is_contact").alias("interaction_contacts"),
                (pl.sum("is_contact") / pl.count()).alias("interaction_contact_ratio"),
            )

            # Объединяем статистики
            interaction_features = interaction_stats.join(
                contact_stats, on=["cookie", "node"], how="left"
            ).with_columns(
                pl.col("interaction_contacts").fill_null(0),
                pl.col("interaction_contact_ratio").fill_null(0),
            )
        else:
            interaction_features = interaction_stats

        # Заполняем пропущенные значения для пар, которых нет в истории
        result = user_item_pairs.join(
            interaction_features, on=["cookie", "node"], how="left"
        ).with_columns(
            [
                pl.col("interaction_count").fill_null(0),
                pl.col("hours_since_last_interaction").fill_null(float("inf")),
                pl.col("interaction_duration_hours").fill_null(0),
            ]
        )

        # Если есть колонки с контактами, заполняем их нулями
        if "interaction_contacts" in result.columns:
            result = result.with_columns(
                pl.col("interaction_contacts").fill_null(0),
                pl.col("interaction_contact_ratio").fill_null(0),
            )

        # Кэшируем промежуточный результат
        if self.use_cache:
            self._cache[cache_key] = interaction_features

        return result

    def prepare_features_for_model(
        self,
        user_item_pairs: pl.DataFrame,
        include_user_features: bool = True,
        include_item_features: bool = True,
    ) -> pl.DataFrame:
        """
        Подготавливает полный набор признаков для модели, включая
        признаки пользователей, объявлений и их взаимодействий

        Args:
            user_item_pairs (pl.DataFrame): DataFrame с парами (cookie, node)
            include_user_features (bool): Включать ли признаки пользователей
            include_item_features (bool): Включать ли признаки объявлений

        Returns:
            pl.DataFrame: DataFrame с полным набором признаков
        """
        # Извлекаем признаки взаимодействия
        features_df = self.extract_user_item_features(user_item_pairs)

        # Добавляем признаки пользователей, если нужно
        if include_user_features:
            unique_users = user_item_pairs["cookie"].unique().to_list()
            user_features = self.extract_user_features(unique_users)

            # Присоединяем признаки пользователей
            features_df = features_df.join(
                user_features, on="cookie", how="left", suffix="_user"
            )

        # Добавляем признаки объявлений, если нужно
        if include_item_features:
            unique_nodes = user_item_pairs["node"].unique().to_list()
            item_features = self.extract_item_features(unique_nodes)

            # Присоединяем признаки объявлений
            features_df = features_df.join(
                item_features, on="node", how="left", suffix="_item"
            )

        # Нормализация числовых признаков, если включена
        if self.normalize_features:
            # Получаем числовые колонки
            numeric_cols = [
                col
                for col in features_df.columns
                if features_df[col].dtype
                in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                and col not in ["cookie", "node"]  # Исключаем идентификаторы
            ]

            # Нормализуем каждую числовую колонку
            for col in numeric_cols:
                col_max = features_df[col].max()
                col_min = features_df[col].min()

                # Избегаем деления на ноль
                if col_max != col_min:
                    features_df = features_df.with_columns(
                        ((pl.col(col) - col_min) / (col_max - col_min)).alias(
                            f"{col}_norm"
                        )
                    )

        return features_df

    def clear_cache(self):
        """Очищает кэш результатов"""
        self._cache = {}
        return self
