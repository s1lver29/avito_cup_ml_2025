from datetime import timedelta
from typing import Type

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


class CFRetriever:
    def __init__(self, model, **params_model):
        self.model = model(**params_model)

        self._user_id_to_index: dict[int, int] | None = None
        self._item_id_to_index: dict[int, int] | None = None
        self._index_to_item_id: dict[int, int] | None = None
        self._sparse_matrix = None

    def _create_mappings(self, user_ids, item_ids):
        """Создает маппинги между ID и индексами для пользователей и товаров"""
        self._user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self._item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self._index_to_item_id = {v: k for k, v in self._item_id_to_index.items()}
        self._index_to_user_id = {v: k for k, v in self._user_id_to_index.items()}

    def fit(
        self,
        users,
        nodes,
        events=None,
        event_weights=None,
        contact_event_boost=10.0,
        contact_events=None,
        is_contact_col=None,
    ):
        """
        Обучает модель RecommenderBase (ALS, BPR)

        Args:
            users (pl.Series): ID пользователей
            nodes (pl.Series): ID элементов (товаров)
            events (pl.Series, optional): Типы событий
            event_weights (dict, optional): Словарь весов для разных типов событий
            contact_event_boost (float): Множитель для контактных событий
            contact_events (list, optional): Список ID контактных событий
            is_contact_col (pl.Series, optional): Колонка с флагом контакта

        Returns:
            self: Возвращает self для возможности цепочки вызовов
        """
        user_ids = users.unique().to_list()
        item_ids = nodes.unique().to_list()

        self._create_mappings(user_ids, item_ids)

        rows = users.replace_strict(self._user_id_to_index).to_list()
        cols = nodes.replace_strict(self._item_id_to_index).to_list()

        # Определение весов взаимодействий
        if events is not None and event_weights:
            values = [event_weights.get(event, 1.0) for event in events.to_list()]

            # Если указаны контактные события, увеличиваем их вес
            if contact_events:
                values = [
                    val * contact_event_boost if events[i] in contact_events else val
                    for i, val in enumerate(values)
                ]
        elif is_contact_col is not None:
            # Если есть прямой признак контакта, используем его для весов
            values = is_contact_col.to_list()
        else:
            # По умолчанию все взаимодействия имеют вес 1
            values = [1] * len(users)

        # Создаем разреженную матрицу взаимодействий
        self._sparse_matrix = csr_matrix(
            (values, (rows, cols)), shape=(len(user_ids), len(item_ids))
        )

        # Обучаем модель
        self.model.fit(self._sparse_matrix)

    def recommend(self, user_ids, n=40, filter_already_liked_items=True):
        """
        Генерирует рекомендации для указанных пользователей

        Args:
            user_ids (list): Список ID пользователей для рекомендаций
            n (int): Количество рекомендаций для каждого пользователя
            filter_already_liked_items (bool): Исключать ли уже взаимодействовавшие товары
            recalculate_user (bool): Пересчитывать ли модель для пользователя

        Returns:
            pl.DataFrame: DataFrame с рекомендациями
        """
        if not self._user_id_to_index:
            raise ValueError("Сначала надо fit.")
        # Преобразуем ID пользователей в индексы
        user_indices = []
        valid_users = []

        for user_id in user_ids:
            if user_id in self._user_id_to_index:
                user_indices.append(self._user_id_to_index[user_id])  # type: ignore
                valid_users.append(user_id)

        if not user_indices:
            return pl.DataFrame({"cookie": [], "node": [], "scores": []})

        # Получаем рекомендации
        user_index_array = np.array(user_indices)
        recommendations, scores = self.model.recommend(
            user_index_array,
            self._sparse_matrix[user_index_array],  # type: ignore
            N=n,
            filter_already_liked_items=filter_already_liked_items,
        )

        # Преобразуем индексы товаров обратно в ID
        node_lists = [
            [self._index_to_item_id[idx] for idx in rec] for rec in recommendations
        ]  # type: ignore

        # Создаем DataFrame с результатами
        df_pred = pl.DataFrame(
            {"node": node_lists, "cookie": valid_users, "scores": scores.tolist()}
        )

        # Разворачиваем списки
        df_pred = df_pred.explode(["node", "scores"])

        return df_pred

    def save_model(self, path: str):
        import pickle

        model_data = {
            "model": self.model,
            "user_id_to_index": self._user_id_to_index,
            "item_id_to_index": self._item_id_to_index,
            "index_to_item_id": self._index_to_item_id,
            "index_to_user_id": self._index_to_user_id,
            "sparse_matrix": self._sparse_matrix,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, model, path):
        """
        Загружает модель из файла

        Args:
            path (str): Путь к файлу модели

        Returns:
            RecommenderBase: Загруженная модель
        """
        import pickle

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        retriever = cls(model)
        retriever.model = model_data["model"]
        retriever._user_id_to_index = model_data["user_id_to_index"]
        retriever._item_id_to_index = model_data["item_id_to_index"]
        retriever._index_to_item_id = model_data["index_to_item_id"]
        retriever._index_to_user_id = model_data["index_to_user_id"]
        retriever._sparse_matrix = model_data["sparse_matrix"]

        return retriever


class PopularItemsRetriever:
    """
    Класс для поиска популярных товаров по различным критериям
    """

    def __init__(self, config=None):
        """
        Инициализирует ретривер популярных товаров

        Args:
            config (dict or omegaconf.DictConfig, optional): Конфигурация ретривера
        """
        self.clickstream_df: pl.DataFrame | None = None
        self.categories_df: pl.DataFrame | None = None

        # Кэш для хранения результатов
        self._cache = {}

        # Параметры по умолчанию
        self.default_days = 7
        self.default_top_k = 100
        self.use_cache = False
        self.trending_window_days = 3
        self.trending_prev_days = 7

        self.methods = []

        self.join_type = "full"
        self.combine_scores = False

        # Применяем конфигурацию, если она указана
        if config:
            self._apply_config(config)

    def _apply_config(self, config):
        """Применяет параметры из конфигурации"""
        if hasattr(config, "default_days"):
            self.default_days = config.default_days
        if hasattr(config, "default_top_k"):
            self.default_top_k = config.default_top_k
        if hasattr(config, "use_cache"):
            self.use_cache = config.use_cache
        if hasattr(config, "trending"):
            if hasattr(config.trending, "window_days"):
                self.trending_window_days = config.trending.window_days
            if hasattr(config.trending, "previous_days"):
                self.trending_prev_days = config.trending.previous_days

        if hasattr(config, "join_type"):
            self.join_type = config.join_type

        if hasattr(config, "methods"):
            for method_config in config.methods:
                self.methods.append(method_config)

        if not self.methods:
            self.methods = [
                {
                    "id": "popular_default",
                    "function": "get_popular_items",
                    "output_column": "score",
                    "params": {
                        "days": self.default_days,
                        "top_k": self.default_top_k,
                        "use_cache": self.use_cache,
                    },
                }
            ]

    def set_data(self, clickstream_df, categories_df=None):
        """
        Устанавливает данные для работы

        Args:
            clickstream_df (pl.DataFrame): DataFrame с данными кликстрима
            categories_df (pl.DataFrame, optional): DataFrame с категориями товаров

        Returns:
            self: Возвращает self для цепочки вызовов
        """
        self.clickstream_df = clickstream_df
        if categories_df is not None:
            self.categories_df = categories_df

        # Сбрасываем кэш при обновлении данных
        self._cache = {}

    def get_popular_items(self, days=None, top_k=None, use_cache=None):
        """
        Получает популярные товары за указанный период

        Args:
            days (int, optional): Количество дней для анализа
            top_k (int, optional): Количество лучших товаров
            use_cache (bool, optional): Использовать кэш или пересчитать

        Returns:
            pl.DataFrame: Таблица с популярными товарами и количеством взаимодействий
        """
        if self.clickstream_df is None:
            raise ValueError("Не установлены данные кликстрима")

        # Используем значения из конфигурации, если параметры не указаны
        days = days if days is not None else self.default_days
        top_k = top_k if top_k is not None else self.default_top_k
        use_cache = use_cache if use_cache is not None else self.use_cache

        cache_key = f"popular_{days}_{top_k}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Определяем временное окно
        cutoff_date = self.clickstream_df["event_date"].max() - timedelta(days=days)
        recent_data = self.clickstream_df.filter(pl.col("event_date") > cutoff_date)

        # Подсчитываем популярность каждого node
        popular_nodes = (
            recent_data.group_by("node")
            .agg(pl.count().alias("score"))
            .sort("score", descending=True)
            .head(top_k)
            .with_columns(
                np.log1p(pl.col("score").cast(pl.Float32))
                / np.log1p(pl.col("score").max())
            )
        )[["node", "score"]]

        # Кэшируем результаты
        if use_cache:
            self._cache[cache_key] = popular_nodes

        return popular_nodes

    def get_popular_items_by_category(self, days=None, top_k=None, use_cache=None):
        """
        Получает популярные товары по категориям за указанный период

        Args:
            days (int, optional): Количество дней для анализа
            top_k (int, optional): Количество лучших товаров на категорию
            use_cache (bool, optional): Использовать кэш или пересчитать

        Returns:
            pl.DataFrame: Таблица с популярными товарами по категориям
        """
        if self.clickstream_df is None or self.categories_df is None:
            raise ValueError("Не установлены данные кликстрима или категорий товаров")

        # Используем значения из конфигурации, если параметры не указаны
        days = days if days is not None else self.default_days
        top_k = (
            top_k if top_k is not None else self.default_top_k // 5
        )  # Меньше товаров на категорию
        use_cache = use_cache if use_cache is not None else self.use_cache

        cache_key = f"popular_by_category_{days}_{top_k}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Определяем временное окно
        cutoff_date = self.clickstream_df["event_date"].max() - timedelta(days=days)
        recent_data = self.clickstream_df.filter(pl.col("event_date") > cutoff_date)

        # Соединяем с категориями
        recent_with_cat = recent_data.join(
            self.categories_df[["node", "category"]].unique(), on="node", how="inner"
        )

        # Подсчитываем популярность по категориям
        popular_by_cat = (
            recent_with_cat.group_by(["category", "node"])
            .agg(pl.count().alias("score"))
            .sort(["category", "score"], descending=[False, True])
        )

        # Берем top_k для каждой категории
        result = popular_by_cat.group_by("category").head(top_k)[
            ["node", "score", "category"]
        ]

        result = result.with_columns(
            np.log1p(pl.col("score").cast(pl.Float32))
            / np.log1p(pl.col("score").max().over("category"))
        )

        # Кэшируем результаты
        if use_cache:
            self._cache[cache_key] = result

        return result

    def get_trending_items(self, days_window=None, previous_days=None, top_k=None):
        """
        Получает тренды - товары, растущие по популярности

        Args:
            days_window (int, optional): Размер окна для текущей популярности (дни)
            previous_days (int, optional): Количество дней для сравнения с предыдущим периодом
            top_k (int, optional): Количество растущих товаров для возврата

        Returns:
            pl.DataFrame: Таблица с растущими товарами
        """
        if self.clickstream_df is None:
            raise ValueError("Не установлены данные кликстрима")

        # Используем значения из конфигурации, если параметры не указаны
        days_window = (
            days_window if days_window is not None else self.trending_window_days
        )
        previous_days = (
            previous_days if previous_days is not None else self.trending_prev_days
        )
        top_k = top_k if top_k is not None else self.default_top_k

        max_date = self.clickstream_df["event_date"].max()

        # Данные для текущего окна
        current_cutoff = max_date - timedelta(days=days_window)
        current_data = self.clickstream_df.filter(pl.col("event_date") > current_cutoff)

        # Данные для предыдущего окна
        prev_start = current_cutoff - timedelta(days=previous_days)
        prev_data = self.clickstream_df.filter(
            (pl.col("event_date") <= current_cutoff)
            & (pl.col("event_date") > prev_start)
        )

        # Вычисляем популярность для текущего и предыдущего окна
        current_pop = current_data.group_by("node").agg(
            pl.count().alias("current_count")
        )
        prev_pop = prev_data.group_by("node").agg(pl.count().alias("previous_count"))

        # Объединяем и вычисляем рост
        trending = (
            current_pop.join(prev_pop, on="node", how="left")
            .with_columns(
                [
                    pl.col("previous_count").fill_null(0),
                    (pl.col("current_count") - pl.col("previous_count")).alias(
                        "growth"
                    ),
                    (
                        (pl.col("current_count") - pl.col("previous_count"))
                        / (pl.col("previous_count") + 1)
                    ).alias("score"),
                ]
            )
            .sort("score", descending=True)
            .head(top_k)
            .with_columns(np.log1p(pl.col("score")) / np.log1p(pl.col("score").max()))
        )

        return trending

    def get_recommendations_from_method(self, method_config, users):
        """
        Получает рекомендации с помощью указанного метода

        Args:
            method_config (dict): Конфигурация метода
            users (list): Список пользователей для рекомендаций
            n_per_method (int): Количество рекомендаций от одного метода

        Returns:
            pl.DataFrame: DataFrame с рекомендациями
        """
        method_name = method_config["function"]
        params = method_config["params"]
        output_column = method_config["output_column"]

        # Вызываем соответствующий метод с параметрами
        method = getattr(self, method_name)
        items_df = method(**params)

        recommendations = pl.DataFrame(
            {
                "cookie": [user for user in users for _ in range(len(items_df))],
                "node": items_df["node"].to_list() * len(users),
                output_column: items_df["score"].to_list() * len(users),
            }
        )

        if not recommendations.is_empty():
            return recommendations

        return pl.DataFrame({"cookie": [], "node": [], "scores": []})

    def recommend(self, users, n=40):
        """
        Генерирует рекомендации популярных товаров для пользователей,
        комбинируя результаты разных методов согласно конфигурации

        Args:
            users (list): Список ID пользователей
            n (int): Количество рекомендаций для каждого пользователя

        Returns:
            pl.DataFrame: DataFrame с рекомендациями
        """
        if not self.methods:
            # Если методы не настроены, используем get_popular_items
            raise ValueError("Нужны методы.")

        # Объединяем рекомендации от всех методов
        all_recommendations = []

        for method_config in self.methods:
            method_recommendations = self.get_recommendations_from_method(
                method_config, users
            )
            all_recommendations.append(method_recommendations)

        if not all_recommendations:
            return pl.DataFrame({"cookie": [], "node": [], "scores": []})

        combined_df: pl.DataFrame = all_recommendations[0]

        for i in range(1, len(all_recommendations)):
            combined_df = combined_df.join(
                all_recommendations[i],
                on=["cookie", "node"],
                how=self.join_type,  # type: ignore
                coalesce=True,
            )

        return combined_df

    def clear_cache(self):
        """Очищает кэш результатов"""
        self._cache = {}
        return self
