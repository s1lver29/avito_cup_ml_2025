import polars as pl
from datetime import timedelta
import concurrent.futures
import tempfile
import os
import gc
import pickle
from pathlib import Path
from catboost.utils import get_gpu_device_count

print(get_gpu_device_count())

import implicit

print(implicit.gpu.HAS_CUDA)


def get_cat_and_text_features_worker(data_dir_str: str, output_path: str):
    """Worker для обработки cat и text features"""
    try:
        print(f"Процесс {os.getpid()} начал обработку cat_and_text_features...")

        df_cat_features = pl.read_parquet(f"{data_dir_str}/cat_features.pq")
        df_text_features = pl.read_parquet(f"{data_dir_str}/text_features.pq")

        cast_types = {
            "category": pl.Int16,
            "location_mode": pl.Int16,
        }
        for i in range(64):
            cast_types[f"emb_{i}"] = pl.Int8

        df_text_features = df_text_features.with_columns(
            [pl.col("title_projection").arr.get(i).alias(f"t_{i}") for i in range(64)]
        )

        df_cat_features = df_cat_features.join(df_text_features, on="item", how="left")

        agg_exprs = [pl.mean(f"t_{i}").alias(f"emb_{i}") for i in range(64)] + [
            pl.col("location").mode().first().alias("location_mode"),
        ]

        result = (
            df_cat_features.group_by(["node", "category"])
            .agg(agg_exprs)
            .cast(cast_types)
        )

        # Сохраняем результат
        result.write_parquet(output_path)
        size_mb = result.estimated_size() / (1024**2)

        return {"status": "success", "size_mb": size_mb, "output_path": output_path}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def prepare_data_splits_worker(
    clickstream_path: str,
    event_path: str,
    valid_days_retriever: int,
    valid_days_reranker: int,
    output_dir: str,
):
    """Worker для разделения данных на train/validation"""
    try:
        print(f"Процесс {os.getpid()} начал разделение данных...")

        df_clickstream = pl.read_parquet(clickstream_path)
        df_event = pl.read_parquet(event_path)

        # Определяем границы для разделения данных
        max_date = df_clickstream["event_date"].max()
        retrivial_threshold = max_date - timedelta(
            days=valid_days_retriever + valid_days_reranker
        )  # type: ignore
        reranker_threshold = max_date - timedelta(days=valid_days_reranker)  # type: ignore

        # Разделяем данные
        df_train_retrivial = df_clickstream.filter(
            pl.col("event_date") < retrivial_threshold
        )
        df_train_reranker = df_clickstream.filter(
            pl.col("event_date").is_between(retrivial_threshold, reranker_threshold)
        )
        df_valid = df_clickstream.filter(pl.col("event_date") > reranker_threshold)

        # Создаем оценочные наборы
        def prepare_eval_set(df_eval_raw, df_train_data):
            df_eval = df_eval_raw[["cookie", "node", "event"]]
            df_eval = df_eval.join(df_train_data, on=["cookie", "node"], how="anti")
            df_eval = df_eval.filter(
                pl.col("event").is_in(
                    df_event.filter(pl.col("is_contact") == 1)["event"].unique()
                )
            )
            df_eval = df_eval.filter(
                pl.col("cookie").is_in(df_train_data["cookie"].unique())
            ).filter(pl.col("node").is_in(df_train_data["node"].unique()))
            df_eval = df_eval.unique(["cookie", "node"])
            return df_eval

        df_valid_eval_retrivial = prepare_eval_set(
            df_train_reranker, df_train_retrivial
        )
        df_valid_eval_reranker = prepare_eval_set(
            df_valid, pl.concat([df_train_retrivial, df_train_reranker])
        )

        # Присоединяем события
        df_train_retrivial = df_train_retrivial.join(df_event, on="event")
        df_train_reranker = df_train_reranker.join(df_event, on="event")
        df_valid = df_valid.join(df_event, on="event")

        # Сохраняем все части
        output_dir = Path(output_dir)
        df_train_retrivial.write_parquet(output_dir / "train_retrivial.pq")
        df_train_reranker.write_parquet(output_dir / "train_reranker.pq")
        df_valid.write_parquet(output_dir / "valid.pq")
        df_valid_eval_retrivial.write_parquet(output_dir / "valid_eval_retrivial.pq")
        df_valid_eval_reranker.write_parquet(output_dir / "valid_eval_reranker.pq")

        return {"status": "success", "output_dir": str(output_dir)}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def train_als_worker(
    train_retrivial_path: str,
    valid_eval_retrivial_path: str,
    model_output_path: str,
    predictions_output_path: str,
):
    """Worker для обучения ALS модели"""
    try:
        print(f"Процесс {os.getpid()} начал обучение ALS...")

        # Импортируем внутри worker
        from implicit.als import AlternatingLeastSquares
        from retrievers import CFRetriever
        import numpy as np

        df_train_retrivial = pl.read_parquet(train_retrivial_path)
        df_valid_eval_retrivial = pl.read_parquet(valid_eval_retrivial_path)

        # Создаем модель ALS
        als_retriever = CFRetriever(
            AlternatingLeastSquares,
            iterations=50,
            factors=100,
            regularization=0.3,
            use_gpu=True,
            random_state=123,
        )

        users = df_train_retrivial["cookie"]
        nodes = df_train_retrivial["node"]
        events = df_train_retrivial["event"]
        eval_users = df_valid_eval_retrivial["cookie"].unique().to_list()

        # Создаем веса
        def create_event_weights(clickstream_df: pl.DataFrame):
            event_count = {
                row[0]: row[1] for row in clickstream_df.group_by("event").len().rows()
            }
            return {
                event_id: 1.0 / np.log1p(count)
                for event_id, count in event_count.items()
            }

        inverse_freq_weights = create_event_weights(df_train_retrivial.select("event"))

        # Обучаем модель
        als_retriever.fit(
            users,
            nodes,
            events,
            inverse_freq_weights,
            100,
            list(inverse_freq_weights.keys()),
        )

        # Получаем предсказания
        df_pred_alt = als_retriever.recommend(eval_users, 700)

        # Сохраняем модель и предсказания
        als_retriever.save_model(model_output_path)
        df_pred_alt.write_parquet(predictions_output_path)

        return {
            "status": "success",
            "model_path": model_output_path,
            "predictions_path": predictions_output_path,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def prepare_features_worker(
    predictions_path: str,
    train_retrivial_path: str,
    cat_features_path: str,
    output_path: str,
):
    """Worker для подготовки признаков"""
    try:
        print(f"Процесс {os.getpid()} начал подготовку признаков...")

        df_pred_alt = pl.read_parquet(predictions_path)
        df_train_retrivial = pl.read_parquet(train_retrivial_path)
        df_cat_features = pl.read_parquet(cat_features_path)

        # Используем функцию prepare_features_for_boosting
        def prepare_features_for_boosting(
            df_retrieval, df_clickstream, df_cat_features
        ):
            max_date = df_clickstream["event_date"].max()
            features = df_retrieval.rename({"scores": "als_score"})

            # Пользовательские признаки
            user_features = (
                df_clickstream.group_by("cookie")
                .agg(
                    [
                        pl.len().alias("user_total_interactions"),
                        pl.n_unique("node").alias("user_unique_nodes"),
                        pl.n_unique("event").alias("user_unique_events"),
                        pl.col("is_contact").sum().alias("user_total_contacts"),
                        (pl.col("is_contact").sum() / pl.len()).alias(
                            "user_contact_ratio"
                        ),
                        pl.col("event_date").min().alias("user_first_interaction"),
                        pl.col("event_date").max().alias("user_last_interaction"),
                    ]
                )
                .with_columns(
                    [
                        (
                            pl.col("user_last_interaction")
                            - pl.col("user_first_interaction")
                        )
                        .dt.total_days()
                        .alias("user_activity_span_days"),
                        (max_date - pl.col("user_last_interaction"))
                        .dt.total_days()
                        .alias("days_since_user_last_activity"),
                        (max_date - pl.col("user_first_interaction"))
                        .dt.total_days()
                        .alias("user_account_age_days"),
                    ]
                )
                .drop(["user_first_interaction", "user_last_interaction"])
            )

            features = features.join(user_features, on="cookie", how="left").fill_null(
                0
            )
            del user_features

            # Активность пользователей за последние дни
            for days in [3, 7, 14, 30]:
                recent_threshold = max_date - timedelta(days=days)
                recent_user_activity = (
                    df_clickstream.filter(pl.col("event_date") > recent_threshold)
                    .group_by("cookie")
                    .agg(
                        [
                            pl.len().alias(f"user_interactions_last_{days}d"),
                            pl.n_unique("node").alias(
                                f"user_unique_nodes_last_{days}d"
                            ),
                            pl.col("is_contact")
                            .sum()
                            .alias(f"user_contacts_last_{days}d"),
                        ]
                    )
                )
                features = features.join(
                    recent_user_activity, on="cookie", how="left"
                ).fill_null(0)
                del recent_user_activity

            # Признаки товаров
            item_features = (
                df_clickstream.group_by("node")
                .agg(
                    [
                        pl.len().alias("item_total_interactions"),
                        pl.n_unique("cookie").alias("item_unique_users"),
                        pl.col("is_contact").sum().alias("item_total_contacts"),
                        (pl.col("is_contact").sum() / pl.len()).alias(
                            "item_contact_ratio"
                        ),
                        pl.col("event_date").min().alias("item_first_seen"),
                        pl.col("event_date").max().alias("item_last_seen"),
                    ]
                )
                .with_columns(
                    [
                        (max_date - pl.col("item_first_seen"))
                        .dt.total_days()
                        .alias("item_age_days"),
                        (max_date - pl.col("item_last_seen"))
                        .dt.total_days()
                        .alias("days_since_item_last_interaction"),
                    ]
                )
                .drop(["item_first_seen", "item_last_seen"])
            )

            features = features.join(item_features, on="node", how="left").fill_null(0)
            del item_features

            # Добавляем информацию о категории товара
            features = features.join(df_cat_features, on="node", how="left").fill_null(
                -1
            )

            # Признаки категорий
            category_features = (
                df_clickstream.join(
                    df_cat_features.select(["node", "category"]).unique(),
                    on="node",
                    how="left",
                )
                .group_by("category")
                .agg(
                    [
                        pl.len().alias("category_total_interactions"),
                        pl.n_unique("node").alias("category_unique_items"),
                        pl.n_unique("cookie").alias("category_unique_users"),
                        pl.col("is_contact").sum().alias("category_total_contacts"),
                        (pl.col("is_contact").sum() / pl.len()).alias(
                            "category_contact_ratio"
                        ),
                    ]
                )
            )

            features = features.join(
                category_features, on=["category"], how="left"
            ).fill_null(-1)
            del category_features

            # Кросс-признаки
            features = features.with_columns(
                [
                    (
                        pl.col("item_total_interactions").log1p()
                        * pl.col("user_total_interactions").log1p()
                    ).alias("item_user_interaction_product"),
                    (pl.col("user_contact_ratio") * pl.col("item_contact_ratio")).alias(
                        "user_item_contact_affinity"
                    ),
                    pl.when(pl.col("item_age_days") < 7)
                    .then(1)
                    .otherwise(0)
                    .alias("is_new_item"),
                    pl.when(pl.col("days_since_user_last_activity") < 3)
                    .then(1)
                    .otherwise(0)
                    .alias("is_recently_active_user"),
                    (
                        pl.col("item_total_interactions")
                        / pl.col("category_total_interactions")
                    ).alias("item_category_popularity_ratio"),
                ]
            )

            return features

        # Подготавливаем признаки
        features = prepare_features_for_boosting(
            df_pred_alt, df_train_retrivial, df_cat_features
        )

        # Сохраняем результат
        features.write_parquet(output_path)

        return {"status": "success", "output_path": output_path}

    except Exception as e:
        return {"status": "error", "error": str(e)}


class DataPipelineManager:
    """Менеджер для управления pipeline обработки данных"""

    def __init__(
        self, data_dir: str, valid_days_retriever: int = 7, valid_days_reranker: int = 7
    ):
        self.data_dir = Path(data_dir)
        self.valid_days_retriever = valid_days_retriever
        self.valid_days_reranker = valid_days_reranker
        self.temp_files = []

    def create_temp_file(self, suffix: str) -> str:
        """Создает временный файл и добавляет его в список для очистки"""
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = temp_file.name
        temp_file.close()
        self.temp_files.append(temp_path)
        return temp_path

    def create_temp_dir(self) -> str:
        """Создает временную директорию"""
        temp_dir = tempfile.mkdtemp()
        self.temp_files.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Очистка всех временных файлов"""
        for temp_path in self.temp_files:
            try:
                if os.path.isdir(temp_path):
                    import shutil

                    shutil.rmtree(temp_path)
                else:
                    os.unlink(temp_path)
            except:
                pass
        self.temp_files.clear()

    def run_pipeline(self, timeout: int = 1800):
        """Запускает полный pipeline обработки данных"""
        try:
            print("=== Запуск pipeline обработки данных ===")

            # Шаг 1: Обработка cat и text features
            print("Шаг 1: Обработка cat и text features...")
            cat_features_path = self.create_temp_file(".pq")

            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    get_cat_and_text_features_worker,
                    str(self.data_dir),
                    cat_features_path,
                )
                result = future.result(timeout=timeout)

                if result["status"] != "success":
                    raise RuntimeError(
                        f"Ошибка обработки cat features: {result['error']}"
                    )

                print(f"✅ Cat features обработаны: {result['size_mb']:.2f} MB")

            # Шаг 2: Разделение данных
            print("Шаг 2: Разделение данных на train/validation...")
            splits_dir = self.create_temp_dir()

            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    prepare_data_splits_worker,
                    str(self.data_dir / "clickstream.pq"),
                    str(self.data_dir / "events.pq"),
                    self.valid_days_retriever,
                    self.valid_days_reranker,
                    splits_dir,
                )
                result = future.result(timeout=timeout)

                if result["status"] != "success":
                    raise RuntimeError(f"Ошибка разделения данных: {result['error']}")

                print("✅ Данные разделены на train/validation")

            # Шаг 3: Обучение ALS
            print("Шаг 3: Обучение ALS модели...")
            model_path = self.create_temp_file(".pkl")
            predictions_path = self.create_temp_file(".pq")

            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    train_als_worker,
                    os.path.join(splits_dir, "train_retrivial.pq"),
                    os.path.join(splits_dir, "valid_eval_retrivial.pq"),
                    model_path,
                    predictions_path,
                )
                result = future.result(timeout=timeout)

                if result["status"] != "success":
                    raise RuntimeError(f"Ошибка обучения ALS: {result['error']}")

                print("✅ ALS модель обучена")

            # Шаг 4: Подготовка признаков
            print("Шаг 4: Подготовка признаков для CatBoost...")
            features_path = self.create_temp_file(".pq")

            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    prepare_features_worker,
                    predictions_path,
                    os.path.join(splits_dir, "train_retrivial.pq"),
                    cat_features_path,
                    features_path,
                )
                result = future.result(timeout=timeout)

                if result["status"] != "success":
                    raise RuntimeError(
                        f"Ошибка подготовки признаков: {result['error']}"
                    )

                print("✅ Признаки подготовлены")

            # Загружаем финальные результаты в основном процессе
            print("Загрузка результатов в основной процесс...")
            df_features = pl.read_parquet(features_path)
            df_train_reranker = pl.read_parquet(
                os.path.join(splits_dir, "train_reranker.pq")
            )

            print(f"✅ Pipeline завершен успешно")
            print(
                f"Размер признаков: {df_features.estimated_size() / (1024**2):.2f} MB"
            )
            print(f"Форма признаков: {df_features.shape}")

            return df_features, df_train_reranker

        except Exception as e:
            print(f"❌ Ошибка в pipeline: {e}")
            return None, None
        finally:
            # Очистка всех временных файлов
            self.cleanup()


def main():
    """Основная функция"""
    DATA_DIR = Path.cwd().parent / "data" / "avito_ml_cup"

    # Создаем менеджер pipeline
    pipeline = DataPipelineManager(
        data_dir=DATA_DIR, valid_days_retriever=7, valid_days_reranker=7
    )

    try:
        # Запускаем pipeline
        df_features, df_train_reranker = pipeline.run_pipeline(timeout=3600)  # 1 час

        if df_features is not None:
            print("=== Pipeline завершен успешно ===")

            # Подготавливаем финальные данные для CatBoost
            df_pred_alt = pl.read_parquet(
                "temp_predictions.pq"
            )  # Можно сохранить отдельно
            df_candidates_train = df_pred_alt.join(
                df_train_reranker[["node", "cookie", "is_contact"]]
                .sort("is_contact", descending=True)
                .unique(["cookie", "node"]),
                on=["cookie", "node"],
                how="left",
            ).fill_null(0)

            return df_features, df_candidates_train
        else:
            print("❌ Pipeline не удался")
            return None, None

    finally:
        # Гарантированная очистка
        pipeline.cleanup()


if __name__ == "__main__":
    df_features, df_candidates_train = main()

    if df_features is not None:
        print("🎉 Данные готовы для обучения CatBoost!")
        print(f"Финальная память: {df_features.estimated_size() / (1024**2):.2f} MB")
    else:
        print("💥 Обработка не удалась")
