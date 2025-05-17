# File: src/model_train_catboost_tuned.py

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error  # Add this import at the top

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics      import mean_squared_error

from catboost import CatBoostRegressor, Pool

# Импорт вашей функции feature_engineering
from preprocess import feature_engineering


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    start_time = time.time()

    # 1) Загрузка и feature engineering
    DATA_PATH = os.path.join("data", "listings.csv")
    df = load_data(DATA_PATH)
    df = feature_engineering(df)

    if "price" not in df.columns:
        raise KeyError("В датафрейме нет столбца 'price'")

    # Целевая переменная
    y = df.pop("price").values
    X = df.copy()

    # 2) Split на train(60%) / valid(20%) / test(20%)
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, random_state=42
    )

    # 3) Подготовка категорий для CatBoost
    #    все object-столбцы считаем категориальными
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for D in (X_train, X_valid, X_test):
        for c in cat_cols:
            D[c] = D[c].fillna("__MISSING__").astype(str)

    # 4) Объединяем train+valid для RandomizedSearchCV
    X_tune = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
    y_tune = np.concatenate([y_train, y_valid], axis=0)

    # 5) Задаём пространство гиперпараметров
    param_dist = {
        "depth":              [4, 6, 8, 10],
        "learning_rate":      [0.01, 0.03, 0.05, 0.1],
        "l2_leaf_reg":        [1, 3, 5, 7, 9],
        "bagging_temperature":[0, 1, 3, 5],
        "border_count":       [32, 64, 128, 254],
        "random_strength":    [1, 3, 5, 10],
        "iterations":         [200, 500, 800, 1200]
    }

    # 6) Запускаем RandomizedSearchCV
    print("\n>>> Hyperparameter tuning for CatBoost <<<")
    cb_base = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False
    )
    cb_search = RandomizedSearchCV(
        estimator=cb_base,
        param_distributions=param_dist,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=2,
        refit=True
    )
    cb_search.fit(X_tune, y_tune, cat_features=cat_cols)

    best_params = cb_search.best_params_
    print(f"\nBest CatBoost params: {best_params}\n")

    # 7) Финальное обучение на всём train+valid
    print(">>> Training final CatBoost on train+valid <<<")
    X_full = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train, y_valid], axis=0)
    # Обновляем категориальные столбцы в X_full (на случай нового индекса)
    for c in cat_cols:
        X_full[c] = X_full[c].fillna("__MISSING__").astype(str)
    # Собираем модель с лучшими параметрами
    final_cb = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=100,
        **best_params
    )
    final_cb.fit(
        Pool(X_full, y_full, cat_features=cat_cols)
    )

    # 8) Оценка на отложенном тесте
    print("\n>>> Evaluating on test set <<<")
    preds_test = final_cb.predict(X_test)
    rmse_test = root_mean_squared_error(y_test, preds_test)
    print(f"Final CatBoost Test RMSE = {rmse_test:.2f}")

    # 9) Сохраняем модель
    os.makedirs("models", exist_ok=True)
    MODEL_PATH = os.path.join("models", "final_catboost.cbm")
    final_cb.save_model(MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")

    print(f"\nTotal elapsed time: {time.time() - start_time:.1f} sec.")


if __name__ == "__main__":
    main()
