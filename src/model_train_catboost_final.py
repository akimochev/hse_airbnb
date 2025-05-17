import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics      import mean_squared_error
from catboost             import CatBoostRegressor, Pool
from sklearn.metrics import root_mean_squared_error  # Add this import at the top

from preprocess import feature_engineering

def main():
    start_time = time.time()

    # 1) Загрузка и удаление служебных полей
    df = pd.read_csv("data/listings.csv")
    for c in ["id", "name", "host_id", "host_name"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # 2) Фиче­рин­г
    df = feature_engineering(df)
    if "price" not in df.columns:
        raise KeyError("Нет столбца 'price' в данных после feature_engineering")

    # 3) Целевая и признаки
    y = df.pop("price").values
    X = df.copy()

    # 4) Сплит 60/20/20
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, random_state=42
    )

    # 5) Категориальные признаки в строковый формат
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for D in (X_train, X_valid, X_test):
        for c in cat_cols:
            D[c] = D[c].fillna("__MISSING__").astype(str)

    # 6) Объединяем train+valid
    X_full = pd.concat([X_train, X_valid], ignore_index=True)
    y_full = np.concatenate([y_train, y_valid], axis=0)

    # 7) Ваши лучшие гиперпараметры
    best_params = {
        "random_strength":     5,
        "learning_rate":       0.1,
        "l2_leaf_reg":         3,
        "iterations":          1200,
        "depth":               4,
        "border_count":        32,
        "bagging_temperature": 1
    }

    # 8) Инициализация и обучение финальной модели
    final_cb = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=100,
        **best_params
    )
    train_pool = Pool(X_full, y_full, cat_features=cat_cols)
    final_cb.fit(train_pool)

    # 9) Оценка на тесте
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)
    preds_test = final_cb.predict(test_pool)
    rmse = root_mean_squared_error(y_test, preds_test)
    print(f"\nFinal CatBoost RMSE on test = {rmse:.2f}")

    # 10) Сохранение модели
    os.makedirs("models", exist_ok=True)
    final_cb.save_model("models/final_catboost.cbm")
    print("Model saved to models/final_catboost.cbm")

    print(f"Elapsed time: {time.time() - start_time:.1f} sec.")


if __name__ == "__main__":
    main()
