import os
import json

import pandas as pd
from catboost import CatBoostRegressor, Pool

from preprocess import feature_engineering


def load_input(path: str):
    """Возвращает список словарей из input.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("input.json должен содержать объект или массив объектов")


def main():
    INPUT_JSON  = "data/input.json"
    OUTPUT_JSON = "data/output.json"
    MODEL_PATH  = "models/final_catboost.cbm"

    # 1) Читаем JSON
    records = load_input(INPUT_JSON)
    df = pd.DataFrame(records)

    # 2) Удаляем служебные колонки, если попали
    for c in ["id", "name", "host_id", "host_name", "price"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # 3) Фиче­рин­г
    df = feature_engineering(df)

    # 4) Категории → str + fillna
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna("__MISSING__").astype(str)

    # 5) Загружаем модель
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    # 6) Для жёсткой совместимости по именам признаков
    feature_names = model.feature_names_
    df = df[feature_names]

    # 7) Предсказание
    pool = Pool(df, cat_features=cat_cols)
    preds = model.predict(pool)

    # 8) Сохраняем результат
    df["pred_price"] = preds
    output = df.to_dict(orient="records")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
