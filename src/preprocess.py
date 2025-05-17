import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляем производные признаки:
    - days_since_review
    - reviews_per_year
    Убираем сырые столбцы, в том числе 'amenities' (если он есть).
    """
    df = df.copy()

    # days_since_review
    if 'last_review' in df.columns:
        df['last_review'] = pd.to_datetime(df['last_review'])
        df['days_since_review'] = (pd.Timestamp('today') - df['last_review']).dt.days
    else:
        df['days_since_review'] = np.nan

    # reviews_per_year
    if 'reviews_per_month' in df.columns:
        df['reviews_per_year'] = df['reviews_per_month'] * 12
    else:
        df['reviews_per_year'] = np.nan

    # удаляем «сырье», которое более не нужно
    drop_cols = [c for c in ['last_review', 'reviews_per_month', 'amenities'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df

def reduce_cardinality(df: pd.DataFrame, cat_cols: list, max_levels: int = 20) -> pd.DataFrame:
    """
    В категории оставляем топ-max_levels частотных значений,
    все остальные помечаем как 'OTHER'.
    """
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            top = df[col].value_counts().nlargest(max_levels).index
            df[col] = df[col].where(df[col].isin(top), other='OTHER')
    return df
