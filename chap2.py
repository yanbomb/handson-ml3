from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_housing_data():
    data_folder = Path(__file__).parent / "data"
    file = "housing/housing.csv"
    fcsv = data_folder / file
    df = None
    if fcsv.exists():
        df = pd.read_csv(fcsv)
    else:
        raise FileNotFoundError(f"cannot find {fcsv}")
    return df


def explore(df: pd.DataFrame):
    print(df.describe())
    print(df.info())
    print(df.ocean_proximity.value_counts())

    df.hist(bins=50, figsize=(12, 8))

    plt.savefig("ch2_hist.png")


def income_to_cat(df: pd.DataFrame) -> pd.DataFrame:
    col = "income_cat"
    df["income_cat"] = pd.cut(
        df.median_income, bins=[0, 1.5, 3.0, 4.5, 6.0, 16], labels=[1, 2, 3, 4, 5]
    )
    print(df[col].value_counts())
    return df, col


def median_house_value_to_cat(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(0, 500000, 100000).tolist()
    bins.append(np.inf)
    labels = np.arange(1, len(bins), 1)
    col = "house_value_cat"
    df[col] = pd.cut(
        df.median_house_value,
        bins=bins,
        labels=labels,
    )
    print(df["house_value_cat"].value_counts())
    return df, col


def combine_cats(df: pd.DataFrame, columns: list[str], new_column: str) -> pd.DataFrame:
    df[new_column] = [f"{i}_{j}" for i, j in zip(df[columns[0]], df[columns[1]])]
    return df


if __name__ == "__main__":
    df = load_housing_data()
    print(df.columns)

    train_set1, test_set1 = train_test_split(df, test_size=0.2, random_state=123)

    # stats
    print(df.info())

    # transform income to categories
    df, col1 = income_to_cat(df)
    df, col2 = median_house_value_to_cat(df)
    col3 = "combined_cat"
    df = combine_cats(df, columns=[col1, col2], new_column=col3)
    print(df[col3].value_counts())

    # statified shuffle splitting
    train_set2, test_set2 = train_test_split(
        df, test_size=0.2, stratify=df[col3], random_state=123
    )
    vca = train_set2[col3].value_counts().sort_index()
    vcb = test_set2[col3].value_counts().sort_index()

    print(len(vca))
    print(len(vcb))
    for i, v in vca.items():
        print(f"{i} -> {v / vca.sum():4f} | {vcb[i] / vcb.sum():4f}")
