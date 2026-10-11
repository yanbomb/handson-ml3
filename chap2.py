import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_housing_data():
    data_folder = Path(__file__).parent / "data"
    file = "housing/housing.csv"
    fcsv = data_folder  / file
    df = None
    if fcsv.exists():
        df = pd.read_csv(fcsv)
    else:
        raise FileNotFoundError(f"cannot find {fcsv}")
    return df


def explore(df:pd.DataFrame):
    print(df.describe())
    print(df.info())
    print(df.ocean_proximity.value_counts())

    df.hist(bins=50,figsize=(12,8))
    
    plt.savefig("ch2_hist.png")


    
if __name__ == "__main__":
    df = load_housing_data()

    # explore(df)


    train_set,test_set = train_test_split(df,test_size=0.2,random_state=123)

    print(type(train_set))
    print(type(test_set))
    print(train_set.describe())
    print(test_set.describe())
    # # print(df)


