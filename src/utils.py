import argparse
from datetime import datetime

import networkx as nx
import pandas as pd


def load_data(data_name):
    data_nw_df = pd.read_csv(f"../rev2data/{data_name}/{data_name}_network.csv", header=None,
                             names=["src", "dest", "rating", "timestamp"])
    data_gt_df = pd.read_csv(f"../rev2data/{data_name}/{data_name}_gt.csv", header=None, names=["id", "label"])

    if data_name != "epinions":
        data_nw_df["timestamp"] = data_nw_df["timestamp"].astype(int).apply(datetime.fromtimestamp)
    else:
        data_nw_df["timestamp"] = pd.to_datetime(data_nw_df["timestamp"])

    data_nw_df["src"] = "u" + data_nw_df["src"].astype(str)
    data_nw_df["dest"] = "p" + data_nw_df["dest"].astype(str)
    data_nw_df["rating"] = (data_nw_df["rating"] - data_nw_df["rating"].min()) / \
        (data_nw_df["rating"].max() - data_nw_df["rating"].min()) * 2 - 1

    # ! label=+1 means a benign user and label=-1 means a fraudster
    data_gt_df["id"] = "u" + data_gt_df["id"].astype(str)
    return data_nw_df, data_gt_df


def split_data_by_time(df, n_splits=10) -> list:
    time_max = df["timestamp"].max() + pd.Timedelta("1 day")
    time_min = df["timestamp"].min()
    time_dlt = (time_max - time_min)/n_splits
    df["split"] = (df["timestamp"] - time_min) // time_dlt
    gps = df.groupby("split")
    return [gps.get_group(i) for i in range(n_splits)]


def build_nx(data_nw_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.from_pandas_edgelist(
        data_nw_df,
        source="src",
        target="dest",
        edge_attr=True,
        create_using=nx.DiGraph(),
    )

    return G


def normalize_dict(d: dict) -> dict:
    keys = list(d.keys())
    values = [d[k] for k in keys]
    maxv = max(values)
    minv = min(values)
    new_values = [(v - minv)/(maxv - minv) for v in values]
    return dict(zip(keys, new_values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test utils")
    parser.add_argument("--data", action="store", type=str, required=True,
                        choices=["alpha", "otc", "amazon", "epinions"])

    args = parser.parse_args()
    print(args)

    data_nw_df, data_gt_df = load_data(args.data)
    sdf = split_data_by_time(data_nw_df)
    print([df.shape for df in sdf])
