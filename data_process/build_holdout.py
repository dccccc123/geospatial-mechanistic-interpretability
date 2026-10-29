"""Build the city-disjoint held-out set.

Keeps the 90 discovery pairs (180 directed prompts) unchanged and builds an
independent held-out set from cities that do not appear in those pairs. For the
hold-out cities it generates all undirected pairs, computes bearing / distance /
direction, samples them in a stratified, balanced way, expands them to directed
prompts, and writes holdout_city_pairs.jsonl with the same schema as
balanced_city_pairs.jsonl.

Pure data processing; does not load the model or use the GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CITY_CSV = ROOT / "data_process" / "cleaned_data" / "china_cities_cleaned.csv"
TRAIN_JSONL = ROOT / "data_process" / "dataset" / "balanced_city_pairs.jsonl"
OUT = ROOT / "data_process" / "dataset" / "holdout_city_pairs.jsonl"


def load_cities():
    df = pd.read_csv(CITY_CSV, encoding="utf-8-sig")
    df = df[["城市名称", "经度", "纬度", "人口"]].copy()
    df["城市名称"] = df["城市名称"].astype(str).str.strip()
    for c in ("经度", "纬度", "人口"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def load_train_cities():
    cities = set()
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cities.add(item["City_A"])
                cities.add(item["City_B"])
    return cities


def direction_from_bearing(bearing):
    b = np.mod(np.asarray(bearing, dtype=np.float64), 360.0)
    return np.where((b >= 315.0) | (b < 45.0), "North",
           np.where((b >= 45.0) & (b < 135.0), "East",
           np.where((b >= 135.0) & (b < 225.0), "South", "West")))


def compute_pairs(cities):
    n = len(cities)
    names = cities["城市名称"].to_numpy()
    lon = np.radians(cities["经度"].to_numpy(dtype=np.float64))
    lat = np.radians(cities["纬度"].to_numpy(dtype=np.float64))
    pop = cities["人口"].to_numpy(dtype=np.float64)

    i, j = np.triu_indices(n, k=1)
    lon1, lat1 = lon[i], lat[i]
    lon2, lat2 = lon[j], lat[j]
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist = 6371.0088 * c

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0

    df = pd.DataFrame({
        "城市A": names[i], "城市B": names[j],
        "A经度": cities["经度"].to_numpy()[i], "A纬度": cities["纬度"].to_numpy()[i],
        "B经度": cities["经度"].to_numpy()[j], "B纬度": cities["纬度"].to_numpy()[j],
        "A人口": pop[i], "B人口": pop[j],
        "距离": dist, "方位角": bearing,
    })
    return df


def stratify(df):
    df = df.copy()
    q33, q66 = df["距离"].quantile([0.33, 0.66]).to_numpy(dtype=np.float64)
    df["Distance_Tier"] = np.where(df["距离"] < q33, "Short",
                          np.where(df["距离"] < q66, "Medium", "Long"))

    pop_a = df[["城市A", "A人口"]].rename(columns={"城市A": "city", "A人口": "pop"})
    pop_b = df[["城市B", "B人口"]].rename(columns={"城市B": "city", "B人口": "pop"})
    pop_unique = pd.concat([pop_a, pop_b], ignore_index=True).dropna()
    pop_unique = pop_unique.groupby("city", as_index=False)["pop"].max()
    thr = float(pop_unique["pop"].quantile(0.70))

    a_high = df["A人口"] >= thr
    b_high = df["B人口"] >= thr
    df["Pop_Tier"] = np.where(a_high & b_high, "High-High",
                     np.where(~a_high & ~b_high, "Low-Low", "High-Low"))

    df["Direction"] = direction_from_bearing(df["方位角"])
    df["Axis_Tier"] = np.where(df["Direction"].isin(["North", "South"]), "NS-Axis", "EW-Axis")
    df["Bucket_Key"] = df["Axis_Tier"] + "_" + df["Distance_Tier"] + "_" + df["Pop_Tier"]
    return df


def balanced_sample(df, k, seed):
    counts = df["Bucket_Key"].value_counts().sort_index()
    kept = counts[counts >= k].index.tolist()
    sampled = (df[df["Bucket_Key"].isin(kept)]
               .groupby("Bucket_Key", group_keys=False)
               .sample(n=k, replace=False, random_state=seed)
               .reset_index(drop=True))
    return sampled, counts, kept


def directed_expand(df):
    forward = df.copy()
    reverse = df.copy()
    reverse["城市A"], reverse["城市B"] = df["城市B"], df["城市A"]
    reverse["A经度"], reverse["B经度"] = df["B经度"], df["A经度"]
    reverse["A纬度"], reverse["B纬度"] = df["B纬度"], df["A纬度"]
    reverse["A人口"], reverse["B人口"] = df["B人口"], df["A人口"]
    reverse["方位角"] = (df["方位角"] + 180.0) % 360.0
    reverse["Direction"] = direction_from_bearing(reverse["方位角"])
    return pd.concat([forward, reverse], ignore_index=True)


def format_output(df):
    result = df.rename(columns={
        "城市A": "City_A", "城市B": "City_B",
        "A经度": "Lon_A", "A纬度": "Lat_A",
        "B经度": "Lon_B", "B纬度": "Lat_B",
        "A人口": "Pop_A", "B人口": "Pop_B",
        "距离": "Distance", "方位角": "Bearing",
    })
    ordered = ["City_A", "City_B", "Lon_A", "Lat_A", "Lon_B", "Lat_B",
               "Pop_A", "Pop_B", "Distance", "Bearing",
               "Distance_Tier", "Pop_Tier", "Axis_Tier", "Bucket_Key", "Direction"]
    return result[ordered]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Undirected pairs sampled per bucket (default 5, matching the training set)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cities = load_cities()
    train_cities = load_train_cities()

    missing = train_cities - set(cities["城市名称"])
    if missing:
        print(f"[Warning] {len(missing)} training cities are not in the city table: {sorted(missing)}")

    holdout = cities[~cities["城市名称"].isin(train_cities)].reset_index(drop=True)

    print(f"[Data] total cities: {len(cities)}")
    print(f"[Data] distinct cities in the training set: {len(train_cities)}")
    print(f"[Data] holdout cities: {len(holdout)}")

    pairs = compute_pairs(holdout)
    print(f"[Data] total holdout undirected pairs: {len(pairs)}")

    labeled = stratify(pairs)
    sampled, counts, kept = balanced_sample(labeled, k=args.k, seed=args.seed)

    print(f"[Sample] buckets meeting k={args.k}: {len(kept)} / {len(counts)}")
    print(f"[Sample] sampled undirected pairs: {len(sampled)}")

    expanded = directed_expand(sampled)
    final = format_output(expanded)

    dist = final["Direction"].value_counts().to_dict()
    print(f"[Sample] directed samples after expansion: {len(final)}")
    print(f"[Sample] direction distribution: {dist}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_json(OUT, orient="records", lines=True, force_ascii=False)
    print(f"[Saved] {OUT}")


if __name__ == "__main__":
    main()
