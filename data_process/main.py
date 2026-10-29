"""Build the balanced directed city-pair dataset from GeoNames.

Filters Chinese cities to prefecture level and above, keeps single- or two-token
city names, computes pairwise great-circle distance and bearing, and produces a
stratified, balanced, direction-expanded dataset as Parquet + JSONL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# GeoNames cities15000 column names
COLUMN_NAMES = [
	"geonameid",
	"name",
	"asciiname",
	"alternatenames",
	"latitude",
	"longitude",
	"feature_class",
	"feature_code",
	"country_code",
	"cc2",
	"admin1_code",
	"admin2_code",
	"admin3_code",
	"admin4_code",
	"population",
	"elevation",
	"dem",
	"timezone",
	"modification_date",
]


def load_and_clean_data(file_path: Path) -> pd.DataFrame:
	"""Read the raw TSV, keep Chinese prefecture-level-and-above cities, drop NaNs."""
	raw_df = pd.read_csv(
		file_path,
		sep="\t",
		header=None,
		names=COLUMN_NAMES,
		low_memory=False,
	)

	# PPLC: national capital, PPLA: provincial capital, PPLA2: prefecture seat
	allowed_feature_codes = {"PPLC", "PPLA", "PPLA2"}
	df = raw_df[["country_code", "feature_code", "name", "longitude", "latitude", "population"]].copy()
	df["country_code"] = df["country_code"].astype(str).str.upper().str.strip()
	df["feature_code"] = df["feature_code"].astype(str).str.upper().str.strip()
	df = df[df["country_code"] == "CN"].copy()
	df = df[df["feature_code"].isin(allowed_feature_codes)].copy()
	df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
	df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
	df["population"] = pd.to_numeric(df["population"], errors="coerce")

	# drop rows with any missing key field
	df = df.dropna(subset=["name", "longitude", "latitude", "population"])
	df = df[df["name"].astype(str).str.strip() != ""].reset_index(drop=True)
	df = df[["name", "longitude", "latitude", "population"]]
	return df


def filter_by_token_length(df: pd.DataFrame, model_name: str = "Qwen/Qwen3-8B") -> pd.DataFrame:
	"""Filter names by tokenizer length; fall back to length-2 if <1000 single-token names."""
	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
	names = df["name"].astype(str).tolist()

	encoded = tokenizer(names, add_special_tokens=False, padding=False, truncation=False)
	token_lengths = np.fromiter((len(ids) for ids in encoded["input_ids"]), dtype=np.int32)

	one_token_mask = token_lengths == 1
	filtered_df = df[one_token_mask].copy()

	if len(filtered_df) < 1000:
		print(f"Only {len(filtered_df)} cities have single-token names; keeping two-token names instead.")
		two_token_mask = token_lengths == 2
		filtered_df = df[two_token_mask].copy()

	filtered_df = filtered_df.reset_index(drop=True)
	return filtered_df


def compute_pairwise_metrics(df: pd.DataFrame) -> pd.DataFrame:
	"""Vectorised pairwise great-circle distance and initial bearing."""
	n = len(df)
	if n < 2:
		raise ValueError("Fewer than 2 cities after filtering; cannot generate city pairs.")

	names = df["name"].to_numpy()
	lon = df["longitude"].to_numpy(dtype=np.float64)
	lat = df["latitude"].to_numpy(dtype=np.float64)
	pop = df["population"].to_numpy(dtype=np.float64)

	# upper-triangular indices for the N(N-1)/2 unique pairs
	i_idx, j_idx = np.triu_indices(n, k=1)

	lon1 = np.radians(lon[i_idx])
	lat1 = np.radians(lat[i_idx])
	lon2 = np.radians(lon[j_idx])
	lat2 = np.radians(lat[j_idx])

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	# Haversine great-circle distance (km)
	a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
	c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
	earth_radius_km = 6371.0088
	distance_km = earth_radius_km * c

	# initial bearing (A -> B, 0-360 deg)
	x = np.sin(dlon) * np.cos(lat2)
	y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
	bearing_deg = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0

	result_df = pd.DataFrame(
		{
			"城市A": names[i_idx],
			"城市B": names[j_idx],
			"A经度": lon[i_idx],
			"A纬度": lat[i_idx],
			"B经度": lon[j_idx],
			"B纬度": lat[j_idx],
			"A人口": pop[i_idx],
			"B人口": pop[j_idx],
			"距离": distance_km,
			"方位角": bearing_deg,
		}
	)

	return result_df


def save_result(df: pd.DataFrame, output_dir: Path) -> Path:
	"""Save the pairwise result as CSV under cleaned_data."""
	output_dir.mkdir(parents=True, exist_ok=True)
	csv_path = output_dir / "city_pairs_qwen_filtered.csv"
	df.to_csv(csv_path, index=False, encoding="utf-8-sig")
	return csv_path


def load_city_pairs(csv_path: Path) -> pd.DataFrame:
	"""Read the generated city-pair CSV."""
	df = pd.read_csv(csv_path)
	required_columns = ["城市A", "城市B", "A经度", "A纬度", "B经度", "B纬度", "A人口", "B人口", "距离", "方位角"]
	missing = [col for col in required_columns if col not in df.columns]
	if missing:
		raise ValueError(f"Input file is missing required columns: {missing}")
	return df


def direction_from_bearing(bearing_series: pd.Series) -> pd.Series:
	"""Map bearing to North/South/East/West."""
	b = np.mod(bearing_series.to_numpy(dtype=np.float64), 360.0)
	direction = np.where(
		(b >= 315.0) | (b < 45.0),
		"North",
		np.where((b >= 45.0) & (b < 135.0), "East", np.where((b >= 135.0) & (b < 225.0), "South", "West")),
	)
	return pd.Series(direction, index=bearing_series.index)


def assign_stratification_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
	"""Add distance-, population-, and axis-tier labels."""
	result = df.copy()

	q33, q66 = result["距离"].quantile([0.33, 0.66]).to_numpy(dtype=np.float64)
	result["Distance_Tier"] = np.where(
		result["距离"] < q33,
		"Short",
		np.where(result["距离"] < q66, "Medium", "Long"),
	)

	city_pop_a = result[["城市A", "A人口"]].rename(columns={"城市A": "city", "A人口": "pop"})
	city_pop_b = result[["城市B", "B人口"]].rename(columns={"城市B": "city", "B人口": "pop"})
	city_pop_map = pd.concat([city_pop_a, city_pop_b], ignore_index=True)
	city_pop_map = city_pop_map.dropna(subset=["city", "pop"])
	city_pop_map["city"] = city_pop_map["city"].astype(str).str.strip()
	city_pop_map = city_pop_map[city_pop_map["city"] != ""]

	# for duplicate city names, use the max population as a stable value
	city_pop_unique = city_pop_map.groupby("city", as_index=False)["pop"].max()
	pop_threshold = float(city_pop_unique["pop"].quantile(0.70))

	result["A_Pop_Level"] = np.where(result["A人口"] >= pop_threshold, "High", "Low")
	result["B_Pop_Level"] = np.where(result["B人口"] >= pop_threshold, "High", "Low")

	result["Pop_Tier"] = np.where(
		(result["A_Pop_Level"] == "High") & (result["B_Pop_Level"] == "High"),
		"High-High",
		np.where(
			(result["A_Pop_Level"] == "Low") & (result["B_Pop_Level"] == "Low"),
			"Low-Low",
			"High-Low",
		),
	)

	result["Direction"] = direction_from_bearing(result["方位角"])
	result["Axis_Tier"] = np.where(result["Direction"].isin(["North", "South"]), "NS-Axis", "EW-Axis")

	stats = {
		"distance_q33": float(q33),
		"distance_q66": float(q66),
		"pop_threshold_q70": pop_threshold,
	}
	return result, stats


def build_bucket_key(df: pd.DataFrame) -> pd.DataFrame:
	"""Build the bucket key: Axis + Distance + Pop."""
	result = df.copy()
	result["Bucket_Key"] = result["Axis_Tier"] + "_" + result["Distance_Tier"] + "_" + result["Pop_Tier"]
	return result


def balanced_sampling(df: pd.DataFrame, k: int = 100, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
	"""Balanced sampling per bucket; buckets smaller than k are dropped."""
	bucket_counts = df["Bucket_Key"].value_counts().sort_index()
	print("Bucket capacity distribution:")
	print(bucket_counts.to_string())

	kept_keys = bucket_counts[bucket_counts >= k].index.tolist()
	if not kept_keys:
		raise ValueError(f"No bucket reaches the sampling threshold K={k}; lower K or relax the filters.")

	sampled_df = (
		df[df["Bucket_Key"].isin(kept_keys)]
		.groupby("Bucket_Key", group_keys=False)
		.sample(n=k, replace=False, random_state=random_state)
		.reset_index(drop=True)
	)
	return sampled_df, bucket_counts


def directed_expand(df: pd.DataFrame) -> pd.DataFrame:
	"""Expand undirected pairs into directed samples with reversed bearing/direction."""
	forward = df.copy()
	forward["Direction"] = direction_from_bearing(forward["方位角"])

	reverse = df.copy()
	reverse["城市A"], reverse["城市B"] = df["城市B"], df["城市A"]
	reverse["A经度"], reverse["B经度"] = df["B经度"], df["A经度"]
	reverse["A纬度"], reverse["B纬度"] = df["B纬度"], df["A纬度"]
	reverse["A人口"], reverse["B人口"] = df["B人口"], df["A人口"]
	reverse["方位角"] = (df["方位角"] + 180.0) % 360.0
	reverse["Direction"] = direction_from_bearing(reverse["方位角"])

	expanded = pd.concat([forward, reverse], ignore_index=True)
	return expanded


def format_output_schema(df: pd.DataFrame) -> pd.DataFrame:
	"""Rename columns to the final schema and order them."""
	result = df.rename(
		columns={
			"城市A": "City_A",
			"城市B": "City_B",
			"A经度": "Lon_A",
			"A纬度": "Lat_A",
			"B经度": "Lon_B",
			"B纬度": "Lat_B",
			"A人口": "Pop_A",
			"B人口": "Pop_B",
			"距离": "Distance",
			"方位角": "Bearing",
		}
	)

	result["Distance_Tier"] = pd.Categorical(result["Distance_Tier"], categories=["Short", "Medium", "Long"], ordered=True)
	result["Pop_Tier"] = pd.Categorical(result["Pop_Tier"], categories=["Low-Low", "High-Low", "High-High"], ordered=True)
	result["Axis_Tier"] = pd.Categorical(result["Axis_Tier"], categories=["NS-Axis", "EW-Axis"], ordered=True)
	result["Direction"] = pd.Categorical(result["Direction"], categories=["North", "East", "South", "West"], ordered=False)

	ordered_columns = [
		"City_A",
		"City_B",
		"Lon_A",
		"Lat_A",
		"Lon_B",
		"Lat_B",
		"Pop_A",
		"Pop_B",
		"Distance",
		"Bearing",
		"Distance_Tier",
		"Pop_Tier",
		"Axis_Tier",
		"Bucket_Key",
		"Direction",
	]
	return result[ordered_columns]


def save_balanced_outputs(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
	"""Save both Parquet and JSONL."""
	output_dir.mkdir(parents=True, exist_ok=True)
	parquet_path = output_dir / "balanced_city_pairs.parquet"
	jsonl_path = output_dir / "balanced_city_pairs.jsonl"

	df.to_parquet(parquet_path, index=False)
	df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
	return parquet_path, jsonl_path


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	input_file = base_dir / "row_data" / "cities15000.txt"
	cleaned_output_dir = base_dir / "cleaned_data"

	cleaned_df = load_and_clean_data(input_file)
	cleaned_output_dir.mkdir(parents=True, exist_ok=True)
	cleaned_df.rename(columns={"name": "城市名称", "longitude": "经度",
	                           "latitude": "纬度", "population": "人口"}) \
		.to_csv(cleaned_output_dir / "china_cities_cleaned.csv",
		        index=False, encoding="utf-8-sig")
	filtered_df = filter_by_token_length(cleaned_df, model_name="Qwen/Qwen3-8B")
	result_df = compute_pairwise_metrics(filtered_df)
	output_path = save_result(result_df, cleaned_output_dir)

	print(f"Cleaned Chinese cities: {len(cleaned_df)}")
	print(f"Cities after token filtering: {len(filtered_df)}")
	print(f"Generated city pairs: {len(result_df)}")
	print(f"Result saved: {output_path}")

	# append stratification/bucketing/sampling/directed-expansion to the CSV logic
	balance_input_file = base_dir / "cleaned_data" / "city_pairs_qwen_filtered.csv"
	balance_output_dir = base_dir / "dataset"
	k = 5

	pairs_df = load_city_pairs(balance_input_file)
	labeled_df, stats = assign_stratification_labels(pairs_df)
	bucketed_df = build_bucket_key(labeled_df)
	sampled_df, bucket_counts = balanced_sampling(bucketed_df, k=k, random_state=42)
	expanded_df = directed_expand(sampled_df)
	final_df = format_output_schema(expanded_df)
	parquet_path, jsonl_path = save_balanced_outputs(final_df, balance_output_dir)

	print(f"Input city pairs: {len(pairs_df)}")
	print(f"Distance quantiles: q33={stats['distance_q33']:.4f}, q66={stats['distance_q66']:.4f}")
	print(f"Population threshold (70th percentile): {stats['pop_threshold_q70']:.0f}")
	print(f"Buckets meeting K={k}: {(bucket_counts >= k).sum()} / {len(bucket_counts)}")
	print(f"Sampled (undirected) count: {len(sampled_df)}")
	print(f"Expanded (directed) count: {len(final_df)}")
	print(f"Parquet saved: {parquet_path}")
	print(f"JSONL saved: {jsonl_path}")


if __name__ == "__main__":
	main()
