import os
import pandas as pd


df = pd.read_csv(os.path.join("starter", "data", "census.csv"))

trailing_white_space_columns = [c for c in df.columns if c.startswith(" ")]
replace_column_names = [c.replace(" ", "") if c.startswith(" ") else c for c in df.columns]
df.columns = replace_column_names

cat_vars = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "salary"
]


df[cat_vars] = df[cat_vars].apply(lambda col: col.str.lstrip(), axis=1)

# TODO: save to same file and use dvc for versioning
df.to_csv(os.path.join("starter", "data", "census_cleaned.csv"))
