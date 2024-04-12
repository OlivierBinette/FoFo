#! /usr/bin/env python3

import pandas as pd
import os

base_dir = "results"
models = os.listdir(base_dir)
scores_filename = "annotations.json"

results = []
for model in models:
    df = pd.read_json(os.path.join(base_dir, model, scores_filename))
    df["model"] = model
    results.append(df)

joined_results = pd.concat(results, axis=0)

joined_results['instruction_no'] = pd.Categorical(joined_results['instruction']).codes

instruction_difficulties = (
    joined_results
    .groupby("instruction_no")
    .agg({"annotation": lambda x: (1-x).mean(skipna=True)})
    .sort_values('annotation', ascending=False)
    .rename(columns={"annotation": "difficulty"})
)

joined_results = joined_results.merge(instruction_difficulties, how="left", on="instruction_no").sort_values("difficulty", ascending=False)

joined_results.to_csv("joined_results.csv", index=False)
