# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr


encounters = pd.read_parquet(
    "htem/PF_PC/connectivity_analysis/graphs/encounter_table.parquet"
    )

# fetch pairs with two encounters
counts = encounters.groupby(["pc_name", "pf_name"])["number"].transform("size")
two = encounters[counts == 2]

two = two.pivot_table(index=["pc_name", "pf_name"],
    columns="number",
    values="has_syn",
    )

# compute correlation
corr = two[0].corr(two[1])
print(f"Pearson R: {corr:.3f}")