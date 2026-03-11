import numpy as np
import glob
import os 
import re 
from PIL import Image
import tifffile as tiff
import pandas as pd

root_paths = glob.glob('./data/*/5_thickness')

dfs = []
for root_path in root_paths:
    name = os.path.split(os.path.split(root_path)[0])[1]

    loc_paths = glob.glob(os.path.join(root_path, f'*.tif'))
    volumn = []
    for loc_path in loc_paths:
        array = np.array(Image.open(loc_path))
        if array.shape[0]>1000:
            volumn.append(array)
    volumn = np.array(volumn)
    unique_values, counts = np.unique(volumn, return_counts=True)


    df = pd.DataFrame({
        "thickness": unique_values,
        name: counts
    })
    dfs.append(df)

df_all = dfs[0]

# Merge all remaining dfs based on "thickness"
for df in dfs[1:]:
    df_all = df_all.merge(df, on="thickness", how="outer")


# Sort by thickness
df_all = df_all.sort_values("thickness")

# Save
df_all.to_csv("thickness_merged.csv", index=False)

print("Saved: thickness_merged.csv")
