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
    hist_dict = {}
    for loc_path in loc_paths:
        print(loc_path)
        array = np.array(Image.open(loc_path))
        unique_values, counts = np.unique(array, return_counts=True)
        
        for value, count in zip(unique_values, counts):
            if value in hist_dict:
                hist_dict[value] += count     # accumulate
            else:
                hist_dict[value] = count      # initialize

    df = pd.DataFrame({
        "thickness": list(hist_dict.keys()),
        name: list(hist_dict.values())
    })
    df = df.sort_values("thickness")
    df.to_csv(name+'.csv', index=False)
    dfs.append(df)

df_all = dfs[0]

# Merge all remaining dfs based on "thickness"
for df in dfs[1:]:
    df_all = df_all.merge(df, on="thickness", how="outer")


# Sort by thickness
df_all = df_all.sort_values("thickness")

# Save
df_all.to_csv("thickness_statictics.csv", index=False)

print("Saved: thickness_merged.csv")
