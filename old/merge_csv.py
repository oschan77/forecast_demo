import glob
import os

import pandas as pd

# Folder containing the CSV files
folder_path = "data/processed/walmart_sales/BySD"

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "walmart_*.csv"))

# List to store individual DataFrames
dataframes = []

for file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Extract the unique_id from the file name
    unique_id = os.path.basename(file).split("_")[1].split(".")[0]

    # Add the unique_id column
    df["unique_id"] = unique_id

    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Rename the columns
# merged_df.rename(columns={"Weekly_Sales": "y", "Date": "ds"}, inplace=True)

# # Reorder the columns
# columns_order = ["unique_id", "Date"] + [
#     col for col in merged_df.columns if col not in ["unique_id", "Date"]
# ]
# merged_df = merged_df[columns_order]

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("merged_darts.csv", index=False)
