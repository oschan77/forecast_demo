import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("merged_nf_mid.csv")

# Map the 'Type' column values
type_mapping = {"A": 0, "B": 1, "C": 2}
df["Type"] = df["Type"].map(type_mapping)

# Save the modified DataFrame to a new CSV file
df.to_csv("merged_nf_mid.csv", index=False)
