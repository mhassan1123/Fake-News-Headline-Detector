import pandas as pd

file_path = "D:/AI_Project/Fake-News-Headline-Detector/data/raw/FakeNewsNet.csv"

df = pd.read_csv(file_path)

print("\nDATASET OVERVIEW")
print("----------------")
print("Total records:", len(df))
print("Columns:", list(df.columns))

print("\nFIRST 5 HEADLINES")
print("-----------------")
print(df[["title", "real"]].head())

print("\nMISSING VALUES")
print("--------------")
print(df.isnull().sum())

print("\nCLASS DISTRIBUTION")
print("------------------")
print(df["real"].value_counts().sort_index())

print("\nDUPLICATE HEADLINES")
print("-------------------")
print(df["title"].duplicated().sum())