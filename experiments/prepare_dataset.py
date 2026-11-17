from src.data_loader import load_and_clean_dataset

input_path = "data/raw/FakeNewsNet.csv"
output_path = "data/processed/cleaned_fakenewsnet.csv"

clean_df = load_and_clean_dataset(input_path, output_path)

# Verify the cleaned dataset
print(
    f"\nFINAL VERIFICATION\n"
    f"------------------\n"
    f"Total records: {len(clean_df)}\n"
    f"Fake headlines: {(clean_df['real'] == 0).sum()}\n"
    f"Real headlines: {(clean_df['real'] == 1).sum()}\n"
    f"Remaining duplicate headlines: "
    f"{clean_df['title'].duplicated().sum()}"
)