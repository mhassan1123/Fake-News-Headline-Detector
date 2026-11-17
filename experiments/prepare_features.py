from src.preprocessing import prepare_features

input_path = "data/processed/cleaned_fakenewsnet.csv"
output_dir = "outputs/features"

prepare_features(input_path, output_dir)