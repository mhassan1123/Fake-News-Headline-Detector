from src.validation_preprocessing import prepare_validation_features

input_path = "data/processed/cleaned_fakenewsnet.csv"
output_dir = "outputs/validation_features"

prepare_validation_features(input_path, output_dir)