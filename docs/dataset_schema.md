
# FakeNewsNet - Dataset Schema

## 1. Dataset Information

- Dataset: FakeNewsNet
- Source: https://huggingface.co/datasets/rickstello/FakeNewsNet
- Original research repository: https://github.com/KaiDMML/FakeNewsNet
- Dataset size: 23,196 records
- Selected version: Hugging Face CSV mirror

## 2. Dataset Columns

| Column | Description | Project Usage |
|---|---|---|
| title | News headline | Model input |
| news_url | News article URL | Not used for prediction |
| source_domain | Publisher domain | Not used for prediction |
| tweet_num | Number of associated tweets | Not used for prediction |
| real | News label | Target variable |

## 3. Label Mapping

- 0 = Fake
- 1 = Real

## 4. Original Dataset Statistics

- Total records: 23,196
- Fake records: 5,755
- Real records: 17,441
- Missing titles: 0
- Missing labels: 0
- Missing URLs: 330
- Missing source domains: 330
- Duplicate headline occurrences beyond first: 1,472
- Distinct repeated headline groups: 963
- Conflicting-label headline groups: 123

## 5. Cleaning Decisions

1. Remove all records belonging to headline groups with conflicting labels.
2. For repeated headlines with consistent labels, retain one occurrence.
3. Preserve the original dataset without modification.
4. Save the cleaned dataset separately.

## 6. Cleaned Dataset Statistics

- Original records: 23,196
- Records removed due to conflicting labels: 364
- Repeated rows removed: 1,231
- Final records: 21,601
- Fake records: 5,200
- Real records: 16,401
- Missing headlines: 0
- Missing labels: 0
- Remaining exact duplicate headlines: 0

## 7. Model Configuration

- Input: Headline (`title`)
- Target: Label (`real`)
- Feature extraction: TF-IDF
- Classification models: Naive Bayes and Logistic Regression
- Train/test split: Same split for both models
- Random seed: Fixed for reproducibility

## 8. Dataset Limitations

- Class distribution is imbalanced.
- Dataset labels may contain original annotation errors.
- Exact duplicate removal does not detect semantically similar headlines.
- Model predictions represent learned dataset patterns, not verified facts.
- The dataset covers political and entertainment news domains.