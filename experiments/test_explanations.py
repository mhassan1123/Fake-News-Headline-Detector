from src.explain import get_influential_words, explain_headline

print("\nGLOBAL INFLUENTIAL WORDS")
print("=" * 60)

global_words = get_influential_words(top_n=10)

print("\nWords associated with Fake predictions:")
for item in global_words["fake_words"]:
    print(f"{item['word']}: {item['coefficient']:.4f}")

print("\nWords associated with Real predictions:")
for item in global_words["real_words"]:
    print(f"{item['word']}: {item['coefficient']:.4f}")

test_headlines = [
    "Government announces new economic policy",
    "Shocking discovery reveals secret government conspiracy",
    "Scientists publish new research on climate change"
]

print("\nINDIVIDUAL HEADLINE EXPLANATIONS")
print("=" * 60)

for headline in test_headlines:
    result = explain_headline(headline)

    print(f"\nHeadline: {result['headline']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Fake probability: {result['fake_probability']:.4f}")
    print(f"Threshold: {result['threshold']:.3f}")

    print("Words supporting prediction:")
    for item in result["supporting_words"]:
        print(
            f"  {item['word']}: "
            f"{item['contribution']:.4f}"
        )

    print("Words opposing prediction:")
    for item in result["opposing_words"]:
        print(
            f"  {item['word']}: "
            f"{item['contribution']:.4f}"
        )

    print(result["explanation_note"])

print("\nEXPLANATION TEST COMPLETED")