from sentiment import analyze_single_sentiment

ref = "I am very excited"

hyp = "I am sad"

result = analyze_single_sentiment(ref, hyp)

print(result)