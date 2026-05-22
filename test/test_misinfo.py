from misinfo import analyze_single_misinfo

ref = "There is no evidence of fraud"

hyp = "There is evidence of fraud"

result = analyze_single_misinfo(ref, hyp)

print(result)