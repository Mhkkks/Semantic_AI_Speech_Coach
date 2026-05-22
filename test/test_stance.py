from stance import analyze_single_stance

ref = "I support renewable energy"

hyp = "I oppose renewable energy"

result = analyze_single_stance(ref, hyp)

print(result)