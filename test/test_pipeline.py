from backend.pipeline import analyze_speech

reference = "There is no evidence of election fraud"

spoken = "There is evidence of election fraud"

result = analyze_speech(
    reference,
    spoken
)

print(result)