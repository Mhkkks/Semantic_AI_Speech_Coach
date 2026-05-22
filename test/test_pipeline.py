from pipeline import analyze_speech

reference = """
I strongly support renewable energy.
"""

audio_path = "sample.wav"

result = analyze_speech(
    reference,
    audio_path
)

print(result)