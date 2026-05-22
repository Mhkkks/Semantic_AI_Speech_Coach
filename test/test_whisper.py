from backend.speech_to_text import transcribe_audio

text = transcribe_audio("sample.wav")

print(text)