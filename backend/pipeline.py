from speech_to_text import transcribe_audio
from wer_utils import compute_wer

from sentiment import analyze_single_sentiment
from stance import analyze_single_stance
from misinfo import analyze_single_misinfo

def analyze_speech(reference_text, audio_path):

    # --------------------------------
    # TRANSCRIBE
    # --------------------------------

    spoken_text = transcribe_audio(audio_path)

    # --------------------------------
    # WER
    # --------------------------------

    wer_score = compute_wer(
        reference_text,
        spoken_text
    )

    # --------------------------------
    # SENTIMENT
    # --------------------------------

    sentiment_results = analyze_single_sentiment(
        reference_text,
        spoken_text
    )

    # --------------------------------
    # STANCE
    # --------------------------------

    stance_results = analyze_single_stance(
        reference_text,
        spoken_text
    )

    # --------------------------------
    # MISINFO
    # --------------------------------

    misinfo_results = analyze_single_misinfo(
        reference_text,
        spoken_text
    )

    # --------------------------------
    # FINAL OUTPUT
    # --------------------------------

    return {

        "spoken_text": spoken_text,

        "wer": wer_score,

        "sentiment": sentiment_results,

        "stance": stance_results,

        "misinfo": misinfo_results
    }