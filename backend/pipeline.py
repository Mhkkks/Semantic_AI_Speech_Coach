from backend.wer_utils import compute_wer
from backend.sentiment import analyze_single_sentiment
from backend.stance import analyze_single_stance
from backend.misinfo import analyze_single_misinfo


def analyze_speech(reference_text, spoken_text):

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
    # ADJUSTED MAS
    # --------------------------------

    if stance_results["stance_flip"]:

        adjusted_mas = max(
            0,
            misinfo_results["mas_score"] - 0.30
        )

    else:

        adjusted_mas = misinfo_results["mas_score"]

    misinfo_results["adjusted_mas"] = adjusted_mas

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