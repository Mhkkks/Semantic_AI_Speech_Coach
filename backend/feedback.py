def generate_feedback(results):

    feedback = []

    if results["wer"] > 0.2:
        feedback.append(
            "High speech distortion detected."
        )

    if results["sentiment"]["sentiment_shift"]:
        feedback.append(
            "Emotional tone changed."
        )

    if results["stance"]["stance_flip"]:
        feedback.append(
            "Speech changed intended meaning."
        )

    if results["misinfo"]["meaning_drift"] > 0.4:
        feedback.append(
            "Strong semantic drift detected."
        )

    return feedback