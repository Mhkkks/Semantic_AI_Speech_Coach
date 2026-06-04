import jiwer
import re

from num2words import num2words

# ============================================================
# NUMBER NORMALIZATION
# ============================================================

replacements = {}

for i in range(99):

    replacements[num2words(i)] = str(i)

    if i == 1 or (i % 10 == 1 and i > 20):
        replacements[num2words(i, to='ordinal')] = str(i) + "st"

    elif i == 2 or (i % 10 == 2 and i > 20):
        replacements[num2words(i, to='ordinal')] = str(i) + "nd"

    elif i == 3 or (i % 10 == 3 and i > 20):
        replacements[num2words(i, to='ordinal')] = str(i) + "rd"

    else:
        replacements[num2words(i, to='ordinal')] = str(i) + "th"

# ============================================================
# TEXT TO NUMBERS
# ============================================================

def do_text_to_numbers(text):

    new_text = text

    for i, j in replacements.items():

        new_text = re.sub(
            r'\b' + i + r'\b',
            j,
            new_text
        )

    return new_text

# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def do_text_to_symbols(text):

    new_text = text

    new_text = re.sub(
        r"\b([0-9]+) percent\b",
        "\\1%",
        new_text
    )

    new_text = re.sub(
        r"\b([0-9]+) dollars{0,1}\b",
        "$\\1",
        new_text
    )

    new_text = re.sub(
        r"\b([0-9]+) euros{0,1}\b",
        "€\\1",
        new_text
    )

    return new_text

# ============================================================
# SANITIZATION
# ============================================================

def do_sanitize(text):

    new_text = text.lower().replace('-', '')

    new_text = do_text_to_numbers(new_text)

    new_text = do_text_to_symbols(new_text)

    return new_text

# ============================================================
# JIWER TRANSFORM
# ============================================================

transformation = jiwer.Compose([

    jiwer.RemoveKaldiNonWords(),

    jiwer.RemovePunctuation(),

    jiwer.ToLowerCase(),

    jiwer.SubstituteRegexes({
        r" '": r"'"
    }),

    jiwer.RemoveWhiteSpace(
        replace_by_space=True
    ),

    jiwer.RemoveMultipleSpaces(),

    jiwer.Strip(),

    jiwer.ReduceToListOfListOfWords()
])

# ============================================================
# MAIN WER FUNCTION
# ============================================================
def compute_wer(reference_text, hypothesis_text):
    return jiwer.wer(reference_text, hypothesis_text)

#def compute_wer(reference_text, hypothesis_text):



    reference_text = do_sanitize(
        reference_text
        .replace('\n', ' ')
        .replace('\r', '')
        .replace(" '", "'")
        .replace("(", "[")
        .replace(")", "]")
    )

    hypothesis_text = do_sanitize(
        hypothesis_text
        .replace('\n', ' ')
        .replace('\r', '')
        .replace(" '", "'")
        .replace("(", "[")
        .replace(")", "]")
    )

    measures = jiwer.compute_measures(

        reference_text,

        hypothesis_text,

        truth_transform=transformation,

        hypothesis_transform=transformation
    )

    return measures["wer"]
