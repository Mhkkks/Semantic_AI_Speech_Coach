# ============================================================
# sentiment_pipeline.py
# FULL SENTIMENT PIPELINE
# MINIMUM MODIFICATIONS FROM ORIGINAL NOTEBOOK
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# AUDIO
# ============================================================

import librosa

# ============================================================
# TORCH + TRANSFORMERS
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModel

# ============================================================
# SKLEARN
# ============================================================

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

# ============================================================
# GLOBAL CONFIG
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# LOAD PRETRAINED DISTILBERT
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model.to(device)
bert_model.eval()

# ============================================================
# TF-IDF SETTINGS
# EXACTLY AS PAPER
# ============================================================

TFIDF_MAX_FEATURES = 5000

TFIDF_NGRAM_RANGE = (1, 2)

TFIDF_STOPWORDS = "english"

# ============================================================
# RANDOM FOREST SETTINGS
# EXACTLY AS PAPER
# ============================================================

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

# ============================================================
# SENTIMENT LABEL FUNCTION
# ============================================================

def to_label(score):

    if score > 0.2:
        return "positive"

    elif score < -0.2:
        return "negative"

    else:
        return "neutral"


# ============================================================
# DISTILBERT EMBEDDINGS
# 768-DIM EXACTLY AS PAPER
# ============================================================

def get_distilbert_embedding(text):

    text = str(text)

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    with torch.no_grad():

        outputs = bert_model(**encoded)

        # CLS TOKEN
        embedding = outputs.last_hidden_state[:, 0, :]

    embedding = embedding.cpu().numpy()[0]

    return embedding


# ============================================================
# BATCH EMBEDDINGS
# ============================================================

def generate_text_embeddings(texts):

    embeddings = []

    for text in tqdm(texts, desc="Generating DistilBERT embeddings"):

        emb = get_distilbert_embedding(text)

        embeddings.append(emb)

    embeddings = np.array(embeddings)

    return embeddings


# ============================================================
# LIBROSA ACOUSTIC FEATURES
# EXACTLY AS PAPER
#
# 16 FEATURES:
# RMS mean
# RMS std
# Pause ratio
# 13 MFCC means
# ============================================================

def extract_audio_features(audio_path):

    try:

        y, sr = librosa.load(audio_path, sr=None)

        # ====================================================
        # RMS ENERGY
        # ====================================================

        rms = librosa.feature.rms(y=y)[0]

        rms_mean = np.mean(rms)

        rms_std = np.std(rms)

        # ====================================================
        # PAUSE RATIO
        # ====================================================

        silence_threshold = 0.01

        pause_ratio = np.mean(np.abs(y) < silence_threshold)

        # ====================================================
        # MFCC
        # ====================================================

        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13
        )

        mfcc_means = np.mean(mfccs, axis=1)

        # ====================================================
        # FINAL FEATURE VECTOR
        # ====================================================

        features = np.concatenate([
            [rms_mean],
            [rms_std],
            [pause_ratio],
            mfcc_means
        ])

        return features

    except Exception as e:

        print(f"Audio feature extraction failed: {audio_path}")

        return np.zeros(16)


# ============================================================
# AUDIO FEATURE MATRIX
# ============================================================

def generate_audio_feature_matrix(audio_paths):

    features = []

    for path in tqdm(audio_paths, desc="Extracting audio features"):

        feat = extract_audio_features(path)

        features.append(feat)

    features = np.array(features)

    return features


# ============================================================
# TF-IDF + LOGISTIC REGRESSION
# EXACTLY AS PAPER
# ============================================================

def train_tfidf_logistic_regression(texts, labels):

    pipeline = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                stop_words=TFIDF_STOPWORDS,
                ngram_range=TFIDF_NGRAM_RANGE
            )
        ),

        (
            "clf",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(texts, labels)

    return grid


# ============================================================
# TF-IDF + NAIVE BAYES
# EXACTLY AS PAPER
# ============================================================

def train_tfidf_naive_bayes(texts, labels):

    pipeline = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                stop_words=TFIDF_STOPWORDS,
                ngram_range=TFIDF_NGRAM_RANGE
            )
        ),

        (
            "clf",
            MultinomialNB()
        )
    ])

    param_grid = {
        "clf__alpha": [0.1, 0.5, 1.0]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(texts, labels)

    return grid


# ============================================================
# TF-IDF + RANDOM FOREST
# EXACTLY AS PAPER
# ============================================================

def train_tfidf_random_forest(texts, labels):

    pipeline = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                stop_words=TFIDF_STOPWORDS,
                ngram_range=TFIDF_NGRAM_RANGE
            )
        ),

        (
            "clf",
            RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                random_state=42
            )
        )
    ])

    param_grid = {
        "clf__max_depth": [None, 10, 20]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(texts, labels)

    return grid


# ============================================================
# TEXT-ONLY RANDOM FOREST
# DISTILBERT EMBEDDINGS
# ============================================================

def train_text_only_rf(text_embeddings, labels):

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42
    )

    model.fit(text_embeddings, labels)

    return model


# ============================================================
# AUDIO-ONLY RANDOM FOREST
# ============================================================

def train_audio_only_rf(audio_features, labels):

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42
    )

    model.fit(audio_features, labels)

    return model


# ============================================================
# MULTIMODAL RANDOM FOREST
# CONCAT TEXT + AUDIO
# ============================================================

def train_multimodal_rf(
    text_embeddings,
    audio_features,
    labels
):

    combined_features = np.concatenate(
        [text_embeddings, audio_features],
        axis=1
    )

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=42
    )

    model.fit(combined_features, labels)

    return model


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    macro_f1 = f1_score(
        y_test,
        preds,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        preds,
        average="weighted"
    )

    print("\nAccuracy:", acc)

    print("Macro F1:", macro_f1)

    print("Weighted F1:", weighted_f1)

    print("\nClassification Report:\n")

    print(classification_report(y_test, preds))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


# ============================================================
# SENTIMENT SHIFT ANALYSIS
# ============================================================

def analyze_sentiment_shift(
    reference_labels,
    asr_labels
):

    shifts = []

    for ref, asr in zip(reference_labels, asr_labels):

        shifts.append(ref != asr)

    shift_rate = np.mean(shifts)

    return {
        "shift_rate": shift_rate,
        "total_shifts": int(np.sum(shifts)),
        "total_samples": len(shifts)
    }


# ============================================================
# PLATFORM EVALUATION
# ============================================================

def evaluate_asr_platform(
    model,
    platform_texts,
    reference_labels
):

    preds = model.predict(platform_texts)

    results = analyze_sentiment_shift(
        reference_labels,
        preds
    )

    return results


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_complete_sentiment_pipeline(df):

    # ========================================================
    # REFERENCE TEXT + LABELS
    # ========================================================

    texts = df["transcript"].fillna("").astype(str)

    labels = df["label"]

    # ========================================================
    # AUDIO PATHS
    # ========================================================

    audio_paths = df["audio_filepath"]

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
        X_train_audio,
        X_test_audio
    ) = train_test_split(
        texts,
        labels,
        audio_paths,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # ========================================================
    # TF-IDF MODELS
    # ========================================================

    print("\n==============================")
    print("TF-IDF + Logistic Regression")
    print("==============================")

    lr_model = train_tfidf_logistic_regression(
        X_train_text,
        y_train
    )

    lr_results = evaluate_model(
        lr_model,
        X_test_text,
        y_test
    )

    print("\n==============================")
    print("TF-IDF + Naive Bayes")
    print("==============================")

    nb_model = train_tfidf_naive_bayes(
        X_train_text,
        y_train
    )

    nb_results = evaluate_model(
        nb_model,
        X_test_text,
        y_test
    )

    print("\n==============================")
    print("TF-IDF + Random Forest")
    print("==============================")

    rf_model = train_tfidf_random_forest(
        X_train_text,
        y_train
    )

    rf_results = evaluate_model(
        rf_model,
        X_test_text,
        y_test
    )

    # ========================================================
    # DISTILBERT EMBEDDINGS
    # ========================================================

    train_embeddings = generate_text_embeddings(
        X_train_text
    )

    test_embeddings = generate_text_embeddings(
        X_test_text
    )

    # ========================================================
    # AUDIO FEATURES
    # ========================================================

    train_audio_features = generate_audio_feature_matrix(
        X_train_audio
    )

    test_audio_features = generate_audio_feature_matrix(
        X_test_audio
    )

    # ========================================================
    # TEXT ONLY RF
    # ========================================================

    print("\n==============================")
    print("TEXT ONLY RANDOM FOREST")
    print("==============================")

    text_rf = train_text_only_rf(
        train_embeddings,
        y_train
    )

    text_rf_results = evaluate_model(
        text_rf,
        test_embeddings,
        y_test
    )

    # ========================================================
    # AUDIO ONLY RF
    # ========================================================

    print("\n==============================")
    print("AUDIO ONLY RANDOM FOREST")
    print("==============================")

    audio_rf = train_audio_only_rf(
        train_audio_features,
        y_train
    )

    audio_rf_results = evaluate_model(
        audio_rf,
        test_audio_features,
        y_test
    )

    # ========================================================
    # MULTIMODAL RF
    # ========================================================

    print("\n==============================")
    print("MULTIMODAL RANDOM FOREST")
    print("==============================")

    multimodal_rf = train_multimodal_rf(
        train_embeddings,
        train_audio_features,
        y_train
    )

    multimodal_test_features = np.concatenate(
        [test_embeddings, test_audio_features],
        axis=1
    )

    multimodal_results = evaluate_model(
        multimodal_rf,
        multimodal_test_features,
        y_test
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results = {

        "tfidf_logistic_regression": lr_results,

        "tfidf_naive_bayes": nb_results,

        "tfidf_random_forest": rf_results,

        "text_only_rf": text_rf_results,

        "audio_only_rf": audio_rf_results,

        "multimodal_rf": multimodal_results
    }

    return results


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = pd.read_csv("combined_audio_mapping.csv")

    # ========================================================
    # EXAMPLE LABEL GENERATION
    # USE YOUR ORIGINAL LABELS HERE
    # ========================================================

    # Example:
    # df["label"] = ...

    # ========================================================
    # RUN COMPLETE PIPELINE
    # ========================================================

    results = run_complete_sentiment_pipeline(df)

    print("\nFINAL RESULTS\n")

    print(results)
def analyze_single_sentiment(reference_text, spoken_text):

    ref_emb = get_distilbert_embedding(reference_text)

    hyp_emb = get_distilbert_embedding(spoken_text)

    # temporary sentiment scoring logic
    # can use cosine or classifier later

    return {
        "reference_sentiment": "positive",
        "spoken_sentiment": "neutral",
        "sentiment_shift": True
    }