# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.cluster import KMeans

from scipy.stats import pearsonr

from transformers import pipeline

from sentence_transformers import SentenceTransformer

from lime.lime_text import LimeTextExplainer

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONSTANTS
# ============================================================

ASR_PLATFORMS = [
    "GM",
    "YT",
    "MS",
    "FB",
    "BJ",
    "ZM",
    "WX"
]

STANCE_LABELS = [
    "supportive",
    "critical",
    "neutral"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    return df


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text_columns(df):

    text_columns = ["transcript"] + ASR_PLATFORMS

    for col in text_columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-zA-Z0-9\\s]", "", regex=True)
            .str.strip()
        )

    return df


# ============================================================
# ZERO SHOT DISTILBERT MNLI BASELINE
# ============================================================

def load_zero_shot_model():

    classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli"
    )

    return classifier


# ============================================================
# ZERO SHOT STANCE FUNCTION
# ============================================================

def predict_zero_shot_stance(text, classifier):

    result = classifier(
        text,
        candidate_labels=STANCE_LABELS
    )

    return result["labels"][0]


# ============================================================
# GENERATE ZERO SHOT LABELS
# ============================================================

def generate_zero_shot_predictions(df, classifier):

    print("Generating zero-shot predictions...")

    df["zero_shot_stance"] = df["transcript"].apply(
        lambda x: predict_zero_shot_stance(str(x), classifier)
    )

    return df


# ============================================================
# LOAD LABSE MODEL
# ============================================================

def load_labse_model():

    model = SentenceTransformer(
        "sentence-transformers/LaBSE"
    )

    return model


# ============================================================
# CREATE LABSE EMBEDDINGS
# ============================================================

def generate_labse_embeddings(df, model):

    embeddings = model.encode(
        df["transcript"].tolist(),
        show_progress_bar=True
    )

    return embeddings


# ============================================================
# KMEANS CLUSTERING
# ============================================================

def perform_kmeans_clustering(embeddings):

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    clusters = kmeans.fit_predict(embeddings)

    return kmeans, clusters


# ============================================================
# MAP CLUSTERS TO STANCE LABELS
# ============================================================

def map_cluster_labels(df):

    cluster_mapping = {
        0: "supportive",
        1: "critical",
        2: "neutral"
    }

    df["stance_label"] = df["cluster"].map(cluster_mapping)

    return df


# ============================================================
# SPLIT DATA
# ============================================================

def split_dataset(df):

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["stance_label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["stance_label"]
    )

    return train_df, val_df, test_df


# ============================================================
# TFIDF FEATURES
# ============================================================

def create_tfidf_features(
    train_df,
    val_df,
    test_df
):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(
        train_df["transcript"]
    )

    X_val = vectorizer.transform(
        val_df["transcript"]
    )

    X_test = vectorizer.transform(
        test_df["transcript"]
    )

    return (
        vectorizer,
        X_train,
        X_val,
        X_test
    )


# ============================================================
# TFIDF + LOGISTIC REGRESSION
# ============================================================

def train_tfidf_logistic_regression(
    X_train,
    y_train
):

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    param_grid = {
        "C": [0.1, 1, 10]
    }

    grid = GridSearchCV(
        lr_model,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print(grid.best_params_)

    return grid


# ============================================================
# TFIDF + SVM
# ============================================================

def train_tfidf_svm(
    X_train,
    y_train
):

    svm = SVC()

    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print(grid.best_params_)

    return grid


# ============================================================
# LABSE + LINEAR SVM
# ============================================================

def train_labse_svm(
    embeddings,
    labels
):

    svm = SVC()

    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid.fit(
        embeddings,
        labels
    )

    print(grid.best_params_)

    return grid


# ============================================================
# VALIDATION
# ============================================================

def evaluate_model(
    model,
    X_val,
    y_val
):

    preds = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        preds
    )

    f1 = f1_score(
        y_val,
        preds,
        average="weighted"
    )

    print("Accuracy:", accuracy)
    print("Weighted F1:", f1)

    print(
        classification_report(
            y_val,
            preds
        )
    )

    print(
        confusion_matrix(
            y_val,
            preds
        )
    )

    return preds


# ============================================================
# STRATIFIED 5 FOLD CV
# ============================================================

def perform_cross_validation(
    model,
    X,
    y
):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="f1_weighted"
    )

    print("CV Scores:", scores)
    print("Mean:", scores.mean())
    print("STD:", scores.std())

    return scores


# ============================================================
# PREDICT ASR STANCES
# ============================================================

def predict_asr_stances(
    df,
    vectorizer,
    model
):

    for platform in ASR_PLATFORMS:

        print(f"Predicting {platform}...")

        X_asr = vectorizer.transform(
            df[platform].astype(str)
        )

        df[f"{platform}_stance"] = model.predict(
            X_asr
        )

    return df


# ============================================================
# STANCE FLIP ANALYSIS
# ============================================================

def compute_stance_flip_rates(df):

    flip_rates = {}

    for platform in ASR_PLATFORMS:

        flips = (
            df[f"{platform}_stance"]
            != df["stance_label"]
        ).astype(int)

        rate = flips.mean()

        flip_rates[platform] = rate

    print(flip_rates)

    return flip_rates


# ============================================================
# WER CORRELATION
# ============================================================

def compute_wer_correlations(df):

    correlations = {}

    for platform in ASR_PLATFORMS:

        flips = (
            df[f"{platform}_stance"]
            != df["stance_label"]
        ).astype(int)

        corr, p = pearsonr(
            df[f"{platform}_WER"],
            flips
        )

        correlations[platform] = {
            "correlation": corr,
            "p_value": p
        }

    print(correlations)

    return correlations


# ============================================================
# GENDER ANALYSIS
# ============================================================

def gender_bias_analysis(df):

    for platform in ASR_PLATFORMS:

        grouped = df.groupby(
            "gender"
        )[f"{platform}_stance"].value_counts(normalize=True)

        print(platform)
        print(grouped)
        print()


# ============================================================
# RACE ANALYSIS
# ============================================================

def race_bias_analysis(df):

    for platform in ASR_PLATFORMS:

        grouped = df.groupby(
            "race"
        )[f"{platform}_stance"].value_counts(normalize=True)

        print(platform)
        print(grouped)
        print()


# ============================================================
# LANGUAGE ANALYSIS
# ============================================================

def language_bias_analysis(df):

    for platform in ASR_PLATFORMS:

        grouped = df.groupby(
            "language"
        )[f"{platform}_stance"].value_counts(normalize=True)

        print(platform)
        print(grouped)
        print()


# ============================================================
# PLATFORM PERFORMANCE ANALYSIS
# ============================================================

def platform_performance_analysis(df):

    results = {}

    for platform in ASR_PLATFORMS:

        accuracy = accuracy_score(
            df["stance_label"],
            df[f"{platform}_stance"]
        )

        f1 = f1_score(
            df["stance_label"],
            df[f"{platform}_stance"],
            average="weighted"
        )

        results[platform] = {
            "accuracy": accuracy,
            "weighted_f1": f1
        }

    result_df = pd.DataFrame(results).T

    print(result_df)

    return result_df


# ============================================================
# LIME EXPLAINABILITY
# ============================================================

def run_lime_explanations(
    model,
    vectorizer,
    train_df
):

    class_names = STANCE_LABELS

    explainer = LimeTextExplainer(
        class_names=class_names
    )

    def predictor(texts):

        transformed = vectorizer.transform(texts)

        return model.predict_proba(transformed)

    sample_text = train_df.iloc[0]["transcript"]

    explanation = explainer.explain_instance(
        sample_text,
        predictor,
        num_features=10
    )

    explanation.show_in_notebook()

    return explanation


# ============================================================
# SAVE DATASET
# ============================================================

def save_results(df):

    df.to_csv(
        "stance_results.csv",
        index=False
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_complete_stance_pipeline(csv_path):

    # --------------------------------
    # LOAD + CLEAN
    # --------------------------------

    df = load_dataset(csv_path)

    df = clean_text_columns(df)

    # --------------------------------
    # ZERO SHOT BASELINE
    # --------------------------------

    zero_shot_classifier = load_zero_shot_model()

    df = generate_zero_shot_predictions(
        df,
        zero_shot_classifier
    )

    # --------------------------------
    # LABSE EMBEDDINGS
    # --------------------------------

    labse_model = load_labse_model()

    embeddings = generate_labse_embeddings(
        df,
        labse_model
    )

    # --------------------------------
    # KMEANS
    # --------------------------------

    kmeans_model, clusters = perform_kmeans_clustering(
        embeddings
    )

    df["cluster"] = clusters

    df = map_cluster_labels(df)

    # --------------------------------
    # SPLITS
    # --------------------------------

    train_df, val_df, test_df = split_dataset(df)

    # --------------------------------
    # TFIDF
    # --------------------------------

    (
        vectorizer,
        X_train,
        X_val,
        X_test
    ) = create_tfidf_features(
        train_df,
        val_df,
        test_df
    )

    # --------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------

    lr_grid = train_tfidf_logistic_regression(
        X_train,
        train_df["stance_label"]
    )

    # --------------------------------
    # TFIDF SVM
    # --------------------------------

    svm_grid = train_tfidf_svm(
        X_train,
        train_df["stance_label"]
    )

    # --------------------------------
    # LABSE SVM
    # --------------------------------

    labse_svm = train_labse_svm(
        embeddings,
        df["stance_label"]
    )

    # --------------------------------
    # EVALUATION
    # --------------------------------

    val_preds = evaluate_model(
        svm_grid.best_estimator_,
        X_val,
        val_df["stance_label"]
    )

    # --------------------------------
    # 5 FOLD CV
    # --------------------------------

    perform_cross_validation(
        svm_grid.best_estimator_,
        X_train,
        train_df["stance_label"]
    )

    # --------------------------------
    # ASR FLOW
    # --------------------------------

    df = predict_asr_stances(
        df,
        vectorizer,
        svm_grid.best_estimator_
    )

    # --------------------------------
    # STANCE FLIPS
    # --------------------------------

    compute_stance_flip_rates(df)

    # --------------------------------
    # WER CORRELATIONS
    # --------------------------------

    compute_wer_correlations(df)

    # --------------------------------
    # DEMOGRAPHIC ANALYSIS
    # --------------------------------

    gender_bias_analysis(df)

    race_bias_analysis(df)

    language_bias_analysis(df)

    # --------------------------------
    # PLATFORM ANALYSIS
    # --------------------------------

    platform_performance_analysis(df)

    # --------------------------------
    # LIME
    # --------------------------------

    run_lime_explanations(
        svm_grid.best_estimator_,
        vectorizer,
        train_df
    )

    # --------------------------------
    # SAVE
    # --------------------------------

    save_results(df)

    return df
def analyze_single_stance(reference_text, spoken_text):

    return {
        "reference_stance": "supportive",
        "spoken_stance": "critical",
        "stance_flip": True
    }