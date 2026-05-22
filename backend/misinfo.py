# ============================================================
# MISINFORMATION PRESERVATION PIPELINE
# PAPER-ALIGNED IMPLEMENTATION
# MINIMAL MODIFICATIONS VERSION
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import random
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm

from flashtext import KeywordProcessor

from scipy.stats import pearsonr

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# GLOBAL CONFIG
# ============================================================

ASR_SYSTEMS = ['YT', 'MS', 'FB', 'GM', 'BJ', 'ZM', 'WX']

TOP_TFIDF_FEATURES = 250000

TOP_SEED_TERMS = 6000

SEMANTIC_NEIGHBORS = 12

SVD_COMPONENTS = 300

SIM_THRESHOLD = 0.55

CHUNK_SIZE = 120

TFIDF_CHUNK_FEATURES = 50000

ALPHA = 0.6
BETA = 0.4


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    return df


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# CLEAN DATAFRAME
# ============================================================

def clean_all_transcripts(df):

    df['transcript'] = df['transcript'].apply(clean_text)

    for col in ASR_SYSTEMS:
        df[col] = df[col].apply(clean_text)

    return df


# ============================================================
# BUILD TFIDF MATRIX
# ============================================================

def build_tfidf_matrix(texts):

    vectorizer = TfidfVectorizer(
        ngram_range=(1,3),
        max_features=TOP_TFIDF_FEATURES,
        stop_words='english'
    )

    X = vectorizer.fit_transform(texts)

    return vectorizer, X


# ============================================================
# GET TOP TFIDF TERMS
# ============================================================

def get_top_tfidf_terms(vectorizer, X):

    scores = np.asarray(X.mean(axis=0)).ravel()

    vocab = np.array(vectorizer.get_feature_names_out())

    top_indices = np.argsort(scores)[::-1][:TOP_SEED_TERMS]

    top_terms = vocab[top_indices]

    return top_terms


# ============================================================
# APPLY SVD
# ============================================================

def apply_svd(X):

    svd = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        random_state=42
    )

    X_reduced = svd.fit_transform(X)

    return svd, X_reduced


# ============================================================
# SEMANTIC NEIGHBOR EXPANSION
# ============================================================

def semantic_neighbor_expansion(
    top_terms,
    vectorizer,
    svd,
    similarity_threshold=SIM_THRESHOLD
):

    vocab = vectorizer.get_feature_names_out()

    vocab_vectors = svd.components_.T

    final_terms = set()

    vocab_index = {
        term:i for i,term in enumerate(vocab)
    }

    for term in tqdm(top_terms):

        if term not in vocab_index:
            continue

        idx = vocab_index[term]

        query_vec = vocab_vectors[idx].reshape(1,-1)

        sims = cosine_similarity(
            query_vec,
            vocab_vectors
        )[0]

        neighbors_idx = np.argsort(sims)[::-1][1:SEMANTIC_NEIGHBORS+1]

        final_terms.add(term)

        for ni in neighbors_idx:

            if sims[ni] >= similarity_threshold:

                neighbor = vocab[ni]

                if (
                    neighbor not in ENGLISH_STOP_WORDS
                    and not neighbor.isnumeric()
                ):
                    final_terms.add(neighbor)

    return final_terms


# ============================================================
# BUILD MISINFORMATION LEXICON
# ============================================================

def build_misinformation_lexicon(df):

    texts = df['transcript'].fillna("").tolist()

    vectorizer, X = build_tfidf_matrix(texts)

    top_terms = get_top_tfidf_terms(vectorizer, X)

    svd, X_reduced = apply_svd(X)

    lexicon = semantic_neighbor_expansion(
        top_terms,
        vectorizer,
        svd
    )

    return lexicon, vectorizer, svd


# ============================================================
# BUILD FLASHTEXT PROCESSOR
# ============================================================

def build_keyword_processor(lexicon):

    kp = KeywordProcessor(case_sensitive=False)

    for term in lexicon:
        kp.add_keyword(term)

    return kp


# ============================================================
# EXTRACT KEYWORDS
# ============================================================

def extract_keywords(text, keyword_processor):

    if pd.isna(text):
        return set()

    kws = keyword_processor.extract_keywords(text)

    return set(kws)


# ============================================================
# KEYWORD METRICS
# ============================================================

def keyword_metrics(ref_set, hyp_set):

    TP = len(ref_set & hyp_set)

    FN = len(ref_set - hyp_set)

    FP = len(hyp_set - ref_set)

    recall = TP / (TP + FN + 1e-9)

    precision = TP / (TP + FP + 1e-9)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (
            2 * precision * recall
        ) / (precision + recall)

    return {
        'TP': TP,
        'FN': FN,
        'FP': FP,
        'Recall': recall,
        'Precision': precision,
        'F1': f1
    }


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def compute_semantic_similarity(ref_text, hyp_text):

    vectorizer = TfidfVectorizer(
        stop_words='english'
    )

    X = vectorizer.fit_transform([
        ref_text,
        hyp_text
    ])

    sim = cosine_similarity(X[0], X[1])[0][0]

    meaning_drift = 1 - sim

    return sim, meaning_drift


# ============================================================
# MAS COMPUTATION
# ============================================================

def compute_mas(recall, semantic_similarity):

    mas = (
        ALPHA * recall
        +
        BETA * semantic_similarity
    )

    return mas


# ============================================================
# EVALUATE ASR SYSTEMS
# ============================================================

def evaluate_asr_systems(df, keyword_processor):

    rows = []

    for system in ASR_SYSTEMS:

        for idx, row in tqdm(df.iterrows(), total=len(df)):

            ref_text = row['transcript']

            hyp_text = row[system]

            ref_keywords = extract_keywords(
                ref_text,
                keyword_processor
            )

            hyp_keywords = extract_keywords(
                hyp_text,
                keyword_processor
            )

            kw = keyword_metrics(
                ref_keywords,
                hyp_keywords
            )

            semantic_similarity, meaning_drift = \
                compute_semantic_similarity(
                    ref_text,
                    hyp_text
                )

            mas = compute_mas(
                kw['Recall'],
                semantic_similarity
            )

            wer_col = f"{system}_WER"

            rows.append({

                'name': row['name'],

                'platform': system,

                'WER': row.get(wer_col, np.nan),

                'TP': kw['TP'],
                'FN': kw['FN'],
                'FP': kw['FP'],

                'Recall': kw['Recall'],
                'Precision': kw['Precision'],
                'F1': kw['F1'],

                'SemanticSimilarity': semantic_similarity,

                'MeaningDrift': meaning_drift,

                'MAS': mas,

                'gender': row.get('gender'),

                'race': row.get('race'),

                'language': row.get('language'),

                'wps': row.get('wps')
            })

    results_df = pd.DataFrame(rows)

    return results_df


# ============================================================
# PEARSON CORRELATION
# ============================================================

def compute_wer_mas_correlation(results_df):

    correlations = []

    for system in ASR_SYSTEMS:

        sub = results_df[
            results_df['platform'] == system
        ]

        sub = sub.dropna(subset=['WER', 'MAS'])

        r, p = pearsonr(
            sub['WER'],
            sub['MAS']
        )

        correlations.append({
            'platform': system,
            'pearson_r': r,
            'p_value': p
        })

    corr_df = pd.DataFrame(correlations)

    return corr_df


# ============================================================
# SPEECH RATE GROUPING
# ============================================================

def assign_speech_rate_group(wps):

    if wps < 2.0:
        return 'slow'

    elif wps < 2.5:
        return 'normal'

    elif wps < 3.0:
        return 'fast'

    else:
        return 'very_fast'


# ============================================================
# DEMOGRAPHIC ANALYSIS
# ============================================================

def demographic_analysis(results_df):

    results_df['speech_group'] = \
        results_df['wps'].apply(
            assign_speech_rate_group
        )

    gender_stats = results_df.groupby(
        'gender'
    )['MAS'].mean()

    race_stats = results_df.groupby(
        'race'
    )['MAS'].mean()

    language_stats = results_df.groupby(
        'language'
    )['MAS'].mean()

    speech_stats = results_df.groupby(
        'speech_group'
    )['MAS'].mean()

    return {
        'gender': gender_stats,
        'race': race_stats,
        'language': language_stats,
        'speech_rate': speech_stats
    }


# ============================================================
# CREATE 120 WORD CHUNKS
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = words[i:i+chunk_size]

        chunk = " ".join(chunk)

        chunks.append(chunk)

    return chunks


# ============================================================
# BUILD CHUNK DATASET
# ============================================================

def build_chunk_dataset(df, keyword_processor):

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        text = row['transcript']

        chunks = create_chunks(text)

        for chunk in chunks:

            kws = extract_keywords(
                chunk,
                keyword_processor
            )

            kw_count = len(kws)

            if kw_count >= 2:
                label = 1

            elif kw_count == 0:
                label = 0

            else:
                continue

            rows.append({
                'chunk': chunk,
                'label': label
            })

    chunk_df = pd.DataFrame(rows)

    return chunk_df


# ============================================================
# TFIDF + SVD FEATURES
# ============================================================

def build_chunk_features(chunk_df):

    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=TFIDF_CHUNK_FEATURES,
        sublinear_tf=True,
        stop_words='english'
    )

    X = tfidf.fit_transform(
        chunk_df['chunk']
    )

    svd = TruncatedSVD(
        n_components=300,
        random_state=42
    )

    X_svd = svd.fit_transform(X)

    y = chunk_df['label'].values

    return tfidf, svd, X_svd, y


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_data(X, y):

    X_train, X_temp, y_train, y_temp = \
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    X_val, X_test, y_val, y_test = \
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(X_train, y_train):

    lr = LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        max_iter=2000
    )

    param_grid = {
        'C': [0.01, 0.1, 1, 10]
    }

    grid = GridSearchCV(
        lr,
        param_grid,
        cv=3,
        scoring='f1_weighted',
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    return best_model, grid


# ============================================================
# XGBOOST
# ============================================================

def train_xgboost(X_train, y_train):

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    param_dist = {

        'max_depth': [3,5,7],

        'learning_rate': [0.01,0.1],

        'n_estimators': [100,300,500],

        'subsample': [0.6,0.8,1.0],

        'scale_pos_weight': [1,2,5]
    }

    random_search = RandomizedSearchCV(

        xgb,

        param_distributions=param_dist,

        n_iter=10,

        cv=3,

        scoring='f1_weighted',

        verbose=1,

        random_state=42,

        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    return best_model, random_search


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(y_test, y_pred)

    rec = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print(classification_report(
        y_test,
        y_pred
    ))

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return {

        'accuracy': acc,

        'precision': prec,

        'recall': rec,

        'f1': f1,

        'confusion_matrix': cm
    }


# ============================================================
# CROSS ASR AGGREGATION
# ============================================================

def aggregate_platform_results(results_df):

    agg = results_df.groupby(
        'platform'
    ).agg({

        'WER': 'mean',

        'Recall': 'mean',

        'Precision': 'mean',

        'F1': 'mean',

        'SemanticSimilarity': 'mean',

        'MeaningDrift': 'mean',

        'MAS': 'mean'

    }).reset_index()

    return agg


# ============================================================
# EXPORT RESULTS
# ============================================================

def export_results(
    results_df,
    agg_df,
    corr_df,
    output_dir='misinfo_outputs'
):

    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(
        f'{output_dir}/misinfo_results.csv',
        index=False
    )

    agg_df.to_csv(
        f'{output_dir}/platform_summary.csv',
        index=False
    )

    corr_df.to_csv(
        f'{output_dir}/wer_mas_correlation.csv',
        index=False
    )


# ============================================================
# FULL PIPELINE
# ============================================================

def run_misinformation_pipeline(csv_path):

    print("Loading Dataset...")
    df = load_dataset(csv_path)

    print("Cleaning Transcripts...")
    df = clean_all_transcripts(df)

    print("Building Lexicon...")
    lexicon, vectorizer, svd = \
        build_misinformation_lexicon(df)

    print("Building FlashText Processor...")
    keyword_processor = \
        build_keyword_processor(lexicon)

    print("Evaluating ASR Systems...")
    results_df = evaluate_asr_systems(
        df,
        keyword_processor
    )

    print("Computing WER-MAS Correlation...")
    corr_df = compute_wer_mas_correlation(
        results_df
    )

    print("Running Demographic Analysis...")
    demographic_stats = demographic_analysis(
        results_df
    )

    print("Building Chunk Dataset...")
    chunk_df = build_chunk_dataset(
        df,
        keyword_processor
    )

    print("Building TFIDF + SVD Features...")
    tfidf, svd_model, X, y = \
        build_chunk_features(chunk_df)

    print("Splitting Data...")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    print("Training Logistic Regression...")
    lr_model, lr_grid = \
        train_logistic_regression(
            X_train,
            y_train
        )

    print("Training XGBoost...")
    xgb_model, xgb_search = \
        train_xgboost(
            X_train,
            y_train
        )

    print("Evaluating Logistic Regression...")
    lr_results = evaluate_model(
        lr_model,
        X_test,
        y_test
    )

    print("Evaluating XGBoost...")
    xgb_results = evaluate_model(
        xgb_model,
        X_test,
        y_test
    )

    print("Aggregating Platform Results...")
    agg_df = aggregate_platform_results(
        results_df
    )

    print("Exporting Results...")
    export_results(
        results_df,
        agg_df,
        corr_df
    )

    print("PIPELINE COMPLETE")

    return {

        'df': df,

        'lexicon': lexicon,

        'results_df': results_df,

        'correlation_df': corr_df,

        'demographic_stats': demographic_stats,

        'chunk_df': chunk_df,

        'lr_model': lr_model,

        'xgb_model': xgb_model,

        'lr_results': lr_results,

        'xgb_results': xgb_results,

        'agg_df': agg_df
    }


# ============================================================
# RUN
# ============================================================

# results = run_misinformation_pipeline(
#     "your_dataset.csv"
# )
def analyze_single_misinfo(reference_text, spoken_text):

    semantic_similarity, drift = \
        compute_semantic_similarity(
            reference_text,
            spoken_text
        )

    mas = compute_mas(
        0.8,
        semantic_similarity
    )

    return {
        "semantic_similarity": semantic_similarity,
        "meaning_drift": drift,
        "mas_score": mas
    }