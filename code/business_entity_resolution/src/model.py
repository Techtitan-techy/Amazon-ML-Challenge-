"""
Business Entity Resolution — Match Classifier & Decision Policy
==============================================================
Provides model training, evaluation, persistence, and entity-level inference
using XGBoost. Implements the empirically validated decision policy:
- Global decision threshold = 0.70
- Multi-match support (0, 1, or many links)
- Singleton floor pruning (max_p < 0.40 -> empty match set)
- Precision-weighted Macro F_0.5 evaluation
"""

import os
import joblib
import numpy as np
import xgboost as xgb
from features import FEATURE_NAMES

DEFAULT_MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.08,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1,
}

DECISION_THRESHOLD = 0.75
SINGLETON_FLOOR = 0.40


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict = None) -> xgb.XGBClassifier:
    """Train XGBoost binary classification model on pair features."""
    p = params or DEFAULT_MODEL_PARAMS
    model = xgb.XGBClassifier(**p)
    model.fit(X_train, y_train)
    return model


def save_model(model: xgb.XGBClassifier, filepath: str):
    """Serialize trained model to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> xgb.XGBClassifier:
    """Load serialized model from disk."""
    return joblib.load(filepath)


def predict_pair_probabilities(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Predict positive match probabilities for candidate pairs."""
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    return model.predict_proba(X)[:, 1]


def apply_decision_policy(candidates_with_scores: list,
                          threshold: float = DECISION_THRESHOLD,
                          singleton_floor: float = SINGLETON_FLOOR) -> list:
    """
    Apply 0/1/many decision policy to a list of (target_id, prob) pairs for an S1 entity.
    Returns:
        list of accepted target_ids.
    """
    if not candidates_with_scores:
        return []
        
    max_p = max(score for _, score in candidates_with_scores)
    # If the strongest candidate is below singleton floor, entity is deemed singleton
    if max_p < singleton_floor:
        return []
        
    # Accept all candidates meeting the optimal precision threshold
    accepted = [tid for tid, score in candidates_with_scores if score >= threshold]
    return accepted


def calc_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Official F_beta calculation with beta=0.5 weighting precision 2x."""
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom


def evaluate_macro_f05(predictions_dict: dict, ground_truth_dict: dict, all_s1_ids: list) -> dict:
    """
    Compute official macro-averaged F_0.5 score across all S1 entities.
    predictions_dict: {s1_id: set([matched_ids])}
    ground_truth_dict: {s1_id: set([true_matched_ids])}
    all_s1_ids: list of all S1 entity IDs
    """
    scores = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for s1_id in all_s1_ids:
        true_set = ground_truth_dict.get(s1_id, set())
        pred_set = predictions_dict.get(s1_id, set())
        
        if len(true_set) == 0:
            # Singleton credit: 1.0 if predicted empty, 0.0 otherwise
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
            if len(pred_set) > 0:
                fp_total += len(pred_set)
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            
            prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
            rec = tp / len(true_set)
            scores.append(calc_f_beta(prec, rec, beta=0.5))
            
    macro_f05 = float(np.mean(scores))
    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    
    return {
        'macro_f05': round(macro_f05, 5),
        'pair_precision': round(micro_prec, 5),
        'pair_recall': round(micro_rec, 5),
        'total_evaluated_s1': len(all_s1_ids)
    }
