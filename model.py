from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    HistGradientBoostingClassifier = None
    LogisticRegression = None
    make_pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


TOKEN_RE = re.compile(r"[a-z0-9]+")

MODALITY_KEYWORDS = {
    "ct": {"ct", "cta", "cat"},
    "mri": {"mri", "mr"},
    "xr": {"xr", "xray", "x", "radiograph"},
    "us": {"us", "ultrasound", "sonogram", "sono"},
    "pet": {"pet"},
    "nm": {"nm", "nuc", "nuclear", "spect"},
    "mam": {"mam", "mammography", "mammogram", "tomo"},
    "fluro": {"fluoro", "fluro", "esophagram"},
}

ANATOMY_GROUPS = {
    "brain": {"brain", "head", "cerebral", "stroke", "neuro", "maxfacial", "sinus"},
    "chest": {"chest", "lung", "thoracic", "thorax", "pulm"},
    "abdomen": {"abd", "abdomen", "pelvis", "aorta", "aaa", "renal", "kidney"},
    "spine": {"spine", "cervical", "thoracic", "lumbar"},
    "breast": {"breast", "mam", "mammography", "tomo", "axilla"},
    "cardiac": {"cardiac", "coronary", "myo", "myocard", "heart"},
    "extremity": {
        "knee",
        "shoulder",
        "elbow",
        "wrist",
        "ankle",
        "foot",
        "hand",
        "hip",
        "femur",
        "humerus",
    },
    "prostate": {"prostate"},
}

NEGATION_TOKENS = {"without", "wo", "non", "nocon", "no", "cntrst"}
CONTRAST_TOKENS = {"with", "w", "wcon", "contrast", "wo/w"}


@dataclass
class LinearModel:
    weights: list[float]
    bias: float


class RelevantPriorsModel:
    def __init__(
        self,
        modality_prior: dict[str, float],
        anatomy_prior: dict[str, float],
        threshold: float,
        linear_model: LinearModel | None = None,
        logistic_model: Any = None,
        hgb_model: Any = None,
    ):
        self.modality_prior = modality_prior
        self.anatomy_prior = anatomy_prior
        self.threshold = threshold
        self.linear_model = linear_model
        self.logistic_model = logistic_model
        self.hgb_model = hgb_model
        # Cache keyed on (c_desc, c_date, p_desc, p_date) -> predicted_is_relevant bool
        self._prediction_cache: dict[tuple[str, str, str, str], bool] = {}

    @property
    def mode(self) -> str:
        if self.logistic_model is not None and self.hgb_model is not None:
            return "ensemble"
        return "linear"

    @staticmethod
    def _safe_date(value: str) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    @lru_cache(maxsize=200000)
    def _normalize(desc: str) -> tuple[str, tuple[str, ...]]:
        normalized = " ".join(TOKEN_RE.findall((desc or "").lower()))
        tokens = tuple(normalized.split())
        return normalized, tokens

    @staticmethod
    def _get_modality(tokens: tuple[str, ...]) -> str:
        token_set = set(tokens)
        for modality, keys in MODALITY_KEYWORDS.items():
            if token_set.intersection(keys):
                return modality
        return "other"

    @staticmethod
    def _get_anatomy_groups(tokens: tuple[str, ...]) -> set[str]:
        token_set = set(tokens)
        groups = set()
        for group, keys in ANATOMY_GROUPS.items():
            if token_set.intersection(keys):
                groups.add(group)
        return groups

    @staticmethod
    def _has_contrast(tokens: tuple[str, ...]) -> int:
        tset = set(tokens)
        if tset.intersection(CONTRAST_TOKENS):
            return 1
        if tset.intersection(NEGATION_TOKENS):
            return -1
        return 0

    def _feature_vector(self, current: dict[str, Any], prior: dict[str, Any]) -> list[float]:
        c_norm, c_tokens = self._normalize(current.get("study_description", ""))
        p_norm, p_tokens = self._normalize(prior.get("study_description", ""))

        c_set = set(c_tokens)
        p_set = set(p_tokens)

        intersect = len(c_set & p_set)
        union = max(1, len(c_set | p_set))
        jaccard = intersect / union

        c_modality = self._get_modality(c_tokens)
        p_modality = self._get_modality(p_tokens)
        same_modality = 1.0 if c_modality == p_modality else 0.0

        c_anat = self._get_anatomy_groups(c_tokens)
        p_anat = self._get_anatomy_groups(p_tokens)
        anat_overlap = 1.0 if c_anat.intersection(p_anat) else 0.0

        c_contrast = self._has_contrast(c_tokens)
        p_contrast = self._has_contrast(p_tokens)
        contrast_match = 1.0 if c_contrast == p_contrast and c_contrast != 0 else 0.0
        contrast_conflict = 1.0 if c_contrast * p_contrast == -1 else 0.0

        c_date = self._safe_date(current.get("study_date", ""))
        p_date = self._safe_date(prior.get("study_date", ""))
        days_apart = 99999
        if c_date and p_date:
            days_apart = abs((c_date - p_date).days)

        within_30 = 1.0 if days_apart <= 30 else 0.0
        within_180 = 1.0 if days_apart <= 180 else 0.0
        within_365 = 1.0 if days_apart <= 365 else 0.0
        within_3y = 1.0 if days_apart <= 365 * 3 else 0.0

        desc_exact = 1.0 if c_norm == p_norm and c_norm != "" else 0.0
        starts_with_same_token = 1.0 if c_tokens and p_tokens and c_tokens[0] == p_tokens[0] else 0.0

        modality_base = self.modality_prior.get(c_modality, 0.3)
        anatomy_base = 0.0
        if c_anat:
            anatomy_base = max((self.anatomy_prior.get(g, 0.25) for g in c_anat), default=0.25)

        return [
            1.0,
            desc_exact,
            jaccard,
            same_modality,
            anat_overlap,
            contrast_match,
            contrast_conflict,
            within_30,
            within_180,
            within_365,
            within_3y,
            starts_with_same_token,
            modality_base,
            anatomy_base,
            math.log1p(days_apart),
        ]

    def _raw_score_from_features(self, features: list[float]) -> float:
        if self.mode == "ensemble":
            row = np.array([features], dtype=np.float64)
            p_lr = float(self.logistic_model.predict_proba(row)[0, 1])
            p_hgb = float(self.hgb_model.predict_proba(row)[0, 1])
            return 0.62 * p_lr + 0.38 * p_hgb

        z = self.linear_model.bias
        for w, xi in zip(self.linear_model.weights, features):
            z += w * xi
        if z >= 0:
            exp_neg = math.exp(-z)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(z)
        return exp_pos / (1.0 + exp_pos)

    def _raw_scores_batch(self, feature_matrix: list[list[float]]) -> list[float]:
        """Compute raw scores for many feature vectors in one sklearn call."""
        if self.mode == "ensemble":
            X = np.clip(np.array(feature_matrix, dtype=np.float64), -1e6, 1e6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p_lr = self.logistic_model.predict_proba(X)[:, 1]
                p_hgb = self.hgb_model.predict_proba(X)[:, 1]
            return list(0.62 * p_lr + 0.38 * p_hgb)
        return [self._raw_score_from_features(f) for f in feature_matrix]

    def _compatibility_adjustment(self, current: dict[str, Any], prior: dict[str, Any], score: float) -> float:
        _, c_tokens = self._normalize(current.get("study_description", ""))
        _, p_tokens = self._normalize(prior.get("study_description", ""))

        c_mod = self._get_modality(c_tokens)
        p_mod = self._get_modality(p_tokens)
        if c_mod == p_mod:
            return score

        c_set = set(c_tokens)
        p_set = set(p_tokens)
        intersect = len(c_set & p_set)
        union = max(1, len(c_set | p_set))
        jaccard = intersect / union

        c_anat = self._get_anatomy_groups(c_tokens)
        p_anat = self._get_anatomy_groups(p_tokens)
        same_anatomy = bool(c_anat.intersection(p_anat))

        allowed_cross_modality = {
            ("mam", "us"),
            ("us", "mam"),
            ("ct", "pet"),
            ("pet", "ct"),
            ("ct", "nm"),
            ("nm", "ct"),
            ("ct", "xr"),
            ("xr", "ct"),
        }

        if (c_mod, p_mod) in allowed_cross_modality:
            return score

        if not same_anatomy:
            return max(0.0, score - 0.25)
        if jaccard < 0.50:
            return max(0.0, score - 0.18)
        return score

    def score(self, current: dict[str, Any], prior: dict[str, Any]) -> float:
        features = self._feature_vector(current, prior)
        raw = self._raw_score_from_features(features)
        return self._compatibility_adjustment(current, prior, raw)

    def predict_case(self, case: dict[str, Any]) -> list[dict[str, Any]]:
        current = case.get("current_study", {})
        case_id = str(case.get("case_id", ""))
        priors = case.get("prior_studies", [])
        if not priors:
            return []
        feature_matrix = [self._feature_vector(current, p) for p in priors]
        raw_scores = self._raw_scores_batch(feature_matrix)
        predictions = []
        for prior, raw in zip(priors, raw_scores):
            score = self._compatibility_adjustment(current, prior, raw)
            predictions.append(
                {
                    "case_id": case_id,
                    "study_id": str(prior.get("study_id", "")),
                    "predicted_is_relevant": bool(score >= self.threshold),
                }
            )
        return predictions

    def _cache_key(self, current: dict[str, Any], prior: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            current.get("study_description", "") or "",
            current.get("study_date", "") or "",
            prior.get("study_description", "") or "",
            prior.get("study_date", "") or "",
        )

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bulk predict with cache: repeated (current, prior) pairs skip recomputation."""
        cases = payload.get("cases", [])
        if not cases:
            return {"predictions": []}

        # Slot for each output prediction; fill from cache or batch inference
        slots: list[dict[str, Any] | None] = []
        uncached_indices: list[int] = []
        uncached_features: list[list[float]] = []
        uncached_meta: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        for case in cases:
            current = case.get("current_study", {})
            case_id = str(case.get("case_id", ""))
            for prior in case.get("prior_studies", []):
                key = self._cache_key(current, prior)
                if key in self._prediction_cache:
                    slots.append({
                        "case_id": case_id,
                        "study_id": str(prior.get("study_id", "")),
                        "predicted_is_relevant": self._prediction_cache[key],
                    })
                else:
                    uncached_indices.append(len(slots))
                    uncached_features.append(self._feature_vector(current, prior))
                    uncached_meta.append((case_id, current, prior))
                    slots.append(None)

        if uncached_features:
            raw_scores = self._raw_scores_batch(uncached_features)
            for idx, (case_id, current, prior), raw in zip(uncached_indices, uncached_meta, raw_scores):
                score = self._compatibility_adjustment(current, prior, raw)
                result = bool(score >= self.threshold)
                self._prediction_cache[self._cache_key(current, prior)] = result
                slots[idx] = {
                    "case_id": case_id,
                    "study_id": str(prior.get("study_id", "")),
                    "predicted_is_relevant": result,
                }

        return {"predictions": [s for s in slots if s is not None]}


class _SGDLogReg:
    def __init__(self, n_features: int, lr: float = 0.03, epochs: int = 7, l2: float = 1e-4):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            e = math.exp(-z)
            return 1.0 / (1.0 + e)
        e = math.exp(z)
        return e / (1.0 + e)

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = self.b + sum(wj * xj for wj, xj in zip(self.w, xi))
                p = self._sigmoid(z)
                grad = p - yi
                self.b -= self.lr * grad
                for j in range(len(self.w)):
                    self.w[j] -= self.lr * (grad * xi[j] + self.l2 * self.w[j])

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [self._sigmoid(self.b + sum(w * x for w, x in zip(self.w, xi))) for xi in X]


def _compute_priors(data: dict[str, Any], truth_lookup: dict[tuple[str, str], int]) -> tuple[dict[str, float], dict[str, float]]:
    modality_stats = defaultdict(lambda: [0, 0])
    anatomy_stats = defaultdict(lambda: [0, 0])

    helper = RelevantPriorsModel(modality_prior={}, anatomy_prior={}, threshold=0.5)

    for case in data.get("cases", []):
        current = case.get("current_study", {})
        _, c_tokens = helper._normalize(current.get("study_description", ""))
        c_modality = helper._get_modality(c_tokens)
        c_groups = helper._get_anatomy_groups(c_tokens)

        for prior in case.get("prior_studies", []):
            key = (str(case.get("case_id", "")), str(prior.get("study_id", "")))
            if key not in truth_lookup:
                continue
            y = truth_lookup[key]
            modality_stats[c_modality][0] += y
            modality_stats[c_modality][1] += 1
            for g in c_groups:
                anatomy_stats[g][0] += y
                anatomy_stats[g][1] += 1

    modality_prior = {k: (v[0] + 1.0) / (v[1] + 2.0) for k, v in modality_stats.items()}
    anatomy_prior = {k: (v[0] + 1.0) / (v[1] + 2.0) for k, v in anatomy_stats.items()}
    return modality_prior, anatomy_prior


def _build_dataset(
    data: dict[str, Any],
    truth_lookup: dict[tuple[str, str], int],
    model_for_features: RelevantPriorsModel,
) -> tuple[list[list[float]], list[int], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    X: list[list[float]] = []
    y: list[int] = []
    case_ids: list[str] = []
    currents: list[dict[str, Any]] = []
    priors: list[dict[str, Any]] = []

    for case in data.get("cases", []):
        current = case.get("current_study", {})
        case_id = str(case.get("case_id", ""))
        for prior in case.get("prior_studies", []):
            key = (case_id, str(prior.get("study_id", "")))
            if key not in truth_lookup:
                continue
            X.append(model_for_features._feature_vector(current, prior))
            y.append(truth_lookup[key])
            case_ids.append(case_id)
            currents.append(current)
            priors.append(prior)

    return X, y, case_ids, currents, priors


def _train_sklearn_ensemble(X_train: list[list[float]], y_train: list[int]) -> tuple[Any, Any]:
    x_train_np = np.array(X_train, dtype=np.float64)
    x_train_np = np.clip(x_train_np, -1e6, 1e6)
    y_train_np = np.array(y_train, dtype=np.int64)

    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.5,
            max_iter=400,
            solver="saga",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )
    logistic_model.fit(x_train_np, y_train_np)

    hgb_model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.06,
        max_iter=220,
        min_samples_leaf=18,
        random_state=42,
    )
    hgb_model.fit(x_train_np, y_train_np)
    return logistic_model, hgb_model


def _tune_threshold(model: RelevantPriorsModel, val_currents: list[dict[str, Any]], val_priors: list[dict[str, Any]], y_val: list[int]) -> float:
    val_scores = [model.score(c, p) for c, p in zip(val_currents, val_priors)]
    best_threshold = 0.5
    best_acc = -1.0

    for t in [x / 100.0 for x in range(5, 96)]:
        correct = 0
        for score, yi in zip(val_scores, y_val):
            pred = 1 if score >= t else 0
            if pred == yi:
                correct += 1
        acc = correct / max(1, len(y_val))
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return best_threshold


def train_from_public_json(json_path: Path) -> RelevantPriorsModel:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    truth_lookup = {
        (str(item["case_id"]), str(item["study_id"])): 1 if item.get("is_relevant_to_current") else 0
        for item in data.get("truth", [])
    }

    modality_prior, anatomy_prior = _compute_priors(data, truth_lookup)

    model = RelevantPriorsModel(
        modality_prior=modality_prior,
        anatomy_prior=anatomy_prior,
        threshold=0.5,
    )

    X, y, case_ids, currents, priors = _build_dataset(data, truth_lookup, model)

    unique_case_ids = sorted(set(case_ids))
    val_case_set = set(unique_case_ids[::5])

    X_train, y_train = [], []
    X_val, y_val = [], []
    val_currents, val_priors = [], []

    for xi, yi, cid, current, prior in zip(X, y, case_ids, currents, priors):
        if cid in val_case_set:
            X_val.append(xi)
            y_val.append(yi)
            val_currents.append(current)
            val_priors.append(prior)
        else:
            X_train.append(xi)
            y_train.append(yi)

    if SKLEARN_AVAILABLE and np is not None:
        logistic_model, hgb_model = _train_sklearn_ensemble(X_train, y_train)
        model.logistic_model = logistic_model
        model.hgb_model = hgb_model
    else:
        learner = _SGDLogReg(n_features=len(X[0]), lr=0.03, epochs=9, l2=1e-4)
        learner.fit(X_train, y_train)
        model.linear_model = LinearModel(weights=learner.w, bias=learner.b)

    model.threshold = _tune_threshold(model, val_currents, val_priors, y_val)
    return model


def save_weights(model: RelevantPriorsModel, out_path: Path) -> None:
    payload = {
        "mode": model.mode,
        "threshold": model.threshold,
        "modality_prior": model.modality_prior,
        "anatomy_prior": model.anatomy_prior,
    }

    if model.mode == "ensemble":
        payload["logistic_model"] = model.logistic_model
        payload["hgb_model"] = model.hgb_model
    else:
        payload["linear_model"] = {
            "weights": model.linear_model.weights,
            "bias": model.linear_model.bias,
        }

    if out_path.suffix == ".json" or joblib is None:
        if payload.get("mode") == "ensemble":
            raise RuntimeError("Cannot serialize sklearn ensemble to JSON. Use a .pkl path.")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        return

    joblib.dump(payload, out_path)


def load_weights(weights_path: Path) -> RelevantPriorsModel:
    if weights_path.suffix == ".json":
        with weights_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        if joblib is None:
            raise RuntimeError("joblib is required to load pickle model artifacts")
        payload = joblib.load(weights_path)

    linear_model = None
    if "linear_model" in payload:
        lm = payload["linear_model"]
        linear_model = LinearModel(weights=[float(x) for x in lm["weights"]], bias=float(lm["bias"]))

    return RelevantPriorsModel(
        modality_prior={str(k): float(v) for k, v in payload.get("modality_prior", {}).items()},
        anatomy_prior={str(k): float(v) for k, v in payload.get("anatomy_prior", {}).items()},
        threshold=float(payload.get("threshold", 0.5)),
        linear_model=linear_model,
        logistic_model=payload.get("logistic_model"),
        hgb_model=payload.get("hgb_model"),
    )
