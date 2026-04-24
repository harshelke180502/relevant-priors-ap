# Relevant Priors API

A FastAPI service that predicts whether each prior radiology study is relevant to a radiologist reading a patient's current examination.

**Challenge**: `relevant-priors-v1`  
**Public eval accuracy**: 91.97% on 27,614 labeled prior exams

---

## How it works

The model uses **feature engineering + an sklearn ensemble** (no LLM calls):

| Feature | Description |
|---------|-------------|
| `desc_exact` | Current and prior descriptions are identical |
| `jaccard` | Token-set Jaccard similarity of study descriptions |
| `same_modality` | Both studies use the same imaging modality (CT, MRI, XR, US, etc.) |
| `anat_overlap` | Both studies cover the same anatomy group (brain, chest, spine, etc.) |
| `contrast_match/conflict` | Contrast usage agrees or conflicts |
| `within_30/180/365/3y` | How recently the prior was taken |
| `starts_with_same_token` | Descriptions share the same first token |
| `modality_base`, `anatomy_base` | Empirical base relevance rates per modality/anatomy |
| `log1p(days_apart)` | Log-scaled temporal distance |

These 15 features feed a **0.62 × LogisticRegression + 0.38 × HistGradientBoosting** ensemble, with a threshold tuned on a held-out validation split. A post-processing compatibility adjustment lowers scores for cross-modality, cross-anatomy pairs.

---

## Project structure

```
.
├── app.py                      # FastAPI server (POST /predict)
├── model.py                    # RelevantPriorsModel + training logic
├── train_and_freeze.py         # Train from public JSON and save model_weights.pkl
├── evaluate_public.py          # Score against the labeled public split
├── model_weights.pkl           # Pre-trained ensemble (joblib)
├── relevant_priors_public.json # Labeled public eval dataset (996 cases)
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

On startup the server loads `model_weights.pkl`. If it does not exist it trains from `relevant_priors_public.json` automatically.

Health check:
```bash
curl http://localhost:8000/
# {"status":"ok"}
```

---

## API

### `POST /predict`

**Request**

```json
{
  "challenge_id": "relevant-priors-v1",
  "schema_version": 1,
  "generated_at": "2026-04-16T12:00:00.000Z",
  "cases": [
    {
      "case_id": "1001016",
      "patient_id": "606707",
      "patient_name": "Andrews, Micheal",
      "current_study": {
        "study_id": "3100042",
        "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
        "study_date": "2026-03-08"
      },
      "prior_studies": [
        {
          "study_id": "2453245",
          "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
          "study_date": "2020-03-08"
        },
        {
          "study_id": "992654",
          "study_description": "CT HEAD WITHOUT CNTRST",
          "study_date": "2021-03-08"
        }
      ]
    }
  ]
}
```

**Response**

```json
{
  "predictions": [
    {"case_id": "1001016", "study_id": "2453245", "predicted_is_relevant": true},
    {"case_id": "1001016", "study_id": "992654",  "predicted_is_relevant": false}
  ]
}
```

One prediction is returned for every prior study in the request. `predicted_is_relevant` is a boolean.

---

## Retraining

```bash
python train_and_freeze.py
```

Trains the ensemble on `relevant_priors_public.json`, tunes the threshold on a 20% validation split, and writes `model_weights.pkl`.

## Evaluating locally

```bash
python evaluate_public.py
# accuracy=0.9197146375 n=27614 threshold=0.61 mode=ensemble
```

---

## Design notes

- **Bulk inference**: all prior studies across all cases in a single request are batched into one numpy array and scored in a single `predict_proba` call per model — well within the 360 s evaluator timeout.
- **Caching**: repeated `(current_description, current_date, prior_description, prior_date)` pairs are cached in memory; retries or duplicate study pairs are returned immediately (~0.015 s for a full dataset retry vs ~0.52 s cold).
- **Logging**: every request logs a short UUID, case count, prior count, and elapsed time for easy evaluator debugging.
- **No LLM dependency**: the service has no external API calls, eliminating per-examination timeout risk.
