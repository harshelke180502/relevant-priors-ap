import json
from pathlib import Path

from model import load_weights


def main() -> None:
    data = json.load(open("relevant_priors_public.json", "r", encoding="utf-8"))
    truth = {
        (str(t["case_id"]), str(t["study_id"])): bool(t["is_relevant_to_current"])
        for t in data["truth"]
    }

    model = load_weights(Path("model_weights.pkl"))

    correct = 0
    n = 0
    for case in data["cases"]:
        for pred in model.predict_case(case):
            key = (pred["case_id"], pred["study_id"])
            if key not in truth:
                continue
            n += 1
            if pred["predicted_is_relevant"] == truth[key]:
                correct += 1

    accuracy = correct / max(1, n)
    print(f"accuracy={accuracy:.10f} n={n} threshold={model.threshold:.2f} mode={model.mode}")


if __name__ == "__main__":
    main()
