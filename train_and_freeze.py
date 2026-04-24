from pathlib import Path

from model import save_weights, train_from_public_json


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "relevant_priors_public.json"
    out_path = root / "model_weights.pkl"

    model = train_from_public_json(data_path)
    save_weights(model, out_path)
    print(f"Saved model weights to {out_path}")


if __name__ == "__main__":
    main()
