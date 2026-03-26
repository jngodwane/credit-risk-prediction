from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "credit_data.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "credit_risk_model.pkl"
TARGET = "default_flag"
RANDOM_STATE = 42