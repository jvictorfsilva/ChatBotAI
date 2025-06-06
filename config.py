import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()

MODEL_NAME = "pierreguillou/gpt2-small-portuguese"
MODEL_OUTPUT_DIR = BASE_DIR / "models" / "fine_tuned_gpt2"

INITIAL_DATA_PATH = BASE_DIR / "data" / "initial_data.csv"
CONVERSATION_LOG_PATH = BASE_DIR / "data" / "conversation_logs.jsonl"

NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-5))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 128))

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
