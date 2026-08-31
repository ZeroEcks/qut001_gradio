import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_DIR = os.getenv("MODEL_WEIGHTS_DIR") or os.path.join(ROOT_DIR, "model_weights")
