import os
import random

import numpy as np
import tensorflow as tf

from backend.app.core.config import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)