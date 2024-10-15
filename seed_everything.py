import random
import os
import numpy as np


def seed_everything(seed):
    """
    Seeds the 3 conceivable randomizers used throughout this repository. Default seed 42 is passed as an argument wherever necessary.
    """
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
