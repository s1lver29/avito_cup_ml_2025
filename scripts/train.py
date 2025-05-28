from catboost.utils import get_gpu_device_count

import implicit.gpu
import polars as pl
import numpy as np
import implicit
from scipy.sparse import csr_matrix
from pathlib import Path
from catboost import CatBoostRanker, Pool
from datetime import timedelta
from hydra import compose, initialize

from tools.retrievers import PopularItemsRetriever, CFRetriever
from tools.experiment_tracker import ExperimentTracker

print(get_gpu_device_count())
print(implicit.gpu.HAS_CUDA)
