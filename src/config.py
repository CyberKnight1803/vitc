import os 
import torch

# PATHS 
PATH_IMAGENET_DATASET = os.environ.get("PATH_IMAGENET_DATASET", "./data/ImageNet")
PATH_IMAGENET_TRAIN = os.environ.get("PATH_IMAGENET_TRAIN", "./data/ImageNet/imagenette2-160/train")
PATH_IMAGENET_VAL = os.environ.get("PATH_IMAGENET_TRAIN", "./data/ImageNet/imagenette2-160/val")
PATH_IMAGENET_CSV = os.environ.get("PATH_IMAGENET_CSV", "./data/ImageNet/imagenette2-160/noisy_imagenette.csv")
 
PATH_EXPERIMENTS = os.environ.get("PATH_EXPERIMENTS", "./runs")

# DATASETS
IMAGENET = "ImageNet"

# HARDWARE 
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(int(os.cpu_count()) / 2)

# PROJECT 
PROJECT_NAME = "VITC"
DEFAULT_EXP_NAME = "test"


# HYPERPARAMETERS
MAX_EPOCHS = 30
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 1e-3

BATCH_SIZE = 64


