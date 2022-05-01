import os 
import argparse

from pytorch_lightning.utilities.seed import seed_everything

from src.config import (
    AVAIL_GPUS,
    BATCH_SIZE,
    DEFAULT_EXP_NAME,
    DROPOUT_RATE,
    IMAGENET,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_WORKERS,
    WEIGHT_DECAY
) 

def main(args):
    pass


if __name__=="__main__":

    parser = argparse.ArgumentParser()


    # Hyper params
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Set no. of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Set batch size")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Set Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Set weight decay")
    parser.add_argument("--dropout_rate", type=float, default=DROPOUT_RATE, help="Set dropout rate")

    # Hardware  
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Set no. of threads")
    parser.add_argument("--gpu", type=int, default=AVAIL_GPUS, help="Set no. of GPUs to use") 

    # Dataset
    parser.add_argument("--dataset", type=str, default=IMAGENET, help="Set dataset name")

    # Experiment
    parser.add_argument("--exp_name", type=str, default=DEFAULT_EXP_NAME, help="Set exp Name")

    args = parser.parse_args()
    main(args)