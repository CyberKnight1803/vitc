import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger

from src.config import PATH_EXPERIMENTS, PROJECT_NAME

def configure_logger(args):
    logger = WandbLogger(
        project=PROJECT_NAME,
        save_dir=PATH_EXPERIMENTS, 
        name=args.exp_name,
    )

    return logger 

def configure_trainer(args, logger):
    trainer = pl.Trainer(
        gpus=args.gpu,
        logger=logger, 
        max_epochs=args.epochs, 
    )

    return trainer 

