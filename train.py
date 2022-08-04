import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from experiment import SegmentationModel3d
 
if __name__ == '__main__':
    
    defaults = {
        'learning_rate': 0.0001,
        'loss': 'dice',
        'alpha': 0.9,
        'blocks': 4,
        'batch_size': 10,
        'initial_features': 64,

        'p_dropout': 0.0,

        'p_affine_or_elastic': 0.0,
        'p_elastic': 0.2,
        'p_affine': 0.8,

        'patch_size': 48,
        'samples_per_volume': 10,
        'queue_length': 80,
        'patch_overlap': 4,
        'random_sample_ratio': 4,

        'log_image_every_n': 3,

        'data_path': '/data/training',
    }
    
    wandb.init(
        project="aneurism_detection",
        config=defaults
    )
    
    hparams = wandb.config._as_dict()
    
    model = SegmentationModel3d(hparams)
    
    wandb_logger = WandbLogger(
        project="aneurism_detection")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{wandb_logger.experiment.name}/",
        # save_top_k=1,
        # monitor="step",
        filename="{epoch:02d}-{avg_train_jaccard:.2f}",
    )
    
    early_stop_callback = EarlyStopping(
        monitor="avg_validation_sensitivity",
        patience=4,
        mode="max")
    
    
    trainer = pl.Trainer(
        #fast_dev_run=True,
        gpus=[0],
        #limit_train_batches=0.02, limit_val_batches=0.1,

        profiler="simple",
        
        callbacks=[checkpoint_callback],        
        
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=3,)

    trainer.fit(model, ckpt_path='checkpoints/pretty-frog-165/epoch=95-avg_train_jaccard=0.00.ckpt' )

