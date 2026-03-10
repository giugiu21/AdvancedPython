import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import pytorch_lightning as pl

#Imports for training and dataset splitting
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

#Importing the Dataset, the Model class, and the LightningModule class
from fmnist import FMNIST
from networks import CNN, MLP
from networks_lightning import MyLightningModule

def get_devices():
    """
    Check available device and return preferred one cuda > mps > cpu.
    """
    cuda_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    if len(cuda_devices) > 0:
        return cuda_devices[0]
    else:
        return "cpu"
        
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    
def build_model(model_name):
    if model_name == "cnn":
        return CNN(num_classes=6)
    elif model_name == "mlp":
        return MLP(num_classes=6)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


if __name__=="__main__":
    """Main method (will be called when executing the file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='whether the model should be trained')
    parser.add_argument('--evaluate', action='store_true', help='whether the model should be evaluated (either requires the path to the parameters of a trained model, or additionally the flag "--train")')
    parser.add_argument('--model-dir', type=str, default='model_checkpoints', help='directory in which the trained model is located/should be stored (assumes that the model parameters are stored in the file "best_model.pth", default="model_checkpoints")')
    parser.add_argument('--checkpoint', type=str, help='Concrete checkpoint. Can only used with evaluation and no training.')
    parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs (default=20)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default=0.001)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size (default=32)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default=42)')
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"], help='model type: "cnn" or "mlp" (default="cnn")')

    #Part 6
    parser.add_argument("--rotate-test", action="store_true", help="Enable random 90-degree rotations for test dataset")
    parser.add_argument("--rotate-train", action="store_true", help="Enable random 90-degree rotations for train and validation datasets")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    if not args.train and not args.evaluate:
        raise ValueError("Please pass at least one of --train or --evaluate.")

    if args.checkpoint is not None and args.train:
        raise ValueError("--checkpoint can only be used with --evaluate and without --train.")

    # Dataset preparation
    dataset = FMNIST(train=True, rotate= args.rotate_train)
    test = FMNIST(train=False, rotate=args.rotate_test)

    train_size = int(0.8 * len(dataset)) # taking only 80% of the dataset for training
    val_size = len(dataset) - train_size #the 20% remaining is for validation

    train, val = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed) #using the predefined seed so that the split is not random everytime we run the code
    )

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    # Training settings
    #Checkpoint stores the model's weights, epochs and optimizer state during training in case it is needed to restore the training up to that point 
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename=f"{args.model}" + "-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    #Early stopping allows to stop the training automatically if there is no significant changes after 5 consecutive epochs
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5
    )

    #This was added so that i could plot the metrics
    logger = CSVLogger(
        save_dir=args.model_dir,
        name=f"{args.model}_logs"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        deterministic=True
    )

    best_checkpoint_path = None

    # Training
    if args.train:
        model = build_model(args.model)
        lm = MyLightningModule(model=model, lr=args.lr)

        # log hyperparameters
        logger.log_hyperparams({
            "model_type": args.model,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed
        })

        trainer.fit(lm, train_loader, val_loader)

        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Best checkpoint: {best_checkpoint_path}")

    
    # Evaluation
    if args.evaluate:
        if args.train:
            checkpoint_path = best_checkpoint_path
        else:
            if args.checkpoint is None:
                raise ValueError("For evaluation without training, please provide --checkpoint.")
            checkpoint_path = args.checkpoint

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        #build the same model type
        model = build_model(args.model)

        lm = MyLightningModule.load_from_checkpoint(
            checkpoint_path,
            model=model,
            lr=args.lr
        )

        trainer.test(lm, dataloaders=test_loader)