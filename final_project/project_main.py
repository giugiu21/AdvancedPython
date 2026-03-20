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
from pytorch_lightning.loggers import TensorBoardLogger

#Part 9 imports
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np

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

    #Part 8
    parser.add_argument("--use-augmentations", action="store_true", help="Enable additional training augmentations (Random Horizontal Flip)")

    #Part 9
    parser.add_argument("--analyze-error-source", action="store_true", help="Compute balanced accuracy and per-class accuracy")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    if not args.train and not args.evaluate:
        raise ValueError("Please pass at least one of --train or --evaluate.")

    if args.checkpoint is not None and args.train:
        raise ValueError("--checkpoint can only be used with --evaluate and without --train.")

    # --------------------------Dataset preparation---------------
    #Loading the dataset and test set throught the dataset class
    dataset = FMNIST(train=True, rotate= args.rotate_train, augment= args.use_augmentations)
    test = FMNIST(train=False, rotate=args.rotate_test)

    #Test-Train split 
    train_size = int(0.8 * len(dataset)) # taking only 80% of the dataset for training
    val_size = len(dataset) - train_size #the 20% remaining is for validation

    train, val = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed) #using the predefined seed so that the split is not random everytime we run the code
    )

    #Loading the training set the validation set and the test set
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    # ----------------------------Training settings------------------
    #Checkpoint stores the model's weights, epochs and optimizer state during training in case it is needed to restore the training up to that point 
    #it is based on validation accuracy
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename=f"{args.model}" + "-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    #Early stopping allows to stop the training automatically if there is no significant changes in validation accuracy after 5 consecutive epochs
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5
    )

    #Setting up the directories so that they are easier to see for exercise 7
    if args.rotate_train and args.rotate_test:
        directory = "train_test_aug"
    elif args.rotate_test:
        directory = "test_aug"
    elif args.use_augmentations:
        directory = "train_aug_part8"
    else:
        directory = "no_aug"

    #TensorFlow logger definition using the predefined directories
    logger = TensorBoardLogger(
        save_dir=args.model_dir,
        name=f"{args.model}_logs",
        version=directory
    )

    #Defining the trainer for the model
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        deterministic=True
    )

    #Initializing the best checkpoint path
    best_checkpoint_path = None

    # -----------------Training-------------------
    if args.train:
        #Building the model and the lightning module
        model = build_model(args.model)
        lm = MyLightningModule(model=model, lr=args.lr)

        #logging the hyperparameters given by user input
        logger.log_hyperparams({
            "model_type": args.model,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed
        })

        #Training the model on the preferences defined
        trainer.fit(lm, train_loader, val_loader)

        #Updating the best checkpoint path
        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Best checkpoint: {best_checkpoint_path}") #Printing it on screen when training is over

    
    # --------------------------Evaluation--------------------
    if args.evaluate:
        #Loading the best checkpoint path for the model when using the --evaluate flag combined with the --train flag
        if args.train:
            checkpoint_path = best_checkpoint_path
        else:
             #Control loop to ensure the checkpoint is given by the user in case we use the flag --evaluate i a different command line than --train flag
            if args.checkpoint is None:
                raise ValueError("For evaluation without training, please provide --checkpoint.")
            checkpoint_path = args.checkpoint
        #Control loop to ensure the checkpoint path exist
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}") 

        #build the same model type
        model = build_model(args.model)

        #Load the lighting module for the model
        lm = MyLightningModule.load_from_checkpoint(
            checkpoint_path,
            model=model,
            lr=args.lr
        )
        #Test the model
        trainer.test(lm, dataloaders=test_loader)

    #--------------------------- Error Analysis-----------------
    #Part 9 
    if args.analyze_error_source:
        #Initializing the predictions and labels
        all_preds = []
        all_labels = []

        #Evaluating the lighting module
        lm.eval()
        #Putting the lighting module to device (gpu is preferred if found)
        device = get_devices()
        lm.to(device)

        with torch.no_grad():
            #Loop over the test data (input-output pairs)
            for x, y in test_loader:
                x = x.to(device) 
                y = y.to(device) 

                #Passing the input to the model
                logits = lm(x)
                #Getting the predictions
                preds = torch.argmax(logits, dim=1)

                #Storing the predictions and all the labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        #Computing the balanced accuracy -- average of recall obtained on each class
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        print(f"Balanced Accuracy: {balanced_accuracy}")

        #Computing the per-class accuracy
        class_accuracy = recall_score(all_labels, all_preds, average=None)

        #For each class print the accuracy score
        for i, accuracy in enumerate(class_accuracy):
            print(f"Class {i} Accuracy: {accuracy:.4f}")

        num_classes = len(np.unique(all_labels)) #total number of classes

        #Bar Plot for per class accuracy
        plt.figure()
        plt.bar(range(num_classes), class_accuracy)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Class")

        plt.tight_layout()
        plt.show()