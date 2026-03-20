import pytorch_lightning as pl
import torch
import torch.nn as nn


class MyLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()

        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss() #Like defined by the task

        #log the hyperparameters including the model type
        self.save_hyperparameters({"model_type": type(model).__name__, "lr": lr})

    #Passing the input to the network
    def forward(self, x):
        return self.model(x)

    #Defining the optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch #input-output pairs
        raw_output = self(x)
        loss = self.loss_fn(raw_output, y)

        #Logging the metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch #input-output pairs
        raw_output = self(x)
        loss = self.loss_fn(raw_output, y)

        predictions = torch.argmax(raw_output, dim=1) #argmax picks the predicted class -> class with higher probability
        accuracy = (predictions == y).float().mean()

        #Logging the metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch #input-output pairs
        raw_output = self(x)
        loss = self.loss_fn(raw_output, y)

        predictions = torch.argmax(raw_output, dim=1) #argmax picks the predicted class -> class with higher probability
        accuracy = (predictions == y).float().mean()

        #Logging the metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss