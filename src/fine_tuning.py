from __future__ import annotations
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    BatchEncoding
)
from transformers.modeling_outputs import BaseModelOutput

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.early_stopping import EarlyStopping

class MultiLayerPerceptron(nn.Module):
    """ A simple and configurable multi-layer perceptron. """

    def __init__(
        self,
        in_features: int,
        n_classes  : int,
        layer_units: list[int],
    ):
        """ Constructor for MLP.

        Args:
            in_features: Number of input features.
            n_classes: Number of output classes.
            layer_units: List of units in each hidden layer.
        """
        super().__init__()

        # Create the layers
        layers    : list[nn.Module] = []
        prev_units: int             = in_features

        # Add hidden layers
        for units in layer_units:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.GELU())
            prev_units = units

        # Add output layer
        layers.append(nn.Linear(prev_units, n_classes))

        # Create the sequential model
        self._hidden_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._hidden_layers(x)

class TextClassification(nn.Module):
    """ Combines a DistilBert model with a MultiLayerPerceptron for text classification. """

    def __init__(
        self,
        base_model:          DistilBertModel,
        classification_head: MultiLayerPerceptron,
        freeze_base_model:   bool = True
        ) -> None:
        super().__init__()
        self.base_model          = base_model
        self.dropout             = nn.Dropout(0.2)
        self.classification_head = classification_head

        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor
        ) -> Tensor:
        """ Take encoded_input from the tokenizer and produce outputs as logits.

        Steps:
        1. Produce outputs using the base Bert model.
        2. Use the last_hidden_state of the transformer layer.
        3. Focus on the output for the first token, being "[CLS]".
        4. Pass this to classification head to obtain prediction logits.
        """
        output:           BaseModelOutput = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        output_cls_token: Tensor          = output.last_hidden_state[:, 0, :] # (batch_size, num_tokens, model_dim)
        output_cls_token                  = self.dropout(output_cls_token)
        logits:           Tensor          = self.classification_head(output_cls_token)
        return logits

def load_pretrained_model(name: str):
    model     = DistilBertModel.from_pretrained(name)
    tokenizer = DistilBertTokenizer.from_pretrained(name)
    return model, tokenizer

def get_label_weights(y_path: str):
    """ Calculate label weights for class balance. """
    y = pd.read_csv(y_path, index_col="ID")
    y_counts = np.unique_counts(y["score"]).counts
    y_weights = 1. / torch.tensor(y_counts, dtype=torch.float32)
    y_weights = y_weights / y_weights.max()
    return y_weights

def load_and_prepare_data(
    X_path:     str,
    y_path:     str,
    tokenizer:  DistilBertTokenizer,
    batch_size: int,
    shuffle:    bool,
    max_length: int   = 512,
    device:     str   = "cpu"
    ):
    X = pd.read_csv(X_path, index_col="ID")
    y = pd.read_csv(y_path, index_col="ID")
    assert len(X) == len(y), "X and y must have the same number of samples"

    def preprocess(text: str) -> str:
        """ Basic text preprocessing by converting to lower case and removing new line character. """
        return (
            text
            .lower()
            .replace("\n", " ")
        )

    # Format input text as: [CLS] <summary> <dialogue> [SEP]
    input_text: list[str] = []
    for _, row in X.iterrows():
        summary  = preprocess(row["section_text"])
        dialogue = preprocess(row["dialogue"])
        text = f"[CLS] {summary} {dialogue} [SEP]"
        input_text.append(text)

    # Use the tokenizer to create embeddings
    encoded_text: BatchEncoding = tokenizer(
        input_text,
        padding        = "max_length",
        truncation     = True,
        max_length     = max_length,
        return_tensors = "pt"
        )

    # Create the dataloader
    input_ids      = encoded_text.input_ids.to(device)
    attention_mask = encoded_text.attention_mask.to(device)
    y_tensor       = torch.tensor(y["score"].values, dtype=torch.long, device=device).squeeze()
    dataset        = TensorDataset(input_ids, attention_mask, y_tensor)
    dataloader     = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train_batch(
    input_ids:      Tensor,
    attention_mask: Tensor,
    y_true:         Tensor,
    model:          TextClassification,
    optimizer:      optim.Optimizer,
    loss_fn:        Callable
    ) -> float:
    """ Performs a forward and backwards pass and returning the loss value. """
    model.train()
    # Forward pass
    logits = model(input_ids, attention_mask)
    loss   = loss_fn(logits, y_true)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Output the value of the loss
    loss_value = loss.detach().cpu().numpy().item()
    return loss_value

def evaluate_batch(
    input_ids:      Tensor,
    attention_mask: Tensor,
    y_true:         Tensor,
    model:          TextClassification,
    loss_fn:        Callable
    ) -> tuple[Tensor, float]:
    """ Performs a inference only forward pass and returns the class prediction and loss value. """
    model.eval()
    with torch.no_grad():
        logits       = model(input_ids, attention_mask)
        loss         = loss_fn(logits, y_true)
        y_pred_proba = nn.functional.softmax(logits, dim=1)
        y_pred       = torch.argmax(y_pred_proba, dim=1)
        loss_value   = loss.detach().cpu().numpy().item()
    return y_pred, loss_value

def train(
    train_data:      DataLoader,
    validation_data: DataLoader,
    model:           TextClassification,
    optimizer:       optim.Optimizer,
    loss_fn:         Callable,
    num_epochs:      int,
    early_stopping:  int,
    save_dir:        str
    ):
    # Initiate loss trackers across epochs
    train_losses = []
    validation_losses = []

    # Create the save path
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Training phase with early stopping
    tracker = EarlyStopping(patience=early_stopping, minimise=True)
    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch:03d}")

        # Loss trackers for batches in this current epoch
        train_batch_losses      = []
        validation_batch_losses = []

        # First, train our model
        train_batches = tqdm(train_data, desc=f"Training")
        for input_ids, attention_mask, y_true in train_batches:
            train_loss = train_batch(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                y_true         = y_true,
                model          = model,
                optimizer      = optimizer,
                loss_fn        = loss_fn
                )
            train_batch_losses.append(train_loss)
            avg_train_loss = np.mean(train_batch_losses).item()
            train_batches.set_postfix({"loss": avg_train_loss})

        # Then, evaluate using validation data
        validation_batches = tqdm(validation_data, desc=f"Validation")
        for input_ids, attention_mask, y_true in validation_batches:
            y_pred, validation_loss = evaluate_batch(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                y_true         = y_true,
                model          = model,
                loss_fn        = loss_fn
                )
            validation_batch_losses.append(validation_loss)
            avg_validation_loss = np.mean(validation_batch_losses).item()
            validation_batches.set_postfix({"loss": avg_validation_loss})

        # Track losses
        avg_train_loss      = np.mean(train_batch_losses).item()
        avg_validation_loss = np.mean(validation_batch_losses).item()

        train_losses.append(avg_train_loss)
        validation_losses.append(avg_validation_loss)

        # Track validation loss with early stopping
        status = tracker.check(avg_validation_loss)
        if status == 2: # Did not improve
            print("Stopping training due to early stopping.")
            break
        elif status == 0: # Improves
            print("Improved, saved model weights.")
            torch.save(model.state_dict(), str(save_path / "model_weights.pth"))
            continue

    # Before we leave, we will save a plot of training_loss vs validation_loss
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(train_losses, c="royalblue", label="Training")
    ax.plot(validation_losses, c="mediumseagreen", label="Validation")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    fig.suptitle("Training and validation loss across epochs")
    fig.tight_layout()
    plt.savefig(str(save_path / "training_loss_graph.png"))

def evaluate(
    eval_data: DataLoader,
    model:     TextClassification,
    loss_fn:   Callable,
    save_dir:  str | None = None,
    plot_name: str        = "confusion_matrix"
    ):
    """ Evaluate the performance of a model. """
    # Load weights if specified
    if save_dir is not None:
        save_path = Path(save_dir)
        if (save_path / "model_weights.pth").exists():
            print("Loaded model weights.")
            model.load_state_dict(torch.load(str(save_path / "model_weights.pth")))

    # Interate through batches of evaluation data and generate predictions.
    y_true             = []
    y_pred             = []
    evaluation_losses  = []
    evaluation_batches = tqdm(eval_data, desc=f"Evaluation")
    for input_ids, attention_mask, batch_y_true in evaluation_batches:
        batch_y_pred, evaluation_loss = evaluate_batch(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            y_true         = batch_y_true,
            model          = model,
            loss_fn        = loss_fn
            )
        y_true.extend(batch_y_true.detach().cpu().numpy())
        y_pred.extend(batch_y_pred.detach().cpu().numpy())
        evaluation_losses.append(evaluation_loss)
        avg_evaluation_loss = np.mean(evaluation_losses).item()
        evaluation_batches.set_postfix({"loss": avg_evaluation_loss})

    # Compute metrics
    accuracy  = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall    = recall_score(y_true, y_pred, average="weighted")

    # Print metrics
    print("\nEvaluation results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    # Plot a confusion matrix
    cm  = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 8))
    gs  = GridSpec(nrows=1, ncols=1)
    ax  = fig.add_subplot(gs[0, 0])
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    fig.tight_layout()
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path / f"{plot_name}.png"))
    else:
        plt.show(block=True)
    plt.close()
