import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.fine_tuning import (
    load_pretrained_model,
    MultiLayerPerceptron,
    TextClassification,
    get_label_weights,
    load_and_prepare_data,
    train,
    evaluate
)

def main():
    """ An example of a general fine-tuning pipeline to classify clinical text using a
    pre-trained model that is based on the DistilBERT. Note that we are using mock labels
    which may not have any clinical relevance and is used here for educational purposes only.
    """
    pretrained_model_name = "distilbert/distilbert-base-uncased"
    # pretrained_model_name = "medicalai/ClinicalBERT"

    run_name = "fine_tuning"

    # Load pre-trained model, define classification head and our final model
    base_model, tokenizer = load_pretrained_model(pretrained_model_name)
    classification_head   = MultiLayerPerceptron(
        in_features = 768,         # BERT models usually have a dimension of 768
        layer_units = [ 64, 32 ],  # Number of units in hidden layers, change as needed
        n_classes   = 3,           # Number of classes, change as needed
        )
    model = TextClassification(base_model, classification_head, freeze_base_model=True)

    # Training specifications
    device         = "mps"
    optimizer      = optim.AdamW(model.parameters(), lr=0.001)
    label_weights  = get_label_weights("mock_labels/mts-dialog/MTS_Dataset_TrainingSet.csv").to(device)
    loss_fn        = nn.CrossEntropyLoss(weight=label_weights)
    sequence_len   = 128      # Maximum number of tokens to use from each sample of text (up to 512)
    batch_size     = 64       # How many samples of text per batch
    num_epochs     = 20       # Maximum number of training epochs
    early_stopping = 3        # Stop if validation loss does not improve after this number of epochs
    save_dir       = f"./output/{run_name}"
    model          = model.to(device)

    # Load train/validation/test sets
    train_data: DataLoader = load_and_prepare_data(
        X_path     = "clinical_notes_corpus/data/mts-dialog/MTS_Dataset_TrainingSet.csv",
        y_path     = "mock_labels/mts-dialog/MTS_Dataset_TrainingSet.csv",
        tokenizer  = tokenizer,
        batch_size = batch_size,
        shuffle    = True,
        max_length = sequence_len,
        device     = device
        )
    validation_data: DataLoader = load_and_prepare_data(
        X_path     = "clinical_notes_corpus/data/mts-dialog/MTS_Dataset_ValidationSet.csv",
        y_path     = "mock_labels/mts-dialog/MTS_Dataset_ValidationSet.csv",
        tokenizer  = tokenizer,
        batch_size = batch_size,
        shuffle    = False,
        max_length = sequence_len,
        device     = device
        )
    test_data: DataLoader = load_and_prepare_data(
        X_path     = "clinical_notes_corpus/data/mts-dialog/MTS_Dataset_Final_200_TestSet_1.csv",
        y_path     = "mock_labels/mts-dialog/MTS_Dataset_Final_200_TestSet_1.csv",
        tokenizer  = tokenizer,
        batch_size = batch_size,
        shuffle    = False,
        max_length = sequence_len,
        device     = device
        )

    # First, check performance before fine tuning
    evaluate(
        eval_data = test_data,
        model     = model,
        loss_fn   = loss_fn,
        save_dir  = save_dir,
        plot_name = "before_finetuning"
        )

    # Perform fine-tuning
    train(
        train_data      = train_data,
        validation_data = validation_data,
        model           = model,
        optimizer       = optimizer,
        loss_fn         = loss_fn,
        num_epochs      = num_epochs,
        early_stopping  = early_stopping,
        save_dir        = save_dir
        )

    # Evaluate fine-tuned model
    evaluate(
        eval_data = test_data,
        model     = model,
        loss_fn   = loss_fn,
        save_dir  = save_dir,
        plot_name = "after_finetuning"
        )

if __name__ == "__main__":
    main()