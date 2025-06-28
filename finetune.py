import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
import numpy as np
from tqdm import tqdm
from utils import get_model
from data import get_dynamic_loader
from sklearn.metrics import accuracy_score

# --------- Custom Dataset Wrapper for Hugging Face ---------
class LoaderDataset(Dataset):
    def __init__(self, dataloader):
        self.dataset = []
        for batch in tqdm(dataloader, desc="Wrapping DataLoader"):
            images, labels = batch
            for img, lbl in zip(images, labels):
                self.dataset.append((img, lbl))
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {"pixel_values": img, "labels": label}
    def __len__(self):
        return len(self.dataset)

# --------- Model Wrapper (if needed) ---------
# If your model returns logits, Trainer will work; else, wrap it:
class MyModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits, labels)
        return {"logits": logits, "loss": loss}

# --------- Main Training Code ---------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 200
    model = get_model(num_classes=num_classes)
    model = MyModelWrapper(model).to(device)

    # Use your data loaders as before:
    train_loader = get_dynamic_loader(class_range=(0, 99), mode="train", batch_size=64)
    val_loader = get_dynamic_loader(class_range=(0, 99), mode="val", batch_size=64)

    # Wrap DataLoaders into Datasets for Trainer
    train_dataset = LoaderDataset(train_loader)
    val_dataset = LoaderDataset(val_loader)

    # Metrics for Trainer
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    OUTPUT_DIR = "./hf_trainer_output"
    NUM_EPOCHS = 100
    LR = 3e-4
    BATCH_SIZE = 64
    LABEL_SMOOTHING = 0.1
    WEIGHT_DECAY = 0.05

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=LABEL_SMOOTHING,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_total_limit=2,
        logging_steps=10,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()
