import logging
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_training():
    logging.info("Starting GPT-2 training process...")

    logging.info(f"Loading data from {config.INITIAL_DATA_PATH}")
    df = pd.read_csv(config.INITIAL_DATA_PATH)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        full_texts = [
            prompt + " " + response + tokenizer.eos_token
            for prompt, response in zip(examples["prompt"], examples["response"])
        ]

        tokenized_inputs = tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH,
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        return tokenized_inputs

    logging.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=train_df.columns.tolist()
    )
    val_dataset = val_dataset.map(
        tokenize_function, batched=True, remove_columns=val_df.columns.tolist()
    )
    logging.info(f"Loading model {config.MODEL_NAME} for Causal LM")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logging.info("Starting fine-tuning...")
    trainer.train()
    logging.info("Training finished.")

    trainer.save_model(str(config.MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(config.MODEL_OUTPUT_DIR))
    logging.info(f"Model saved to {config.MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    run_training()
