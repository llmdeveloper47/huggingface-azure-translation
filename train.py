import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import mlflow
import argparse

# Start Logging
mlflow.set_experiment("machine-translation")
mlflow.start_run()

checkpoint = "t5-small"
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

metric = evaluate.load("sacrebleu")

def load_dataset():
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.2)



def load_artifacts():
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model

tokenizer,model = load_artifacts()

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def finetune_model(tokenized_books, model_artifact_path,mlflow_artifact_path ):
    training_args = Seq2SeqTrainingArguments(
    output_dir=model_artifact_path,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    
    components = {
            "model": model,
            "tokenizer": tokenizer,
        }
    
    mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path=mlflow_artifact_path,
        )
    

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_artifact_path", type=str, help="path to save finetuned transformer model")
    parser.add_argument("--mlflow_artifact_path", type=str, help="path to save mlflow model")
    args = parser.parse_args()
    return args


mlflow.end_run()

if __name__ == "__main__":

    args = argument_parser()

    books = load_dataset()
    

    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs


    tokenized_books = books.map(preprocess_function, batched=True)
    
    finetune_model(tokenized_books, args.model_artifact_path, args.mlflow_artifact_path)
