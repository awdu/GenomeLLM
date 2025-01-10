import pandas as pd
import os
import numpy as np

from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score

import transformers
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, TrainingArguments
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset

###########################################################
#		CLASSES
###########################################################
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, texts, labels, tokenizer):

        super(SupervisedDataset, self).__init__()

        sequences = [text for text in texts]

        output = tokenizer(
            sequences,
            add_special_tokens=True,
            max_length=310,
            padding="longest",
            return_tensors="pt",
            truncation=True
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i]
        )





##############################################################
#			FUNCTIONS
##############################################################
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": matthews_corrcoef(
            labels, predictions
        ),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def process_dataset(dataset_dir):
	print("\n\n\n")
	print(dataset_dir)

	trainFile = ""
	testFile = ""
	for file in os.listdir(dataset_dir):
		if ("test.tsv" in file):
			testFile = dataset_dir + "/" + file
		if ("train.tsv" in file):
			trainFile = dataset_dir + "/" + file
	train_dataIn = pd.read_csv(trainFile, sep='\t')
	test_dataIn = pd.read_csv(testFile, sep='\t')

	
	train = train_dataIn.sample(frac=0.8, random_state=0)
	validation = train_dataIn.drop(train.index)
	test = test_dataIn.sample(frac=1.0, random_state=0)

	train = train.reset_index(drop=True)
	validation = validation.reset_index(drop=True)
	test = test.reset_index(drop=True)
	

	tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
	model = AutoModelForSequenceClassification.from_pretrained("PoetschLab/GROVER")

	train_dataset = SupervisedDataset(train.sequence, train.label, tokenizer)
	test_dataset = SupervisedDataset(test.sequence, test.label, tokenizer)
	val_dataset = SupervisedDataset(validation.sequence, validation.label, tokenizer)

	train_args = TrainingArguments(seed = 42,
				output_dir="/run/tmp/",
				per_device_train_batch_size=16,
				eval_strategy="epoch",
				learning_rate=0.000001,
				num_train_epochs=4
				)
	trainer = transformers.Trainer(
				model=model,
				tokenizer=tokenizer,
				compute_metrics=compute_metrics,
				train_dataset=train_dataset,
				eval_dataset=val_dataset,
				args = train_args
				)
	trainer.train()

	results = trainer.evaluate(eval_dataset=test_dataset)
	print(results)

################################################################
#			MAIN
################################################################
def main():
	NTdataDir = "/run/datasets/NTdatasets/Instadeep"
	datasets = ["enhancers", "H3K27me3",  "H3K4me3", "promoter_all", "splice_sites_all", "enhancers_types", "H3K36me3", "H3K9ac", "promoter_no_tata", "splice_sites_donors", "H2AFZ", "H3K4me1", "H3K9me3", "promoter_tata", "H3K27ac", "H3K4me2", "H4K20me1", "splice_sites_acceptors"]

	for dataset in datasets:
		currentDataset = NTdataDir + "/" + dataset
		process_dataset(currentDataset)

main()
