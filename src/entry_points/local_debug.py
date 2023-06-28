"""
This is the train script.

This script contains all steps required to train a Huggingface model.
"""
import argparse  # to parse arguments from passed in the hyperparameters
import datetime
import json  # to open the json file with labels
import logging  # module for displaying relevant information in the logs
import os  # to manage environmental variables
import pathlib
import random
import sys  # to access to some variables used or maintained by the interpreter
from copy import deepcopy
from typing import Optional  # for type hints

import numpy as np
import pandas as pd  # home of the DataFrame construct, _the_ most important object for Data Science
import torch  # library to work with PyTorch tensors and to figure out if we have a GPU available
import transformers
from datasets import (  # required tools to create, load and process our audio dataset
    Audio, Dataset, load_dataset)

sys.path.append('..')
from gdsc_eval import (  # functions to create predictions and evaluate them
    compute_metrics, make_predictions)
from preprocessing import preprocess_audio_arrays

BACKBONE_DIM = 128
DROPOUT_PROB = 0.5


class AstDropout(torch.nn.Module):
    def __init__(
            self,
            backbone: torch.nn.Module,
            ast_output_dim,
            output_dim,
            label2id: dict,
            id2label: dict,
            dropout_prob=0.5,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.dense = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(in_features=ast_output_dim, out_features=output_dim),
        )
        self.num_labels = output_dim
        self.label2id=label2id,
        self.id2label=id2label,

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):
        ret = self.backbone.forward(
            input_values,
            head_mask,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict
        )
        ret["logits"] = self.dense(ret["logits"])
        print("#"*100)
        print("beggining")
        print(list(self.parameters())[0].view(-1)[0:5])
        print("end")
        print(list(self.parameters())[-1].view(-1)[0:5])
        return ret


class FocalLossTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = FocalLoss(alpha=1, gamma=.25)
        self.num_classes = 66

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        one_hoted_labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        outputs = model(**inputs) ## currently translate to input_values=inputs["input_values"]

        logits = outputs.logits
        loss = self.loss_fn(logits, one_hoted_labels)
        return (loss, outputs) if return_outputs else loss



class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute the binary cross-entropy loss
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Compute the modulating factor
        modulating_factor = torch.exp(-self.gamma * targets * inputs.sigmoid() - self.gamma * (1 - targets) * (1 - inputs.sigmoid()))
        # Compute the focal loss
        focal_loss = self.alpha * (1 - inputs.sigmoid()).pow(self.gamma) * bce_loss * modulating_factor
        return focal_loss.mean()


class AugmentedDataset(Dataset):
    def __init__(self, *args, augs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.augs = [] if augs is None else augs

    @classmethod
    def from_dataset(cls, other, augs=None):
        aug = cls(other.data, augs=augs, info=other.info)
        return aug

    def __getitem__(self, index):
        item = super().__getitem__(index)
        aug = item
        for augmentation in self.augs:
            aug = augmentation(aug)
        return aug

class NoiseAug:
    def __init__(self, p, noise_ratio=0.01):
        self.noise_ratio = noise_ratio
        self.proba = p

    def __call__(self, data):
        if random.random() > self.proba:
            return data
        array = data['audio']['array']
        noise = np.random.randn(len(array))
        augmented_array = array + noise * self.noise_ratio
        augmented_array = augmented_array.astype(type(array[0]))
        data['audio']['array'] = augmented_array
        return data


class ShiftAug:
    def __init__(self, p: float, len_percent: int, direction: str):
        self.proba = p
        self.percent = len_percent
        self.direction = direction
        assert 0.0 < self.percent < 1.0, 'Audio len percentage should be between 0.0 and 1.0.'
        assert 0.0 <= self.proba <= 1.0, 'Augmentation probability should be between 0.0 and 1.0.'
        assert self.direction in ['right', 'left', 'both'], 'Shift direction should have one of those values: \'right\', \'left\', \'both\'.'

    def __call__(self, data):
        if  random.random() > self.proba:
            return data
        array = data['audio']['array']
        max_shift = int(len(array) * self.percent)
        shift = random.randint(0, max_shift)

        if self.direction == 'left':
            shift = -shift
        elif self.direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(array, shift)
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        data['audio']['array'] = augmented_data
        return data

def get_best_checkpoint_path(local_path: str):
    save_dir = pathlib.Path(local_path)
    ckpt_dirs = list(map(str, save_dir.glob("checkpoint*")))
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[-1]))
    last_ckpt = ckpt_dirs[-1]
    last_ckpt
    state = transformers.TrainerState.load_from_json(last_ckpt + "/trainer_state.json")

    best_checkopot_num = state.best_model_checkpoint.split("-")[-1]
    best_checkopot_num
    best_ckpt = last_ckpt.rstrip("1234567890") + best_checkopot_num
    return best_ckpt


def get_feature_extractor(model_name: str,
                          train_dataset_mean: Optional[float] = None,
                          train_dataset_std: Optional[float] = None
    ) -> transformers.ASTFeatureExtractor:
    """
    Retrieves a feature extractor for audio signal processing.

    Args:
        model_name (str): The name of the pre-trained model to use.
        train_dataset_mean (float, optional): The mean value of the training dataset. Defaults to None.
        train_dataset_std (float, optional): The standard deviation of the training dataset. Defaults to None.

    Returns:
        ASTFeatureExtractor: An instance of the ASTFeatureExtractor class.

    """
    if all((train_dataset_mean, train_dataset_std)):
        feature_extractor = transformers.ASTFeatureExtractor.from_pretrained(model_name, mean=train_dataset_mean, std=train_dataset_std)
        logger.info(f" feature extractor loaded with dataset mean: {train_dataset_mean} and standard deviation: {train_dataset_std}")
    else:
        feature_extractor = transformers.ASTFeatureExtractor.from_pretrained(model_name)
        logger.info(" at least one of the optional arguments (mean, std) is missing")
        logger.info(f" feature extractor loaded with default dataset mean: {feature_extractor.mean} and standard deviation: {feature_extractor.std}")

    return feature_extractor

def preprocess_data_for_training(
    dataset_path: str,
    sampling_rate: int,
    feature_extractor: transformers.ASTFeatureExtractor,
    fe_batch_size: int,
    dataset_name: str,
    shuffle: bool = False,
    extract_file_name: bool = True,
    augmentations=None) -> Dataset:
    """
    Preprocesses audio data for training.

    Args:
        dataset_path (str): The path to the dataset.
        sampling_rate (int): The desired sampling rate for the audio.
        feature_extractor (ASTFeatureExtractor): The feature extractor to use for preprocessing.
        fe_batch_size (int): The batch size for feature extraction.
        dataset_name (str, optional): The name of the dataset. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        extract_file_name (bool, optional): Whether to extract paths from audio features. Defaults to True.

    Returns:
        dataset: The preprocessed dataset.

    """
    dataset = load_dataset("audiofolder", data_dir=dataset_path).get('train') # loading the dataset
    if augmentations:
        dataset = AugmentedDataset.from_dataset(dataset, augs=augmentations)

    # perform shuffle if specified
    if shuffle:
        dataset = dataset.shuffle(seed=42)

    logger.info(f" loaded {dataset_name} dataset length is: {len(dataset)}")

    if extract_file_name:
        remove_metadata = lambda x: x.endswith(".wav")
        extract_file_name = lambda x: x.split('/')[-1]

        dataset_paths = list(dataset.info.download_checksums.keys())
        dataset_paths = list(filter(remove_metadata, dataset_paths))
        dataset_paths = list(map(extract_file_name, dataset_paths))
        dataset = dataset.add_column("file_name", dataset_paths)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    logger.info(f" {dataset_name} dataset sampling rate casted to: {sampling_rate}")

    dataset_encoded = dataset.map(
        lambda x: preprocess_audio_arrays(x, 'audio', 'array', feature_extractor),
        remove_columns="audio",
        batched=True,
        batch_size=fe_batch_size
    )

    logger.info(f" done extracting features for {dataset_name} dataset")

    return dataset_encoded

class TrainMetricsTrackerCallback(transformers.TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent from our jupyter notebook are passed as command-line arguments to the script
    # preprocessing hyperparameters
    parser.add_argument("--sampling_rate", type=int, default=16000)                        # sampling rate to which we will cast audio files
    parser.add_argument("--fe_batch_size", type=int, default=32)                           # feature extractor batch size
    parser.add_argument("--train_dataset_mean", type=float, default=None)                  # mean value of spectrograms of our data
    parser.add_argument("--train_dataset_std", type=float, default=None)                   # standard deviation value of spectrograms of our resampled data

    # training hyperparameters
    parser.add_argument("--model_name", type=str)                                          # name of the pretrained model from HuggingFace
    parser.add_argument("--learning_rate", type=float, default=5e-5)                       # learning rate
    parser.add_argument("--epochs", type=int, default=4)                                   # number of training epochs
    parser.add_argument("--train_batch_size", type=int, default=32)                        # training batch size
    parser.add_argument("--eval_batch_size", type=int, default=64)                         # evaluation batch size
    parser.add_argument("--patience", type=int, default=2)                                 # early stopping - how many epoch without improvement will stop the training
    parser.add_argument("--train_dir", type=str, default="train")                          # folder name with training data
    parser.add_argument("--val_dir", type=str, default="val")                              # folder name with validation data
    parser.add_argument("--test_dir", type=str, default="test")                            # folder name with test data

    parser.add_argument("--data_channel", type=str, default=os.environ.get("SM_CHANNEL_DATA", "local")) # directory where input data from S3 is stored
    parser.add_argument("--output_dir", type=str, default=os.environ.get('SM_MODEL_DIR', "local"))      # output directory. This directory will be saved in the S3 bucket


    args, _ = parser.parse_known_args()                    # parsing arguments from the notebook
    if args.data_channel == "local":
        args.data_channel = "../../data"
        args.output_dir = f"../../models/DEBUG_{datetime.datetime.now().timestamp()}"

    train_path = f"{args.data_channel}/{args.train_dir}"   # directory of our training dataset on the instance
    val_path = f"{args.data_channel}/{args.val_dir}"       # directory of our validation dataset on the instance
    test_path = f"{args.data_channel}/{args.test_dir}"     # directory of our test dataset on the instance

    # Set up logging which allows to print information in logs
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Torch version")
    logger.info(torch.__version__)
    logger.info("Torch sees CUDA?")
    logger.info(torch.cuda.is_available())

    # Load json file with label2id mapping
    with open(f'{args.data_channel}/labels.json', 'r') as f:
        labels = json.load(f)

    # Create mapping from label to id and id to label
    label2id, id2label = dict(), dict()
    for k, v in labels.items():
        label2id[k] = str(v)
        id2label[str(v)] = k

    augmentations = [
        NoiseAug(0.3, noise_ratio=0.005),
        ShiftAug(0.3, 0.06, 'both'),
    ]


    num_labels = len(label2id)  # define number of labels


    # If mean or std are not passed it will load Featue Extractor with the default settings.
    feature_extractor = get_feature_extractor(
        args.model_name, args.train_dataset_mean, args.train_dataset_std)

    # creating train and validation datasets
    train_dataset_encoded = preprocess_data_for_training(
        dataset_path=train_path,
        sampling_rate=args.sampling_rate,
        feature_extractor=feature_extractor,
        fe_batch_size=args.fe_batch_size,
        dataset_name="train",
        shuffle=True,
        extract_file_name=False,
        augmentations=augmentations,
    )

    val_dataset_encoded = preprocess_data_for_training(
        dataset_path=val_path,
        sampling_rate=args.sampling_rate,
        feature_extractor=feature_extractor,
        fe_batch_size=args.fe_batch_size,
        dataset_name="validation"
    )
    if args.data_channel != "../../data":
        test_dataset_encoded = preprocess_data_for_training(
            dataset_path=test_path,
            sampling_rate=args.sampling_rate,
            feature_extractor=feature_extractor,
            fe_batch_size=args.fe_batch_size,
            dataset_name="test"
            )

    # Download model from model hub
    backbone = transformers.ASTForAudioClassification.from_pretrained(
        args.model_name,
        # num_labels=num_labels,
        num_labels=BACKBONE_DIM,
        # label2id=label2id,
        # id2label=id2label,
        ignore_mismatched_sizes=True
    )

    model = AstDropout(
        backbone,
        BACKBONE_DIM,
        num_labels,
        label2id=label2id,
        id2label=id2label,
        dropout_prob=DROPOUT_PROB,
    )

    # Define training arguments for the purpose of training
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,                          # directory for saving model checkpoints and logs
        num_train_epochs=args.epochs,                        # number of epochs
        per_device_train_batch_size=args.train_batch_size,   # number of examples in batch for training
        per_device_eval_batch_size=args.eval_batch_size,     # number of examples in batch for evaluation
        evaluation_strategy="epoch",                         # makes evaluation at the end of each epoch
        learning_rate=args.learning_rate,                    # learning rate
        optim="adamw_torch",                                 # optimizer
        # warmup_ratio=0.1,                                  # warm up to allow the optimizer to collect the statistics of gradients
        logging_steps=10,                                    # number of steps for logging the training process - one step is one batch; float denotes ratio of the global training steps
        load_best_model_at_end = True,                       # whether to load or not the best model at the end of the training
        metric_for_best_model="eval_loss",                   # claiming that the best model is the one with the lowest loss on the val set
        save_strategy = 'epoch',                             # saving is done at the end of each epoch
        disable_tqdm=True                                    # disable printing progress bar to reduce amount of logs
    )

    early_stopping_callback = transformers.EarlyStoppingCallback(
        early_stopping_patience = args.patience)

    # Create Trainer instance
    trainer = FocalLossTrainer(
    # trainer = transformers.Trainer(
        model=model,                                                                 # passing our model
        args=training_args,                                                          # passing the above created arguments
        compute_metrics=compute_metrics,                                             # passing the compute_metrics function that we imported from gdsc_eval module
        train_dataset=train_dataset_encoded,                                         # passing the encoded train set
        eval_dataset=val_dataset_encoded,                                            # passing the encoded val set
        tokenizer=feature_extractor,                                                 # passing the feature extractor
        callbacks = [early_stopping_callback]                                        # adding early stopping to avoid overfitting
    )
    trainer.add_callback(TrainMetricsTrackerCallback(trainer))

    # Train the model
    logger.info(f" starting training proccess for {args.epochs} epoch(s)")
    trainer.train()

    # Prepare predictions on the validation set for the purpose of error analysis
    logger.info("training job done. Preparing predictions for validation set.")


    # set evaluation mode (e.g. disable dropout)
    model.eval()
    # use gpu for inference if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    val_dataset_encoded.set_format(type='torch', columns=['input_values'])
    val_dataset_encoded = val_dataset_encoded.map(
        lambda x: make_predictions(
            x['input_values'], model, device, x['label']),
        batched = True,
        batch_size=args.eval_batch_size
    )

    # Keeping only the important columns for the csv file
    val_dataset_encoded = val_dataset_encoded.remove_columns(['input_values'])
    val_dataset_encoded = val_dataset_encoded.to_pandas()

    val_dataset_encoded['loss'] = (
        val_dataset_encoded['loss']
        .apply(lambda x: x[0])
    ) # extract floats

    pred_val_save_path = os.path.join(args.output_dir, "prediction_val.csv")
    val_dataset_encoded.to_csv(pred_val_save_path, index = False) # saving the file with validation predictions

    logger.info(" predictions for the validation set done and saved")
    logger.info(" preparing predictions for test set.")

    if args.data_channel != "../../data":
        # Preparing predictions for test set and saving them in the output directory
        test_dataset_encoded.set_format(type='torch', columns=['input_values'])
        test_dataset_encoded = test_dataset_encoded.map(
            lambda x: make_predictions(x['input_values'], model, device),
            batched = True,
            batch_size=args.eval_batch_size
        )

        # Keeping only the important columns for the csv file
        test_dataset_encoded = test_dataset_encoded.remove_columns(['input_values'])
        test_dataset_encoded_df = test_dataset_encoded.to_pandas()


        pred_test_save_path = os.path.join(args.output_dir, "prediction_test.csv")
        test_dataset_encoded_df.to_csv(pred_test_save_path, index = False)  # saving the file with test predictions

        logger.info(" prepared predictions for test set and saved it to the output directory. Training job completed")
