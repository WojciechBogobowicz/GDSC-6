"""
This is the train script.

This script contains all steps required to train a Huggingface model.
"""
import argparse  # to parse arguments from passed in the hyperparameters
import datetime
import json  # to open the json file with labels
import logging  # module for displaying relevant information in the logs
import os  # to manage environmental variables
import sys  # to access to some variables used or maintained by the interpreter
from typing import Optional, Tuple  # for type hints

import pandas as pd  # home of the DataFrame construct, _the_ most important object for Data Science
import torch  # library to work with PyTorch tensors and to figure out if we have a GPU available
import transformers
from datasets import (  # required tools to create, load and process our audio dataset
    Audio, Dataset, load_dataset)
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

sys.path.append('..')
import gc

import numpy as np

from gdsc_eval import (  # functions to create predictions and evaluate them
    compute_metrics, make_predictions)
from preprocessing import calculate_stats, preprocess_audio_arrays


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
        feature_extractor = transformers.ASTFeatureExtractor.from_pretrained(
            model_name, mean=train_dataset_mean, std=train_dataset_std)
        logger.info(
            f" feature extractor loaded with dataset mean: {train_dataset_mean} and standard deviation: {train_dataset_std}")
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
    extract_file_name: bool = True) -> Dataset:
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
    dataset = dataset.map(lambda x: calculate_stats(
        x, audio_field='audio',
        array_field='array',
        feature_extractor=feature_extractor
    ), batched=True)

    dataset_mean = np.mean(dataset['mean'])
    dataset_std = np.mean(dataset['std'])

    feature_extractor = get_feature_extractor(
        args.model_name, dataset_mean, dataset_std)

    logger.info(f" {dataset_name} dataset sampling rate casted to: {sampling_rate}")

    dataset_encoded: Dataset = dataset.map(
        lambda x: preprocess_audio_arrays(x, 'audio', 'array', feature_extractor),
        remove_columns="audio",
        batched=True,
        batch_size=fe_batch_size
    )

    logger.info(f" done extracting features for {dataset_name} dataset")
    return Dataset.from_dict(dataset_encoded[:100])

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

    num_labels = len(label2id)  # define number of labels


    # If mean or std are not passed it will load Featue Extractor with the default settings.
    feature_extractor = get_feature_extractor(args.model_name, args.train_dataset_mean, args.train_dataset_std)

    # feature_extractor = transformers.ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-16-16-0.442", do_normalize=False)

    # creating train and validation datasets
    # train_dataset_encoded = preprocess_data_for_training(
    #     dataset_path=train_path,
    #     sampling_rate=args.sampling_rate,
    #     feature_extractor=feature_extractor,
    #     fe_batch_size=args.fe_batch_size,
    #     dataset_name="train",
    #     shuffle=True,
    #     extract_file_name=False
    # )

    # val_dataset_encoded = preprocess_data_for_training(
    #     dataset_path=val_path,
    #     sampling_rate=args.sampling_rate,
    #     feature_extractor=feature_extractor,
    #     fe_batch_size=args.fe_batch_size,
    #     dataset_name="validation"
    # )
    # if args.data_channel != "../../data":
    #     test_dataset_encoded = preprocess_data_for_training(
    #         dataset_path=test_path,
    #         sampling_rate=args.sampling_rate,
    #         feature_extractor=feature_extractor,
    #         fe_batch_size=args.fe_batch_size,
    #         dataset_name="test"
    #         )

    def objective(  # sampling rate
        sampling_rate_base: hp.randint('sampling_rate_base', 44 - 16),
        # sampling_rate_base: hp.choice('sampling_rate_base', (50,)),
        # hidden_act: hp.choice("hidden_act", ("gelu", "relu", "gelu_new")),
        # hidden_dropout_prob: hp.uniform("hidden_dropout_prob", 0, 0.5),
        # attention_probs_dropout_prob: hp.uniform("attention_probs_dropout_prob", 0, 0.5),
        # initializer_range: hp.uniform("initializer_range", 0.005, 0.1),
    ):
        # Download model from model hub
        model = transformers.ASTForAudioClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            # hidden_act=hidden_act,
            # hidden_dropout_prob=hidden_dropout_prob,
            # attention_probs_dropout_prob=attention_probs_dropout_prob,
            # initializer_range=initializer_range,
        )

        sampling_rate = (int(sampling_rate_base) + 16) * 1000

        train_dataset_encoded = preprocess_data_for_training(
            dataset_path=train_path,
            sampling_rate=sampling_rate,
            feature_extractor=feature_extractor,
            fe_batch_size=args.fe_batch_size,
            dataset_name="train",
            shuffle=True,
            extract_file_name=False
        )

        val_dataset_encoded = preprocess_data_for_training(
            dataset_path=val_path,
            sampling_rate=sampling_rate,
            feature_extractor=feature_extractor,
            fe_batch_size=args.fe_batch_size,
            dataset_name="validation"
        )

        if args.data_channel != "../../data":
            test_dataset_encoded = preprocess_data_for_training(
                dataset_path=test_path,
                sampling_rate=sampling_rate,
                feature_extractor=feature_extractor,
                fe_batch_size=args.fe_batch_size,
                dataset_name="test"
            )

        # Define training arguments for the purpose of training
        training_args = transformers.TrainingArguments(
            output_dir=args.output_dir,                          # directory for saving model checkpoints and logs
            num_train_epochs=1, #args.epochs,                        # number of epochs
            per_device_train_batch_size=args.train_batch_size,   # number of examples in batch for training
            per_device_eval_batch_size=args.eval_batch_size,     # number of examples in batch for evaluation
            evaluation_strategy="epoch",                         # makes evaluation at the end of each epoch
            learning_rate=args.learning_rate,                    # learning rate
            optim="adamw_torch",                                 # optimizer
            warmup_ratio=0.1,                                    # warm up to allow the optimizer to collect the statistics of gradients
            logging_steps=10,                                    # number of steps for logging the training process - one step is one batch; float denotes ratio of the global training steps
            # load_best_model_at_end = True,                       # whether to load or not the best model at the end of the training
            metric_for_best_model="eval_f1",                     # claiming that the best model is the one with the lowest loss on the val set
            save_strategy = 'no', #'epoch',                             # saving is done at the end of each epoch
            disable_tqdm=True                                    # disable printing progress bar to reduce amount of logs
        )

        early_stopping_callback = transformers.EarlyStoppingCallback(
            early_stopping_patience = args.patience)


        # Create Trainer instance
        trainer = transformers.Trainer(
            model=model,                                                                 # passing our model
            args=training_args,                                                          # passing the above created arguments
            compute_metrics=compute_metrics,                                             # passing the compute_metrics function that we imported from gdsc_eval module
            train_dataset=train_dataset_encoded,                                         # passing the encoded train set
            eval_dataset=val_dataset_encoded,                                            # passing the encoded val set
            tokenizer=feature_extractor,                                                 # passing the feature extractor
            # callbacks = [early_stopping_callback]                                        # adding early stopping to avoid overfitting
        )
        # trainer.add_callback(CustomCallback(trainer))

        # Train the model
        logger.info(f" starting training proccess for {args.epochs} epoch(s)")
        trainer.train()

        # Prepare predictions on the validation set for the purpose of error analysis
        logger.info("training job done. Preparing predictions for validation set.")
        return {"loss": -trainer.evaluate()['eval_accuracy'], 'status': STATUS_OK}
        # return 1 - trainer.state.best_metric

    # trials = Trials()

    best = fmin(
        fn=objective,
        space="annotated",
        algo=tpe.suggest,
        max_evals=6,
        trials=None, #trials,
        trials_save_file=""  #TODO: uzupełnić albo wywalić?
    )
    # best["hidden_act"] = activates[best["hidden_act"]]
    best['sampling_rate_base'] = int(best['sampling_rate_base'])
    logger.info("bayes optimization done :)")
    logger.info(f"Best found parameters are: {best}")


    best_hparams_path = os.path.join(args.output_dir, "best_hparams.json")
    json_object = json.dumps(dict(best), indent=4)
    with open(best_hparams_path, "w") as outfile:
        outfile.write(json_object)

    logger.info(f"Best hyper parameters saved to {best_hparams_path}")


    # trials_path = os.path.join(args.output_dir, "trials.pkl")
    # with open(trials_path, "wb") as f:
    #     pickle.dump(trials, f)




# from hyperopt import hp, Trials, fmin, tpe
# import matplotlib.pyplot as plt

# def model(x, y, z):
#     return 5*x**2-10*y**4+z**6+10

# def loss_fn(t, p):
#     return abs(t - p)


# def objective(
#         x: hp.uniform('x', -10, 10),
#         y: hp.uniform('y', -10, 10),
#         z: hp.uniform('z', -10, 10)
#     ):
#     """Objective function to minimize"""
#     output = model(x,y,z)
#     loss = loss_fn(output, -10)
#     return loss



# trials = Trials()

# best = fmin(
#     fn=objective,
#     space="annotated",
#     algo=tpe.suggest,
#     max_evals=2,
#     trials=trials
# )

# import pickle
# with open("test.pkl", "wb") as f:
#     pickle.dump(trials, f)

# print("Best hyperparameters:", best)
# print("Best objective value:", min(trials.losses()))
# plt.plot(list(trials.losses()))
# print(*[t for t in trials.trials], sep="\n#\n")
# plt.show()