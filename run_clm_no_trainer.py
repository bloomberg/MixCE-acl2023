#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from collections import OrderedDict
from itertools import chain

# from pathlib import Path

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset

# from huggingface_hub import Repository
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
)

# from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version

from gpt2 import GPT2MIXModel
from metrics import compute_diversity_repetition

logger = get_logger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the testing data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        help="How to reduce MLE loss.",
    )
    parser.add_argument(
        "--mixing_ratio",
        type=float,
        default=0.5,
        help="The mixing ratio of mixces.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="The max length of input.",
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=50,
        help="The length of prompt to keep for sampling.",
    )
    parser.add_argument(
        "--eval_prompt_length",
        type=int,
        default=50,
        help="The length of prompt to keep for evaluation.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether do train",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether do evaluation",
    )
    args = parser.parse_args()

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert (
            args.output_dir is not None
        ), "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir)
        if args.with_tracking
        else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # if args.hub_model_id is None:
            #     repo_name = get_full_repo_name(
            #         Path(args.output_dir).name, token=args.hub_token
            #     )
            # else:
            #     repo_name = args.hub_model_id
            # repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    #     if "validation" not in raw_datasets.keys():
    #         raw_datasets["validation"] = load_dataset(
    #             args.dataset_name,
    #             args.dataset_config_name,
    #             split=f"train[:{args.validation_split_percentage}%]",
    #         )
    #         raw_datasets["train"] = load_dataset(
    #             args.dataset_name,
    #             args.dataset_config_name,
    #             split=f"train[{args.validation_split_percentage}%:]",
    #         )
    # else:
    data_files = {}
    dataset_args = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
    raw_datasets = load_dataset(
        extension,
        name=args.dataset_name,
        data_files=data_files,
        cache_dir=args.cache_dir,
        **dataset_args,
    )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{args.validation_split_percentage}%]",
            **dataset_args,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{args.validation_split_percentage}%:]",
            **dataset_args,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently

    if not os.path.isdir(args.model_name_or_path):
        print(
            f"ERROR: please download {args.model_name_or_path} as instructed in README"
        )
        exit()

    # if evaluation only, find pretrained model
    if not args.do_train and args.do_eval:
        if not os.path.isdir(args.output_dir):
            print(f"ERROR: coundn't find {args.output_dir}")
            exit()
        elif not os.path.isdir(f"{args.output_dir}/best"):
            print(f"ERROR: coundn't find {args.output_dir}/best")
            exit()

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    config.prompt_length = args.prompt_length
    config.reduction = args.reduction
    config.mixing_ratio = args.mixing_ratio

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # if tokenizer has no pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding = True
        tokenizer.truncation = True

    if args.model_name_or_path:
        model_params = torch.load(f"{args.model_name_or_path}/pytorch_model.bin")
        if "lm.transformer" not in list(model_params.keys())[0]:
            # rename the parameters from gpt-2 pretrained checkpoints so that they can be loaded to GPT2MIXModel
            new_model_params = OrderedDict()
            for key in model_params:
                new_model_params[f"lm.transformer.{key}"] = model_params[key]
            torch.save(new_model_params, f"{args.model_name_or_path}/pytorch_model.bin")
        model = GPT2MIXModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(
            dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
        )
        logger.info(
            f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params"
        )

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        # add EOS after each text
        output["input_ids"] = [
            item[: args.max_length - 1] + [tokenizer.eos_token_id]
            for item in output["input_ids"]
        ]
        output["attention_mask"] = [
            item[: args.max_length - 1] + [1] for item in output["attention_mask"]
        ]
        return output

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Packing: concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        remainder_length = total_length % block_size
        padding_length = block_size - remainder_length
        concatenated_examples["input_ids"].extend(
            [tokenizer.pad_token_id] * padding_length
        )
        concatenated_examples["attention_mask"].extend([0] * padding_length)
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        assert total_length % block_size == 0
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    if block_size > 0:  # we always set block_size=0, i.e., do not use packing
        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
    else:
        lm_datasets = tokenized_datasets

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    config.median_length = int(
        np.median([len(item) for item in train_dataset["input_ids"]])
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=DataCollatorWithPadding(tokenizer),
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=DataCollatorWithPadding(tokenizer),
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ].value
            accelerator.init_trackers("clm_no_trainer", experiment_config)

    if args.do_train:
        # Train!
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(args.max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if (
                args.resume_from_checkpoint is not None
                or args.resume_from_checkpoint != ""
            ):
                accelerator.print(
                    f"Resumed from checkpoint: {args.resume_from_checkpoint}"
                )
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[
                    -1
                ]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        best_loss = 100000
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                losses.append(
                    accelerator.gather(loss.repeat(args.per_device_eval_batch_size))
                )

            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]
            eval_loss = torch.mean(losses).item()
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(
                f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}"
            )

            res = {
                "perplexity": perplexity,
                "eval_loss": eval_loss,
                "train_loss": total_loss.item() / len(train_dataloader),
                "epoch": epoch,
                "step": completed_steps,
            }
            if args.with_tracking:
                accelerator.log(
                    res,
                    step=completed_steps,
                )

            if eval_loss < best_loss:
                best_loss = eval_loss
                output_dir = "best"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                os.system(f"rm {output_dir}/*")
                accelerator.save_state(output_dir)  # save the best model

        # output_dir = f"epoch_{epoch}"
        # if args.output_dir is not None:
        #     output_dir = os.path.join(args.output_dir, output_dir)
        # accelerator.save_state(output_dir)  # save the last model

    if args.do_eval:
        output_dir = os.path.join(args.output_dir, "best")
        accelerator.load_state(output_dir)
        model.eval()
        model.config.baseline = True  # evaluate on baseline mode
        model.config.reduction = "mean"  # get ppl per token

        for dsplit, eval_dataset, eval_dataloader in [
            ("dev", eval_dataset, eval_dataloader),
            ("test", test_dataset, test_dataloader),
        ]:
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                losses.append(
                    accelerator.gather(loss.repeat(args.per_device_eval_batch_size))
                )
            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]
            eval_loss = torch.mean(losses).item()
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = "inf"
            metrics = {"eval_loss": eval_loss, "perplexity": perplexity}
            print(metrics)
            scores, data = evaluate(
                model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                eval_dataloader=eval_dataloader,
                eval_batch_size=args.per_device_eval_batch_size,
                device=model.device,
                model_name="gpt2-large",
                prompt_length=args.eval_prompt_length,
            )
            metrics.update(scores)
            print(dsplit, metrics)
            # write output metric scores
            path = os.path.join(args.output_dir, f"{dsplit}_results.json")
            with open(path, "w") as f:
                json.dump(metrics, f, indent=4, sort_keys=True)

            # write out generations
            for key in data:
                with open(f"{args.output_dir}/{dsplit}.{key}", "w") as f:
                    f.write(
                        "\n".join(
                            [
                                gen.replace("\n", " ").replace("\r", " ")
                                for gen in data[key]
                            ]
                        )
                    )


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    eval_dataset,
    eval_dataloader,
    eval_batch_size,
    device,
    prompt_length=50,
):
    logger.info("***** Running Generation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    human_gens, sample_gens, sample_gens1, sample_gens2 = [], [], [], []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"]
            max_length = max(input_ids.size()[1], prompt_length + 1)
            human_gens.extend(
                tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            )
            if prompt_length == 0:
                input_ids = (
                    torch.zeros((input_ids.size()[0], 1), dtype=torch.long)
                    + tokenizer.eos_token_id
                )
            else:
                input_ids = input_ids[:, :prompt_length]
            # unbiased sampling
            out_ids = model.lm.generate(
                inputs=input_ids.to(device),
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                num_beams=1,
                repetition_penalty=1.0,
                min_length=0,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
            )
            sample_gens.extend(
                tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            )
            # unbiased sampling 1
            out_ids = model.lm.generate(
                inputs=input_ids.to(device),
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                num_beams=1,
                repetition_penalty=1.0,
                min_length=0,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
            )
            sample_gens1.extend(
                tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            )
            # unbiased sampling 2
            out_ids = model.lm.generate(
                inputs=input_ids.to(device),
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                num_beams=1,
                repetition_penalty=1.0,
                min_length=0,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
            )
            sample_gens2.extend(
                tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            )

    assert (
        len(human_gens)
        == len(sample_gens)
        == len(eval_dataset)
        == len(sample_gens1)
        == len(sample_gens2)
    )
    # compute diversity and repetition (though only diversity is reported in our paper)
    logger.info("***** Compute metrics *****")
    data = {
        "human": human_gens,
        "sample": sample_gens,
        "sample1": sample_gens1,
        "sample2": sample_gens2,
    }
    scores = compute_diversity_repetition(data, tokenizer)
    return scores, data


if __name__ == "__main__":
    main()
