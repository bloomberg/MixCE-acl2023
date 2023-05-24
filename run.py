# Copyright 2023 Bloomberg Finance L.P.
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

import datetime
import os

current_time = datetime.datetime.now().strftime("%Y%m%d%H")

data_sets = {
    "wikitext": {
        "train100K": "data/wikitext/wikitext-103-raw-v1.train100K.txt",
        "train50K": "data/wikitext/wikitext-103-raw-v1.train50K.txt",
        "train25K": "data/wikitext/wikitext-103-raw-v1.train25K.txt",
        "train10K": "data/wikitext/wikitext-103-raw-v1.train10K.txt",
        "valid": "data/wikitext/wikitext-103-raw-v1.validation.txt",
        "test": "data/wikitext/wikitext-103-raw-v1.test.txt",
    },
    "webtext": {
        "train100K": "data/webtext/webtext.train100K.txt",
        "train50K": "data/webtext/webtext.train50K.txt",
        "train10K": "data/webtext/webtext.train10K.txt",
        "train25K": "data/webtext/webtext.train25K.txt",
        "valid": "data/webtext/webtext.valid.txt",
        "test": "data/webtext/webtext.test.txt",
    },
    "writingPrompts": {
        "train100K": "data/writingPrompts/writingPrompts.train100K.txt",
        "train50K": "data/writingPrompts/writingPrompts.train50K.txt",
        "train10K": "data/writingPrompts/writingPrompts.train10K.txt",
        "train25K": "data/writingPrompts/writingPrompts.train25K.txt",
        "valid": "data/writingPrompts/writingPrompts.valid.txt",
        "test": "data/writingPrompts/writingPrompts.test.txt",
    },
}


def run_no_trainer(
    model="gpt2",
    data_set="wikitext",
    prompt_length=50,
    block_size=0,
    train_batch_size=4,
    accumulation=8,
    eval_batch_size=8,
    seed=42,
    learning_rate=5e-5,
    epoch=5,
    reduction="mean",
    mixing_ratio=0.5,
    training_size="25K",
):
    train_file, valid_file, test_file = (
        data_sets[data_set][f"train{training_size}"],
        data_sets[data_set]["valid"],
        data_sets[data_set]["test"],
    )
    for file in [train_file, valid_file, test_file]:
        if not os.path.exists(file):
            print(f"ERROR: {file} does not exist!")
            exit()
    output_dir = (
        f"train/{current_time}_{data_set}_train{training_size}_{model}_mix{mixing_ratio}"
        f"_pl{prompt_length}_bl{block_size}"
        f"_batch{train_batch_size * accumulation}_lr{learning_rate}"
        f"_epoch{epoch}_reduce{reduction}_seed{seed} "
    )
    os.system(
        f"python run_clm_no_trainer.py --model_name_or_path {model} "
        f"--dataset_name {data_set} --cache_dir train --train_file {train_file} "
        f"--validation_file {valid_file} --test_file {test_file} "
        f"--per_device_train_batch_size {train_batch_size} --with_tracking "
        f"--gradient_accumulation_steps {accumulation} --seed {seed} --reduction {reduction} "
        f"--per_device_eval_batch_size {eval_batch_size} --do_train --do_eval "
        f"--prompt_length {prompt_length} --mixing_ratio {mixing_ratio} "
        f"--block_size {block_size} "
        f"--output_dir {output_dir} "
        f"--num_train_epochs {epoch} --learning_rate {learning_rate} "
    )


def run_no_trainer_eval(
    timestamp,
    model="gpt2",
    data_set="wikitext",
    prompt_length=50,
    block_size=0,
    train_batch_size=4,
    accumulation=8,
    eval_batch_size=32,
    seed=42,
    learning_rate=5e-5,
    epoch=5,
    reduction="mean",
    mixing_ratio=0.5,
    training_size="25K",
):
    train_file, valid_file, test_file = (
        data_sets[data_set][f"train{training_size}"],
        data_sets[data_set]["valid"],
        data_sets[data_set]["test"],
    )
    for file in [train_file, valid_file, test_file]:
        if not os.path.exists(file):
            print(f"ERROR: {file} does not exist!")
            exit()
    output_dir = (
        f"train/{timestamp}_{data_set}_train{training_size}_{model}_mix{mixing_ratio}"
        f"_pl{prompt_length}_bl{block_size}"
        f"_batch{train_batch_size * accumulation}_lr{learning_rate}"
        f"_epoch{epoch}_reduce{reduction}_seed{seed} "
    )
    os.system(
        f"python run_clm_no_trainer.py --model_name_or_path {model} "
        f"--dataset_name {data_set} --cache_dir train --train_file {train_file} "
        f"--validation_file {valid_file} --test_file {test_file} "
        f"--per_device_train_batch_size {train_batch_size} --with_tracking "
        f"--gradient_accumulation_steps {accumulation} --seed {seed} --reduction {reduction} "
        f"--per_device_eval_batch_size {eval_batch_size} --do_eval "
        f"--prompt_length {prompt_length} --mixing_ratio {mixing_ratio} "
        f"--block_size {block_size} "
        f"--output_dir {output_dir} "
        f"--num_train_epochs {epoch} --learning_rate {learning_rate} "
    )


def run_no_trainer_turn_topp(
    timestamp,
    model="gpt2",
    data_set="wikitext",
    prompt_length=50,
    block_size=0,
    train_batch_size=4,
    accumulation=8,
    eval_batch_size=32,
    seed=42,
    learning_rate=5e-5,
    epoch=5,
    reduction="mean",
    mixing_ratio=0.5,
    training_size="25K",
):
    train_file, valid_file, test_file = (
        data_sets[data_set][f"train{training_size}"],
        data_sets[data_set]["valid"],
        data_sets[data_set]["test"],
    )
    for file in [train_file, valid_file, test_file]:
        if not os.path.exists(file):
            print(f"ERROR: {file} does not exist!")
            exit()
    output_dir = (
        f"train/{timestamp}_{data_set}_train{training_size}_{model}_mix{mixing_ratio}"
        f"_pl{prompt_length}_bl{block_size}"
        f"_batch{train_batch_size * accumulation}_lr{learning_rate}"
        f"_epoch{epoch}_reduce{reduction}_seed{seed} "
    )
    os.system(
        f"python run_clm_no_trainer_tune_topp.py --model_name_or_path {model} "
        f"--dataset_name {data_set} --cache_dir train --train_file {train_file} "
        f"--validation_file {valid_file} --test_file {test_file} "
        f"--per_device_train_batch_size {train_batch_size} --with_tracking "
        f"--gradient_accumulation_steps {accumulation} --seed {seed} --reduction {reduction} "
        f"--per_device_eval_batch_size {eval_batch_size} --do_eval "
        f"--prompt_length {prompt_length}  --mixing_ratio {mixing_ratio} "
        f"--block_size {block_size} "
        f"--output_dir {output_dir}  "
        f"--num_train_epochs {epoch} --learning_rate {learning_rate} "
    )


if __name__ == "__main__":
    training_size = "50K"
    model = "gpt2-large"
    train_batch_size, accumulation, eval_batch_size = (
        2,
        32,
        32,
    )  # 4, 16, 32 for gpt2-medium, 8, 8, 32 for gpt2-small
    for dataset in ["wikitext", "webtext", "writingPrompts"]:
        for mixing_ratio in [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]:
            run_no_trainer(
                data_set=dataset,
                model=model,
                epoch=5,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                mixing_ratio=mixing_ratio,
                accumulation=accumulation,
                reduction="mean",
                training_size=training_size,
            )
            # run_no_trainer_eval(timestamp="2022110901", data_set=dataset, model=model,
            #                     epoch=5, train_batch_size=train_batch_size,
            #                     mixing_ratio=mixing_ratio, eval_batch_size=eval_batch_size,
            #                     accumulation=accumulation, reduction="mean", training_size=training_size)
            # run_no_trainer_turn_topp(timestamp="2022111101", data_set=dataset, model=model,
            #                          epoch=5, train_batch_size=train_batch_size,
            #                          mixing_ratio=mixing_ratio,
            #                          eval_batch_size=eval_batch_size, accumulation=accumulation,
            #                          reduction="mean", training_size=training_size)
