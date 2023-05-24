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

import json
import os

import numpy as np
import torch

from metrics import (
    compute_coherence,
    compute_controlled_coherence,
    compute_controlled_mauve,
    compute_mauve,
)
from utils import set_seed

set_seed(42)
device = torch.device("cuda:0")


def get_synthetic_results(setting, vocab, objective):
    # e.g., setting, vocab, object = 'webtext', '20', 'two_kls'
    with open("synthetic_logs/synthetic_models.json", "r") as f:
        synthetic_models = json.load(f)

    all_res = {}
    for model in synthetic_models[setting][vocab][objective]:
        print(model)
        with open(f"synthetic_logs/{model}/all_best_metrics.json", "r") as f:
            all_best_metrics = json.load(f)
        for eta in all_best_metrics["train_eta"]:
            if eta not in all_res:
                all_res[eta] = {"avg_0s": [], "avg_js": []}
            for metric in all_best_metrics["train_eta"][eta]:
                all_res[eta][metric].append(
                    float(all_best_metrics["train_eta"][eta][metric])
                )
    all_res_list = []
    for eta in all_res:
        all_res_list.append([eta])
        for metric in ["avg_js", "avg_0s"]:
            all_res_list[-1].append(np.mean(all_res[eta][metric]))
    print(sorted(all_res_list, key=lambda x: x[1]))  # sorted by avg_js


def compute_mauve_coherence(data_set, model_size, data_size, max_length, batch_size=16):
    # e.g., data_set, model_size, data_size, max_length = 'wikitext', 'large', '50K', 512
    with open("train/models.json", "r") as f:
        models = json.load(f)

    for ratio in models[data_set][model_size][data_size]:
        model = models[data_set][model_size][data_size][ratio]
        directory = f"train/{model}"
        for data_split in ["dev", "test"]:
            with open(f"{directory}/{data_split}.human", "r") as f:
                human_gens = [line.strip() for line in f.readlines()]
            with open(f"{directory}/{data_split}.sample", "r") as f:
                sample_gens = [line.strip() for line in f.readlines()]
            with open(f"{directory}/{data_split}.sample1", "r") as f:
                sample_gens1 = [line.strip() for line in f.readlines()]
            with open(f"{directory}/{data_split}.sample2", "r") as f:
                sample_gens2 = [line.strip() for line in f.readlines()]
            data = {
                "human": human_gens,
                "sample": sample_gens,
                "sample1": sample_gens1,
                "sample2": sample_gens2,
            }
            scores = compute_coherence(
                data, batch_size=batch_size, max_text_length=max_length, device=device
            )
            scores.update(
                compute_mauve(
                    data,
                    batch_size=batch_size,
                    max_text_length=max_length,
                    device=device,
                    featurize_model_name="gpt2-large-mauve",
                )
            )
            path = os.path.join(
                directory, f"{data_split}_mauve_coherence_{max_length}.json"
            )
            with open(path, "w") as f:
                json.dump(scores, f, indent=4, sort_keys=True)


def compute_controlled_mauve_coherence(
    data_set, model_size, data_size, max_length, batch_size=16
):
    # e.g., data_set, model_size, data_size, max_length = 'wikitext', 'large', '50K', 100
    with open("train/models.json", "r") as f:
        models = json.load(f)

    for ratio in models[data_set][model_size][data_size]:
        model = models[data_set][model_size][data_size][ratio]
        directory = f"train/{model}"
        for data_split in ["dev", "test"]:
            with open(f"{directory}/{data_split}.human", "r") as f:
                human_gens = [line.strip() for line in f.readlines()]
            with open(f"{directory}/{data_split}.sample", "r") as f:
                sample_gens = [line.strip() for line in f.readlines()]
            with open(f"{directory}/{data_split}.sample1", "r") as f:
                sample_gens1 = [line.strip() for line in f.readlines()]
            with open(f"{directory}/{data_split}.sample2", "r") as f:
                sample_gens2 = [line.strip() for line in f.readlines()]
            data = {
                "human": human_gens,
                "sample": sample_gens,
                "sample1": sample_gens1,
                "sample2": sample_gens2,
            }
            scores = compute_controlled_mauve(
                data,
                batch_size=batch_size,
                max_text_length=max_length,
                device=device,
                featurize_model_name="gpt2-large-mauve",
            )
            scores.update(
                compute_controlled_coherence(
                    data,
                    batch_size=batch_size,
                    max_text_length=max_length,
                    device=device,
                )
            )
            path = os.path.join(
                directory, f"{data_split}_controlled_mauve_coherence_{max_length}.json"
            )
            with open(path, "w") as f:
                json.dump(scores, f, indent=4, sort_keys=True)
