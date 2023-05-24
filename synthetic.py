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
import json
import pickle as pkl

import numpy as np
import torch
from scipy.spatial import distance
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

import tensorflow as tf

from utils import set_seed

MAX_LENGTH = 500  # max length of sampling from P and Q


def synthetic_data(
    vocab_size=20,
    data_size=55000,
    eval_data_size=5000,
    transition_matrix=None,
    first_token=None,
    zero_percent=0.5,
    real_dataset=None,
):
    if transition_matrix is None:
        if real_dataset:
            with open(f"data/{real_dataset}_transition_matrices.pkl", "rb") as f:
                real_transition_matrices = pkl.load(f)
            transition_matrix = real_transition_matrices[vocab_size]
        else:
            # bigram LM transition matrix
            # dividing 2 makes the distribution more skewed
            transition_matrix = np.random.dirichlet(
                np.ones(vocab_size) / 2, size=vocab_size
            )
            # randomly make zero_percent transition probs 0
            for i in range(vocab_size - 1):
                for j in range(vocab_size - 1):
                    if np.random.random() < zero_percent:
                        transition_matrix[i][j] = 0
            # renormalize transition matrix
            transition_matrix = transition_matrix / np.expand_dims(
                transition_matrix.sum(axis=1), axis=1
            )

        # the last row in transition matrix is not useful because after EOS, there won't be any tokens
        transition_matrix[-1] = np.zeros_like(transition_matrix[-1])
        first_token = [1 / vocab_size] * vocab_size

    # sample data
    data, length, data_log_p = [], [], []
    for _ in tqdm(range(data_size)):
        example, w, log_p = [], None, []
        while w != vocab_size:
            if len(example) == 0:
                w = np.argmax(np.random.multinomial(1, first_token)) + 1
                log_p.append(np.log(first_token[w - 1] + 1e-30))
            else:
                probs = transition_matrix[w - 1]
                if (
                    len(example) == MAX_LENGTH - 1
                ):  # force the sequence to end at the max length
                    w = vocab_size
                else:
                    w = np.argmax(np.random.multinomial(1, probs)) + 1
                log_p.append(np.log(probs[w - 1] + 1e-30))
            example.append(w)
        data.append(example)
        length.append(len(example))
        data_log_p.append(sum(log_p))
    print(len(data))
    print(max(length), min(length), np.mean(length))
    return (
        data[eval_data_size:],
        data[:eval_data_size],
        data_log_p[eval_data_size:],
        data_log_p[:eval_data_size],
        transition_matrix,
        first_token,
    )


class BigramLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        transition_matrix,
        first_token,
        device,
        hidden_size=50,
        dropout_p=0.1,
    ):
        super(BigramLM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device
        self.max_length = MAX_LENGTH

        self.transition_matrix = torch.log(
            torch.tensor(transition_matrix, requires_grad=False) + 1e-30
        ).to(device)
        self.first_token = torch.log(
            torch.tensor(first_token, requires_grad=False) + 1e-30
        ).to(device)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_size, self.vocab_size),
        )
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, seq_x):
        # seq_x starts with bos
        return [self.linear_relu_stack(self.embedding(x)) for x in seq_x]

    def get_learned_transition_matrix(self):
        logits = self.linear_relu_stack(self.embedding.weight)
        return torch.softmax(logits, dim=1)

    def greedy_search(self, prompt):
        finish = torch.zeros_like(prompt[0]).to(self.device)
        for label in list(prompt):
            finish += (1 - finish) * (label == self.vocab_size - 1).long()
        gen_labels, gen_one_hot_labels, gen_log_q = list(prompt), [], []
        while not torch.all(finish.bool()) and len(gen_labels) < self.max_length:
            logits = self.linear_relu_stack(self.embedding(gen_labels[-1]))
            label = torch.argmax(logits, dim=-1)
            gen_labels.append(label * (1 - finish))
            one_hot_label = nn.functional.one_hot(
                gen_labels[-1], self.vocab_size
            ).double()
            gen_one_hot_labels.append(one_hot_label)
            gen_log_q.append(
                (self.logsoftmax(logits) * one_hot_label).sum(dim=-1) * (1 - finish)
            )
            finish += (1 - finish) * (label == self.vocab_size - 1).long()
        gen_log_q = torch.cat(gen_log_q, dim=1).sum(dim=1)  # sum across seq length
        return gen_labels, None, gen_log_q

    def sample(self, prompt, gumbel=False):
        finish = torch.zeros_like(prompt[0]).to(self.device)
        for label in list(prompt):
            finish += (1 - finish) * (label == self.vocab_size - 1).long()
        gen_labels, gen_one_hot_labels, gen_log_q = list(prompt), [], []
        while not torch.all(finish.bool()) and len(gen_labels) < self.max_length:
            if gumbel and len(gen_one_hot_labels) != 0:
                logits = self.linear_relu_stack(
                    torch.matmul(gen_one_hot_labels[-1].float(), self.embedding.weight)
                )
            else:
                logits = self.linear_relu_stack(self.embedding(gen_labels[-1]))
            if gumbel:
                one_hot_label = torch.nn.functional.gumbel_softmax(
                    logits, dim=-1, tau=1, hard=True
                ).double()
                label = torch.argmax(one_hot_label, dim=-1)
                gen_labels.append(label * (1 - finish))
            else:
                probs = torch.softmax(logits.squeeze(), dim=-1)
                label = torch.multinomial(probs, 1)
                gen_labels.append(label * (1 - finish))
                one_hot_label = nn.functional.one_hot(
                    gen_labels[-1], self.vocab_size
                ).double()
            gen_one_hot_labels.append(one_hot_label)
            gen_log_q.append(
                (self.logsoftmax(logits) * one_hot_label).sum(dim=-1) * (1 - finish)
            )
            finish += (1 - finish) * (label == self.vocab_size - 1).long()
        gen_log_q = torch.cat(gen_log_q, dim=1).sum(dim=1)  # sum across seq length
        return gen_labels, gen_one_hot_labels if gumbel else None, gen_log_q

    def compute_log_q(
        self, seq_logits, seq_labels, seq_one_hot_labels=None, reduction="sum"
    ):
        # seq_labels do not start with bos
        seq_log_q = []
        finish = torch.zeros_like(seq_labels[0]).to(self.device)
        length = torch.zeros_like(seq_labels[0]).to(self.device)
        for i, (seq_logit, seq_label) in enumerate(zip(seq_logits, seq_labels)):
            if seq_one_hot_labels is not None:
                one_hot_label = seq_one_hot_labels[i]
            else:
                one_hot_label = nn.functional.one_hot(
                    seq_label, self.vocab_size
                ).double()
            seq_log_q.append(
                (self.logsoftmax(seq_logit) * one_hot_label).sum(dim=-1) * (1 - finish)
            )
            length += 1 - finish
            finish += (1 - finish) * (seq_label == self.vocab_size - 1).long()
        if reduction == "none":
            log_q = torch.cat(seq_log_q, dim=1)
        else:
            log_q = torch.cat(seq_log_q, dim=1).sum(dim=1)  # sum across seq length
        return log_q, length

    def compute_log_p(self, seq_labels, seq_one_hot_labels=None):
        # seq_labels do not start with bos
        finish = torch.zeros_like(seq_labels[0]).to(self.device)
        seq_log_p, prev_probs = [], self.first_token.unsqueeze(dim=0).unsqueeze(dim=0)
        for i, seq_label in enumerate(seq_labels):
            if seq_one_hot_labels is not None:
                one_hot_label = seq_one_hot_labels[i]
            else:
                one_hot_label = nn.functional.one_hot(
                    seq_label, self.vocab_size
                ).double()
            log_p = (one_hot_label * prev_probs).sum(dim=-1) * (1 - finish)
            seq_log_p.append(log_p)
            prev_probs = torch.matmul(one_hot_label.detach(), self.transition_matrix)
            finish += (1 - finish) * (seq_label == self.vocab_size - 1).long()
        log_p = torch.cat(seq_log_p, dim=1).sum(dim=1)  # sum across seq length
        return log_p

    def log_prob_truncation(self, log_prob, limit=30):
        # to avoid overflow when using exp
        truncated_log_prob = torch.minimum(log_prob, torch.zeros_like(log_prob) + limit)
        truncated_log_prob = torch.maximum(
            truncated_log_prob, torch.zeros_like(log_prob) - limit
        )
        return truncated_log_prob

    def loss_mle(self, seq_labels):
        seq_logits = self.forward(seq_labels)
        log_q, length = self.compute_log_q(seq_logits[:-1], seq_labels[1:])
        loss = -log_q.mean()
        return loss

    def forward_kl(self, seq_labels, log_p):
        # actually the same as loss_mle because log_p is a constant and will not affect optimization
        seq_logits = self.forward(seq_labels)
        log_q, _ = self.compute_log_q(seq_logits[:-1], seq_labels[1:])
        logp_logq = log_p - log_q
        return logp_logq.mean()

    def reverse_xen(self, seq_labels, gumbel=False):
        gen_labels, gen_one_hot_labels, log_q = self.sample(
            seq_labels[:1], gumbel=gumbel
        )
        log_p = self.compute_log_p(
            gen_labels[1:], seq_one_hot_labels=gen_one_hot_labels
        )
        loss = -log_p.mean()
        return loss

    def reverse_kl_sample_q(self, seq_labels, gumbel=False):
        # sample from q
        gen_labels, gen_one_hot_labels, log_q = self.sample(
            seq_labels[:1], gumbel=gumbel
        )
        log_p = self.compute_log_p(
            gen_labels[1:], seq_one_hot_labels=gen_one_hot_labels
        )
        loss = (log_q - log_p).mean()
        return loss

    def js_divergence_eta_sample_q(self, seq_labels, log_p, eta=0.5, gumbel=False):
        # sample from p
        seq_logits = self.forward(seq_labels)
        log_q, _ = self.compute_log_q(seq_logits[:-1], seq_labels[1:])
        logq_logp = log_q - log_p
        log_m = log_p + torch.log(
            eta + (1 - eta) * torch.exp(self.log_prob_truncation(logq_logp))
        )
        logp_logm = log_p - log_m

        # sample from q
        gen_labels, gen_one_hot_labels, log_q = self.sample(
            seq_labels[:1], gumbel=gumbel
        )
        log_p = self.compute_log_p(
            gen_labels[1:], seq_one_hot_labels=gen_one_hot_labels
        )
        logq_logp = log_q - log_p
        log_m = log_p + torch.log(
            eta + (1 - eta) * torch.exp(self.log_prob_truncation(logq_logp))
        )
        logq_logm = log_q - log_m

        loss = eta * logp_logm.mean() + (1.0 - eta) * logq_logm.mean()
        return loss

    def qlogq(self, seq_labels):
        seq_logits = self.forward(seq_labels)
        log_q, _ = self.compute_log_q(seq_logits[:-1], seq_labels[1:], reduction="none")
        with torch.no_grad():
            q = torch.exp(log_q.detach())
        loss = -(q * log_q).sum(dim=1)
        return loss.mean()

    def qlogq_mix(self, seq_labels, eta=0.5):
        seq_logits = self.forward(seq_labels)
        log_q, _ = self.compute_log_q(seq_logits[:-1], seq_labels[1:], reduction="none")
        with torch.no_grad():
            q = torch.exp(log_q.detach())
        qlogq_loss = -(q * log_q).sum(dim=1).mean()
        mle_loss = -log_q.sum(dim=1).mean()
        loss = eta * mle_loss + (1.0 - eta) * qlogq_loss
        return loss.mean()


def compare_parameters(learned_transition_matrix, transition_matrix, first_token):
    # 1: -> remove the first column, which is for the first token
    # :-1 -> remove the last row because after EOS there won't be any tokens
    learned_M = learned_transition_matrix[:-1][:, 1:]
    gold_M = np.concatenate(([first_token], transition_matrix[:-1]), axis=0)
    # average of values at positions where gold_M = zero
    zero_pos = (gold_M == 0.0).astype(int)
    avg_0s = np.sum(learned_M * zero_pos) / np.sum(zero_pos)
    # js
    avg_js = np.square(distance.jensenshannon(learned_M, gold_M, axis=1)).mean()
    return avg_0s, avg_js


def main(
    data=None,
    loss_func="mle",
    train_eta=0.5,
    seed=42,
    vocab_size=21,
    batch_size=1000,
    eval_batch_size=None,
    epochs=50,
    train_data_size=50000,
    eval_data_size=5000,
    summary_writer=None,
    output_dir=None,
    zero_percent=0.5,
    real_dataset=None,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed=seed)

    if eval_batch_size is None:
        eval_batch_size = batch_size
    data_size = train_data_size + eval_data_size
    # vocab size -1: not counting BOS
    fix_train, dev, fix_train_log_p, dev_log_p, transition_matrix, first_token = (
        synthetic_data(
            vocab_size=vocab_size - 1,
            data_size=data_size,
            eval_data_size=eval_data_size,
            zero_percent=zero_percent,
            real_dataset=real_dataset,
        )
        if data is None
        else data
    )
    # add PAD token to transition matrix and first token
    new_transition_matrix = np.zeros((vocab_size, vocab_size))
    new_transition_matrix[1:, 1:] = transition_matrix
    new_first_token = np.zeros(vocab_size)
    new_first_token[1:] = first_token

    set_seed(seed=seed)
    model = BigramLM(
        vocab_size=vocab_size,
        transition_matrix=new_transition_matrix,
        first_token=new_first_token,
        device=device,
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    print("====Transition Matrix Evaluation====")
    learned_transition_matrix = (
        model.get_learned_transition_matrix().detach().cpu().numpy()
    )
    avg_0s, avg_js = compare_parameters(
        learned_transition_matrix, transition_matrix, first_token
    )

    print("[%d] avg_0s: %.3f, avg_js: %.3f" % (0, avg_0s, avg_js))

    if summary_writer:
        with summary_writer.as_default():
            tf.summary.scalar("avg_0s", avg_0s, step=0)
            tf.summary.scalar("avg_js", avg_js, step=0)

    train_indexs = [i for i in range(len(fix_train))]
    patience, best_dev_loss, best_metrics = 5, 1e30, None
    for epoch in tqdm(range(epochs)):
        np.random.shuffle(train_indexs)
        train = [fix_train[i] for i in train_indexs]
        train_log_p = [fix_train_log_p[i] for i in train_indexs]

        running_loss, steps = 0, 0
        for bi in tqdm(range(len(train) // batch_size)):
            batch = train[bi * batch_size : (bi + 1) * batch_size]
            batch_log_p = train_log_p[bi * batch_size : (bi + 1) * batch_size]
            max_length = max(map(len, batch)) + 1
            # padding
            input_ids = torch.tensor(
                [[0] + item + [0] * (max_length - 1 - len(item)) for item in batch]
            ).to(device)
            batch_log_p = torch.tensor(batch_log_p).to(device)

            model.train()
            if "mle" in loss_func:
                loss = model.loss_mle(input_ids.split(split_size=1, dim=1))
            elif loss_func == "two_xens":
                forward_xen = model.loss_mle(input_ids.split(split_size=1, dim=1))
                reverse_xen = model.reverse_xen(
                    input_ids.split(split_size=1, dim=1), gumbel=True
                )
                loss = train_eta * forward_xen + (1.0 - train_eta) * reverse_xen
            elif loss_func == "qlogq_mix":
                loss = model.qlogq_mix(
                    input_ids.split(split_size=1, dim=1), eta=train_eta
                )
            elif loss_func == "forward_kl":
                loss = model.forward_kl(
                    input_ids.split(split_size=1, dim=1), batch_log_p
                )
            elif loss_func == "reverse_kl":
                loss = model.reverse_kl_sample_q(
                    input_ids.split(split_size=1, dim=1), gumbel=True
                )
            elif "js" in loss_func:
                loss = model.js_divergence_eta_sample_q(
                    input_ids.split(split_size=1, dim=1),
                    batch_log_p,
                    eta=train_eta,
                    gumbel=True,
                )
            elif "two_kls" in loss_func:
                forward_kl = (
                    model.forward_kl(input_ids.split(split_size=1, dim=1), batch_log_p)
                    if train_eta > 0.0
                    else 0.0
                )
                reverse_kl = (
                    model.reverse_kl_sample_q(
                        input_ids.split(split_size=1, dim=1), gumbel=True
                    )
                    if train_eta < 1.0
                    else 0.0
                )
                loss = train_eta * forward_kl + (1.0 - train_eta) * reverse_kl
            else:
                print("Not defined loss function!")
                exit()

            running_loss += loss  # extract the loss value

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            steps += 1

        running_loss /= steps

        print("[%d] training loss: %.3f" % (epoch + 1, running_loss.item()))
        with summary_writer.as_default():
            tf.summary.scalar("training_loss", running_loss.item(), step=epoch + 1)

        model.eval()
        with torch.no_grad():
            print("====Dev Evaluation====")
            dev_loss, steps = 0, 0
            for bi in tqdm(range(len(dev) // eval_batch_size + 1)):
                batch = dev[bi * eval_batch_size : (bi + 1) * eval_batch_size]
                batch_log_p = dev_log_p[
                    bi * eval_batch_size : (bi + 1) * eval_batch_size
                ]
                if len(batch) == 0:
                    continue
                max_length = max(map(len, batch)) + 1
                # padding
                input_ids = torch.tensor(
                    [[0] + item + [0] * (max_length - 1 - len(item)) for item in batch]
                ).to(device)
                batch_log_p = torch.tensor(batch_log_p).to(device)

                # loss on dev
                if "mle" in loss_func:
                    loss = model.loss_mle(input_ids.split(split_size=1, dim=1))
                elif loss_func == "two_xens":
                    forward_xen = model.loss_mle(input_ids.split(split_size=1, dim=1))
                    reverse_xen = model.reverse_xen(
                        input_ids.split(split_size=1, dim=1)
                    )
                    loss = train_eta * forward_xen + (1.0 - train_eta) * reverse_xen
                elif loss_func == "qlogq_mix":
                    loss = model.qlogq_mix(
                        input_ids.split(split_size=1, dim=1), eta=train_eta
                    )
                elif "js" in loss_func:
                    loss = model.js_divergence_eta_sample_q(
                        input_ids.split(split_size=1, dim=1), batch_log_p, eta=train_eta
                    )
                elif "two_kls" in loss_func:
                    forward_kl = (
                        model.forward_kl(
                            input_ids.split(split_size=1, dim=1), batch_log_p
                        )
                        if train_eta > 0.0
                        else 0.0
                    )
                    reverse_kl = (
                        model.reverse_kl_sample_q(input_ids.split(split_size=1, dim=1))
                        if train_eta < 1.0
                        else 0.0
                    )
                    loss = train_eta * forward_kl + (1.0 - train_eta) * reverse_kl

                dev_loss += loss.item()
                steps += 1

            dev_loss = dev_loss / steps
            print("[%d] dev loss: %.3f" % (epoch + 1, dev_loss))

            print("====Transition Matrix Evaluation====")
            learned_transition_matrix = (
                model.get_learned_transition_matrix().detach().cpu().numpy()
            )
            avg_0s, avg_js = compare_parameters(
                learned_transition_matrix, transition_matrix, first_token
            )
            print("[%d] avg_0s: %.3f, avg_js: %.3f" % (epoch + 1, avg_0s, avg_js))

            if dev_loss < best_dev_loss:  # model selection is based on loss on dev
                best_dev_loss = dev_loss
                best_metrics = {"avg_0s": avg_0s, "avg_js": avg_js, "epoch": epoch + 1}
                if output_dir:
                    model_state = model.state_dict()
                    model_state["transition_matrix"] = torch.tensor(
                        new_transition_matrix
                    )
                    model_state["first_token"] = torch.tensor(new_first_token)
                    torch.save(model_state, f"{output_dir}/best.pt")
                patience = 5
            else:
                patience -= 1

            if summary_writer:
                with summary_writer.as_default():
                    tf.summary.scalar("dev_loss", dev_loss, step=epoch + 1)
                    tf.summary.scalar("avg_0s", avg_0s, step=epoch + 1)
                    tf.summary.scalar("avg_js", avg_js, step=epoch + 1)

        if patience <= 0:
            break

    return best_metrics, (
        fix_train,
        dev,
        fix_train_log_p,
        dev_log_p,
        transition_matrix,
        first_token,
    )


def run(config, sweep, experiment_name, current_time=None):
    """
    :param sweep: e.g., {"seed": [42, 7, 777]}
    """
    if current_time is None:
        current_time = datetime.datetime.now().strftime("%Y%m%d%H")
    log_dir = f"synthetic_logs/{current_time}_{experiment_name}"
    data = None
    all_best_metrics = {}
    for sweep_key in sweep:
        all_best_metrics[sweep_key] = {}
        for sweep_index, sweep_value in enumerate(sweep[sweep_key]):
            all_best_metrics[sweep_key][str(sweep_value)] = {}
            if sweep_key in ["vocab_size", "seed", "eval_data_size", "train_data_size"]:
                data = None  # can not reuse data
            config[sweep_key] = sweep_value
            print(config)
            sub_experiment_name = "-".join([f"{key}_{config[key]}" for key in config])
            summary_output_dir = f"{log_dir}/{sub_experiment_name}"
            summary_writer = tf.summary.create_file_writer(summary_output_dir)
            best_metrics, data = main(
                data=data,
                summary_writer=summary_writer,
                output_dir=summary_output_dir,
                **config,
            )
            with summary_writer.as_default():
                for key in best_metrics:
                    try:
                        tf.summary.scalar(
                            "best_" + key,
                            best_metrics[key],
                            step=int(sweep_value * 100)
                            if sweep_value <= 1.0
                            else int(sweep_value),
                        )
                    except Exception:
                        tf.summary.scalar(
                            "best_" + key,
                            best_metrics[key],
                            step=sweep_index,
                            description=sweep_value,
                        )
            all_best_metrics[sweep_key][str(sweep_value)]["avg_0s"] = best_metrics[
                "avg_0s"
            ]
            all_best_metrics[sweep_key][str(sweep_value)]["avg_js"] = best_metrics[
                "avg_js"
            ]
    with open(
        f"synthetic_logs/{current_time}_{experiment_name}/all_best_metrics.json", "w"
    ) as f:
        json.dump(all_best_metrics, f)
    return all_best_metrics


if __name__ == "__main__":
    real_dataset = None  # or 'webtext'
    for zero_percent in [0.5]:
        for vocab_size in [21, 51, 101, 501, 1001]:
            for seed in [7, 42, 777, 4222, 99999]:
                for loss_func, train_eta in [
                    # two_xens refers to mixces* (use the data distribution P via gumbel softmax)
                    ("two_xens", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]),
                    (
                        "qlogq_mix",
                        [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0],
                    ),  # qlogq_mix refers to mixces
                    # two_kls is the mixture of two kls, 0.0=reverse KL, 1.0=forward KL=mle
                    ("two_kls", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]),
                    ("js", [0.01, 0.1, 0.5, 0.9, 0.99]),
                ]:
                    config = {
                        "loss_func": loss_func,
                        "seed": seed,
                        "vocab_size": vocab_size,
                        "zero_percent": zero_percent,
                        "real_dataset": real_dataset,
                    }
                    if real_dataset:
                        experiment_name = (
                            f"{real_dataset}_{loss_func}_seed{seed}_vocab{vocab_size}"
                        )
                    else:
                        experiment_name = f"random_{zero_percent}_zero_{loss_func}_seed{seed}_vocab{vocab_size}"
                    best_metrics = run(
                        config,
                        {"train_eta": train_eta},
                        experiment_name=experiment_name,
                    )
