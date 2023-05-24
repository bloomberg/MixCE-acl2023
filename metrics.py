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

import os

import mauve
import numpy as np
from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import check_repetition_one


def diversity(words, max_n=4):
    # only consider ngrams n<=4
    ngrams = {}
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            if n > 2 and len(set(words[i : i + n])) == 1:
                continue
            ngram = " ".join(words[i : i + n])
            if ngram not in ngrams:
                ngrams[ngram] = 0
            ngrams[ngram] += 1
    return len(ngrams) / (sum([ngrams[k] for k in ngrams]) + 1e-15)


def repetition(words, k=3):
    # repeat for >= 3 times
    repeated_phrases = check_repetition_one(words, k=k)
    repetition_length, repeated_phrase_lengths = 0, []
    for key in repeated_phrases:
        phrase = key.split(" ")
        repeated_phrase_lengths.append(len(phrase))
        repetition_length += len(phrase) * repeated_phrases[key][0]
    repetition_rate = repetition_length / (len(words) + 1e-10)
    max_repeated_phase_length = (
        max(repeated_phrase_lengths) if repeated_phrase_lengths else 0
    )
    return repetition_rate, max_repeated_phase_length


def compute_diversity_repetition(data, tokenizer):
    def _helper(texts):
        diversities, repetition_rates, repeated_phrase_lengths = [], [], []
        for text in tqdm(texts):
            words = tokenizer.tokenize(text)
            diversities.append(diversity(words))
            repetition_rate, max_repeated_phase_length = repetition(words)
            repetition_rates.append(repetition_rate)
            if max_repeated_phase_length > 0:
                repeated_phrase_lengths.append(max_repeated_phase_length)
        return {
            "diversity": np.mean(diversities),
            "repetition_rate": np.mean(repetition_rates),
            "repeated_phrase_length": np.mean(repeated_phrase_lengths),
        }

    scores = {}
    for model_name in data:
        res = _helper(data[model_name])
        for key in res:
            scores[f"{model_name}_{key}"] = res[key]
    return scores


def compute_mauve(
    data, batch_size, max_text_length, device, featurize_model_name="gpt2-large-mauve"
):
    if not os.path.isdir(featurize_model_name):
        print(
            f"ERROR: please get {featurize_model_name} first following the instruction in README"
        )
        exit()
    device_id = device.index
    p_features = mauve.get_features_from_input(
        None,
        None,
        data["human"],
        featurize_model_name=featurize_model_name,
        max_len=max_text_length,
        device_id=device_id,
        name="p",
        verbose=True,
        batch_size=batch_size,
        use_float64=False,
    )
    scores = {}
    for model_name in data:
        if model_name != "human":
            out = mauve.compute_mauve(
                p_features=p_features,
                q_text=data[model_name],
                max_text_length=max_text_length,
                verbose=False,
                device_id=device_id,
                batch_size=batch_size,
                featurize_model_name=featurize_model_name,
            )
            scores[f"{model_name}_mauve"] = out.mauve
    return scores


def compute_coherence(data, device, batch_size, max_text_length):
    simcse = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    human_texts = data["human"]
    scores = {}
    for model_name in data:
        if model_name == "human":
            continue
        model_texts = data[model_name]
        prompts, continuations = [], []
        for human_text, model_text in zip(human_texts, model_texts):
            humen_tokens = human_text.strip().split(" ")
            model_tokens = model_text.strip().split(" ")
            common = 0
            for baseline_word, ours_word in zip(humen_tokens, model_tokens):
                if baseline_word == ours_word:
                    common += 1
                else:
                    break
            prompt = " ".join(model_tokens[:common])
            continuation = " ".join(model_tokens[common:])
            prompts.append(prompt)
            continuations.append(continuation)
        prompt_embs = simcse.encode(
            prompts,
            return_numpy=True,
            device=device,
            batch_size=batch_size,
            max_length=max_text_length,
        )
        conti_embs = simcse.encode(
            continuations,
            return_numpy=True,
            device=device,
            batch_size=batch_size,
            max_length=max_text_length,
        )
        coherence = []
        for prompt_emb, conti_emb in zip(prompt_embs, conti_embs):
            coherence.append(cosine_similarity([prompt_emb], [conti_emb])[0][0])
        scores[f"{model_name}_coherence"] = float(np.mean(coherence))
    return scores


def compute_controlled_mauve(
    data, batch_size, max_text_length, device, featurize_model_name="gpt2-large-mauve"
):
    if not os.path.isdir(featurize_model_name):
        print(
            f"ERROR: please get {featurize_model_name} first following the instruction in README"
        )
        exit()

    def _sample_text(texts, sample_size=10000):
        num_tokens = int(max_text_length / 1.35)
        possible_samples = []
        for i, text in enumerate(texts):
            tokens = text.strip().split()
            possible_samples.extend([(i, j) for j in range(len(tokens) - num_tokens)])
        if len(possible_samples) < sample_size:
            print("ERROR: Can't get enough samples!")
            return False
        np.random.shuffle(possible_samples)
        samples = possible_samples[:sample_size]
        sample_texts = []
        for i, j in samples:
            sample_texts.append(" ".join(texts[i].strip().split()[j : j + num_tokens]))
        return sample_texts

    device_id = device.index
    scores = {}
    human_texts = _sample_text(data["human"])
    if not human_texts:
        return {"ERROR": "Can't get enough samples!"}
    p_features = mauve.get_features_from_input(
        None,
        None,
        human_texts,
        featurize_model_name=featurize_model_name,
        max_len=512,
        device_id=device_id,
        name="p",
        verbose=True,
        batch_size=batch_size,
        use_float64=False,
    )
    for model_name in data:
        model_texts = _sample_text(data[model_name])
        if not model_texts:
            return {"ERROR": "Can't get enough samples!"}
        out = mauve.compute_mauve(
            p_features=p_features,
            q_text=model_texts,
            max_text_length=512,
            verbose=False,
            device_id=device_id,
            batch_size=batch_size,
            featurize_model_name=featurize_model_name,
        )
        scores[f"{model_name}_mauve"] = out.mauve
    return scores


def compute_controlled_coherence(data, device, batch_size, max_text_length):
    def _sample_text(texts, sample_size=10000):
        num_tokens = int(max_text_length / 1.35)
        num_prompt_tokens = int(50 / 1.35)
        possible_samples = []
        for i, text in enumerate(texts):
            tokens = text.strip().split()
            possible_samples.extend([(i, j) for j in range(len(tokens) - num_tokens)])
        if len(possible_samples) < sample_size:
            print("ERROR: Can't get enough samples!")
            return False, False
        np.random.shuffle(possible_samples)
        samples = possible_samples[:sample_size]
        prompts, continuations = [], []
        for i, j in samples:
            tokens = texts[i].strip().split()
            prompts.append(" ".join(tokens[j : j + num_prompt_tokens]))
            continuations.append(
                " ".join(tokens[j + num_prompt_tokens : j + num_tokens])
            )
        return prompts, continuations

    simcse = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    scores = {}
    for model_name in data:
        prompts, continuations = _sample_text(data[model_name])
        if not prompts:
            return {"ERROR": "Can't get enough samples!"}
        prompt_embs = simcse.encode(
            prompts,
            return_numpy=True,
            device=device,
            batch_size=batch_size,
            max_length=max_text_length,
        )
        conti_embs = simcse.encode(
            continuations,
            return_numpy=True,
            device=device,
            batch_size=batch_size,
            max_length=max_text_length,
        )
        coherence = []
        for prompt_emb, conti_emb in zip(prompt_embs, conti_embs):
            coherence.append(cosine_similarity([prompt_emb], [conti_emb])[0][0])
        scores[f"{model_name}_coherence"] = float(np.mean(coherence))
    return scores
