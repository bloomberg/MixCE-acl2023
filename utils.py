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

import random
from collections import OrderedDict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_repetition(generations, vocab_size, prompt_length):
    repetition_rate = []
    for generation in generations:
        generation = generation[prompt_length - 1 :]
        try:
            last = generation.index(vocab_size - 1) + 1
        except Exception:
            last = len(generation)
        generation = generation[:last]
        res = check_repetition_one(generation)
        repetition_length = 0
        for key in res:
            repetition_length += len(key.split(" ")) * res[key][0]
        repetition_rate.append(repetition_length / len(generation))
    return np.mean(repetition_rate)


def check_repetition_one(generation, k=2):
    # repetition: one span (word, phrase, etc.) repeats >= k times
    # return: {repeated span: repetition times}
    res = {}
    word_locations = OrderedDict()
    for loc, w in enumerate(generation):
        if w not in word_locations:
            word_locations[w] = []
        word_locations[w].append(loc)
    # visited_locs = set()
    for w in word_locations:
        pre_loc = word_locations[w][0]
        distance, span, start = None, None, None
        repeat = 0
        for loc in word_locations[w][1:]:
            # if loc in visited_locs:
            #     pre_loc = loc
            #     continue
            if (
                distance is None
                or loc - pre_loc != distance
                or generation[pre_loc:loc] != span
            ):
                if repeat > 0 and generation[pre_loc : pre_loc + len(span)] == span:
                    pre_loc += len(span)
                    repeat += 1
                if repeat >= k:
                    # visited_locs.update([i for i in range(pre_loc - repeat * len(span), pre_loc)])
                    same_key = None
                    for key in res:
                        if set(key.split(" ")) == set(span):
                            same_key = key
                            break
                    if same_key is not None:
                        if start < res[same_key][1] and repeat >= res[same_key][0]:
                            del res[same_key]
                            res[" ".join(map(str, span))] = (repeat, start)
                    else:
                        res[" ".join(map(str, span))] = (repeat, start)
                distance = loc - pre_loc
                span = generation[pre_loc:loc]
                start = pre_loc
                repeat = 1
            else:
                repeat += 1
            pre_loc = loc
        if repeat > 0 and generation[loc : loc + len(span)] == generation[pre_loc:loc]:
            loc += len(span)
            repeat += 1
        if repeat >= k:
            # visited_locs.update([i for i in range(loc - repeat * len(span), loc)])
            same_key = None
            for key in res:
                if set(key.split(" ")) == set(span):
                    same_key = key
                    break
            if same_key is not None:
                if start < res[same_key][1] and repeat >= res[same_key][0]:
                    del res[same_key]
                    res[" ".join(map(str, span))] = (repeat, start)
            else:
                res[" ".join(map(str, span))] = (repeat, start)
    return res
