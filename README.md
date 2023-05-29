# MixCE

This repository contains the code and data for the following paper:

[MixCE: Training Autoregressive Language Models by Mixing the Forward and Reverse Cross-Entropies](https://arxiv.org/abs/2305.16958)

```
@inproceedings{zhang2023mixce,
  title={MixCE: Training Autoregressive Language Models by Mixing Forward and Reverse Cross-Entropies},
  author={Zhang, Shiyue and Wu, Shijie and İrsoy, Ozan and Lu, Steven and Bansal, Mohit and Dredze, Mark and Rosenberg, David},
  booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics},
  year={2023}
}
```

**code author:** Shiyue Zhang

## Requirements

- Python 3 (tested with Python 3.9.5)
- Install required packages:

```bash
python -m pip install -r requirements.txt
```

Optional: To avoid any version clashes with existing packages, you may want to perform the installation
under a virtual environment:

```bash
python -m venv yourenv
. yourenv/bin/activate  # for bash, might be something else for your particular shell
python -m pip install -r requirements.txt
```

## Synthetic Experiments

### Quick Start

[**synthetic.py**](./synthetic.py) is the script for running synthetic experiments.
Running experiments is very simple, just run:

```
python synthetic.py
```

Configurations (like seed, vocab size, etc.) can be specified and changed within the script and under `if __name__ == '__main__':`.

### Configurations

There are a few important configurations within [synthetic.py](./synthetic.py) that determine what kind of synthetic
experiments you can run:

**real_dataset**: if it is `None`, the transition matrix will be randomly initialized; or if it is `'webtext'`,
the transition matrix will be initialized from the pre-computed [transition matrices on webtext](./data/webtext_transition_matrices.pkl).

**zero_percent**: determines how many values in the transition matrix are 0. For example, if `zero_percent==0.5`, then 50% probabilities in the transition matrix are 0.

**vocab_size**: the vocabulary size. We test 21, 51, 101, 501, or 1001. Note that 21 means we have 20
normal tokens (including EOS) and 1 PAD token.

**seed**: We run 5 seeds (7, 42, 777, 4222, 99999) for each experiment.

**loss_func**: We test 4 loss functions: (1) `'two_xens'`: it is denoted as MixCE\* in our paper and uses the gold data
distribution P and gumbel softmax; (2) `'qlogq_mix'`: it is our approximated MixCE loss function; (3) `'two_kls'`:
the mixture of two KL divergences; (4) `'js'`: js divergence.

**train_eta**: The mixing ratio for those loss functions. If `train_eta==1.0` for `'two_xens'`, it is MLE. If `train_eta==1.0`
for `'two_kls'`, it is forward KL (also equals MLE). If `train_eta==0.0` for `'two_kls'`, it is the reverse KL.
We use a general definition of JS (see [this paper](https://arxiv.org/abs/1511.05101) for more details), and
JS converges to 0 when `train_eta` gets closer to 0.0 or 1.0. When train_eta=0.5, it is the normal definition of
JS divergence.

### Metrics & Evaluation

We evaluate synthetically trained bigram LMs by comparing the learned transition matrix against the gold
transition matrix. We use two metrics:

(1) **avg. js**: we compute the js divergence between each row of the gold and learned
transition matrices and average across rows.

(2) **avg. 0s**: we get the values from the learned matrix at gold probability=0 positions and then average them.

The `compare_parameters()` function in [synthetic.py](./synthetic.py) is used for computing these two metrics.

### Models

Models will all be saved under the `synthetic_logs/` directory.
Each model directory's name starts with the datetime that experiment was run. Under the model
directory, you will also find the TensorBoard event files, as well as an `all_best_metrics.json` that saves the best metrics scores
for each mixing ratio. See examples under [synthetic_logs/](./synthetic_logs).

Model evaluation is conducted after each epoch, and the best checkpoint is selected based on the loss on the dev set.

### Get results

Eventually, for each experiment, we average the results from 5 seeds; and for each
objective, we choose the best mixing ratio based on avg. js.

`get_synthetic_results()` in [results.py](./results.py) is a function used to average results from 5 seeds
and sort results of different mixing ratios accorrding to avg. js.

To use `get_synthetic_results()`, you need to first prepare [synthetic_models.json](./synthetic_logs/synthetic_models.json)
to specify the model directories. An example is shown in [synthetic_models.json](./synthetic_logs/synthetic_models.json).
Then you can get the result of the experiment that uses webtext initialized transition matrix, vocab=20 and objective=two_kls
by running `get_synthetic_results('webtext', '20', 'two_kls')`.

## GPT-2 Experiments

### Preparation

#### Prepare data

**Detokenizer.** You first need to download `detokenizer.perl` from Moses [here](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl),
and place it under the path `data/detokenizer.perl` because the following Python scripts depend on it.

Then:

```
cd data
python wikitext_data.py
python webtext_data.py
curl https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz | tar xvzf -
python writingprompts_data.py
```

The preprocessed data will be saved under `data/wikitext`, `data/webtext`, and `data/writingPrompts`.

#### Download GPT-2 models

Clone GPT-2 models using `git lfs` following the instruction provided by Hugging Face.

```
git lfs install
git clone https://huggingface.co/gpt2
```

gpt2 is the smallest GPT-2 model. We also experiment with gpt2-medium and gpt2-large. gpt2-large is used in computing MAUVE, so please download them too:

```
git clone https://huggingface.co/gpt2-medium
git clone https://huggingface.co/gpt2-large
```

Make a copy of gpt2-large for MAUVE:

```
cp -r gpt2-large gpt2-large-mauve
```

Because we will directly write to gpt2-large, which will affect MAUVE computation.

## Quick start

You can simply start running experiments by doing:

```
python run.py
```

Configurations can be manually specified within `run.py`. See an example under `if __name__ == '__main__'`.

## Configurations & Files

There are a few important configurations in [**run.py**](./run.py):

**training_size**: The training data size, we test `'10K'`, `'25K'`, `'50K'`, and `'100K'`; by default we use `'50K'`.

**model**: It can be `'gpt2'`, `'gpt2-meidum'`, or `'gpt2-large'`.

**dataset**: It can be `"wikitext"`, `"webtext"`, or `"writingPrompts"`.

**mixing_ratio**: We search through `[0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]` and choose the best `mixing_ratio`
based on dev set MAUVE score.

**train_batch_size, accumulation, eval_batch_size**: These configs should be determined by the platform you use. We use
one single Tesla V100 GPU (32G memory), and the recommended configs in this setting are in `run.py`.

There are one dict and three functions in [**run.py**](./run.py):

**data_sets{}**: It saves the paths of data files.

**run_no_trainer()**: The function used for training and evaluating models.

**run_no_trainer_eval()**: The function used for model evaluation only.

**run_no_trainer_turn_topp()**: The function used for tuning top-p sampling's p.

Besides [**run.py**](./run.py), I introduce here the other important Python scripts for model training and evaluation:

[**gpt2.py**](./gpt2.py) (the most essential file) contains a **GPT2MIXModel model class that implements our MixCE loss function**.

[**run_clm_no_trainer.py**](./run_clm_no_trainer.py) is the script to train and evaluate GPT-2 models.

[**run_clm_no_trainer_tune_topp.py**](./run_clm_no_trainer_tune_topp.py) is similar to `run_clm_no_trainer.py`, except that it is only used for tuning the hyperparameter p of top-p sampling.

[**metircs.py**](./metrics.py) contains the metrics we use to evaluate model generations.

## Models

Models will be saved under `train/` directory.

Each model directory's name starts with the datetime that experiment was run.
Under the model directory, we save the **best** checkpoint (selected based on dev loss).

`dev/test.sample`, `dev/test.sample1`, `dev/test.sample2`, and `dev/test.human` are 3 unbiased sampling generations and human text.

`dev/test_results.json` save the results of perplexity, diversity, and repetition.

After tuning p for top-p sampling, `dev/test.topp(p=*)` are top-p sampling generations with different p values.

After computing MAUVE and coherence (see next section for details), `dev/test_mauve_coherence_*.json` have the MAUVE and
coherence scores with different max lengths.

After computing controlled MAUVE and coherence (see next section for details), `dev/test_controlled_mauve_coherence_*.json`
are controlled MAUVE and coherence scores with different max lengths.

## Metrics & Evaluation

We report the scores of 6 metrics in our paper:

**perplexity** is computed along with model training/evaluation (see [**run_clm_no_trainer.py**](./run_clm_no_trainer.py)).

**diversity** is implemented by the `diversity()` function in [**metircs.py**](./metrics.py), and it is also computed
along with model training/evaluation by calling the `compute_diversity_repetition()` function in [**run_clm_no_trainer.py**](./run_clm_no_trainer.py).
Note that repetition is another metric we implemented but did not report in our paper; it checks what percent of the text is repetition loops and also return the repetitive phrase length.

**MAUVE** and **coherence** are computed in a post-hoc manner by using saved generation files. `compute_mauve()` and
`compute_coherence()` in [**metrics.py**](./metrics.py) are two helper functions to compute MAUVE and coherence.
They are called by the `compute_mauve_coherence()` function in [**results.py**](./results.py). To use `compute_mauve_coherence()`,
you must first prepare the [**models.json**](./train/models.json) to specify model directory names for evaluation.

Similarly, **controlled-MAUVE** and **controlled-coherence** can also be computed in a post-hoc manner by `compute_controlled_mauve_coherence()`
function in [**results.py**](./results.py).

## Pretrained models

| Dataset        | Model Size | Training Data Size | Objective       | Hugging Face hub name                             |
| -------------- | ---------- | ------------------ | --------------- | ------------------------------------------------ |
| wikitext       | gpt2-large | 50K                | MLE             | shiyue/wikitext_train50K_gpt2-large_mix1.0       |
| wikitext       | gpt2-large | 50K                | MixCE (eta=0.1) | shiyue/wikitext_train50K_gpt2-large_mix0.1       |
| webtext        | gpt2-large | 50K                | MLE             | shiyue/webtext_train50K_gpt2-large_mix1.0        |
| webtext        | gpt2-large | 50K                | MixCE (eta=0.3) | shiyue/webtext_train50K_gpt2-large_mix0.3        |
| writingPrompts | gpt2-large | 50K                | MLE             | shiyue/writingPrompts_train50K_gpt2-large_mix1.0 |
| writingPrompts | gpt2-large | 50K                | MixCE (eta=0.7) | shiyue/writingPrompts_train50K_gpt2-large_mix0.7 |

Try pretrained models in the following ways:

```
>>> from gpt2 import GPT2MIXModel
>>> from transformers import GPT2Tokenizer
>>> model = GPT2MIXModel.from_pretrained("shiyue/wikitext_train50K_gpt2-large_mix1.0")
>>> tokenizer = GPT2Tokenizer.from_pretrained('shiyue/wikitext_train50K_gpt2-large_mix1.0')
>>> text = "Hey, how are you?"
>>> encoded_input = tokenizer(text, return_tensors='pt')
>>> model.eval()
>>> out_ids = model.lm.generate(inputs=encoded_input["input_ids"], max_length=50, do_sample=True)
>>> print(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
```

## Contributions

We :heart: contributions.

Have you had a good experience with this project? Why not share some love and contribute code, or just let us know about any issues you had with it?

We welcome issue reports [here](../../issues); be sure to choose the proper issue template for your issue, so that we can be sure you're providing us with the necessary information.

Before sending a [Pull Request](../../pulls), please make sure you read our
[Contribution Guidelines](https://github.com/bloomberg/.github/blob/main/CONTRIBUTING.md).

## Notices

The following two files are borrowed and adopted from the `transformers` repository, and therefore retain their original copyrights.

### **run_clm_no_trainer.py**

This is originally picked up from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py. On top of it, we have applied the following modifications:

- Added the following additional arguments for parsing:
  - `--test_file`
  - `--reduction`
  - `--mixing_ratio`
  - `--max_length`
  - `--prompt_length`
  - `--eval_prompt_length`
  - `--cache_dir`
  - `--do_train`
  - `--do_eval`
- Commented out some unused code blocks for "`push_to_hub`" option.
- Handle tokenizer possibly not having pad token.
- Modify model loading block to use GPT2MIXModel for pretrained GPT2 models.
- Logic to add EOS after each text.
- Use `DataCollatorWithPadding` instead of the default collator.
- Add evaluation logic for "`do_eval`" option, most of which goes into the new function '`evaluate()`'.

### **run_clm_no_trainer_tune_topp.py**

This file is further modified from `run_clm_no_trainer.py` (see above) by changing how the `generate()` function is invoked to enable tuning for `top_p` option.

## Code of Conduct

This project has adopted a [Code of Conduct](https://github.com/bloomberg/.github/blob/main/CODE_OF_CONDUCT.md).
If you have any concerns about the Code, or behavior which you have experienced in the project, please
contact us at opensource@bloomberg.net.
