import os

from datasets import load_dataset
from tqdm import tqdm


def preprocess():
    # run locally
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    for datasplit in ["train", "test", "validation"]:
        outputs = []
        for item in dataset[datasplit]["text"]:
            item = item.replace("\n", " ").replace("\r", " ").strip()
            if item and item[0] != "=":
                outputs.append(item)
        print(datasplit, len(outputs))
        with open(f"wikitext/wikitext-103-raw-v1.{datasplit}.txt", "w") as f:
            f.write("\n".join(outputs))

    # remove @ and detokenization
    for datasplit in ["train", "test", "validation"]:
        # remove @
        os.system(
            f"sed 's/ @//g' wikitext/wikitext-103-raw-v1.{datasplit}.txt > wikitext/tmp.txt"
        )
        os.system(
            f"sed 's/@ //g' wikitext/tmp.txt > wikitext/wikitext-103-raw-v1.{datasplit}.txt"
        )

        # detokenization
        os.system(
            f"perl detokenizer.perl -l en -q < wikitext/wikitext-103-raw-v1.{datasplit}.txt "
            f"> wikitext/wikitext-103-raw-v1.{datasplit}.detok"
        )
        os.system(
            f"mv wikitext/wikitext-103-raw-v1.{datasplit}.detok "
            f"wikitext/wikitext-103-raw-v1.{datasplit}.txt"
        )

        # remove length< 50 and > 1000
        after_filter = []
        with open(f"wikitext/wikitext-103-raw-v1.{datasplit}.txt", "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            words = line.split()
            if 50 < len(words) < 1000:
                after_filter.append(line)
        with open(f"wikitext/wikitext-103-raw-v1.{datasplit}.txt", "w") as f:
            f.write("\n".join(after_filter))

    with open("wikitext/wikitext-103-raw-v1.train.txt", "r") as f:
        train = [line.strip() for line in f.readlines()]
    with open("wikitext/wikitext-103-raw-v1.test.txt", "r") as f:
        test = [line.strip() for line in f.readlines()]
    with open("wikitext/wikitext-103-raw-v1.validation.txt", "r") as f:
        validation = [line.strip() for line in f.readlines()]

    remain_test = 5000 - len(test)
    remain_validation = 5000 - len(validation)
    test += train[:remain_test]
    validation += train[remain_test : remain_test + remain_validation]
    train = train[remain_test + remain_validation :]

    with open("wikitext/wikitext-103-raw-v1.train.txt", "w") as f:
        f.write("\n".join(train))
    with open("wikitext/wikitext-103-raw-v1.test.txt", "w") as f:
        f.write("\n".join(test))
    with open("wikitext/wikitext-103-raw-v1.validation.txt", "w") as f:
        f.write("\n".join(validation))

    train10K = train[:10000]
    with open("wikitext/wikitext-103-raw-v1.train10K.txt", "w") as f:
        f.write("\n".join(train10K))
    train25K = train[:25000]
    with open("wikitext/wikitext-103-raw-v1.train25K.txt", "w") as f:
        f.write("\n".join(train25K))
    train50K = train[:50000]
    with open("wikitext/wikitext-103-raw-v1.train50K.txt", "w") as f:
        f.write("\n".join(train50K))
    train100K = train[:100000]
    with open("wikitext/wikitext-103-raw-v1.train100K.txt", "w") as f:
        f.write("\n".join(train100K))


if __name__ == "__main__":
    preprocess()
