import os

from tqdm import tqdm


def preprocess():
    for name in ["train", "valid", "test"]:
        outputs = []
        with open("writingPrompts/" + name + ".wp_source") as f:
            prompts = f.readlines()
        with open("writingPrompts/" + name + ".wp_target") as f:
            stories = f.readlines()
        for prompt, story in tqdm(zip(prompts, stories)):
            prompt = prompt.strip()
            story = (
                story.replace("<newline>", " ")
                .replace("``", '"')
                .replace("''", '"')
                .strip()
            )
            outputs.append(f"{prompt} {story}")
        with open(f"writingPrompts/writingPrompts.{name}.txt", "w") as f:
            f.write("\n".join(outputs))

        # detokenization
        os.system(
            f"perl detokenizer.perl -l en -q < writingPrompts/writingPrompts.{name}.txt "
            f"> writingPrompts/writingPrompts.{name}.detok"
        )
        os.system(
            f"mv writingPrompts/writingPrompts.{name}.detok "
            f"writingPrompts/writingPrompts.{name}.txt"
        )

        # remove length<50 >1000
        after_filter = []
        with open(f"writingPrompts/writingPrompts.{name}.txt", "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            words = line.split()
            if 50 < len(words) < 1000:
                after_filter.append(line)
        with open(f"writingPrompts/writingPrompts.{name}.txt", "w") as f:
            f.write("\n".join(after_filter))

    with open("writingPrompts/writingPrompts.train.txt", "r") as f:
        train = [line.strip() for line in f.readlines()]
    with open("writingPrompts/writingPrompts.test.txt", "r") as f:
        test = [line.strip() for line in f.readlines()]
    with open("writingPrompts/writingPrompts.valid.txt", "r") as f:
        validation = [line.strip() for line in f.readlines()]

    test = test[:5000]
    validation = validation[:5000]
    train = train

    with open("writingPrompts/writingPrompts.train.txt", "w") as f:
        f.write("\n".join(train))
    with open("writingPrompts/writingPrompts.test.txt", "w") as f:
        f.write("\n".join(test))
    with open("writingPrompts/writingPrompts.valid.txt", "w") as f:
        f.write("\n".join(validation))

    train10K = train[:10000]
    with open("writingPrompts/writingPrompts.train10K.txt", "w") as f:
        f.write("\n".join(train10K))
    train25K = train[:25000]
    with open("writingPrompts/writingPrompts.train25K.txt", "w") as f:
        f.write("\n".join(train25K))
    train50K = train[:50000]
    with open("writingPrompts/writingPrompts.train50K.txt", "w") as f:
        f.write("\n".join(train50K))
    train100K = train[:100000]
    with open("writingPrompts/writingPrompts.train100K.txt", "w") as f:
        f.write("\n".join(train100K))


if __name__ == "__main__":
    preprocess()
