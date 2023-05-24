import json

import requests
from tqdm import tqdm


def fetch_data():
    for ds in ["webtext"]:
        for split in ["train", "valid", "test"]:
            filename = ds + "." + split + ".jsonl"
            r = requests.get(
                "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/"
                + filename,
                stream=True,
            )

            with open(filename, "wb") as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(
                    ncols=100,
                    desc="Fetching " + filename,
                    total=file_size,
                    unit_scale=True,
                ) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)


def preprocess():
    for datasplit in ["train", "valid", "test"]:
        outputs = []
        with open(f"webtext/webtext.{datasplit}.jsonl", "r") as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            text = line["text"].replace("\n", " ").strip()
            # remove length< 50 >1000
            if 50 < len(text.split()) < 1000:
                outputs.append(text)

        with open(f"webtext/webtext.{datasplit}.txt", "w") as f:
            f.write("\n".join(outputs))

    with open("webtext/webtext.train.txt", "r") as f:
        train = [line.strip() for line in f.readlines()]
    with open("webtext/webtext.test.txt", "r") as f:
        test = [line.strip() for line in f.readlines()]
    with open("webtext/webtext.valid.txt", "r") as f:
        validation = [line.strip() for line in f.readlines()]

    remain_test = 5000 - len(test)
    remain_validation = 5000 - len(validation)
    test += train[:remain_test]
    validation += train[remain_test : remain_test + remain_validation]
    train = train[remain_test + remain_validation :]

    with open("webtext/webtext.train.txt", "w") as f:
        f.write("\n".join(train))
    with open("webtext/webtext.test.txt", "w") as f:
        f.write("\n".join(test))
    with open("webtext/webtext.valid.txt", "w") as f:
        f.write("\n".join(validation))

    train10K = train[:10000]
    with open("webtext/webtext.train10K.txt", "w") as f:
        f.write("\n".join(train10K))
    train25K = train[:25000]
    with open("webtext/webtext.train25K.txt", "w") as f:
        f.write("\n".join(train25K))
    train50K = train[:50000]
    with open("webtext/webtext.train50K.txt", "w") as f:
        f.write("\n".join(train50K))
    train100K = train[:100000]
    with open("webtext/webtext.train100K.txt", "w") as f:
        f.write("\n".join(train100K))


if __name__ == "__main__":
    fetch_data()
    preprocess()
