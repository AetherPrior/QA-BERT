# QA-BERT

A Question-Answering model based off of HuggingFace's transformer models, using the [haystack](https://github.com/deepset-ai/haystack) library.  
This is a rewrite and improvement upon the original code written using [cdqa](https://github.com/cdqa-suite/cdQA), a now outdated library.

# Method

- I have used a Dense-passage-retrieval System as a retriever for QA, and and existing BERT Models that function as document embedders and query embedders

- We did _not_ initially train the model at all, but I'm researching into ways to be able to train it on several similar datasets to improve the performance.

  - For example, there are ways to [train the DPR](https://haystack.deepset.ai/docs/latest/tutorial9md) using [medical QA datasets](https://github.com/abachaa/Existing-Medical-QA-Datasets).

  - I'm trying to find one that's more relevant to our field.

- Colab is NOT the way to go IMO for any decent code, but I'm constrained by the lack of a GPU on my system for speed.
  - The initial code was written in colab, after which I've exported it, and changed it
  - Python code can actually be run on colab by mounting it to the drive and calling it from colab
  - The model still works on the cpu however, but parallelism appears to be an issue.

# Setup

Clone the repository:

```
git clone git@github.com:AetherPrior/qa-bert-haystack.git
OR
git clone https://github.com/AetherPrior/qa-bert-haystack.git
```

Create a new `virtualenv`

```
python -m venv /my/venv
source /my/venv/bin/activate
```

Install the required libraries

```
pip install -r requirements.txt
```

**NOTE:** There is no support for `python3.9` as of yet from `haystack`. Recommend against using any version `>=python3.8` for running the code

# Running

```
cd src
python main.py --text-dir /path/to/textbooks
```

please be patient! First runs can take a huge amount of time.
Currently, the whole program is command line based.

# Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

Additionally, the docs for `haystack` are quite terse. A lot of breaking changes happen over a very small time-frame, leaving many questions on their issue-tracker no longer valid. Recommend downloading the source code with the release that you're using to resolve most errors.
