---
title: QA-BERT
created: '2021-03-21T19:52:22.795Z'
modified: '2021-03-21T21:20:35.752Z'
---

# QA-BERT
A Question-Answering model based off of HuggingFace's transformer models, using the [haystack](!https://github.com/deepset-ai/haystack) library.  
This is a rewrite and improvement upon the original code written using [cdqa](!https://github.com/cdqa-suite/cdQA), a now outdated library. 

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
run ```python main.py```

# Contributing
(tbd CONTRIBUTING.md)

Additionally, the docs for `haystack` are quite terse, and it appears to not be a well-known library. A lot of breaking changes happen over a very small time-frame, leaving many questions on their issue-tracker no longer valid. Recommend downloading the source code with the release that you're using to resolve most errors.

