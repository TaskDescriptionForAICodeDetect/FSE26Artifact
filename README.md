## FSE26 Artifact

This repository is the reproducibility artifact for our FSE'26 paper. It contains:
- benchmark: tasks, description, and solutions
- baseline: implementations of comparison methods and experiment scripts used in the paper

## Benchmark

The `benchmark/` directory organizes evaluation tasks and datasets, along with unified solutions.

## Baselines

The `baseline/` directory includes several categories of methods covered in the paper, including:
- CodeRewrite: code rewriting and exam pass-rate evaluation
- Embedding: code representation learning (with AST utilities)
- FeatureEng: feature engineering and feature extraction
- FineTune/GPTSniffer: classifier fine-tuning and feature extraction for GPT provenance/detection
- PERPLEXITY: DetectGPT, DetectCodeGPT, Entropy, LogRank, and related perplexity/perturbation/ranking methods
- Zeroshot: zero-shot judging/scoring workflow
- baseutils, LLMRequest, etc.: common I/O, statistics, request, and task utilities

## Directory structure (two levels)

### baseline/

```text
baseline/
  baseutils/
    ioutils.py
    stasticutils.py
    taskUtil.py
  CodeRewrite/
    ablation_part_cr.py
    crUtil.py
    examPass/
    main.py
  Embedding/
    main.py
    utils/
  FeatureEng/
    codeUtil.py
    main.py
  FineTune/
    GPTSniffer/
  LLMRequest/
    askLLM.py
  PERPLEXITY/
    DetectCodeGPT/
    DetectGPT/
    Entropy/
    inferutil.py
    LogRank/
    utils.py
  Zeroshot/
    GPTJudge/
```

### benchmark/

```text
benchmark/
  tasks/
```


