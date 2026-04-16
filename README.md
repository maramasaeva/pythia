# pythia

a moral judgement machine. transformer models (RoBERTa, BERT) and SVMs for moral classification on open-source social media data. SHAP explanations for interpretability.

master's thesis project (2022), advanced master of AI @ KU Leuven.

## what it does

pythia replicates and extends the [Delphi](https://delphi.allenai.org/) moral judgment system: given a textual description of an action or situation, it predicts whether it's morally acceptable, unacceptable, or ambiguous. the project explores multiple ML approaches and compares their effectiveness.

## approaches

| model | approach | result |
|-------|----------|--------|
| **classical classifiers** | SVM, random forest, logistic regression with TF-IDF features | best performance; interpretable via SHAP |
| **RoBERTa** | fine-tuned transformer for moral classification | abandoned — low accuracy on this dataset |
| **SVM + WordNet (composite)** | SVM with WordNet synonyms as additional features | revealed a data leakage fault during cross-validation |
| **SVM + WordNet (OOV)** | replace out-of-vocabulary words with WordNet synonyms | improved generalization on unseen vocabulary |

## tech stack

- **scikit-learn** — classical classifiers (SVM, random forest, logistic regression)
- **pytorch + transformers** — RoBERTa fine-tuning
- **SHAP** — model interpretability and feature importance
- **spacy + WordNet** — NLP preprocessing, synonym expansion
- **pandas + numpy** — data processing

## project structure

```
Classifiers_with_SHAP.ipynb     # classical classifiers + SHAP interpretability
RoBERTa_model.ipynb             # transformer approach (abandoned)
SVC_WN_COMP_model.ipynb         # SVM + WordNet composite features
SVC_WN_OOV_model.ipynb          # SVM + WordNet OOV replacement
dataprep_notebooks/             # data preparation and preprocessing
prepped_data/                   # cleaned datasets
results/                        # experiment outputs
Pythia_in_Python_MaretaMasaeva.pdf  # full thesis document
```

## key findings

- classical classifiers with TF-IDF outperformed transformer models on this task
- SHAP analysis revealed which moral concepts drive classification decisions
- WordNet synonym expansion improved robustness to unseen vocabulary
- the composite feature approach exposed a methodological fault — a useful negative result

## author

**mara masaeva** — ai engineer, music producer, writer
[site](https://messier-systems.vercel.app) · [github](https://github.com/maramasaeva)
