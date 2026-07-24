# Dataset setup

The datasets are not redistributed. Download them from their original sources
and keep the filenames shown below.

## Default of Credit Card Clients

Source:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Expected path:

```text
data/default_credit_card/default of credit card clients.xls
```

## Give Me Some Credit

Source:
https://www.kaggle.com/c/GiveMeSomeCredit

Expected training path:

```text
data/give_me_some_credit/cs-training.csv
```

The competition test file has no public target labels and is not used for the
reported experiments.

## German Credit

Source:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Expected path:

```text
data/german_credit/german.data-numeric
```

## Verification

After placing the files, run:

```bash
python main_german.py --method logistic_regression --seed 42
```

The loaders print the train, validation, and test shapes and raise an explicit
error when a required file is missing.
