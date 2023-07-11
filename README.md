# Fair-Online-Dating-Recommendation

This repository is the implementation for paper "Fair Online Dating Recommendations for Sexually Fluid Users via Leveraging Opposite Gender Interaction Ratio" where we observe the existence of group unfairness in online dating recommendation and propose re-weighting and re-ranking strategies to mitigate group data imbalance and group calibration imbalance.

## Description

Use _split_dataset.py_ to preprocess and split the datasets into train/val/test.

Train the model with _main.py_ or _main_reweigh.py_.

Test the model performance with _test_model.py_.

Relevant implementation for re-ranking strategy is in _rerank.py_.

## Acknowledgement

The code is developed based on this repository https://github.com/YuWVandy/CAGCN.
