# Fair-Online-Dating-Recommendation

This repository is the implementation for paper "Fair Online Dating Recommendations for Sexually Fluid Users via Leveraging Opposite Gender Interaction Ratio" where we observe the existence of group unfairness in online dating recommendation and propose re-weighting and re-ranking strategies to mitigate group data imbalance and group calibration imbalance.

## Description

Use _split_dataset.py_ to preprocess and split the datasets into train/val/test.

Train the model with _main.py_ or _main_reweigh.py_.

Test the model performance with _test_model.py_.

Relevant implementation for re-ranking strategy is in _rerank.py_.

<!--## Acknowledgement

The code is developed based on this repository https://github.com/YuWVandy/CAGCN.-->

## Dating-Related Datasets

We provide several dataset links that are suitable for investigating dating-related research. In this paper, we use Líbímseti.cz dataset as it provides valuable interaction and gender information.

**Líbímseti.cz**: http://konect.cc/networks/libimseti/

OKCupid: https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles

Lovoo: https://www.kaggle.com/datasets/jmmvutu/dating-app-lovoo-user-profiles

Speed dating: https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment

eHarmony dataset: https://github.com/fiatveritas/eHarmony_Project

One China dataset: https://figshare.com/articles/dataset/Gender-specific_preference_in_online_dating/6429443/1?file=11827379



