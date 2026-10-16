"""Unpickling compatibility for saved XGBoost artifacts.

Saved XGBoost estimators wrap CompatibleXGBClassifier, which pickle stores by
module path (model_definitions or run_final_pipeline). Importing this module
registers a run_final_pipeline alias pointing at the class below, so artifacts
saved under either module path load in this project.
"""
from __future__ import annotations

import sys
import types

from sklearn.base import BaseEstimator
from sklearn.utils import ClassifierTags
from xgboost import XGBClassifier as NativeXGBClassifier


class CompatibleXGBClassifier(NativeXGBClassifier):
    """Local tags adapter for XGBoost 2.1.2 with modern scikit-learn.

    Only estimator metadata changes; fit, prediction, and booster parameters
    remain the installed XGBoost implementation. No environment patch is made.
    Upstream fixed this compatibility problem in XGBoost 2.1.4.
    """

    def __sklearn_tags__(self):
        tags = BaseEstimator.__sklearn_tags__(self)
        tags.estimator_type = "classifier"
        tags.target_tags.required = True
        tags.classifier_tags = ClassifierTags()
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = True
        tags.no_validation = True
        return tags


if "run_final_pipeline" not in sys.modules:
    _alias = types.ModuleType("run_final_pipeline")
    _alias.CompatibleXGBClassifier = CompatibleXGBClassifier
    sys.modules["run_final_pipeline"] = _alias
