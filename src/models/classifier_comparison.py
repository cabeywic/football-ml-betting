from typing import Dict
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

class ClassifierComparison:
    def __init__(self, classifiers: Dict[str, ClassifierMixin], use_standard_scaler: bool = True):
        if use_standard_scaler:
            # Wrap classifiers that benefit from scaled data in a pipeline that includes StandardScaler
            for name, clf in classifiers.items():
                if name in {'KNN', 'SVC'}:
                    classifiers[name] = make_pipeline(StandardScaler(), clf)
        self.classifiers = classifiers
        self.scores = {}

    def fit_and_score(self, X_train, y_train, X_test, y_test, score_func=precision_score, **kwargs):
        for name, clf in self.classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)
            score = score_func(y_test, y_pred, **kwargs)
            self.scores[name] = score

    def plot_scores(self):
        classifier_names = list(self.scores.keys())
        scores = list(self.scores.values())

        plt.figure(figsize=(12, 6))
        plt.bar(classifier_names, scores)
        plt.xlabel('Classifiers')
        plt.ylabel('Score')
        plt.title('Comparison of Scores of Different Classifiers')
        plt.show()

    def best_classifier(self, greater_is_better=True) -> str:
        if greater_is_better:
            return max(self.scores, key=self.scores.get)
        else:
            return min(self.scores, key=self.scores.get)

    def top_n_classifiers(self, n, greater_is_better=True):
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=greater_is_better)
        top_n_names = [name for name, score in sorted_scores[:n]]
        top_n_classifiers = {name: self.classifiers[name] for name in top_n_names}
        return top_n_classifiers