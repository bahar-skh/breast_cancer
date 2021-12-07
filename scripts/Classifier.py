# Copyright: Bahar (Fatemeh) Safikhani

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced


from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
)

class Classifier:
    def __init__(self, estimator, random_state, keep_percentile, gen_report=True):
        self.estimator = estimator
        self.rs = random_state
        self.gen_report = gen_report
        self.statistics = {}
        self.CI = {}
        
        self.model = make_pipeline(
            SelectPercentile(chi2, percentile=keep_percentile),
            SMOTE(sampling_strategy='minority', random_state=self.rs),
            StandardScaler(with_mean=True, with_std=True),
            estimator
        )
        
        self.estimator_name = list(self.model.named_steps.keys())[3]
        
        if self.gen_report:
            print("=================================================================")
            print(f"===================== {self.estimator_name} =====================")
            print("=================================================================")
            print(f"{self.estimator_name} instance created successfully!\n")
        
    def hp_optim(self, X, y, param_grid, scoring, cv, verbose=False):
        new_param_grid = {}
        for key in param_grid.keys():
            new_param_grid[self.estimator_name + '__' + key] = param_grid[key]
    
        grid = GridSearchCV(self.model, new_param_grid, scoring=scoring, cv=cv, refit=False, n_jobs=-1)
        grid.fit(X, y)
        self.best_params_ = grid.best_params_
        
        if self.gen_report:
            print("=========================== HP Tuning ===========================")
            print(f"hyperparameter tuning finished successfully!\n")
            print(f"best hyperparameters found by grid search:\n{self.best_params_}\n")
        
        if verbose:
            means = grid.cv_results_['mean_test_score']
            stds = grid.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, grid.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
        self.model.set_params(**self.best_params_)
        
        if self.gen_report:
            print(f"{self.estimator_name} model parameters updated to the best found values!\n")
        
    def fit_model(self, X, y):
        self.model.fit(X, y)
        self.selected_features_indices = self.model[0].get_support(indices=True)
        self.selected_features = X.columns[self.selected_features_indices]
        
        if self.gen_report:
            print("=========================== Training ============================")
            print("model is successfully trained on training data.\n")
            print(f"indices of the selected features in SelectKBest:\n{self.selected_features_indices}\n")
            print(f"selected features in SelectPercentile:\n{list(self.selected_features)}\n")
            
    def  bootstrap_acc(self, X_train_valid, y_train_valid, bootstraps=100, fold_ratio=0.7):
        gen_report = self.gen_report
        self.gen_report = False
        
        statistics = []
        for _ in range(bootstraps):
            X_train = X_train_valid.sample(frac=fold_ratio, replace=True)
            y_train = y_train_valid.loc[X_train.index]
            valid_indices = [index for index in X_train_valid.index if index not in X_train.index]
            X_valid = X_train_valid.loc[valid_indices]
            y_valid = y_train_valid.loc[valid_indices]
            
            self.fit_model(X_train, y_train)
            pred = self.model.predict(X_valid)
            score = accuracy_score(y_valid, pred)
            
            statistics.append(score)
        self.statistics['accuracy'] = statistics   
        self.gen_report = gen_report

    def  bootstrap_auc(self, X_train_valid, y_train_valid, bootstraps=100, fold_ratio=0.7):
        gen_report = self.gen_report
        self.gen_report = False
        
        statistics = []
        for _ in range(bootstraps):
            X_train = X_train_valid.sample(frac=fold_ratio, replace=True)
            y_train = y_train_valid.loc[X_train.index]
            valid_indices = [index for index in X_train_valid.index if index not in X_train.index]
            X_valid = X_train_valid.loc[valid_indices]
            y_valid = y_train_valid.loc[valid_indices]
            
            self.fit_model(X_train, y_train)
            pred_prob = self.model.predict_proba(X_valid)
            score = roc_auc_score(pd.get_dummies(y_valid), pred_prob)
            
            statistics.append(score)
        self.statistics['roc_auc'] = statistics   
        self.gen_report = gen_report

    def get_confidence_interval(self, alpha=0.95):
        for method, statistics in self.statistics.items():
            mean = np.mean(statistics)
            lower = np.quantile(statistics, (1-alpha) / 2)
            upper = np.quantile(statistics, alpha + (1-alpha) / 2)
            self.CI[method] = [mean, lower, upper]

            if self.gen_report:
                print("======================= Boot Strappping =======================")
                print(f"scoring metric: {method}")
                print(f"mean scores = {mean:.6f}")
                print(f"confidence interval = [{lower:.6f}, {upper:.6f}]\n")
                    
    def cross_validate(self, X, y, scoring, cv):
        self.cvscores = cross_val_score(self.model, X, y, scoring=scoring, cv=cv)
        
        if self.gen_report:
            print("======================= Cross Validation ========================")
            print(f"scoring metric: '{scoring}'")
            print(f"scores = {self.cvscores}")
            print(f"mean score = {np.mean(self.cvscores)}\n")
        return self.cvscores
        
    def results(self, X_valid, y_valid, **kwargs):
        if len(kwargs) > 1:
            raise "wrong number of inputs"
        elif len(kwargs) == 1:
            self.gen_report = kwargs['gen_report']
        else:
            pass
        
        self.pred = self.model.predict(X_valid)
        self.pred_prob = self.model.predict_proba(X_valid)
        
        if self.gen_report:
            print("============================ Results ============================")
            print(f"accuracy score = {accuracy_score(y_valid, self.pred)}")
            print("*****************************************************************")
            print(f"roc area under the curve = {roc_auc_score(pd.get_dummies(y_valid), self.pred_prob)}")
            print("*****************************************************************")
            print(f"average precision (AP) from prediction scores = {average_precision_score(pd.get_dummies(y_valid), self.pred_prob)}")
            print("*****************************************************************")
            print("confusion matrix plot:")
            cm = confusion_matrix(y_valid, self.pred, labels=self.model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_).plot(cmap=plt.cm.Blues)
            plt.show()
            print("*****************************************************************")
            print(f"classification report:\n\n{classification_report(y_valid, self.pred)}")
            print("*****************************************************************")
            print(f"classification report based on metrics used with imbalanced dataset:\n")
            print(f"{classification_report_imbalanced(y_valid, self.pred)}")
            print("*****************************************************************")
            print("roc plot:\n")
            self.roc_figure = self.roc_plot(X_valid, y_valid)
            plt.show()
            print("*****************************************************************")
            print("precision recall plot:\n")
            self.pr_figure = self.pr_plot(X_valid, y_valid)
            plt.show()
            print("*****************************************************************")
            print("callibration plot:\n")
            self.cal_figure = self.calibration_plot(X_valid, y_valid)
            plt.show()
        
    def roc_plot(self, X_valid, y_valid):
        self.pred = self.model.predict(X_valid)
        self.pred_prob = self.model.predict_proba(X_valid)
        
        plt.figure(1)
        curve_function = roc_curve
        auc_roc = roc_auc_score(pd.get_dummies(y_valid), self.pred_prob)
        label = f"{self.estimator_name}, AUC = {auc_roc:.4f}"
        xlabel = "False positive rate"
        ylabel = "True positive rate"
        
        a, b, _ = roc_curve(y_valid, self.pred_prob[:,1], pos_label='M')
        
        plt.plot([0, 1], [0, 1], 'k--')
        fig = plt.plot(a, b, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.3, 1), fancybox=True, ncol=1)
        return fig
    
    def pr_plot(self, X_valid, y_valid):
        self.pred = self.model.predict(X_valid)
        self.pred_prob = self.model.predict_proba(X_valid)
        
        plt.figure(2)
        precision, recall, _ = precision_recall_curve(y_valid, self.pred_prob[:,1], pos_label='M')
        average_precision = average_precision_score(pd.get_dummies(y_valid), self.pred_prob)
        label = f"{self.estimator_name}, Average Precision = {average_precision:.4f}"
        fig = plt.step(recall, precision, where='post', label=label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(bbox_to_anchor=(1.3, 1), fancybox=True, ncol=1)
        return fig
    
    def calibration_plot(self, X_valid, y_valid):
        self.pred = self.model.predict(X_valid)
        self.pred_prob = self.model.predict_proba(X_valid)
        
        plt.figure()
        fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, self.pred_prob[:,1], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        fig = plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.tight_layout()
        plt.show()
        return fig