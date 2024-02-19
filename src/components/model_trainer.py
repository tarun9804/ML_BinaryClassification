import os.path

from src.logger import logging
from src.components.data_ingestion import DataIngestion
import pandas as pd
import joblib

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay  # need to check
from sklearn.model_selection import cross_val_score


class ModelTrainer:
    def __init__(self):
        self.obj_path=os.path.join("artefact","best_model.pkl")
        self.models = {
            "LogisticRegression": LogisticRegression(),
            "RandomForestClassifier": RandomForestClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "SVC": SVC(),
            "KNeighborsClassifier": KNeighborsClassifier()
        }
        self.params = {
            "LogisticRegression": {},
            "RandomForestClassifier": {'n_estimators': [8, 16, 32, 64, 128, 256]},
            "AdaBoostClassifier": {'learning_rate': [.1, .01, .05]},
            "GradientBoostingClassifier": {  # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]},
            "DecisionTreeClassifier": {'criterion': ['gini', 'entropy', 'log_loss'],
                                       # 'splitter':['best','random'],
                                       # 'max_features':['sqrt','log2'],
                                       },
            "SVC": {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
            "KNeighborsClassifier": {'leaf_size': list(range(1, 20)), 'n_neighbors': list(range(1, 20)), 'p': [1, 2]}

        }

    def run_model(self):
        obj = DataIngestion()
        x_train, x_test, y_train, y_test = obj.run_data_ingestion()
        res = pd.DataFrame(columns=["model",
                                    "train_accuracy_score",
                                    "test_accuracy_score",
                                    "train_roc_auc_score",
                                    "test_roc_auc_score",
                                    "tn",
                                    "fp",
                                    "fn",
                                    "tp",
                                    "tn-test",
                                    "fp-test",
                                    "fn-test",
                                    "tp-test"])
        j = 0;
        for i in self.models:
            model = self.models[i]
            h_param = self.params[i]
            print("running ", i)
            gs = GridSearchCV(model, h_param, cv=3)
            gs.fit(x_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)
            x_train_pred = model.predict(x_train)
            x_test_pred = model.predict(x_test)
            res.loc[j, "model"] = i
            # scoring accuracy_score,roc_auc_score,confusion_matrix,cross_val_score
            res.loc[j, "train_accuracy_score"] = accuracy_score(y_train, x_train_pred)
            res.loc[j, "test_accuracy_score"] = accuracy_score(y_test, x_test_pred)
            res.loc[j, "train_roc_auc_score"] = roc_auc_score(y_train, x_train_pred)
            res.loc[j, "test_roc_auc_score"] = roc_auc_score(y_test, x_test_pred)
            cm = confusion_matrix(y_train, x_train_pred)
            res.loc[j, "tn"], res.loc[j, "fp"], res.loc[j, "fn"], res.loc[j, "tp"] = cm.flatten()
            cm = confusion_matrix(y_test, x_test_pred)
            res.loc[j, "tn-test"], res.loc[j, "fp-test"], res.loc[j, "fn-test"], res.loc[j, "tp-test"] = cm.flatten()
            j = j + 1
        print(res)
        res = res.sort_values(by="test_roc_auc_score", ascending=False)
        best_model = res["model"][0]
        with open(self.obj_path, "wb") as f:
            joblib.dump(self.models[best_model],f)


if __name__ == "__main__":
    obj = ModelTrainer()
    obj.run_model()