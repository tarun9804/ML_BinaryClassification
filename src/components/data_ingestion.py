import os
from dataclasses import dataclass
from src.logger import logging
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class DataingestionConfig:
    raw_data_path = "https://raw.githubusercontent.com/tarun9804/misc/main/Datasets/Classification/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    train_file_path = os.path.join("artefact", "test.csv")
    test_file_path = os.path.join("artefact", "train.csv")
    preproc_obj_path = os.path.join("artefact","preprocobj.pkl")


class DataIngestion:
    def __init__(self):
        self.ingestion_path_config = DataingestionConfig()
        os.makedirs(os.path.dirname(self.ingestion_path_config.test_file_path), exist_ok=True)

    def run_data_ingestion(self):
        logging.info("Processing started")
        df = pd.read_csv(self.ingestion_path_config.raw_data_path)
        logging.info("data downloaded")

        train, test = train_test_split(df, test_size=0.2)
        logging.info("split done")
        train.to_csv(self.ingestion_path_config.train_file_path, index=False)
        test.to_csv(self.ingestion_path_config.test_file_path, index=False)
        logging.info("train,test data saved, DONE")

        target_train = train["Selector"]
        train.drop(columns=["Selector"],inplace=True)
        target_test = test["Selector"]
        test.drop(columns=["Selector"],inplace=True)

        num_col = train.select_dtypes(exclude='object').columns
        cat_col = train.select_dtypes(include='object').columns
        pp_obj = self.pre_proc_obj(num_col, cat_col);
        logging.info("preproc object created")
        x_train = pp_obj.fit_transform(train)
        logging.info("train data transofrmed")
        x_test = pp_obj.transform(test)
        logging.info("test data transformed")
        with open(self.ingestion_path_config.preproc_obj_path,"wb") as f:
            joblib.dump(pp_obj, f)

        logging.info("Pre processing obj saved")
        return x_train, x_test, target_train, target_test

    def pre_proc_obj(self, num_col, cat_col):
        num_pipeline = Pipeline([("impute1", SimpleImputer(strategy="mean")),
                                 ("scaling",StandardScaler())])
        cat_pipeline = Pipeline([("impute2", SimpleImputer(strategy="most_frequent")),
                                 ("Onehot", OneHotEncoder()),
                                 ("scaling",StandardScaler(with_mean=False))])
        ct = ColumnTransformer([
            ("cat_imputer", cat_pipeline, cat_col),
            ("num_imputer", num_pipeline, num_col)
        ])
        return ct;


if __name__ == "__main__":
    obj = DataIngestion()
    obj.run_data_ingestion()

