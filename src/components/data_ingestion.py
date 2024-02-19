import os
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


@dataclass
class DataingestionConfig:
    raw_data_path = "https://raw.githubusercontent.com/tarun9804/misc/main/Datasets/Classification/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    train_file_path = os.path.join("artefact", "test.csv")
    test_file_path = os.path.join("artefact", "train.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_path_config = DataingestionConfig()
        os.makedirs(os.path.dirname(self.ingestion_path_config.test_file_path), exist_ok=True)

    def run_data_ingestion(self):
        df = pd.read_csv(self.ingestion_path_config.raw_data_path)
        num_col = df.select_dtypes(exclude='object').columns
        cat_col = df.select_dtypes(include='object').columns
        pp_obj = self.pre_proc_obj(num_col, cat_col);
        x = pp_obj.fit_transform(df)
        df_n = pd.DataFrame(x, columns=cat_col.append(num_col))
        train, test = train_test_split(df_n, test_size=0.2)
        train.to_csv(self.ingestion_path_config.train_file_path, index=False)
        test.to_csv(self.ingestion_path_config.test_file_path, index=False)

    def pre_proc_obj(self, num_col, cat_col):
        num_pipeline = Pipeline([("impute1", SimpleImputer(strategy="mean"))])
        cat_pipeline = Pipeline([("impute2", SimpleImputer(strategy="most_frequent"))])
        ct = ColumnTransformer([
            ("cat_imputer", cat_pipeline, cat_col),
            ("num_imputer", num_pipeline, num_col)
        ])
        return ct;


if __name__ == "__main__":
    obj = DataIngestion()
    obj.run_data_ingestion()
