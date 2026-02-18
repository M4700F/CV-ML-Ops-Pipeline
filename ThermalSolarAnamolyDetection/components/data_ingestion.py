import os
import sys
import shutil
import kagglehub
from ThermalSolarAnamolyDetection.logger import logging
from ThermalSolarAnamolyDetection.exception import AppException
from ThermalSolarAnamolyDetection.entity.config_entity import DataIngestionConfig
from ThermalSolarAnamolyDetection.entity.artifacts_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AppException(e, sys)

    def download_data(self) -> str:
        """
        Downloads the dataset from Kaggle using kagglehub.
        Returns the path where kagglehub saved the dataset.
        """
        try:
            logging.info("Downloading dataset from Kaggle: pkdarabi/solarpanel")

            kaggle_dataset_path = kagglehub.dataset_download("pkdarabi/solarpanel")

            logging.info(f"Dataset downloaded by kagglehub to: {kaggle_dataset_path}")
            return kaggle_dataset_path

        except Exception as e:
            raise AppException(e, sys)

    def copy_data_to_feature_store(self, kaggle_dataset_path: str) -> str:
        """
        Copies the downloaded Kaggle dataset into the configured feature store directory.
        Returns the feature store path.
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path, exist_ok=True)

            logging.info(f"Copying dataset from {kaggle_dataset_path} to {feature_store_path}")

            for item in os.listdir(kaggle_dataset_path):
                src = os.path.join(kaggle_dataset_path, item)
                dst = os.path.join(feature_store_path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

            logging.info(f"Dataset successfully copied to feature store: {feature_store_path}")
            return feature_store_path

        except Exception as e:
            raise AppException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            kaggle_dataset_path = self.download_data()
            feature_store_path = self.copy_data_to_feature_store(kaggle_dataset_path)

            data_ingestion_artifact = DataIngestionArtifact(
                data_download_path=kaggle_dataset_path,  # repurposed: raw kaggle download path
                feature_store_path=feature_store_path
            )

            logging.info("Exited initiate_data_ingestion method of DataIngestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise AppException(e, sys)