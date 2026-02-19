import os
import sys
import shutil
import yaml
from ultralytics import YOLO
from ThermalSolarAnamolyDetection.logger import logging
from ThermalSolarAnamolyDetection.exception import AppException
from ThermalSolarAnamolyDetection.entity.config_entity import ModelTrainerConfig
from ThermalSolarAnamolyDetection.entity.artifacts_entity import (DataValidationArtifact,
                                                                   ModelTrainerArtifact)


class ModelTrainer:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(),
        feature_store_path: str = "artifacts/data_ingestion/feature_store",
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_validation_artifact = data_validation_artifact
            self.feature_store_path = feature_store_path
        except Exception as e:
            raise AppException(e, sys)

    def _prepare_custom_yaml(self) -> str:
        """
        Reads the dataset's data.yaml, updates paths and nc,
        writes a new data.yaml for training, and returns its path.
        """
        try:
            dataset_yaml_path = os.path.join(self.feature_store_path, "data.yaml")
            with open(dataset_yaml_path, 'r') as f:
                dataset_config = yaml.safe_load(f)

            num_classes = dataset_config['nc']
            class_names = dataset_config['names']
            logging.info(f"Number of classes: {num_classes}")
            logging.info(f"Class names: {class_names}")

            # Build absolute paths for train/val/test
            train_path = os.path.abspath(os.path.join(self.feature_store_path, "train", "images"))
            val_path   = os.path.abspath(os.path.join(self.feature_store_path, "valid", "images"))
            test_path  = os.path.abspath(os.path.join(self.feature_store_path, "test",  "images"))

            custom_data_yaml = {
                'train': train_path,
                'val':   val_path,
                'test':  test_path,
                'nc':    num_classes,
                'names': class_names,
            }

            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            custom_yaml_path = os.path.join(self.model_trainer_config.model_trainer_dir, "custom_data.yaml")
            with open(custom_yaml_path, 'w') as f:
                yaml.dump(custom_data_yaml, f, default_flow_style=False, sort_keys=False)

            logging.info(f"Custom data.yaml created at: {custom_yaml_path}")
            return custom_yaml_path

        except Exception as e:
            raise AppException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception("Data validation failed. Cannot proceed with model training.")

            # Step 1: Prepare custom data.yaml with correct local paths
            custom_yaml_path = self._prepare_custom_yaml()

            # Step 2: Load pretrained YOLO model
            logging.info(f"Loading pretrained weights: {self.model_trainer_config.weight_name}")
            model = YOLO(self.model_trainer_config.weight_name)

            # Step 3: Train
            logging.info("Starting model training...")
            model.train(
                data=custom_yaml_path,
                epochs=self.model_trainer_config.no_epochs,
                imgsz=640,
                batch=self.model_trainer_config.batch_size,
                project=self.model_trainer_config.model_trainer_dir,
                name="yolo26-solarpanel",
            )
            logging.info("Model training completed.")

            # Step 4: Copy best.pt to model_trainer_dir
            trained_model_source = os.path.join(
                self.model_trainer_config.model_trainer_dir,
                "yolo26-solarpanel", "weights", "best.pt"
            )
            trained_model_dest = os.path.join(
                self.model_trainer_config.model_trainer_dir, "best.pt"
            )
            shutil.copy(trained_model_source, trained_model_dest)
            logging.info(f"Best model saved to: {trained_model_dest}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_dest
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)