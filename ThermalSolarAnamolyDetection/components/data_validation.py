import os, sys
from ThermalSolarAnamolyDetection.logger import logging
from ThermalSolarAnamolyDetection.exception import AppException
from ThermalSolarAnamolyDetection.entity.config_entity import DataValidationConfig
from ThermalSolarAnamolyDetection.entity.artifacts_entity import (DataIngestionArtifact,
                                                                   DataValidationArtifact)


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            
        except Exception as e:
            raise AppException(e, sys)

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True
            all_files = os.listdir(self.data_ingestion_artifact.feature_store_path)

            logging.info(f"Files found in feature store: {all_files}")
            logging.info(f"Required files: {self.data_validation_config.required_file_list}")

            missing_files = []
            for required_file in self.data_validation_config.required_file_list:
                if required_file not in all_files:
                    validation_status = False
                    missing_files.append(required_file)

            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            with open(self.data_validation_config.valid_status_file_dir, 'w') as f:
                f.write(f"Validation status: {validation_status}\n")
                if missing_files:
                    f.write(f"Missing files: {missing_files}\n")

            if missing_files:
                logging.info(f"Missing required files: {missing_files}")
            else:
                logging.info("All required files are present.")

            return validation_status

        except Exception as e:
            raise AppException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info("Entered initiate_data_validation method of DataValidation class")
        try:
            status = self.validate_all_files_exist()
            data_validation_artifact = DataValidationArtifact(validation_status=status)

            logging.info("Exited initiate_data_validation method of DataValidation class")
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact

        except Exception as e:
            raise AppException(e, sys)