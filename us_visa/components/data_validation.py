import json
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.constants import SCHEMA_FILE_PATH
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_articraft: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        self.data_ingestion = data_ingestion_articraft
        self.data_validation = data_validation_config
        self.read_schema = read_yaml_file(SCHEMA_FILE_PATH)
        
        
    def validate_df_columns(self, dataFrame: pd.DataFrame):
        try:
            status = len(dataFrame.columns) == len(self.read_schema["columns"])
            logging.info(f"Is required column present : {status}")
            return status
        except Exception as e:
            raise USvisaException(e, sys)
        
    def is_column_exsist(self, dataFrame: pd.DataFrame) -> bool:
        try:
            dataFrame_column = dataFrame.columns
            numerical_col = []
            categorical_col = []
            for col in self.read_schema['numerical_columns']:
                if col not in dataFrame_column:
                    numerical_col.append(col)
                    
            if len(numerical_col) > 0:
                logging.info(f"Missing numerical columns are : {numerical_col}")
            
            for col in self.read_schema['categorical_columns']:
                if col not in dataFrame_column:
                    categorical_col.append(col)
                    
            if len(categorical_col) > 0:
                logging.info(f"Missing categorical columns are : {categorical_col}")
            
            return False if len(numerical_col) > 0 or len(categorical_col) > 0 else True
        except Exception as e:
            raise USvisaException(e, sys)
        
    @staticmethod
    def read_csv(file_path: str):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)
        
    def detect_dataset_drift(self, reference: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """Detect drift using Evidently DataDriftPreset and save YAML report"""
        try:
            drift_report = Report(
                metrics=[DataDriftPreset()],
                include_tests=True  # auto-generate pass/fail tests
            )

            report_result = drift_report.run(reference_data=reference, current_data=current_df)
            report_dict = report_result.dict()
            write_yaml_file(self.data_validation.data_validation_file_path, content=report_dict)

            # Extract dataset drift from DriftedColumnsCount metric
            dataset_drift = False
            for metric in report_dict.get("metrics", []):
                if metric.get("metric_name", "").startswith("DriftedColumnsCount"):
                    value = metric.get("value", {})
                    n_features = value.get("number_of_columns")
                    n_drifted = value.get("number_of_drifted_columns")
                    dataset_drift = value.get("dataset_drift")
                    logging.info(f"{n_drifted}/{n_features} drifted columns detected.")
                    logging.info(f"Dataset drift detected: {dataset_drift}")
                    break

            return dataset_drift
        except Exception as e:
            raise USvisaException(e, sys)
        
    def initiate_validation_pipeline(self):
        try:
            
            validation_error_msg = ""
            
            train_df, test_df = (DataValidation.read_csv(file_path=self.data_ingestion.trained_file_path),
                                (DataValidation.read_csv(file_path=self.data_ingestion.test_file_path)))
            status = self.validate_df_columns(dataFrame=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                 validation_error_msg += f"Columns are missing in training dataframe."
            status = self.validate_df_columns(dataFrame=test_df)
            
            logging.info(f"All required columns present in test dataframe: {status}")
            if not status:
                 validation_error_msg += f"Columns are missing in testing dataframe."
            status = self.is_column_exsist(dataFrame=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in taining dataframe."
            status = self.is_column_exsist(dataFrame=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in testing dataframe."
                
            validation_status = len(validation_error_msg) == 0
            
            if validation_status:
                drift_report = self.detect_dataset_drift(train_df, test_df)
                if drift_report:
                    logging.info("Drift Detected")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")
                
            return DataValidationArtifact(
                validation_error_msg=validation_error_msg,
                validation_status=validation_status,
                validation_error_file_path=self.data_validation.data_validation_file_path
            )
                
        except Exception as e:
            raise USvisaException(e, sys)