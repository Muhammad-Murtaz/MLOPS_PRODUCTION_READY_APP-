from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from us_visa.exception import USvisaException
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.logger import logging
import sys
import pandas as pd
from typing import Optional
from us_visa.entity.s3_estimator import USvisaEstimator
from dataclasses import dataclass
from us_visa.entity.estimator import USvisaModel
from us_visa.entity.estimator import TargetValueMapping

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_best_model(self) -> Optional[USvisaEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            usvisa_estimator = USvisaEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if usvisa_estimator.is_model_present(model_path=model_path):
                return usvisa_estimator
            return None
        except Exception as e:
            raise  USvisaException(e,sys)

    def evaluate_model(self) -> ModelEvaluationArtifact:
        try:
            test_df = pd.read_csv(self.model_ingestion_articraft.test_file_path)
            X = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())

            trained_model_f1 = self.model_training_articraft.metrics.f1_score
            
            # Load best model from S3
            best_model = self.get_model()
            
            if best_model is None:
                logging.info("No existing model found in S3. Accepting current model.")
                # If no model in S3, accept the current one
                return ModelEvaluationResponse(
                    is_model_accepted=True,
                    best_model_f1_score=None,
                    difference=0,
                    trained_f1_score=trained_model_f1
                )
            
            # Evaluate best model
            y_pred_best = best_model.predict(X)
            best_model_f1 = f1_score(y, y_pred_best)
            
            logging.info(f"Trained model F1: {trained_model_f1}, Best model F1: {best_model_f1}")
            
            # Check if improvement is significant
            improvement_threshold = self.model_evaluate_config.improvement_threshold
            
            is_accepted = (trained_model_f1 - best_model_f1) > improvement_threshold
            
            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                best_model_f1_score=best_model_f1,
                difference=(trained_model_f1 - best_model_f1),
                trained_f1_score=trained_model_f1
            )

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e