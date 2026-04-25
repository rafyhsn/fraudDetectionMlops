from dataValidation import validateData
from dataPreprocessing import preprocessData
from modelTraining import trainAllModels
from driftDetection import simulateTimeBasedDrift
from modelRetraining import retrainIfRecallDrops
from explainabilityAnalysis import runExplainabilityAnalysis


def runMainPipeline():
    print("Starting complete MLflow MLOps pipeline...")

    validateData()
    preprocessData()
    trainAllModels()
    simulateTimeBasedDrift()
    retrainIfRecallDrops()
    runExplainabilityAnalysis()

    print("Complete MLflow MLOps pipeline finished successfully.")


if __name__ == "__main__":
    runMainPipeline()
