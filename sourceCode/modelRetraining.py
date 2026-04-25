import os
import joblib
import mlflow
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


trackingUri = "http://127.0.0.1:5000"
experimentName = "ieeeFraudDetectionMlops"
recallThreshold = 0.80

os.makedirs("sourceCode/modelFiles", exist_ok=True)
os.makedirs("reportsFolder", exist_ok=True)


def retrainIfRecallDrops():
    mlflow.set_tracking_uri(trackingUri)
    mlflow.set_experiment(experimentName)

    print("Loading processed training data...")
    xTrain = pd.read_csv("dataFolder/xTrain.csv")
    xTest = pd.read_csv("dataFolder/xTest.csv")
    yTrain = pd.read_csv("dataFolder/yTrain.csv").squeeze()
    yTest = pd.read_csv("dataFolder/yTest.csv").squeeze()

    bestModelPath = "sourceCode/modelFiles/lightgbmBalancedModel.pkl"
    currentModel = joblib.load(bestModelPath)

    print("Checking current model recall...")
    yPrediction = currentModel.predict(xTest)
    currentRecall = recall_score(yTest, yPrediction, zero_division=0)

    print(f"Current recall: {currentRecall:.4f}")
    print(f"Recall threshold: {recallThreshold:.4f}")

    if currentRecall < recallThreshold:
        print("Recall dropped below threshold. Retraining started...")

        combinedX = pd.concat([xTrain, xTest], axis=0)
        combinedY = pd.concat([yTrain, yTest], axis=0)

        newXTrain, newXTest, newYTrain, newYTest = train_test_split(
            combinedX,
            combinedY,
            test_size=0.20,
            random_state=99,
            stratify=combinedY
        )

        retrainedModel = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.06,
            num_leaves=40,
            class_weight="balanced",
            random_state=99,
            n_jobs=-1
        )

        retrainedModel.fit(newXTrain, newYTrain)

        newPrediction = retrainedModel.predict(newXTest)
        newProbability = retrainedModel.predict_proba(newXTest)[:, 1]

        newPrecision = precision_score(newYTest, newPrediction, zero_division=0)
        newRecall = recall_score(newYTest, newPrediction, zero_division=0)
        newF1 = f1_score(newYTest, newPrediction, zero_division=0)
        newAuc = roc_auc_score(newYTest, newProbability)

        retrainedModelPath = "sourceCode/modelFiles/retrainedLightgbmModel.pkl"
        joblib.dump(retrainedModel, retrainedModelPath)

        retrainingSummary = {
            "triggerReason": "recallBelowThreshold",
            "oldRecall": currentRecall,
            "newRecall": newRecall,
            "newPrecision": newPrecision,
            "newF1Score": newF1,
            "newAucRoc": newAuc
        }

        pd.DataFrame([retrainingSummary]).to_csv(
            "reportsFolder/retrainingSummary.csv",
            index=False
        )

        with mlflow.start_run(run_name="thresholdBasedRetraining"):
            mlflow.log_param("triggerReason", "recallBelowThreshold")
            mlflow.log_param("retrainingStrategy", "thresholdBased")
            mlflow.log_metric("oldRecall", currentRecall)
            mlflow.log_metric("newRecall", newRecall)
            mlflow.log_metric("newPrecision", newPrecision)
            mlflow.log_metric("newF1Score", newF1)
            mlflow.log_metric("newAucRoc", newAuc)
            mlflow.lightgbm.log_model(retrainedModel, artifact_path="model")
            mlflow.log_artifact("reportsFolder/retrainingSummary.csv")

        print("Retraining completed.")
        print(retrainingSummary)

    else:
        print("Recall is above threshold. Retraining not required.")

        noRetrainingSummary = {
            "triggerReason": "recallAboveThreshold",
            "currentRecall": currentRecall,
            "recallThreshold": recallThreshold,
            "retrainingRequired": False
        }

        pd.DataFrame([noRetrainingSummary]).to_csv(
            "reportsFolder/retrainingSummary.csv",
            index=False
        )

        with mlflow.start_run(run_name="retrainingCheckNoAction"):
            mlflow.log_param("triggerReason", "recallAboveThreshold")
            mlflow.log_param("retrainingStrategy", "thresholdBased")
            mlflow.log_metric("currentRecall", currentRecall)
            mlflow.log_metric("recallThreshold", recallThreshold)
            mlflow.log_artifact("reportsFolder/retrainingSummary.csv")


if __name__ == "__main__":
    retrainIfRecallDrops()
