import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from imblearn.under_sampling import RandomUnderSampler


trackingUri = "http://127.0.0.1:5000"
experimentName = "ieeeFraudDetectionMlops"
modelOutputFolder = "sourceCode/modelFiles"
plotOutputFolder = "sourceCode/plotFiles"

os.makedirs(modelOutputFolder, exist_ok=True)
os.makedirs(plotOutputFolder, exist_ok=True)


def loadProcessedData():
    xTrain = pd.read_csv("dataFolder/xTrain.csv")
    xTest = pd.read_csv("dataFolder/xTest.csv")
    yTrain = pd.read_csv("dataFolder/yTrain.csv").squeeze()
    yTest = pd.read_csv("dataFolder/yTest.csv").squeeze()

    return xTrain, xTest, yTrain, yTest


def createConfusionMatrixPlot(yTest, yPrediction, runName):
    confusionMatrix = confusion_matrix(yTest, yPrediction)

    display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
    display.plot()

    plotPath = f"{plotOutputFolder}/{runName}ConfusionMatrix.png"
    plt.title(f"{runName} Confusion Matrix")
    plt.savefig(plotPath, bbox_inches="tight")
    plt.close()

    return plotPath


def evaluateModel(model, xTest, yTest, runName):
    yPrediction = model.predict(xTest)

    if hasattr(model, "predict_proba"):
        yProbability = model.predict_proba(xTest)[:, 1]
    else:
        yProbability = yPrediction

    precisionValue = precision_score(yTest, yPrediction, zero_division=0)
    recallValue = recall_score(yTest, yPrediction, zero_division=0)
    f1Value = f1_score(yTest, yPrediction, zero_division=0)
    aucValue = roc_auc_score(yTest, yProbability)

    confusionMatrixPath = createConfusionMatrixPlot(
        yTest,
        yPrediction,
        runName
    )

    metrics = {
        "precision": precisionValue,
        "recall": recallValue,
        "f1Score": f1Value,
        "aucRoc": aucValue
    }

    return metrics, confusionMatrixPath


def logRun(model, modelType, runName, xTrain, yTrain, xTest, yTest, params):
    with mlflow.start_run(run_name=runName):
        model.fit(xTrain, yTrain)

        metrics, confusionMatrixPath = evaluateModel(
            model,
            xTest,
            yTest,
            runName
        )

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(confusionMatrixPath)

        if modelType == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif modelType == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        localModelPath = f"{modelOutputFolder}/{runName}.pkl"
        joblib.dump(model, localModelPath)

        print(f"\n{runName} completed")
        print(metrics)

        return {
            "runName": runName,
            "modelPath": localModelPath,
            **metrics
        }


def trainAllModels():
    mlflow.set_tracking_uri(trackingUri)
    mlflow.set_experiment(experimentName)

    xTrain, xTest, yTrain, yTest = loadProcessedData()

    imbalanceRatio = (yTrain == 0).sum() / (yTrain == 1).sum()

    results = []

    xgboostStandard = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    results.append(
        logRun(
            xgboostStandard,
            "xgboost",
            "xgboostStandardModel",
            xTrain,
            yTrain,
            xTest,
            yTest,
            {
                "model": "XGBoost",
                "imbalanceStrategy": "none",
                "costSensitive": False
            }
        )
    )

    xgboostCostSensitive = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalanceRatio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    results.append(
        logRun(
            xgboostCostSensitive,
            "xgboost",
            "xgboostCostSensitiveModel",
            xTrain,
            yTrain,
            xTest,
            yTest,
            {
                "model": "XGBoost",
                "imbalanceStrategy": "classWeight",
                "costSensitive": True,
                "scalePosWeight": imbalanceRatio
            }
        )
    )

    lightgbmModel = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.08,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    results.append(
        logRun(
            lightgbmModel,
            "lightgbm",
            "lightgbmBalancedModel",
            xTrain,
            yTrain,
            xTest,
            yTest,
            {
                "model": "LightGBM",
                "imbalanceStrategy": "classWeight",
                "costSensitive": True
            }
        )
    )

    print("\nApplying undersampling strategy...")
    underSampler = RandomUnderSampler(random_state=42)
    xTrainUnderSampled, yTrainUnderSampled = underSampler.fit_resample(
        xTrain,
        yTrain
    )

    xgboostUnderSampled = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    results.append(
        logRun(
            xgboostUnderSampled,
            "xgboost",
            "xgboostUnderSampledModel",
            xTrainUnderSampled,
            yTrainUnderSampled,
            xTest,
            yTest,
            {
                "model": "XGBoost",
                "imbalanceStrategy": "randomUnderSampling",
                "costSensitive": False
            }
        )
    )

    hybridModel = Pipeline(
        steps=[
            ("featureSelection", SelectKBest(mutual_info_classif, k=80)),
            (
                "randomForest",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    results.append(
        logRun(
            hybridModel,
            "sklearn",
            "hybridRandomForestFeatureSelectionModel",
            xTrain,
            yTrain,
            xTest,
            yTest,
            {
                "model": "RandomForest + SelectKBest",
                "imbalanceStrategy": "classWeight",
                "costSensitive": True,
                "selectedFeatures": 80
            }
        )
    )

    resultsDataFrame = pd.DataFrame(results)
    resultsDataFrame.to_csv("reportsFolder/modelComparisonResults.csv", index=False)

    bestModelRow = resultsDataFrame.sort_values(
        by=["recall", "aucRoc"],
        ascending=False
    ).iloc[0]

    print("\nBest model based on recall then AUC:")
    print(bestModelRow)

    joblib.dump(
        bestModelRow.to_dict(),
        f"{modelOutputFolder}/bestModelSummary.pkl"
    )


if __name__ == "__main__":
    trainAllModels()
