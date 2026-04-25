import os
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import shap


trackingUri = "http://127.0.0.1:5000"
experimentName = "ieeeFraudDetectionMlops"

os.makedirs("reportsFolder", exist_ok=True)
os.makedirs("sourceCode/plotFiles", exist_ok=True)


def runExplainabilityAnalysis():
    mlflow.set_tracking_uri(trackingUri)
    mlflow.set_experiment(experimentName)

    print("Loading best model and test data...")
    model = joblib.load("sourceCode/modelFiles/lightgbmBalancedModel.pkl")
    xTest = pd.read_csv("dataFolder/xTest.csv")

    print("Creating sample for SHAP...")
    shapSample = xTest.sample(
        n=min(1000, len(xTest)),
        random_state=42
    )

    print("Generating feature importance...")
    featureImportance = pd.DataFrame({
        "featureName": xTest.columns,
        "importanceValue": model.feature_importances_
    }).sort_values(
        by="importanceValue",
        ascending=False
    )

    featureImportance.to_csv(
        "reportsFolder/featureImportance.csv",
        index=False
    )

    topFeatures = featureImportance.head(20)

    plt.figure(figsize=(10, 7))
    plt.barh(
        topFeatures["featureName"][::-1],
        topFeatures["importanceValue"][::-1]
    )
    plt.title("Top 20 Feature Importance - LightGBM")
    plt.xlabel("Importance Value")
    plt.ylabel("Feature Name")
    plt.tight_layout()

    featureImportancePlotPath = "sourceCode/plotFiles/featureImportancePlot.png"
    plt.savefig(featureImportancePlotPath)
    plt.close()

    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shapValues = explainer.shap_values(shapSample)

    if isinstance(shapValues, list):
        shapValuesToPlot = shapValues[1]
    else:
        shapValuesToPlot = shapValues

    plt.figure()
    shap.summary_plot(
        shapValuesToPlot,
        shapSample,
        show=False,
        max_display=20
    )

    shapPlotPath = "sourceCode/plotFiles/shapSummaryPlot.png"
    plt.savefig(shapPlotPath, bbox_inches="tight")
    plt.close()

    with mlflow.start_run(run_name="explainabilityAnalysis"):
        mlflow.log_artifact("reportsFolder/featureImportance.csv")
        mlflow.log_artifact(featureImportancePlotPath)
        mlflow.log_artifact(shapPlotPath)

    print("Explainability analysis completed.")
    print("Saved: reportsFolder/featureImportance.csv")
    print("Saved: sourceCode/plotFiles/featureImportancePlot.png")
    print("Saved: sourceCode/plotFiles/shapSummaryPlot.png")
    print("\nTop 10 important features:")
    print(featureImportance.head(10))


if __name__ == "__main__":
    runExplainabilityAnalysis()
