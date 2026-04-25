import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt


trackingUri = "http://127.0.0.1:5000"
experimentName = "ieeeFraudDetectionMlops"
recallThreshold = 0.75

os.makedirs("reportsFolder", exist_ok=True)
os.makedirs("sourceCode/plotFiles", exist_ok=True)


def calculatePopulationStabilityIndex(expectedValues, actualValues, bucketCount=10):
    expectedBins = pd.qcut(expectedValues, q=bucketCount, duplicates="drop")
    actualBins = pd.cut(actualValues, bins=expectedBins.cat.categories)

    expectedDistribution = expectedBins.value_counts(normalize=True).sort_index()
    actualDistribution = actualBins.value_counts(normalize=True).sort_index()

    actualDistribution = actualDistribution.replace(0, 0.0001)
    expectedDistribution = expectedDistribution.replace(0, 0.0001)

    psiValue = (
        (actualDistribution - expectedDistribution)
        * (actualDistribution / expectedDistribution).apply(
            lambda value: __import__("numpy").log(value)
        )
    ).sum()

    return psiValue


def simulateTimeBasedDrift():
    mlflow.set_tracking_uri(trackingUri)
    mlflow.set_experiment(experimentName)

    print("Loading merged dataset...")
    dataFrame = pd.read_csv("dataFolder/mergedTrainData.csv")

    print("Creating time-based split using TransactionDT...")
    dataFrame = dataFrame.sort_values("TransactionDT")

    splitIndex = int(len(dataFrame) * 0.70)

    earlierData = dataFrame.iloc[:splitIndex]
    laterData = dataFrame.iloc[splitIndex:]

    print(f"Earlier data shape: {earlierData.shape}")
    print(f"Later data shape: {laterData.shape}")

    earlierFraudRate = earlierData["isFraud"].mean()
    laterFraudRate = laterData["isFraud"].mean()

    print(f"Earlier fraud rate: {earlierFraudRate:.4f}")
    print(f"Later fraud rate: {laterFraudRate:.4f}")

    driftSummary = []

    importantColumns = [
        "TransactionAmt",
        "TransactionDT",
        "card1",
        "addr1",
        "D1"
    ]

    for columnName in importantColumns:
        if columnName in dataFrame.columns:
            cleanEarlier = earlierData[columnName].dropna()
            cleanLater = laterData[columnName].dropna()

            if len(cleanEarlier) > 0 and len(cleanLater) > 0:
                try:
                    psiValue = calculatePopulationStabilityIndex(
                        cleanEarlier,
                        cleanLater
                    )

                    driftSummary.append({
                        "featureName": columnName,
                        "psiValue": psiValue
                    })

                    print(f"{columnName} PSI: {psiValue:.4f}")

                except Exception as error:
                    print(f"Could not calculate PSI for {columnName}: {error}")

    driftDataFrame = pd.DataFrame(driftSummary)
    driftDataFrame.to_csv("reportsFolder/driftSummary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(driftDataFrame["featureName"], driftDataFrame["psiValue"])
    plt.xticks(rotation=30)
    plt.title("Feature Drift PSI Values")
    plt.xlabel("Feature")
    plt.ylabel("PSI")
    plt.tight_layout()

    driftPlotPath = "sourceCode/plotFiles/featureDriftPsiPlot.png"
    plt.savefig(driftPlotPath)
    plt.close()

    with mlflow.start_run(run_name="timeBasedDriftSimulation"):
        mlflow.log_metric("earlierFraudRate", earlierFraudRate)
        mlflow.log_metric("laterFraudRate", laterFraudRate)

        if len(driftDataFrame) > 0:
            mlflow.log_metric("maximumPsiValue", driftDataFrame["psiValue"].max())

        mlflow.log_artifact("reportsFolder/driftSummary.csv")
        mlflow.log_artifact(driftPlotPath)

    print("Drift simulation completed.")
    print("Saved: reportsFolder/driftSummary.csv")
    print("Saved: sourceCode/plotFiles/featureDriftPsiPlot.png")


if __name__ == "__main__":
    simulateTimeBasedDrift()
