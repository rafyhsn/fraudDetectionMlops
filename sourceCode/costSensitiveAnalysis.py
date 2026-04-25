import pandas as pd


def analyzeCostSensitiveImpact():
    resultsPath = "reportsFolder/modelComparisonResults.csv"

    dataFrame = pd.read_csv(resultsPath)

    print("\nModel Comparison:\n")
    print(dataFrame[["runName", "precision", "recall", "f1Score", "aucRoc"]])

    print("\n--- Cost Sensitive Analysis ---\n")

    standardModel = dataFrame[
        dataFrame["runName"] == "xgboostStandardModel"
    ].iloc[0]

    costSensitiveModel = dataFrame[
        dataFrame["runName"] == "xgboostCostSensitiveModel"
    ].iloc[0]

    print("Standard Model:")
    print(standardModel[["precision", "recall"]])

    print("\nCost-Sensitive Model:")
    print(costSensitiveModel[["precision", "recall"]])

    recallIncrease = (
        costSensitiveModel["recall"] - standardModel["recall"]
    )

    precisionDrop = (
        standardModel["precision"] - costSensitiveModel["precision"]
    )

    print("\nImpact Analysis:")
    print(f"Recall Increase: {recallIncrease:.4f}")
    print(f"Precision Drop: {precisionDrop:.4f}")

    print("\nBusiness Interpretation:")
    print(
        "Higher recall means more fraud cases are detected, reducing financial loss."
    )
    print(
        "Lower precision means more false alarms, which increases operational cost."
    )
    print(
        "In fraud detection, missing fraud (false negative) is more costly than false alarms."
    )


if __name__ == "__main__":
    analyzeCostSensitiveImpact()
