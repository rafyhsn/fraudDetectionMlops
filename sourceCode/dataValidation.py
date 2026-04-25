import pandas as pd


def validateData():
    dataPath = "dataFolder/mergedTrainData.csv"

    dataFrame = pd.read_csv(dataPath)

    requiredColumns = ["TransactionID", "isFraud", "TransactionAmt"]

    missingRequiredColumns = [
        columnName for columnName in requiredColumns
        if columnName not in dataFrame.columns
    ]

    if missingRequiredColumns:
        raise ValueError(f"Missing required columns: {missingRequiredColumns}")

    missingValuePercentage = dataFrame.isnull().mean() * 100
    duplicateRows = dataFrame.duplicated().sum()

    print("Data validation completed.")
    print(f"Dataset shape: {dataFrame.shape}")
    print(f"Duplicate rows: {duplicateRows}")
    print("\nTop missing value columns:")
    print(missingValuePercentage.sort_values(ascending=False).head(20))

    validationSummary = {
        "rowCount": dataFrame.shape[0],
        "columnCount": dataFrame.shape[1],
        "duplicateRows": int(duplicateRows),
        "targetDistribution": dataFrame["isFraud"].value_counts().to_dict()
    }

    print("\nValidation summary:")
    print(validationSummary)

    return validationSummary


if __name__ == "__main__":
    validateData()
