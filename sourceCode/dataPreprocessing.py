import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocessData():
    inputPath = "dataFolder/mergedTrainData.csv"

    outputFolder = "dataFolder"
    encoderFolder = "sourceCode/modelFiles"

    os.makedirs(outputFolder, exist_ok=True)
    os.makedirs(encoderFolder, exist_ok=True)

    print("Loading merged dataset...")
    dataFrame = pd.read_csv(inputPath)

    print("Removing columns with more than 80% missing values...")
    missingPercentage = dataFrame.isnull().mean()
    columnsToDrop = missingPercentage[missingPercentage > 0.80].index.tolist()

    dataFrame = dataFrame.drop(columns=columnsToDrop)

    print(f"Dropped columns: {len(columnsToDrop)}")
    print(f"Remaining columns: {dataFrame.shape[1]}")

    print("Separating target column...")
    targetColumn = "isFraud"

    y = dataFrame[targetColumn]
    x = dataFrame.drop(columns=[targetColumn])

    if "TransactionID" in x.columns:
        x = x.drop(columns=["TransactionID"])

    print("Handling missing values...")
    numericColumns = x.select_dtypes(include=["int64", "float64"]).columns
    categoricalColumns = x.select_dtypes(include=["object"]).columns

    for columnName in numericColumns:
        x[columnName] = x[columnName].fillna(x[columnName].median())

    for columnName in categoricalColumns:
        x[columnName] = x[columnName].fillna("missingValue")

    print("Encoding categorical columns...")
    labelEncoders = {}

    for columnName in categoricalColumns:
        labelEncoder = LabelEncoder()
        x[columnName] = labelEncoder.fit_transform(x[columnName].astype(str))
        labelEncoders[columnName] = labelEncoder

    print("Adding basic feature engineering...")
    if "TransactionAmt" in x.columns:
        x["transactionAmountLog"] = x["TransactionAmt"].apply(
            lambda value: 0 if value <= 0 else __import__("numpy").log1p(value)
        )

    if "TransactionDT" in x.columns:
        x["transactionHour"] = (x["TransactionDT"] // 3600) % 24
        x["transactionDay"] = x["TransactionDT"] // (3600 * 24)

    print("Splitting train and test data...")
    xTrain, xTest, yTrain, yTest = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Saving processed files...")
    xTrain.to_csv(f"{outputFolder}/xTrain.csv", index=False)
    xTest.to_csv(f"{outputFolder}/xTest.csv", index=False)
    yTrain.to_csv(f"{outputFolder}/yTrain.csv", index=False)
    yTest.to_csv(f"{outputFolder}/yTest.csv", index=False)

    joblib.dump(labelEncoders, f"{encoderFolder}/labelEncoders.pkl")
    joblib.dump(columnsToDrop, f"{encoderFolder}/columnsToDrop.pkl")
    joblib.dump(x.columns.tolist(), f"{encoderFolder}/featureColumns.pkl")

    print("Preprocessing completed.")
    print(f"xTrain shape: {xTrain.shape}")
    print(f"xTest shape: {xTest.shape}")
    print("Fraud distribution in training:")
    print(yTrain.value_counts())


if __name__ == "__main__":
    preprocessData()
