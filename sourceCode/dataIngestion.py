import os
import pandas as pd


def loadAndMergeData():
    trainTransactionPath = "dataFolder/trainTransactionReduced.csv"
    trainIdentityPath = "dataFolder/trainIdentity.csv"
    outputPath = "dataFolder/mergedTrainData.csv"

    print("Loading reduced transaction data...")
    transactionDataFrame = pd.read_csv(trainTransactionPath)

    print("Loading identity data...")
    identityDataFrame = pd.read_csv(trainIdentityPath)

    print("Merging datasets on TransactionID...")
    mergedDataFrame = transactionDataFrame.merge(
        identityDataFrame,
        on="TransactionID",
        how="left"
    )

    os.makedirs("dataFolder", exist_ok=True)
    mergedDataFrame.to_csv(outputPath, index=False)

    print(f"Merged data saved to: {outputPath}")
    print(f"Final shape: {mergedDataFrame.shape}")

    return mergedDataFrame


if __name__ == "__main__":
    loadAndMergeData()
