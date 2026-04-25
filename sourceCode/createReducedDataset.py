import os
import pandas as pd

windowsDatasetPath = "/mnt/e/semester 8/MLOps/A4/dataset"
localDataPath = "dataFolder"

sampleFraction = 0.30
randomState = 42

os.makedirs(localDataPath, exist_ok=True)

fileConfig = {
    "train_transaction.csv": {
        "outputName": "trainTransactionReduced.csv",
        "useSample": True
    },
    "test_transaction.csv": {
        "outputName": "testTransactionReduced.csv",
        "useSample": True
    },
    "train_identity.csv": {
        "outputName": "trainIdentity.csv",
        "useSample": False
    },
    "test_identity.csv": {
        "outputName": "testIdentity.csv",
        "useSample": False
    },
    "sample_submission.csv": {
        "outputName": "sampleSubmission.csv",
        "useSample": False
    }
}

for inputFileName, config in fileConfig.items():
    inputFilePath = os.path.join(windowsDatasetPath, inputFileName)
    outputFilePath = os.path.join(localDataPath, config["outputName"])

    print(f"Reading: {inputFilePath}")

    dataFrame = pd.read_csv(inputFilePath)

    if config["useSample"]:
        dataFrame = dataFrame.sample(
            frac=sampleFraction,
            random_state=randomState
        )

    dataFrame.to_csv(outputFilePath, index=False)

    print(f"Saved: {outputFilePath}")
    print(f"Rows: {len(dataFrame)}")
    print("-" * 50)

print("Reduced dataset creation completed.")
