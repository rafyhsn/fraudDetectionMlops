import os


def test_required_folders_exist():
    assert os.path.exists("sourceCode")
    assert os.path.exists("dataFolder")
    assert os.path.exists("reportsFolder")


def test_required_files_exist():
    assert os.path.exists("sourceCode/dataValidation.py")
    assert os.path.exists("sourceCode/modelTraining.py")
    assert os.path.exists("sourceCode/inferenceApi.py")
