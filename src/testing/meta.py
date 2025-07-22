"""
A
"""
from pathlib import Path
from src.utility.io.model import load_model
from src.utility.io.dataset import load_pytorch_dataset

dataset = load_pytorch_dataset(Path(
    "C:/Users/luisb/Documents/GitHub/RNA_con_MILP_y_metaheuristicas"
    "/Data/Breast Cancer Winsconsin (Diagnostic)/results/datasets/test_pt/"
    "bcw_best.pt"
))
model = load_model(Path(
    "C:/Users/luisb/Documents/GitHub/RNA_con_MILP_y_metaheuristicas"
    "/Data/Breast Cancer Winsconsin (Diagnostic)/results/models/model_best"
), "bcw")
print(dataset)
print(model)
