from sklearn.metrics import roc_auc_score, average_precision_score
from ood_metrics import calc_metrics
import numpy as np

root = "result/BDD_OOD" 

OOD = np.load(f"{root}/result/OOD_scores.npy")
ID = np.load(f"{root}/result/ID_scores.npy")


scores = np.append(OOD, ID)
labels = np.append(np.ones_like(OOD), np.zeros_like(ID))


result = calc_metrics(scores, labels)
print(f"AUROC: {result['auroc']}")
print(f"FPR@95: {result['fpr_at_95_tpr']}")
print(f"Detection Error: {result['detection_error']}")
print(f"AUPR ID: {result['aupr_in']}")
print(f"AUPR OOD: {result['aupr_out']}")