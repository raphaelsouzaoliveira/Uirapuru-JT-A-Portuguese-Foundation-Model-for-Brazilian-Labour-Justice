import datasets
from sklearn.metrics import average_precision_score

import evaluate

class AUCPRCMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Compute average precision (AP) from prediction scores.",
            citation="", # Add appropriate citation if available
            inputs_description="""
                predictions: Predicted sequences (e.g., lists of probability estimates of the positive class).
                references: Ground truth sequences (e.g., lists of integers or strings).
            """,
            features=datasets.Features({
                "predictions": datasets.Sequence(datasets.Value("float")), # Or Value("string")
                "references": datasets.Sequence(datasets.Value("int32")), # Or Value("string")
            })
        )

    
    def _compute(self, predictions, references, average="macro"):
        return {
            "auc_prc": float(
                average_precision_score(references, predictions, average=average)
            )
        }
# Example Usage:
#auc_prc = AUCPRCMetric()
#results = auc_prc.compute(predictions=[[0.8,0.8,0.8], [0.1,0.2,0.7]], references=[[0,1,1], [1,1,1]])
#print(results)
