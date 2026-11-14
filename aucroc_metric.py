import datasets
from sklearn.metrics import roc_auc_score
import evaluate

class AUCROCMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.",
            citation="", 
            inputs_description="""
                predictions: Predicted sequences (e.g., lists of probability estimates of the positive class).
                references: Ground truth sequences (e.g., lists of integers).
            """,
            features=datasets.Features({
                # Para classificação multirrótulo, as predições e referências são arrays 1D por documento
                "predictions": datasets.Sequence(datasets.Value("float")), 
                "references": datasets.Sequence(datasets.Value("int32")), 
            })
        )

    def _compute(self, predictions, references, sample_weight=None, average="macro"):
        return {
            "auc_roc": float(
                # Chamada da função do scikit-learn
                roc_auc_score(references, predictions, sample_weight=sample_weight, average=average)
            )
        }
        
# Example Usage:
#auc_roc = AUCROCMetric()
#results = auc_roc.compute(predictions=[[0.8,0.8,0.8], [0.1,0.2,0.7]], references=[[0,1,1], [1,1,1]])
#print(results)
