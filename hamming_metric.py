import datasets
from sklearn.metrics import hamming_loss

import evaluate

class HammingMetric(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Computes the Hamming distance between two sequences.",
            citation="", # Add appropriate citation if available
            inputs_description="""
                predictions: Predicted sequences (e.g., lists of integers or strings).
                references: Ground truth sequences (e.g., lists of integers or strings).
            """,
            features=datasets.Features({
                "predictions": datasets.Sequence(datasets.Value("int32")), # Or Value("string")
                "references": datasets.Sequence(datasets.Value("int32")), # Or Value("string")
            })
        )

    #def _compute(self, predictions, references):
    #    total_diff = 0
    #    total_elements = 0
    #    for pred, ref in zip(predictions, references):            
    #        min_len = min(len(pred), len(ref))            
    #        total_elements += min_len
    #        for i in range(min_len):
    #            if pred[i] != ref[i]:
    #                total_diff += 1
    #        # Account for differing lengths if desired, or handle as error
    #        # For basic Hamming distance, typically sequences are assumed to be of equal length.
    #        # If lengths differ, this implementation only compares up to the shorter length.

    #    if total_elements == 0:
    #        return {"hamming_distance": 0.0} # Or handle as an error/special case
    #    return {"hamming_distance": total_diff / total_elements}
    def _compute(self, predictions, references, sample_weight=None):
        return {
            "hamming_loss": float(
                hamming_loss(references, predictions, sample_weight=sample_weight)
            )
        }
# Example Usage:
#hamming_metric = HammingMetric()
#results = hamming_metric.compute(predictions=[[1,1,1], [0,0,1]], references=[[0,1,1], [1,1,1]])
#print(results)
