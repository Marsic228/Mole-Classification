from pathlib import Path
import json

def calculate_class_metrics(matrix, class_names):

    result = {}

    for class_index, class_name in enumerate(class_names):
        true_positive = matrix[class_index][class_index]
        
        false_positive = 0
        for row_index in range(len(matrix)):
            if row_index != class_index:
                false_positive += matrix[row_index][class_index]
        
        false_negative = 0
        for column_index in range(len(matrix[class_index])):
            if column_index != class_index:
                false_negative += matrix[class_index][column_index]
        
        if true_positive + false_positive == 0:
            precision = 0.0
        else:
            precision = true_positive / (true_positive + false_positive)

        if true_positive + false_negative == 0:
            recall = 0.0
        else:
            recall = true_positive / (true_positive + false_negative)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        result[class_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
        
        
    return result


def load_confusion_matrix_report(path):
    path = Path(path)
    with open(path, "r") as file:
        loaded = json.load(file)
    return loaded


def save_class_metrics_report(report, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(report, file, indent=2)



if __name__ == "__main__":
    confusion_matrix = load_confusion_matrix_report("confusion_matrix_report.json")
    matrix = confusion_matrix["matrix"]
    class_names = confusion_matrix["class_names"]
    class_metrics = calculate_class_metrics(matrix, class_names)
    class_metrics_report = {
    "max_batches": confusion_matrix["max_batches"],
    "total_checked": confusion_matrix["total_checked"],
    "class_names": class_names,
    "class_metrics": class_metrics,
    }
    save_class_metrics_report(class_metrics_report, "class_metrics_report.json")
    

        

        

    # 7. Save metrics for this class into the result dictionary

    # 8. Return the result dictionary