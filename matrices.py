import numpy as np
def mapping_y_true(y_true):
    y_test = []
    for rows in y_true:
            row_l=list(rows)
            max_index=row_l.index(max(row_l))
            y_test.append(max_index)
    return y_test

def compute_confusion_matrix(y_true, y_pred, num_classes=3):
    y_true = mapping_y_true(y_true)
    # Initialize matrix with zeros
    matrix = []
    for i in range(num_classes):
        matrix.append([0] * num_classes)
    
    # Fill matrix
    for i in range(len(y_true)):
        t = int(y_true[i])
        p = int(y_pred[i])
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t][p] += 1
            
    return np.array(matrix)

def compute_accuracy(y_true, y_pred):
    
    y_true = mapping_y_true(y_true)
    if len(y_true) == 0:
        return 0
        
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
            
    return (correct / len(y_true)) * 100

def compute_binary_metrics(y_true, y_pred, class_index):
   
    y_true = mapping_y_true(y_true)
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_true)):
        is_true = (y_true[i] == class_index)
        is_pred = (y_pred[i] == class_index)

        if is_true and is_pred:
            TP += 1
        elif not is_true and not is_pred:
            TN += 1
        elif not is_true and is_pred:
            FP += 1
        elif is_true and not is_pred:
            FN += 1
            
    return TP, TN, FP, FN
