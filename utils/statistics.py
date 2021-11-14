def calc_precision(true_pos, false_pos):
    '''
    Calculate precision from number of true positives and false positives

    Args:
        true_pos (int): number of true positives
        false_pos (int): number of false positives

    Returns:
        (float || str): precision
    '''
    return true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else "DIV_BY_0"

def calc_recall(true_pos, ground_truth):
    '''
    Calculate recall from number of true positives and ground truth

    Args:
        true_pos (int): number of true positives
        ground_truth (int): actual number of positives

    Returns:
        (float || str): recall
    '''
    return true_pos / ground_truth if ground_truth > 0 else "DIV_BY_0"

def calc_f1(precision, recall):
    '''
    Calculate f1 from precision and recall

    Args:
        precision (float || str)
        recall (float || str)

    Returns:
        (float || str): f1
    '''
    if precision == "DIV_BY_0" or recall == "DIV_BY_0" or (precision + recall) == 0:
        return "DIV_BY_0"
    else:
        return 2 * (precision * recall) / (precision + recall)

def calc_statistics(true_pos, false_pos, ground_truth):
    '''
    Calculate precision, recall and f1

    Args:
        true_pos (int): number of true positives
        false_pos (int): number of false positives
        ground_truth (int): actual number of positives

    Returns:
        (float || str): precision
        (float || str): recall
        (float || str): f1
    '''
    precision = calc_precision(true_pos, false_pos)
    recall = calc_recall(true_pos, ground_truth)
    f1 = calc_f1(precision, recall)

    return precision, recall, f1
