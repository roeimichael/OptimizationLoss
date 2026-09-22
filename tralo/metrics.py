"""Classification metrics from explicit predictions; no allocation or selection."""


def classification_metrics(actual, predicted, n_classes, constrained_classes):
    """Use all declared classes, including absent ones; undefined ratios are zero.

    Confusion rows are true classes, columns are predicted classes.
    F1(c) = 2 TP(c) / (number truly c + number predicted c).
    cc_f1 is None when no constrained classes are declared.
    """
    if type(n_classes) is not int or n_classes < 1:
        raise ValueError('n_classes must be a positive integer')
    try:
        actual, predicted, constrained = list(actual), list(predicted), list(constrained_classes)
    except TypeError as exc:
        raise ValueError('labels, predictions and constrained classes must be iterable') from exc
    if not actual or len(actual) != len(predicted):
        raise ValueError('actual and predicted must have equal nonzero length')
    if any(type(c) is not int or not 0 <= c < n_classes
           for c in actual + predicted + constrained):
        raise ValueError('class indices must be integers in [0, n_classes)')
    if len(set(constrained)) != len(constrained):
        raise ValueError('constrained classes must be unique')
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for truth, prediction in zip(actual, predicted):
        matrix[truth][prediction] += 1
    rows = []
    for c in range(n_classes):
        tp = matrix[c][c]
        support = sum(matrix[c])
        selected = sum(row[c] for row in matrix)
        rows.append({
            'class': c, 'true_positive': tp, 'support': support,
            'predicted_count': selected,
            'precision': tp / selected if selected else 0.0,
            'recall': tp / support if support else 0.0,
            'f1': 2 * tp / (support + selected) if support + selected else 0.0,
        })
    return {
        'confusion': matrix, 'per_class': rows,
        'accuracy': sum(matrix[c][c] for c in range(n_classes)) / len(actual),
        'macro_f1': sum(row['f1'] for row in rows) / n_classes,
        'cc_f1': sum(rows[c]['f1'] for c in constrained) / len(constrained)
        if constrained else None,
    }
