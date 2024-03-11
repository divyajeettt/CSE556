import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

def labelwise_f1(y_true, y_pred,label_encoder):
    y_pred = label_encoder.inverse_transform(y_pred)
    y_true = label_encoder.inverse_transform(y_true)
    y_pred = [c.split('_')[1] if '_' in c else c for c in y_pred]
    y_true = [c.split('_')[1] if '_' in c else c for c in y_true]
    le = LabelEncoder()
    le.fit(y_true+y_pred)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    scores = f1_score(y_true, y_pred, average=None)
    return dict(zip(le.classes_, scores))
