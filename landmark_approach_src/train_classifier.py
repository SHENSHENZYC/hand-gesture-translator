import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC


def _train_rf_clf(X_train, X_val, y_train, y_val):
    ds = [2, 3, 4, 5]
    best_avg_f1 = 0
    best_model = None
    for d in ds:
        rf_clf = RandomForestClassifier(max_depth=d, oob_score=True, random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_val)
        avg_f1 = f1_score(y_val, y_pred, average='micro')

        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_model = rf_clf
    
    return best_model, best_avg_f1


def _train_svc(X_train, X_val, y_train, y_val):
    Cs = [0.01, 0.1, 1, 10, 100]
    best_avg_f1 = 0
    best_model = None
    for C in Cs:
        svc = SVC(C=C)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_val)
        avg_f1 = f1_score(y_val, y_pred, average='micro')
    
        if avg_f1 < best_avg_f1:
            best_avg_f1 = avg_f1
            best_model = svc
    
    return best_model, best_avg_f1


def main():
    """Entry point of the program."""
    ROOT = '.' if os.path.basename(os.getcwd()) == 'my_project' else '..'
    LANDMARK_FILE = os.path.join(ROOT, 'data/landmarks.pkl')

    with open(LANDMARK_FILE, 'rb') as f:
        dataset = pickle.load(f)
    
    data = np.array(dataset['data'])
    labels_ascii = np.array([ord(label) for label in dataset['labels']])

    # train-val-test split
    X_train_val, X_test, y_train_val, y_test= train_test_split(data, labels_ascii, test_size=0.2, shuffle=True, stratify=labels_ascii)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=True, stratify=y_train_val)

    # train best random forest
    best_rf_clf, rf_f1_val = _train_rf_clf(X_train, X_val, y_train, y_val)
    # train best support vector machines
    best_svc, svc_f1_val = _train_svc(X_train, X_val, y_train, y_val)

    bob = best_rf_clf if rf_f1_val >= svc_f1_val else best_svc
    f1_test = f1_score(y_test, bob.predict(X_test), average='micro')
    acc_test = accuracy_score(y_test, bob.predict(X_test))
    print(f'Final model: {type(bob).__name__}\n\ttest f1 score: {f1_test:.4f}\n\ttest accuracy: {acc_test * 100: .2f}%')

    if not os.path.exists(os.path.join(ROOT, 'model')):
        os.path.mkdir(os.path.join(ROOT, 'model'))

    with open(os.path.join(ROOT, 'model', 'best_landmark_clf.pkl'), 'wb') as f:
        pickle.dump(bob, f)


if __name__ == '__main__':
    main()
