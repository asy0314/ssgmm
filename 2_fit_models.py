import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support, log_loss

from ssgmm import SemiSupervisedGMM

np.seterr(all='raise')

NUM_CLASSES = 22
DATA_ROOT = './data'
MODEL_DIR = './models'
RESULT_DIR = './results'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def fit_models(X_train, y_train, X_val, y_val, X_test, y_test, random_state):

    # Combine training and validation data
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.hstack([y_train, y_val])

    # Define validation fold indices: -1 for training samples, 0 for validation samples
    validation_fold = np.array([-1] * len(X_train) + [0] * len(X_val))
    ps = PredefinedSplit(test_fold=validation_fold)

    # Define hyper-parameter grids
    param_grids = {
        'Lasso': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
        'Ridge': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 1, 10]
        },
        'RF': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10]
        }
    }

    # Initialize models
    models = {
        'Lasso': LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=random_state),
        'Ridge': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000, random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state),
        'RF': RandomForestClassifier(random_state=random_state)
    }
    
    # Perform hyper-parameter tuning
    best_models = {}
    for model_name, model in models.items():
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            scoring='f1_macro',
            cv=ps,  # Use predefined split
            verbose=1,
            n_jobs=-1,
            refit=False,  # set refit=False to exclude the validation set
        )
        grid_search.fit(X_train_val, y_train_val)  # Use combined train+validation set

        # since we set refit=False, train a model once again with the best params
        best_params = grid_search.best_params_
        best_model = model.set_params(**best_params)
        best_model.fit(X_train, y_train)
        best_models[model_name] = best_model

    return best_models
    

for SEED in range(10):
    print(f'Seed {SEED}')

    data = np.load(os.path.join(DATA_ROOT, f'seed_{SEED}_full.npz'))
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    best_models = fit_models(X_train, y_train, X_val, y_val, X_test, y_test, random_state=SEED)
    ssgmm_diag = SemiSupervisedGMM(K=NUM_CLASSES, covariance_type='diagonal')
    ssgmm_diag.fit(X_train, y_train)
    best_models['SSGMM_DIAG'] = ssgmm_diag
    ssgmm_full = SemiSupervisedGMM(K=NUM_CLASSES, covariance_type='full')
    ssgmm_full.fit(X_train, y_train)
    best_models['SSGMM_FULL'] = ssgmm_full
    
    y_score_val_diag = ssgmm_diag.predict_proba(X_val)
    y_pred_val_diag = np.argmax(y_score_val_diag, axis=1)
    f1_diag = f1_score(y_val, y_pred_val_diag, average='macro')
    y_score_val_full = ssgmm_full.predict_proba(X_val)
    y_pred_val_full = np.argmax(y_score_val_full, axis=1)
    f1_full = f1_score(y_val, y_pred_val_full, average='macro')
    if f1_full > f1_diag:
        best_models['SSGMM_TUNE'] = ssgmm_full
    else:
        best_models['SSGMM_TUNE'] = ssgmm_diag

    with open(os.path.join(MODEL_DIR, f'models_seed_{SEED}_full.pkl'), 'wb') as f:
        pickle.dump(best_models, f, pickle.HIGHEST_PROTOCOL)
    
    # Evaluate on the test set
    with open(os.path.join(RESULT_DIR, f'test_seed_{SEED}_full.csv'), 'w') as f:
        f.write(f'Model,Accuracy,Precision,Recall,F1,LogLoss,AUC\n')
        for model_name, model in best_models.items():
            y_score = model.predict_proba(X_test)
            y_pred = np.argmax(y_score, axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
            logloss = log_loss(y_test, y_score)
            roc_auc = roc_auc_score(y_test, y_score, average="macro", multi_class="ovr")
            f.write(f'{model_name},{accuracy},{precision},{recall},{f1},{logloss},{roc_auc}\n')

    for num_labeled_per_class in range(1, 7):
        data = np.load(os.path.join(DATA_ROOT, f'seed_{SEED}_labeled_{num_labeled_per_class}.npz'))
        X_train_l = data['X_train_l']
        y_train_l = data['y_train_l']
        X_train_u = data['X_train_u']
        y_train_u = data['y_train_u']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']

        best_models = fit_models(X_train_l, y_train_l, X_val, y_val, X_test, y_test, random_state=SEED)
        ssgmm_diag = SemiSupervisedGMM(K=NUM_CLASSES, covariance_type='diagonal')
        ssgmm_diag.fit(X_train_l, y_train_l, X_train_u)
        best_models['SSGMM_DIAG'] = ssgmm_diag
        ssgmm_full = SemiSupervisedGMM(K=NUM_CLASSES, covariance_type='full')
        ssgmm_full.fit(X_train_l, y_train_l, X_train_u)
        best_models['SSGMM_FULL'] = ssgmm_full

        y_score_val_diag = ssgmm_diag.predict_proba(X_val)
        y_pred_val_diag = np.argmax(y_score_val_diag, axis=1)
        f1_diag = f1_score(y_val, y_pred_val_diag, average='macro')
        y_score_val_full = ssgmm_full.predict_proba(X_val)
        y_pred_val_full = np.argmax(y_score_val_full, axis=1)
        f1_full = f1_score(y_val, y_pred_val_full, average='macro')
        if f1_full > f1_diag:
            best_models['SSGMM_TUNE'] = ssgmm_full
        else:
            best_models['SSGMM_TUNE'] = ssgmm_diag

        with open(os.path.join(MODEL_DIR, f'models_seed_{SEED}_labeled_{num_labeled_per_class}.pkl'), 'wb') as f:
            pickle.dump(best_models, f, pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(RESULT_DIR, f'test_seed_{SEED}_labeled_{num_labeled_per_class}.csv'), 'w') as f:
            f.write(f'Model,Accuracy,Precision,Recall,F1,LogLoss,AUC\n')
            for model_name, model in best_models.items():
                y_score = model.predict_proba(X_test)
                y_pred = np.argmax(y_score, axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
                logloss = log_loss(y_test, y_score)
                roc_auc = roc_auc_score(y_test, y_score, average="macro", multi_class="ovr")
                f.write(f'{model_name},{accuracy},{precision},{recall},{f1},{logloss},{roc_auc}\n')
    
    for i in range(1, 10):
        num_labeled_per_class = round((i * 0.1 * len(y_train)) / NUM_CLASSES)
        data = np.load(os.path.join(DATA_ROOT, f'seed_{SEED}_labeled_{num_labeled_per_class}.npz'))
        X_train_l = data['X_train_l']
        y_train_l = data['y_train_l']
        X_train_u = data['X_train_u']
        y_train_u = data['y_train_u']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']

        best_models = fit_models(X_train_l, y_train_l, X_val, y_val, X_test, y_test, random_state=SEED)
        ssgmm_diag = SemiSupervisedGMM(K=NUM_CLASSES, covariance_type='diagonal')
        ssgmm_diag.fit(X_train_l, y_train_l, X_train_u)
        best_models['SSGMM_DIAG'] = ssgmm_diag
        ssgmm_full = SemiSupervisedGMM(K=NUM_CLASSES, covariance_type='full')
        ssgmm_full.fit(X_train_l, y_train_l, X_train_u)
        best_models['SSGMM_FULL'] = ssgmm_full

        y_score_val_diag = ssgmm_diag.predict_proba(X_val)
        y_pred_val_diag = np.argmax(y_score_val_diag, axis=1)
        f1_diag = f1_score(y_val, y_pred_val_diag, average='macro')
        y_score_val_full = ssgmm_full.predict_proba(X_val)
        y_pred_val_full = np.argmax(y_score_val_full, axis=1)
        f1_full = f1_score(y_val, y_pred_val_full, average='macro')
        if f1_full > f1_diag:
            best_models['SSGMM_TUNE'] = ssgmm_full
        else:
            best_models['SSGMM_TUNE'] = ssgmm_diag
        
        with open(os.path.join(MODEL_DIR, f'models_seed_{SEED}_labeled_{num_labeled_per_class}.pkl'), 'wb') as f:
            pickle.dump(best_models, f, pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(RESULT_DIR, f'test_seed_{SEED}_labeled_{num_labeled_per_class}.csv'), 'w') as f:
            f.write(f'Model,Accuracy,Precision,Recall,F1,LogLoss,AUC\n')
            for model_name, model in best_models.items():
                y_score = model.predict_proba(X_test)
                y_pred = np.argmax(y_score, axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
                logloss = log_loss(y_test, y_score)
                roc_auc = roc_auc_score(y_test, y_score, average="macro", multi_class="ovr")
                f.write(f'{model_name},{accuracy},{precision},{recall},{f1},{logloss},{roc_auc}\n')
