import glob
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support, log_loss
from tqdm.auto import tqdm

from ssgmm import SemiSupervisedGMM

np.seterr(all='raise')

NUM_CLASSES = 22
DATA_DIR = './data'
MODEL_DIR = './models_by_features'
RESULT_DIR = './results_by_features'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

feature_rank = np.array([6, 4, 2, 1, 0, 3, 5])

list_data_files = glob.glob(os.path.join(DATA_DIR, '*.npz'))
for data_file_path in tqdm(list_data_files):
    data = np.load(data_file_path)

    file_name = os.path.basename(data_file_path).split('.')[0]
    name_split = file_name.split('_')
    seed = int(name_split[1])
    if name_split[-1] == 'full':
        X_train_l = data['X_train']
        y_train_l = data['y_train']
        X_train_u = None
        y_train_u = None
        num_labeled_per_class = 72
    else:
        num_labeled_per_class = int(name_split[-1])
        X_train_l = data['X_train_l']
        y_train_l = data['y_train_l']
        X_train_u = data['X_train_u']
        y_train_u = data['y_train_u']

    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    for num_features in range(1, len(feature_rank) + 1):
        print(f"Seed: {seed} Labels: {num_labeled_per_class} Features: {num_features}")
        X_train_l_reduced = X_train_l[:, feature_rank[:num_features]]
        if X_train_u is not None:
            X_train_u_reduced = X_train_u[:, feature_rank[:num_features]]
        else:
            X_train_u_reduced = None
        X_val_reduced = X_val[:, feature_rank[:num_features]]
        X_test_reduced = X_test[:, feature_rank[:num_features]]

        best_models = dict()
        ssgmm_diag = SemiSupervisedGMM(K=NUM_CLASSES, max_iter=10000, covariance_type='diagonal')
        ssgmm_diag.fit(X_train_l_reduced, y_train_l, X_train_u_reduced)
        best_models['SSGMM_DIAG'] = ssgmm_diag
        
        ssgmm_full = SemiSupervisedGMM(K=NUM_CLASSES, max_iter=10000, covariance_type='full')
        ssgmm_full.fit(X_train_l_reduced, y_train_l, X_train_u_reduced)
        best_models['SSGMM_FULL'] = ssgmm_full

        y_score_val_diag = ssgmm_diag.predict_proba(X_val_reduced)
        y_pred_val_diag = np.argmax(y_score_val_diag, axis=1)
        f1_diag = f1_score(y_val, y_pred_val_diag, average='macro')
        
        y_score_val_full = ssgmm_full.predict_proba(X_val_reduced)
        y_pred_val_full = np.argmax(y_score_val_full, axis=1)
        f1_full = f1_score(y_val, y_pred_val_full, average='macro')
        
        if f1_full > f1_diag:
            best_models['SSGMM_TUNE'] = ssgmm_full
        else:
            best_models['SSGMM_TUNE'] = ssgmm_diag

        with open(os.path.join(MODEL_DIR, f'models_seed_{seed}_labeled_{num_labeled_per_class}_features_{num_features}.pkl'), 'wb') as f:
            pickle.dump(best_models, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(RESULT_DIR, f'test_seed_{seed}_labeled_{num_labeled_per_class}_features_{num_features}.csv'), 'w') as f:
            f.write(f'Model,Accuracy,Precision,Recall,F1,LogLoss,AUC\n')
            for model_name, model in best_models.items():
                y_score = model.predict_proba(X_test_reduced)
                y_pred = np.argmax(y_score, axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
                logloss = log_loss(y_test, y_score)
                roc_auc = roc_auc_score(y_test, y_score, average="macro", multi_class="ovr")
                f.write(f'{model_name},{accuracy},{precision},{recall},{f1},{logloss},{roc_auc}\n')
