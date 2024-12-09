import os
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_ROOT = './data'
DATA_FILE = os.path.join(DATA_ROOT, 'crop_recommendation.csv')

df = pd.read_csv(DATA_FILE)

label_to_crop_map = dict(enumerate(sorted(df['label'].unique().tolist())))
num_classes = len(label_to_crop_map.keys())
print(f'Num Classes: {num_classes}')
crop_to_label_map = {v: k for k, v in label_to_crop_map.items()}

with open(os.path.join(DATA_ROOT, 'label_to_crop_map.pkl'), 'wb') as f:
    pickle.dump(label_to_crop_map, f, pickle.HIGHEST_PROTOCOL)

X = df.iloc[:, :-1].values
y = df['label'].map(crop_to_label_map).values
print(f'Num Total Data: {len(y)}')

for SEED in range(10):
    print(f'Seed {SEED}')

    # data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
    
    # data scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    np.savez(os.path.join(DATA_ROOT, f'seed_{SEED}_full.npz'), X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    # labeled and unlabeled
    for num_labeled_per_class in range(1, 7):
        X_train_l, X_train_u, y_train_l, y_train_u = train_test_split(X_train, y_train, train_size=num_labeled_per_class * num_classes, stratify=y_train, random_state=SEED)
        np.savez(os.path.join(DATA_ROOT, f'seed_{SEED}_labeled_{num_labeled_per_class}.npz'), X_train_l=X_train_l, y_train_l=y_train_l, X_train_u=X_train_u, y_train_u=y_train_u, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    for i in range(1, 10):
        num_labeled_per_class = round((i * 0.1 * len(y_train)) / num_classes)
        X_train_l, X_train_u, y_train_l, y_train_u = train_test_split(X_train, y_train, train_size=num_labeled_per_class * num_classes, stratify=y_train, random_state=SEED)
        np.savez(os.path.join(DATA_ROOT, f'seed_{SEED}_labeled_{num_labeled_per_class}.npz'), X_train_l=X_train_l, y_train_l=y_train_l, X_train_u=X_train_u, y_train_u=y_train_u, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
