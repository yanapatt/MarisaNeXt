# -*- coding: utf-8 -*-
"""test.ipynb

Original file is located at
    https://colab.research.google.com/drive/1bizN4E85TpuRD9DbGbl-CUt3LavGWdU4
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score
import os

def get_true_label(dataloader, is_onehot=False):
    """
        This function gets the true labels from the dataloader.
        Args:
            - dataloader (torch.utils.data.DataLoader): PyTorch DataLoader
            - is_onehot (bool): Whether the labels are one-hot encoded (Default is False)
        Returns:
            - y_true (np.array): True labels
    """
    true_labels = []
    with torch.inference_mode():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                labels = batch[1]
            else:
                raise ValueError('Loader format not supported. Expected (inputs, labels).')

            if labels.ndim > 1:
                if labels.shape[1] == 1:
                    # shape (Batch_size, 1) => 1D (Batch_size,)
                    labels = labels.view(-1)
                elif is_onehot:
                    # One-Hot Encoding => Class Index
                    labels = torch.argmax(labels, dim=1)

            true_labels.append(labels.detach().cpu().numpy())

    if len(true_labels) == 0:
        print("[Warning] DataLoader is empty. Returning an empty array.")
        return np.array([])

    y_true = np.concatenate(true_labels, axis=0)
    return y_true

def evaluate_preds(y_true, y_probs, task='multiclass', average='macro', save_path=None):
    """
        This function evaluates the predictions of a model.
        Metrics:
            - Log Loss
            - Accuracy
            - F1 Score
            - Precision
            - Recall
        Args:
            - y_true (np.array): True labels
            - y_probs (np.array): Predicted probabilities (From model prediction)
            - task (str): Type of task (Include 'multiclass' and 'binary') Default is 'multiclass'
            - average (str): Type of average (Include 'macro', 'micro', 'weighted') Default is 'macro'
            - save_path (str): Path to save the metrics (Optional) Default is None
        Returns:
            - y_pred (np.array): Predicted labels
            - metrics (dict): Dictionary of metrics
    """
    assert task in ['multiclass', 'binary'], "task must be 'multiclass' or 'binary'"
    assert average in ['macro', 'micro', 'weighted'], "average must be 'macro', 'micro', or 'weighted'"

    if task == 'multiclass':
        y_pred = np.argmax(y_probs, axis=1)
        calc_average = average
    else: # task == 'binary'
        if y_probs.ndim == 2 and y_probs.shape[1] == 2:
            y_pred = np.argmax(y_probs, axis=1)
        else:
            y_pred = (y_probs >= 0.5).astype(int).flatten()
        calc_average = 'binary'

    print(f"[*] Task Type: {task.capitalize()} | Average: {calc_average}")

    if task == 'multiclass':
        y_probs = np.clip(y_probs, a_min=1e-15, a_max=None)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

    metrics = {}
    try:
        metrics['loss'] = log_loss(y_true, y_probs)
    except Exception:
        metrics['loss'] = None

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average=calc_average, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, average=calc_average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=calc_average, zero_division=0)

    if save_path:
        pd.DataFrame([metrics]).to_csv(save_path, index=False)
        print(f"[Logs] Saved Metrics to: {save_path}")

    return y_pred, metrics

def evaluate_preds_kfold(base_dir, n_splits, y_true, weight_type='best', task='multiclass', average='macro'):
    """
        This function evaluates the predictions of a model for each fold.
        Args:
            - base_dir (str): Base directory of the experiment
            - n_splits (int): Number of splits
            - y_true (np.array): True labels
            - weight_type (str): Type of weight (Include 'best' and 'final') Default is 'best'
            - task (str): Type of task (Include 'multiclass' and 'binary') Default is 'multiclass'
            - average (str): Type of average (Include 'macro', 'micro', 'weighted') Default is 'macro'
        Returns:
            - df_final (pd.DataFrame): DataFrame of metrics for each fold

        Base directory tree
        - base_dir
            |
            |- fold_1
                |- fold_1_test_probs_best_weight.csv
            |- fold_2
                |- fold_2_test_probs_best_weight.csv
                ...
                .
                .
                ...
            |- fold_n_splits
                |- fold_n_splits_test_probs_best_weight.csv
    """
    assert weight_type in ['best', 'final'], "weight_type must be 'best' or 'final'"
    assert task in ['multiclass', 'binary'], "task must be 'multiclass' or 'binary'"
    assert average in ['macro', 'micro', 'weighted'], "average must be 'macro', 'micro', or 'weighted'"

    all_fold_metrics = []

    for fold_num in range(1, n_splits + 1):
        fold_dir = os.path.join(base_dir, f"fold_{fold_num}")
        prob_path = os.path.join(base_dir, f"fold_{fold_num}", f"fold_{fold_num}_test_probs_{weight_type}_weight.csv")

        if not os.path.exists(prob_path):
            raise FileNotFoundError(f"[Error] Probabilities file not found: {prob_path}")

        y_probs = pd.read_csv(prob_path).values
        print(f"[*] Fold {fold_num}/{n_splits} | Shape: {y_probs.shape}")

        # Set save_path to None
        _, metrics = evaluate_preds(y_true=y_true,
                                    y_probs=y_probs,
                                    task=task,
                                    average=average
                                    )

        metrics_with_fold = {'Fold': f'Fold {fold_num}'}
        metrics_with_fold.update(metrics)
        all_fold_metrics.append(metrics_with_fold)

        test_perf_path = os.path.join(fold_dir, f"fold_{fold_num}_test_perf_{weight_type}_weight.csv")
        pd.DataFrame([metrics_with_fold]).to_csv(test_perf_path, index=False)

    df = pd.DataFrame(all_fold_metrics)
    numeric_df = df.drop('Fold', axis=1)

    mean_series = numeric_df.mean()
    std_series = numeric_df.std()

    mean_row = {'Fold': 'Mean'}
    mean_row.update(mean_series.to_dict())

    std_row = {'Fold': 'STD'}
    std_row.update(std_series.to_dict())

    summary_df = pd.DataFrame([mean_row, std_row])
    df_final = pd.concat([df, summary_df], ignore_index=True)

    print(f"\n{'-'*100}")
    print(f"K-Fold Test Set Evaluation Results (Individual Models):")
    print(df_final.to_string(index=False))
    print(f"{'-'*100}")

    summary_dir = os.path.join(base_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    summary_test_perf_path = os.path.join(summary_dir, f"kfold_test_perf_{weight_type}_weight.csv")
    df_final.to_csv(summary_test_perf_path, index=False)
    print(f"[Logs] Saved K-Fold Test Summary to {summary_test_perf_path}")

    return df_final

def pred_records(y_true, y_pred, y_probs, filepaths, idx_to_class=None, save_dir=None, prefix='model'):
    """
        This function creates a dataframe of the correct and incorrect predictions.
        Args:
            - y_true (np.array): True labels
            - y_pred (np.array): Predicted labels
            - y_probs (np.array): Predicted probabilities
            - filepaths (list): List of filepaths
            - idx_to_class (dict): Dictionary of index to class (Optional)
            - save_dir (str): Directory to save the dataframe (Optional)
            - prefix (str): Prefix for the filename (Optional)
        Returns:
            - df_correct (pd.DataFrame): Dataframe of correct predictions
            - df_incorrect (pd.DataFrame): Dataframe of incorrect predictions
    """
    correct_records = []
    incorrect_records = []

    for filepath, pred_idx, true_idx, probs in zip(filepaths, y_pred, y_true, y_probs):
        pred_int = int(pred_idx)
        true_int = int(true_idx)

        pred_label = idx_to_class[pred_int] if idx_to_class else str(pred_int)
        true_label = idx_to_class[true_int] if idx_to_class else str(true_int)

        if y_probs.ndim > 1:
            prob_val = float(probs[pred_int])
        else:
            prob_val = float(probs) if pred_int == 1 else 1.0 - float(probs)

        record = {
            "Filepaths": filepath,
            "Pred_Label": pred_label,
            "True_Label": true_label,
            "Pred_Index": pred_int,
            "True_Index": true_int,
            "Prob": prob_val
        }

        if pred_int == true_int:
            correct_records.append(record)
        else:
            incorrect_records.append(record)

    df_correct = pd.DataFrame(correct_records)
    df_incorrect = pd.DataFrame(incorrect_records)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        correct_path = os.path.join(save_dir, f"{prefix}_correct_preds.csv")
        incorrect_path = os.path.join(save_dir, f"{prefix}_incorrect_preds.csv")

        df_correct.to_csv(correct_path, index=False)
        df_incorrect.to_csv(incorrect_path, index=False)

    return df_correct, df_incorrect