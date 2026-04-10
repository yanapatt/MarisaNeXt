# -*- coding: utf-8 -*-
"""visualization.ipynb

Original file is located at
    https://colab.research.google.com/drive/1qdA24uu-gq1thNBZLbI6x51XAInqO_wI
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import itertools
import pathlib
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def matplotlib_font(font_dir, prefer_name=None):
    """
        Set the default font for matplotlib display.
        Args:
            - font_dir (str): Path to the directory containing the font files.
            - prefer_name (str): Optional, the font with this name will be selected.
        Returns:
            - str: The name of the selected font.
    """
    font_dir = pathlib.Path(font_dir)
    if not font_dir.exists():
        raise FileNotFoundError(f'Font directory not found: {font_dir}')

    font_files = font_manager.findSystemFonts(fontpaths=[str(font_dir)])
    if not font_files:
        raise RuntimeError(f'No font files found in {font_dir}')

    for font in font_files:
        font_manager.fontManager.addfont(font)

    selected_font = None
    for font in font_files:
        name = font_manager.FontProperties(fname=font).get_name()

        if prefer_name and prefer_name.lower() in name.lower():
            selected_font = name
            break

        if selected_font is None:
            selected_font = name

    matplotlib.rcParams['font.family'] = selected_font
    return selected_font

def plot_lr_curve(history, figsize=(12, 5), fontsize=16, title=None, save_path=None):
    """
        For plotting and visualization the learning curve of a model.
        Args:
            - history (dict): The history of the model training.
            - figsize (tuple): The size of the figure.
            - fontsize (int): The size of the font.
            - title (str): Optional, the title of the figure.
            - save_path (str): Optional, the path to save the figure.
        Returns:
            - None
    """
    epochs = range(1, len(history['accuracy']) + 1)
    fig = plt.figure(figsize=figsize, facecolor='white')

    matlab_blue = '#0072BD'
    matlab_orange = '#D95319'

    def apply_matlab_style(ax):
        ax.grid(axis='y', color='#E5E5E5', linestyle='-', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(axis='both', colors='#333333', width=0.8, length=4)

    # Loss
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, history['loss'], color=matlab_blue, label='Training loss', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], color=matlab_orange, label='Validation loss', linewidth=1.5)
    ax1.set_title('Training and Validation Loss', fontsize=fontsize)
    ax1.set_xlabel('Epochs', fontsize=fontsize)
    ax1.set_ylabel('Loss', fontsize=fontsize)
    ax1.legend(frameon=True, edgecolor='#CCCCCC', fontsize=fontsize-2)
    ax1.tick_params(labelsize=fontsize-2)
    apply_matlab_style(ax1)

    # Accuracy
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, history['accuracy'], color=matlab_blue, label='Training accuracy', linewidth=1.5)
    ax2.plot(epochs, history['val_accuracy'], color=matlab_orange, label='Validation accuracy', linewidth=1.5)
    ax2.set_title('Training and Validation Accuracy', fontsize=fontsize)
    ax2.set_xlabel('Epochs', fontsize=fontsize)
    ax2.set_ylabel('Accuracy', fontsize=fontsize)
    ax2.legend(frameon=True, edgecolor='#CCCCCC', fontsize=fontsize-2)
    ax2.tick_params(labelsize=fontsize-2)
    apply_matlab_style(ax2)

    if title:
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.text(0.5, 0.05, f'{title}', ha='center', fontsize=fontsize)
    else:
        plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}/lr_curve.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/lr_curve.svg', format='svg', bbox_inches='tight')

    plt.show()

def plot_cv_summary_curve(histories, figsize=(12, 5), fontsize=16, title=None, save_path=None):
    """
    For plotting and visualization the summarized learning curve of a K-Fold CV model.
    Args:
        - histories (list): A list of history dictionaries from all folds.
        - figsize (tuple): The size of the figure.
        - fontsize (int): The size of the font.
        - title (str): Optional, the title of the figure.
        - save_path (str): Optional, the path to save the figure.
    """
    # 1. Find min epochs
    min_epochs = min(len(h['loss']) for h in histories)
    epochs = range(1, min_epochs + 1)

    # 2. Fetch history from every fold (with same length epoch)
    losses = np.array([h['loss'][:min_epochs] for h in histories])
    val_losses = np.array([h['val_loss'][:min_epochs] for h in histories])
    accuracies = np.array([h['accuracy'][:min_epochs] for h in histories])
    val_accuracies = np.array([h['val_accuracy'][:min_epochs] for h in histories])

    # 3. Calc Mean and Standard Deviation
    mean_loss, std_loss = np.mean(losses, axis=0), np.std(losses, axis=0)
    mean_val_loss, std_val_loss = np.mean(val_losses, axis=0), np.std(val_losses, axis=0)
    mean_acc, std_acc = np.mean(accuracies, axis=0), np.std(accuracies, axis=0)
    mean_val_acc, std_val_acc = np.mean(val_accuracies, axis=0), np.std(val_accuracies, axis=0)

    fig = plt.figure(figsize=figsize, facecolor='white')
    matlab_blue = '#0072BD'
    matlab_orange = '#D95319'

    def apply_matlab_style(ax):
        ax.grid(axis='y', color='#E5E5E5', linestyle='-', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(axis='both', colors='#333333', width=0.8, length=4)

    # Loss
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, mean_loss, color=matlab_blue, label='Training loss (Mean)', linewidth=2)
    ax1.plot(epochs, mean_val_loss, color=matlab_orange, label='Validation loss (Mean)', linewidth=2)

    ax1.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=matlab_blue, alpha=0.15, label='Training loss (STD)')
    ax1.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color=matlab_orange, alpha=0.15, label='Validation loss (STD)')

    ax1.set_title('Average Training and Validation Loss', fontsize=fontsize)
    ax1.set_xlabel('Epochs', fontsize=fontsize)
    ax1.set_ylabel('Loss', fontsize=fontsize)
    ax1.legend(frameon=True, edgecolor='#CCCCCC', fontsize=fontsize-2)
    ax1.tick_params(labelsize=fontsize-2)
    apply_matlab_style(ax1)

    # Accuracy
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, mean_acc, color=matlab_blue, label='Training accuracy (Mean)', linewidth=2)
    ax2.plot(epochs, mean_val_acc, color=matlab_orange, label='Validation accuracy (Mean)', linewidth=2)

    ax2.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=matlab_blue, alpha=0.15, label='Training accuracy (STD)')
    ax2.fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color=matlab_orange, alpha=0.15, label='Validation accuracy (STD)')

    ax2.set_title('Average Training and Validation Accuracy', fontsize=fontsize)
    ax2.set_xlabel('Epochs', fontsize=fontsize)
    ax2.set_ylabel('Accuracy', fontsize=fontsize)
    ax2.legend(frameon=True, edgecolor='#CCCCCC', fontsize=fontsize-2)
    ax2.tick_params(labelsize=fontsize-2)
    apply_matlab_style(ax2)

    if title:
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.text(0.5, 0.05, f'{title}', ha='center', fontsize=fontsize)
    else:
        plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}/cv_lr_curve.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/cv_lr_curve.svg', format='svg', bbox_inches='tight')

    plt.show()

def plot_confusion_matrix(cm, class_names=None, norm=False, cmap=plt.cm.Blues, task='multiclass',
                          figsize=(12, 5), fontsize=16, title=None, save_path=None):
    """
        For plotting and visualization the confusion matrix of a model.
        Args:
            - cm (numpy.ndarray): The confusion matrix.
            - class_names (list): Optional, the names of the classes.
            - norm (bool): Optional, whether to normalize the confusion matrix.
            - cmap (matplotlib.colors.Colormap): Optional, the colormap to use.
            - task (str): Optional, the task of the model (multiclass or binary).
            - figsize (tuple): Optional, the size of the figure.
            - fontsize (int): Optional, the size of the font.
            - title (str): Optional, the title of the figure.
            - save_path (str): Optional, the path to save the figure.
        Returns:
            - None
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError('Confusion matrix must be a 2D square matrix')
    n_classes = cm.shape[0]

    if task == 'binary' and n_classes != 2:
        raise ValueError('For binary classification, the confusion matrix must have shape (2, 2)')

    if class_names is None:
        if task == 'binary':
            class_names = ['Negative', 'Positive']
        else:
            class_names = [f'Class {i}' for i in range(n_classes)]

    if len(class_names) != n_classes:
        raise ValueError(f'Number of class names ({len(class_names)}) does not match the number of classes ({n_classes})')

    cm = cm.astype('float')
    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total if total > 0 else 0.0
    misclassified = 1.0 - accuracy

    if norm:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm = cm / row_sum

    suffix = ' ' + '(Normalized)' if norm else ''
    plot_title = f'{title} - Confusion Matrix{suffix}' if title else f'Confusion Matrix{suffix}'

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(plot_title, fontsize=fontsize)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=fontsize)
    plt.yticks(tick_marks, class_names, fontsize=fontsize)

    fmt = '.2f' if norm else '.0f'
    thresh = cm.max() / 2.0 if cm.size else 0.0

    for i, j in itertools.product(range(n_classes), range(n_classes)):
        plt.text(
            j, i,
            format(cm[i, j], fmt),
            ha='center',
            va='center',
            fontsize=fontsize,
            color='white' if cm[i, j] > thresh else 'black'
        )

    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel(
        f'Predicted label\nAccuracy: {accuracy:.4f} | Misclass: {misclassified:.4f}',
        fontsize=fontsize
    )
    plt.tight_layout()

    if save_path:
        suffix = 'norm' if norm else 'unnorm'
        plt.savefig(f'{save_path}/conf_matrix_{suffix}.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/conf_matrix_{suffix}.svg', format='svg', bbox_inches='tight')

    plt.show()

def plot_roc_auc_curve(y_true, y_pred_probs, class_names=None, task='multiclass',
                       legend_ncol=1, width_ratios=[3, 1], figsize=(12, 5),
                       fontsize=16, title=None, save_path=None):
    """
        For plotting and visualization the ROC AUC curve of a model.
        Args:
            - y_true (numpy.ndarray): The true labels.
            - y_pred_probs (numpy.ndarray): The predicted probabilities.
            - class_names (list): Optional, the names of the classes.
            - task (str): Optional, the task of the model (multiclass or binary).
            - legend_ncol (int): Optional, the number of columns in the legend.
            - width_ratios (list): Optional, the width ratios of the subplots.
            - figsize (tuple): Optional, the size of the figure.
            - fontsize (int): Optional, the size of the font.
            - title (str): Optional, the title of the figure.
            - save_path (str): Optional, the path to save the figure.
        Returns:
            - None
    """
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': width_ratios}, facecolor='white')

    def apply_matlab_style(ax):
        ax.grid(axis='y', color='#E5E5E5', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', color='#E5E5E5', linestyle='-', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(axis='both', colors='#333333', width=0.8, length=4)

    if task == 'binary':
        if y_true.ndim != 1:
            raise ValueError('For binary classification, y_true should be a 1D array')

        if y_pred_probs.ndim == 2:
            if y_pred_probs.shape[1] != 2:
                raise ValueError('For binary classification, y_pred_probs should have shape (n_samples, 2)')
            y_score = y_pred_probs[:, 1]
        else:
            y_score = y_pred_probs

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax_main.plot(fpr, tpr, lw=3, color='#0072BD', label=f'ROC curve (AUC = {roc_auc:.4f})') # ใช้สีฟ้า MATLAB

    else:
        # multiclass
        if y_pred_probs.ndim != 2:
            raise ValueError('For multiclass classification, y_pred_probs should be a 2D array')
        n_classes = y_pred_probs.shape[1]

        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]

        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 0.9, n_classes)]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax_main.plot(fpr, tpr, lw=2, color=colors[i], label=f'{class_names[i]} (AUC = {roc_auc:.4f})')

        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax_main.plot(fpr_micro, tpr_micro, linestyle='--', lw=2, color='#333333', # เปลี่ยนสีเทาเข้ม
                     label=f'Micro-average (AUC = {auc_micro:.4f})')

        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)

        mean_tpr /= n_classes
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0

        auc_macro = auc(all_fpr, mean_tpr)
        ax_main.plot(all_fpr, mean_tpr, label=f'Macro-average (AUC = {auc_macro:.4f})',
                     color='#D95319', linestyle=':', lw=3) # ใช้สีส้ม MATLAB

    ax_main.plot([0, 1], [0, 1], color='#CCCCCC', lw=2, linestyle='--')
    apply_matlab_style(ax_main)

    ax_main.set_xlim([0.0, 1.0])
    ax_main.set_ylim([0.0, 1.05])
    ax_main.set_xlabel('False Positive Rate', fontsize=fontsize)
    ax_main.set_ylabel('True Positive Rate', fontsize=fontsize)
    ax_main.tick_params(labelsize=fontsize)

    handles, labels = ax_main.get_legend_handles_labels()
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center left',
                     ncol=legend_ncol, fontsize=fontsize, frameon=False)

    if title:
        plot_title = f'{title} - Receiver Operating Characteristic (ROC)'
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(plot_title, fontsize=fontsize + 2, fontweight='bold', ha='center', y=0.99)
    else:
        ax_main.set_title('Receiver Operating Characteristic (ROC)', fontsize=fontsize)
        plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}/roc_auc_curve.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}/roc_auc_curve.svg', format='svg', bbox_inches='tight')

    plt.show()

def vis_images(df, filepaths_col, label_col, n_rows=2, n_cols=5, target_size=(224, 224),
               figsize=(15, 6), fontsize=14, title=None, save_path=None):
    """
        For visualizing a subset of images from a dataframe.
        Args:
            - df (pandas.DataFrame): The dataframe containing the images.
            - filepaths_col (str): The name of the column containing the filepaths.
            - label_col (str): The name of the column
            - n_rows (int): Optional, the number of rows in the grid.
            - n_cols (int): Optional, the number of columns in the grid.
            - target_size (tuple): Optional, the target size of the images.
            - figsize (tuple): Optional, the size of the figure.
            - fontsize (int): Optional, the size of the font.
            - title (str): Optional, the title of the figure.
            - save_path (str): Optional, the path to save the figure.
        Returns:
            - None
    """

    total_slots = n_rows * n_cols
    subset = df.iloc[:total_slots]
    total_images = len(subset)

    fig = plt.figure(figsize=figsize)

    if title:
        fig.suptitle(title, fontsize=fontsize + 2, fontweight='bold', y=0.97)

    img_width = 0.15
    img_height = 0.35
    spacing = 0.02

    for r in range(n_rows):
        row_start = r * n_cols
        row_end = min(row_start + n_cols, total_images)
        row_items = row_end - row_start

        if row_items <= 0:
            break

        total_row_width = row_items * img_width + (row_items - 1) * spacing
        start_x = (1.0 - total_row_width) / 2

        y_pos = 0.55 - r * 0.45

        for c in range(row_items):
            idx = row_start + c
            row = subset.iloc[idx]
            img_path = pathlib.Path(row[filepaths_col])

            ax = fig.add_axes([
                start_x + c * (img_width + spacing),
                y_pos,
                img_width,
                img_height
            ])

            img = Image.open(img_path).convert('RGB').resize(target_size)
            ax.imshow(img)

            ax.text(
                0.5, -0.1,
                f'{row[label_col]}',
                fontsize=fontsize,
                ha='center',
                va='center',
                transform=ax.transAxes
            )

            ax.axis('off')

    if save_path:
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.svg', bbox_inches='tight')

    plt.show()