# -*- coding: utf-8 -*-
"""preprocessing.ipynb

Original file is located at
    https://colab.research.google.com/drive/1PTEgSehBpoY_JbHlCyasPKOyhues0TQo
"""

import os
import pathlib
import numpy as np
from PIL import Image, ImageFile
import pickle
from tqdm import tqdm
from imagededup.methods import PHash, CNN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize
from collections import defaultdict

def verify_images(data_dir, check_solid=True):
    """
        For verify images in exists directory
            1. Check solid color image
            2. Check corrupted image
        Args:
            - data_dir: The exists images directory
            - check_solid: bool default (True)
        Return:
            - report_dict: dict
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    report_dict = {
        'filepaths': [],
        'filename': [],
        'label': [],
        'issue': []
    }

    data_dir = pathlib.Path(data_dir)
    Image.init()
    extensions = {ext.lower() for ext in Image.EXTENSION.keys()}

    img_files = []
    found_extensions = set()

    for path in data_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in extensions:
            img_files.append(path)
            found_extensions.add(path.suffix)

    if not img_files:
        print(f'[Log]: No images found in {data_dir}')
        return report_dict

    print(f"[Log]: Extensions detect {', '.join(sorted(found_extensions))}")
    print(f'[Log]: Total images found: {len(img_files)}')

    def log_issue(path, issue_type):
        report_dict['filepaths'].append(str(path))
        report_dict['filename'].append(path.name)
        report_dict['label'].append(path.parent.name)
        report_dict['issue'].append(issue_type)

    for img_path in tqdm(img_files, desc='Verifying images'):
        try:
            # Check solid color image
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')

                if check_solid:
                    extrema = img_rgb.getextrema()

                    if all(min_val == max_val for min_val, max_val in extrema):
                        log_issue(img_path, 'solidColor')

        except Exception as e:
            # Check corruption image
            log_issue(img_path, 'corrupted')

    total_issues = len(report_dict['filepaths'])
    print('\n--- Summary ---')
    print(f'Total issues found: {total_issues}')

    return report_dict

def find_hasher(data_dir, output_path=None, extensions=('.jpg', '.jpeg', '.png')):
    """
        Modify from: https://github.com/idealo/imagededup
        Find pHash for each image by phasher from Imagededup Library
        Args:
            - data_dir: The exists images directory
            - output_path: The exists destination path default (None)
            - extensions: tuple default (('.jpg', '.jpeg', '.png'))
        Return:
            - hashes_dict: dict
    """
    phasher = PHash()
    hashes_dict = {}
    target_folders = []

    for root, dirs, files in os.walk(data_dir):
        has_images = any(f.lower().endswith(extensions) for f in files)
        if has_images:
            target_folders.append(root)
    print(f'Found {len(target_folders)} folders containing images.')

    for folder_path in target_folders:
        label_name = os.path.basename(folder_path)

        try:
            raw_encodings = phasher.encode_images(image_dir=folder_path)
            safe_encodings = {}
            for filename, hash_val in raw_encodings.items():
                full_path_key = os.path.join(folder_path, filename)
                safe_encodings[full_path_key] = hash_val

            if label_name in hashes_dict:
                hashes_dict[label_name].update(safe_encodings)
            else:
                hashes_dict[label_name] = safe_encodings

        except Exception as e:
            print(f'Error processing folder {folder_path}: {e}')
            pass

    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(hashes_dict, f)
        print(f'Saved hashes for {len(hashes_dict)} unique classes to {output_path}')

    return hashes_dict

def find_duplicates(hashes_dict):
    """
        Find duplicate images by phasher
        Args:
            - hashes_dict: dict
        Return:
            - intra_report: dict
            - inter_report: dict
    """
    reverse_index = defaultdict(list)
    total_images = 0

    for label, file_dict in tqdm(hashes_dict.items(), desc='Processing Classes'):
        for full_path, hash_val in file_dict.items():
            reverse_index[hash_val].append({
                'label': label,
                'full_path': full_path
            })
            total_images += 1

    print(f'\nFound {total_images} images in total.')

    # Duplicate images within the same class. (Intra Duplicate)
    intra_report = {
        'filepaths': [], 'filename': [], 'label': [], 'issue': [], 'hash_group': [], 'keep': []
    }

    # Duplicate images between classes. (Inter Duplicate)
    inter_report = {
        'filepaths': [], 'filename': [], 'label': [], 'issue': [], 'hash_group': [], 'conflicting_labels': []
    }

    for hash_val, file_list in reverse_index.items():
        # Not found duplicate image
        if len(file_list) < 2:
            continue

        labels_found = set(item['label'] for item in file_list)
        if len(labels_found) == 1:
            # (Intra Duplicate)
            sorted_files = sorted(file_list, key=lambda x: x['full_path'])

            for i, item in enumerate(sorted_files):
                intra_report['filepaths'].append(item['full_path'])
                intra_report['filename'].append(os.path.basename(item['full_path']))
                intra_report['label'].append(item['label'])
                intra_report['issue'].append('intraDuplicate')
                intra_report['hash_group'].append(hash_val)
                intra_report['keep'].append(True if i == 0 else False) # Keep First Index (otherwise duplicate)
        else:
            # (Inter Duplicate)
            # Need verify again!
            conflicting_str = ", ".join(sorted(list(labels_found)))

            for i, item in enumerate(file_list):
                inter_report['filepaths'].append(item['full_path'])
                inter_report['filename'].append(os.path.basename(item['full_path']))
                inter_report['label'].append(item['label'])
                inter_report['issue'].append('interDuplicate')
                inter_report['hash_group'].append(hash_val)
                inter_report['conflicting_labels'].append(conflicting_str)

    return intra_report, inter_report

def find_outliers(data_dir, seed=42, contamination=0.05, extensions=('.jpg', '.jpeg', '.png')):
    """
        Modify from: https://github.com/idealo/imagededup
        IsolationForest Paper: https://doi.org/10.1145/2133360.2133363
        MobileNetV3 Paper: https://arxiv.org/abs/1905.02244

        This is an outlier checking process that considers the vector obtained from the CNN encoder and then passes the obtained vector into the IsolationForest.
        The CNN model used is from the imagededup library and checks for outliers only within the same class, seeing if there are any images within the same class that are different from the others.
        Args:
            - data_dir: The exists images directory
            - seed: int for random state default (42)
            - contamination: float default (0.05)
            - extensions: tuple default (('.jpg', '.jpeg', '.png'))
        Return:
            - report_dict: dict
    """
    cnn = CNN() # MobileNetV3 Default From imagededup library
    report_dict = {
        'filepaths': [],
        'filename': [],
        'label': [],
        'issue': [],
        'score': []
    }
    target_folders = []

    for root, dirs, files in os.walk(data_dir):
        has_images = any(f.lower().endswith(extensions) for f in files)
        if has_images:
            target_folders.append(root)

    print(f'Found {len(target_folders)} folders containing images.')

    for folder_path in target_folders:
        label_name = os.path.basename(folder_path)

        try:
            encodings = cnn.encode_images(image_dir=folder_path)
            filenames = list(encodings.keys())
            vectors = np.array(list(encodings.values()))
            vectors_norm = normalize(vectors, norm='l2')

            clf = IsolationForest(
                contamination=contamination,
                n_estimators=200, # Fixed n_estimators at 200
                random_state=seed,
                n_jobs=-1)

            clf.fit(vectors_norm)
            preds = clf.predict(vectors_norm)
            scores = clf.decision_function(vectors_norm)

            for i, pred in enumerate(preds):
                # Negative Score
                if pred == -1:
                    full_path = os.path.join(folder_path, filenames[i])

                    report_dict['filepaths'].append(full_path)
                    report_dict['filename'].append(filenames[i])
                    report_dict['label'].append(label_name)
                    report_dict['issue'].append('candidateOutlier')
                    report_dict['score'].append(scores[i])

            print(f'Class {label_name}: Found {list(preds).count(-1)} candidate outlier.')
        except Exception as e:
            print(f'Error processing folder {folder_path}: {e}')

    return report_dict