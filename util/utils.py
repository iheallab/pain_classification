import gc

import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing.dummy import Pool
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.data import SensorDataset
from sklearn.metrics import roc_curve
import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
from os import mkdir, getcwd
from models.CNN import IMU_CNN
from models.CNN_LSTM import IMU_CNNLSTM
from models.CNN_Transformers import IMU_CNNTransformers
from models.CNN_Transformers_skip import CNN_Transf_Skip
import numpy as np
import sys
import pandas as pd
import time
from scipy import stats as st
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score
from sklearn.metrics import roc_auc_score
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


def plot_loss_func(sample_count, loss_vals, loss_fig_path):
    plt.figure()
    plt.plot(sample_count, loss_vals)
    plt.grid()
    plt.title('Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)


def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")


def plot_cf(conf_matrix):
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(20, 20), cmap=plt.cm.Greens,
                                    colorbar=True,
                                show_absolute=False,
                                show_normed=True)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    #plt.show()
    plt.savefig('/home/jsenadesouza/DA-healthy2patient/results/EMBC/conf.png')


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


def get_metrics(y_true, y_pred, n_classes):
    roc = 0
    if n_classes == 2:
        roc = roc_auc_score(y_true, y_pred, average="macro")
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1-score': f1_score(y_true, y_pred, average=None, labels=[0,1]),
        'f1-score_macro': f1_score(y_true, y_pred, average="macro", labels=[0, 1]),
        'recall': recall_score(y_true, y_pred, average=None, zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average="macro", zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'confusion_matrix_norm_true': confusion_matrix(y_true, y_pred, normalize='true'),
        'precision': precision_score(y_true, y_pred, average=None, zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average="macro", zero_division=0),
        'roc_auc': roc
    }


def print_metrics(logger, n_classes, cum_acc, cum_recall, cum_precision, cum_auc, cum_f1, cum_recall_macro, cum_precision_macro,
                  cum_f1_macro):
    current_acc = np.array(cum_acc)
    current_auc = np.array(cum_auc)
    current_recall_macro = np.array(cum_recall_macro)
    current_prec_macro = np.array(cum_precision_macro)
    current_f1_macro = np.array(cum_f1_macro)

    ci_mean = st.t.interval(0.95, len(current_acc) - 1, loc=np.mean(current_acc), scale=st.sem(current_acc))
    ci_recall_macro = st.t.interval(0.95, len(current_recall_macro) - 1, loc=np.mean(current_recall_macro),
                                    scale=st.sem(current_recall_macro))
    ci_prec_macro = st.t.interval(0.95, len(current_prec_macro) - 1, loc=np.mean(current_prec_macro),
                                  scale=st.sem(current_prec_macro))
    ci_f1_macro = st.t.interval(0.95, len(current_f1_macro) - 1, loc=np.mean(current_f1_macro),
                                scale=st.sem(current_f1_macro))

    logger.info('accuracy: {:.2f} ± {:.2f}\n'.format(np.mean(current_acc) * 100,
                                                          abs(np.mean(current_acc) - ci_mean[0]) * 100))

    logger.info('recall_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall_macro) * 100,
                                                              abs(np.mean(current_recall_macro) - ci_recall_macro[
                                                                  0]) * 100))
    logger.info('precision_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec_macro) * 100,
                                                                 abs(np.mean(current_prec_macro) - ci_prec_macro[
                                                                     0]) * 100))
    logger.info('f1-score_macro: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1_macro) * 100,
                                                                abs(np.mean(current_f1_macro) - ci_f1_macro[
                                                                    0]) * 100))
    if n_classes == 2:
        for class_ in range(n_classes):
            ci_auc = st.t.interval(0.95, len(current_auc) - 1, loc=np.mean(current_auc), scale=st.sem(current_auc))
            logger.info('roc_auc: {:.2f} ± {:.2f}\n'.format(np.mean(current_auc) * 100,
                                                            abs(np.mean(current_auc) - ci_auc[0]) * 100))
            logger.info(f"Class: {class_}")

            current_f1 = np.array(cum_f1)[:, class_]
            current_recall = np.array(cum_recall)[:, class_]
            current_prec = np.array(cum_precision)[:, class_]

            ci_f1 = st.t.interval(0.95, len(current_f1) - 1, loc=np.mean(current_f1), scale=st.sem(current_f1))
            ci_recall = st.t.interval(0.95, len(current_recall) - 1, loc=np.mean(current_recall),
                                      scale=st.sem(current_recall))
            ci_prec = st.t.interval(0.95, len(current_prec) - 1, loc=np.mean(current_prec), scale=st.sem(current_prec))

            logger.info('recall: {:.2f} ± {:.2f}\n'.format(np.mean(current_recall) * 100,
                                                                abs(np.mean(current_recall) - ci_recall[0]) * 100))
            logger.info('precision: {:.2f} ± {:.2f}\n'.format(np.mean(current_prec) * 100,
                                                                   abs(np.mean(current_prec) - ci_prec[0]) * 100))
            logger.info('f1-score: {:.2f} ± {:.2f}\n'.format(np.mean(current_f1) * 100,
                                                             abs(np.mean(current_f1) - ci_f1[0]) * 100))

    else:
        for pair, rocauc in current_auc.items():
            ci_pair_auc = st.t.interval(0.95, len(rocauc) - 1, loc=np.mean(rocauc), scale=st.sem(rocauc))
            logger.info('roc_auc {}: {:.2f} ± {:.2f}\n'.format(pair, np.mean(rocauc) * 100,
                                                            abs(np.mean(rocauc) - ci_pair_auc[0]) * 100))
                #


def validation(model, loader, device, criterion, n_classes, use_cuda=True):
    model.eval()
    y_true = []
    y_pred = []
    loss_total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)

            loss = criterion(preds, targets)

            loss_total += loss.item()

            if use_cuda:
                targets = targets.cpu()

            preds = torch.argmax(preds, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    return loss_total / len(loader), get_metrics(y_true, y_pred, n_classes)


def create_output_dir(name):
    out_dir = join("/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/IMUTransformers/", name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger():
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime()), ".log"])

        # Creating logs' folder is needed
        log_path = create_output_dir('out')

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)


def set_model(config, device, n_classes):
    if config.get("use_model") == "cnn":
        model = IMU_CNN(config, n_classes).to(device)
    elif config.get("use_model") == "cnn_transformer":
        model = IMU_CNNTransformers(config, n_classes).to(device)
    elif config.get("use_model") == "lstm":
        model = IMU_CNNLSTM(config).to(device)
    elif config.get("use_model") == "skip":
        print("Using skip model")
        model = CNN_Transf_Skip(config).to(device)
    else:
        raise ValueError("Model not supported")

    return model


def set_patient_map():
    # create a map between the subject_deiden_id and the patient id
    patient_map = {}
    patient_enrollment = pd.read_excel('/data/daily_data/patient_id_mapping.xlsx', engine='openpyxl')

    for row in patient_enrollment.itertuples():
        patient_map[row.patient_id] = row.subject_deiden_id

    return patient_map


def split_folders(y, experiment):
    folds_pat = []
    if experiment == "zero":
        folds_pat.append(
            ['P023', 'P013', 'I051A', '49', '48', '100', 'I045A', 'P051', 'I028A', '112', '92', '83', 'P037', '22', '29',
             '35', '64', '58', 'P046', 'I001A', 'I034A'])
        folds_pat.append(
            ['I021A', 'I043A', 'I019A', 'I044A', 'I008A', '51', '106', 'P004', 'P007', '82', '69', 'P054', 'P055', '63',
             '41', '4', '89', 'I027A', 'P024', '17'])
        folds_pat.append(
            ['P038', 'P021', 'P017', 'P067', 'I033A', 'I050A', 'P052', '40', 'P015', '109', '90', 'I049A', '103', '14',
             '81', '60', '20', 'I047A', 'P006', '32'])
        folds_pat.append(
            ['I052A', '98', 'P029', 'I006A', 'I037A', 'P057', 'I004A', 'P042', 'I018A', 'P028', '25', 'P003', 'I025A', '39',
             '52', '28', '18', 'I053A', 'I023A', '8'])
        folds_pat.append(
            ['P010', 'I026A', '95', 'I042A', 'P063', '88', '66', '50', '93', '87', '65', '44', 'I022A', 'P070', '47', '26',
             '75', 'P009', '13', '15'])
    elif experiment == "mildxsevere":
        folds_pat.append(
            ['I052A', 'P003', 'P007', 'P055', 'I008A', 'I026A', 'I052A', '32', 'P024', 'P010'])
        folds_pat.append(
            ['P023', 'P015', 'I021A', 'P037', 'I018A', 'P017', 'P023', '13', 'P006', 'P009', 'I047A'])
        folds_pat.append(
            ['P038', '17', 'P057', 'P054', 'I019A', 'P038', 'I034A', 'I023A', 'I001A', 'I027A'])
        folds_pat.append(
            ['18', '15'])
        folds_pat.append(
            ['28', 'I050A', 'I053A', '8', 'P046'])

    k = len(folds_pat)
    folds_idx = [[] for x in range(k)]
    for i in range(k):
        for pat in folds_pat[i]:
            idxs = np.where(y[:, -1] == pat)[0]
            folds_idx[i].extend(list(idxs))

    folders = [[[],[]] for x in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                folders[i][0].extend(folds_idx[j])
            else:
                folders[i][1].extend(folds_idx[j])

            assert set(folders[i][0]).isdisjoint(set(folders[i][1]))  # train and test has to be disjoint
    return folders


def get_labels(X, y, y_target, prev_pain, experiment):
    if experiment == "zero":
        print(np.unique(y_target, return_counts=True))
        prev_pain_t = np.array([0 if x == 0 else 1 for x in prev_pain])
        yy_t = np.array([0 if x == 0 else 1 for x in y_target])
        num_folders = 5
        n_classes = 2
    elif experiment == "mildxsevere":
        idx = np.where(y_target != 0)
        print(np.unique(y_target[idx], return_counts=True))
        prev_pain_t = np.array([0 if x <= 5 else 1 for x in prev_pain[idx]])
        yy_t = np.array([0 if x <= 5 else 1 for x in y_target[idx]])
        X = np.asarray(X[idx])
        y = np.asarray(y[idx])
        num_folders = 5
        n_classes = 2
    else:
        sys.exit("Experiment not supported")
    return X, y, yy_t, prev_pain_t, num_folders, n_classes


def Find_Optimal_Cutoff(target, predicted):  # Youden index
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]


    return list(roc_t['threshold'])


def split_data(X, y, y_target, labels2idx, patient_splits, folder_idx, logger=None):
    # split samples based on patients k-fold cross validation
    test_index, val_index, train_index = [], [], []
    for patient in patient_splits[folder_idx]:
        test_index.extend(list(np.where(y[:, -1] == patient)[0]))
    for patient in patient_splits[folder_idx + 1]:
        val_index.extend(list(np.where(y[:, -1] == patient)[0]))
    train_index = np.setdiff1d(np.arange(y.shape[0]), np.concatenate([test_index, val_index]))

    train_data, train_labels = X[train_index].squeeze(), y_target[train_index].squeeze()
    test_data, test_labels = X[test_index].squeeze(), y_target[test_index].squeeze()
    val_data, val_labels = X[val_index].squeeze(), y_target[val_index].squeeze()

    train_labels = np.array([labels2idx[label] for label in train_labels])
    test_labels = np.array([labels2idx[label] for label in test_labels])
    val_labels = np.array([labels2idx[label] for label in val_labels])

    if logger:
        logger.info(f"Folder {folder_idx + 1}")
        logger.info(f"Train data: {get_class_distribution(np.unique(train_labels, return_counts=True))}")
        logger.info(f"Test data: {get_class_distribution(np.unique(test_labels, return_counts=True))}")
        logger.info(f"Val data: {get_class_distribution(np.unique(val_labels, return_counts=True))}")

    return train_data, train_labels, test_data, test_labels, val_data, val_labels


def get_loaders(batch_size, sample_start, train_data, train_labels, test_data, test_labels, val_data=None, val_labels=None, weighted_sampler=False):
    train_set = SensorDataset(train_data, train_labels, sample_start, dataaug=True)
    test_set = SensorDataset(test_data, test_labels, sample_start)
    if val_data is not None and val_labels is not None:
        val_set = SensorDataset(val_data, val_labels, sample_start)

    if weighted_sampler:
        class_sample_count = np.array(
            [len(np.where(train_labels == t)[0]) for t in np.arange(0, len(np.unique(train_labels)))])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  pin_memory=True, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)

    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True)
    if val_data is not None and val_labels is not None:
        val_loader = DataLoader(val_set, batch_size=1, pin_memory=True)

        return train_loader, test_loader, val_loader
    else:
        return train_loader, test_loader


def load_data(filepath, clin_variable_target):
    # Load data
    dataset = np.load(filepath, allow_pickle=True)
    X = dataset["X"]
    y = dataset["y"]
    y_col_names = list(dataset['y_col_names'])
    col_idx_target = y_col_names.index(clin_variable_target)

    X, y = clean(X, y, col_idx_target)
    y_target = y[:, col_idx_target]#.astype(float)

    # not using pain level 0
    # idxs = np.where(y_target != 0)[0]
    # X_new, y_new, y_target_new = X[idxs], y[idxs], y_target[idxs]
    #
    # y_classes = [0 if yy < 5 else 1 for yy in y_target_new]
    # y_classes = np.array(y_classes)

    return X, y, y_target, y_col_names


def magnitude(sample):
    mag_vector = []
    for s in sample:
        mag_vector.append(math.sqrt(sum([s[0] ** 2, s[1] ** 2, s[2] ** 2])))
    return mag_vector


def load_data_mag(filepath, clin_variable_target):
    # Load data
    dataset = np.load(filepath, allow_pickle=True)
    X = dataset["X"][:500,:,:]
    y = dataset["y"][:500]
    y_col_names = list(dataset['y_col_names'])
    col_idx_target = y_col_names.index(clin_variable_target)

    X, y = clean(X, y, col_idx_target)
    y_target = y[:, col_idx_target]

    X_trasp = np.transpose(np.squeeze(X), (0, 1, 2))
    print("Extracting Features")
    start = time.time()
    with Pool(200) as p:
        X_feat = p.map(magnitude, X_trasp)
    end = time.time()
    print(f"{end - start:.4} seconds passed.")
    X = np.array(X_feat)

    return X, y, y_target


def rescale(x):
    x = (x - np.min(x) / (np.max(x) - np.min(x))) - 0.5
    return x


def clean(X, y, col_target):
    if '-1' in np.unique(y[:, col_target]):
        idxs = np.argwhere(np.array(y[:, col_target]) != '-1')
        X = X[idxs]
        y = y[idxs]
    return np.squeeze(X), np.squeeze(y)


def set_logger(filename):
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logging
    logger = logging.getLogger()
    logger.addHandler(console)

    return logger


def get_class_distribution(class_info):
    names, quant = class_info
    str = ""
    for name, q in zip(names, quant):
        str += f"{name}: {q/sum(quant)*100:.2f}% "
    return str
