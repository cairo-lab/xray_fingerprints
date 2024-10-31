import copy
#import hypertune
import logging
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import sklearn.metrics as skmets
import sys
import time
import torch
import pkg_resources

from trainer import inputs
from trainer import model

log = logging.getLogger(__name__)

SHOW_MEM = False
STOPPING_CRITERIA_METRIC = ''
MODEL_TYPE = None


def init_batch_stats():
    batch_stats = {
        'labels': [],
        'outputs': [],
        'image_names': [],
        'running_loss': 0.}
    return batch_stats


def write_msgpack(data, filename):
    def msgpack_encoder(obj):
        if isinstance(obj, Path):
            return {'__Path__': True, 'as_str': str(obj)}
        return obj

    with open(filename, "wb") as outfile:
        import msgpack
        import msgpack_numpy as m  # encoding/decoding routines that enable the serialization/deserialization of data types provided by numpy
        m.patch()
        packed = msgpack.packb(data, default=msgpack_encoder)
        outfile.write(packed)


def sanity_check_metrics(metrics):
    # Sanity check: no metric should be NaN
    for metric in metrics:
        if np.isnan(metrics[metric]):
            raise Exception("%s is a nan, something is weird about your predictor" % metric)


def get_relevant_metrics(args):
    if args.model_type == 'regression':
        return ['loss', 'r2']
    elif args.model_type == 'classification':
        return ['acc', 'f1', 'auc', 'tp', 'sens', 'spec']
    elif args.model_type == 'multiclass':
        return ['acc', 'adj_balanced_acc']


def regression_metrics(y, yhat):
    r = pearsonr(y, yhat)[0]
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    cc = skmets.r2_score(y, yhat)
    metrics = {'r': r, 'rmse': rmse, 'negative_rmse': -rmse, 'r^2': r ** 2, 'R2': cc}
    sanity_check_metrics(metrics)
    return metrics

def binary_class_metrics(y, yhat):
    bin_preds = 1.*(yhat >= 0.5)
    tn, fp, fn, tp = skmets.confusion_matrix(y, bin_preds).ravel()
    acc = skmets.accuracy_score(y, bin_preds)
    f1 = skmets.f1_score(y_true=y, y_pred=bin_preds)
    auc = skmets.roc_auc_score(y_score=yhat, y_true=y)
    auprc = skmets.average_precision_score(y_score=yhat, y_true=y)  # aka  AUPRC
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    metrics = {'acc': acc, 'auc': auc, 'auprc': auprc, 'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'sens': sens,
               'spec': spec}
    sanity_check_metrics(metrics)
    return metrics


def multiclass_metrics(y, yhat_score, prediction_labels):
    _, y_true = torch.from_numpy(np.float64(y)).topk(1, dim=1)
    yhat_score = torch.softmax(torch.from_numpy(np.float64(yhat_score)), dim=1)
    _, yhat_pred = yhat_score.topk(1, dim=1)
    labels = range(len(prediction_labels))
    report = skmets.classification_report(y_true, yhat_pred, labels=labels, target_names=prediction_labels,
                                          output_dict=True, zero_division=0.0)
    confusion_matrix = skmets.confusion_matrix(y_true, yhat_pred, labels=labels)
    balanced_acc = skmets.balanced_accuracy_score(y_true, yhat_pred)
    adj_balanced_acc = skmets.balanced_accuracy_score(y_true, yhat_pred, adjusted=True)
    if 'accuracy' not in report:
        report['accuracy'] = (np.argmax(yhat_score, 1) == np.argmax(y, 1)).mean()
    metrics = {'acc': report['accuracy'],
               'class_report': report,
               'confusion_matrix': confusion_matrix,
               'balanced_acc': balanced_acc,
               'adj_balanced_acc': adj_balanced_acc,
               }
    return metrics

def assess_performance(y, yhat, prediction_labels):
    """
    Return standard metrics of performance give y and yhat. yhat should never be binary, even for binarized scores.
    """
    if MODEL_TYPE == 'regression':
        return regression_metrics(y, yhat)
    elif MODEL_TYPE == 'classification':
        return binary_class_metrics(y, yhat)
    elif MODEL_TYPE == 'multiclass':
        return multiclass_metrics(y, yhat, prediction_labels)


def calc_epoch_loss(batch_stats, dataset_sizes, prediction_labels, save_names=False):
    metrics_for_epoch = {}
    epoch_loss = batch_stats['running_loss'] / dataset_sizes

    outputs = np.array(batch_stats['outputs'])
    labels = np.array(batch_stats['labels'])

#    assert len(outputs) == dataset_sizes
#    log.info('%s epoch loss: %2.4f; RMSE %2.4f; correlation %2.4f (n=%i); CC %2.4f' %
#             (target_name, epoch_loss, correlation_and_rmse['rmse'], correlation_and_rmse['r'], len(labels), correlation_and_rmse['R2']))

    metrics = assess_performance(labels, outputs, prediction_labels)
    metrics_for_epoch['epoch_loss'] = epoch_loss
    metrics_for_epoch['yhat'] = outputs
    metrics_for_epoch['y'] = labels
    for metric in metrics:
        metrics_for_epoch[metric] = metrics[metric]

    if save_names:
        metrics_for_epoch['image_names'] = batch_stats['image_names']
        metrics_for_epoch['targets'] = np.array(batch_stats['targets'])
        metrics_for_epoch['preds'] = np.array(batch_stats['preds'])
    return metrics_for_epoch


def forward_pass(cnn_model, loss_criterion, data, batch_stats, device, save_names=False):
    # Send data to GPU
    image = data['image'].to(device, non_blocking=True)
    target = data['target'].to(device, non_blocking=True)
#    weight = data['weight'].to(device, non_blocking=True)
    if SHOW_MEM:
        log.info("data: {:.1f}MB".format(torch.cuda.memory_allocated(device) / 1024 ** 2))

    # Score batch
    outputs = cnn_model(image)
    if SHOW_MEM:
        log.info("forward: {:.1f}MB".format(torch.cuda.memory_allocated(device) / 1024 ** 2))

    if outputs.shape == target.shape:
        loss = loss_criterion(input=outputs, target=target) # binary
    else:
        loss = loss_criterion(input=outputs, target=target.squeeze()) #, weight=weight)  # multiclass
    if SHOW_MEM:
        log.info("loss: {:.1f}MB".format(torch.cuda.memory_allocated() / 1024 ** 2))

    # keep track of everything for correlations
    batch_stats['labels'] += list(target.data.cpu().numpy().squeeze())
    batch_stats['outputs'] += list(outputs.data.cpu().to(torch.float16).numpy().squeeze())
    if save_names:
        batch_stats['image_names'] += list(data['name'])
        targets = data['target']
        if 'targets' in batch_stats: # we want everything
            batch_stats['targets'] = torch.cat((batch_stats['targets'], targets), dim=0)
        else:
            batch_stats['targets'] = targets
        if 'preds' in batch_stats:
            batch_stats['preds'] = torch.cat((batch_stats['preds'], outputs.data.cpu().to(torch.float16)), dim=0)
        else:
            batch_stats['preds'] = outputs.data.cpu().to(torch.float16)
    return loss, batch_stats


def train(cnn_model, train_loader, criterion, scheduler, optimizer, device):
    """Create the training loop for one epoch. Read the data from the
     dataloader, calculate the loss, and update the DNN. Lastly, display some
     statistics about the performance of the DNN during training.

    Args:
      cnn_model: The neural network that you are training, based on nn.Module
      train_loader: The training dataset
      criterion: The loss function used during training
      optimizer: The selected optmizer to update parameters and gradients
      device: GPU or CPU
    """
    log.info('')
    log.info('Train:')
    cnn_model.train()

    # keep track of all labels + outputs to compute the final metrics.
    batch_stats = init_batch_stats()

    dl_time = 0
    dl_start = time.time()
    for batch_index, data in enumerate(train_loader):
        dl_time += time.time() - dl_start

        # forward
        optimizer.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        loss, batch_stats = forward_pass(cnn_model, criterion, data, batch_stats, device)
        # backward + optimize
        loss.backward()
        if SHOW_MEM:
            log.info("backward: {:.1f}MB".format(torch.cuda.memory_allocated(device) / 1024 ** 2))

        optimizer.step()
        scheduler.step()
        if SHOW_MEM:
            log.info("step: {:.1f}MB".format(torch.cuda.memory_allocated(device) / 1024 ** 2))
        dl_start = time.time()
        batch_stats['running_loss'] += loss.item() * data['image'].size(0)

    log.info("Total seconds taken for dataloader: {:2.4f}".format(dl_time))
    return calc_epoch_loss(batch_stats, train_loader.dataset.__len__(), train_loader.dataset.one_hot_labels)


def evaluate(cnn_model, test_loader, criterion, scheduler, epoch, device, report_metric=False, save_names=False):
    """Test / validate the DNNs performance with a test / val dataset.
     Read the data from the dataloader and calculate the loss. Lastly,
     display some statistics about the performance of the DNN during testing.

    Args:
      cnn_model: The neural network that you are testing
      test_loader: The test / validation dataset
      criterion: The loss function
      scheduler: The Learning rate scheduler
      epoch: The current epoch that the training loop is on
      device: GPU or CPU
      report_metric: Whether to report metrics for hyperparameter tuning
      save_names: Bool flag on whether image names should be saved in the epoch stats
    """
    log.info('')
    log.info('Validate:')
    cnn_model.eval()

    # keep track of all labels + outputs to compute the final metrics.
    batch_stats = init_batch_stats()

    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            loss, batch_stats = forward_pass(cnn_model, criterion, data, batch_stats, device, save_names)
            batch_stats['running_loss'] += loss.item() * data['image'].size(0)

#    if report_metric:
#        # Uses hypertune to report metrics for hyperparameter tuning.
#        hpt = hypertune.HyperTune()
#        hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='test_loss',
#                                                metric_value=batch_stats['running_loss'] / test_loader.dataset.__len__(),
#                                                global_step=epoch)

    metrics_for_test = calc_epoch_loss(batch_stats, test_loader.dataset.__len__(), test_loader.dataset.one_hot_labels, save_names)

#    if scheduler:
        # Change the learning rate.
#        scheduler.step()
#        scheduler.step(metrics_for_test[STOPPING_CRITERIA_METRIC])

    return metrics_for_test


def print_layer_magnitudes(cnn_model, layer_magnitudes):
    # small helper method so we can make sure the right layers are being trained.
    log.debug("\n***\nPrinting layer magnitudes")
    log.debug('Name\t\tMagnitude\tDelta (from prior epoch)')
    for name, param in cnn_model.named_parameters():
        magnitude = np.linalg.norm(param.data.cpu())
        if param not in layer_magnitudes:
            layer_magnitudes[param] = magnitude
            log.debug("%s:\t\t%2.5f" % (name, magnitude))
        else:
            old_magnitude = layer_magnitudes[param]
            delta_magnitude = magnitude - old_magnitude
            log.debug("%s:\t\t%2.5f\t%2.5f" % (name, magnitude, delta_magnitude))
            layer_magnitudes[param] = magnitude


def print_run_conditions(args):
    log.info('Python ver.:{}'.format(sys.version))
    log.info('Package vers:\n{}'.format({pkg.key: pkg.version for pkg in pkg_resources.working_set}))
    if hasattr(args, 'region'):
        log.info('Region: {}'.format(args.region))
        log.info('Image URI: {}'.format(args.master_image_uri))
        log.info('Machine type: {}'.format(args.master_machine_type))
        log.info('Accelerator: {}'.format(args.master_accelerator))
        log.info('')
    log.info('*************************')
    log.info('`cuda` available: {}'.format(args.cuda_availability))
    log.info('Current Device: {}'.format(args.device))
    log.info('*************************')
    log.info('')
    log.info('Job Dir: {}'.format(args.job_dir))
    log.info('Label file: {}'.format(args.label_file))
    log.info('Images dir: {}'.format(args.images_dir))
    log.info('Seed: {}'.format(args.seed))
    log.info('NUM_WORKERS_TO_USE: {}'.format(inputs.NUM_WORKERS_TO_USE))
    log.info('Num epochs: {}'.format(args.num_epochs))
    log.info('Batch size: {}'.format(args.batch_size))
    log.info('Learning Rate: {}'.format(args.learning_rate))
    if MODEL_TYPE == 'classification':
        log.info('BINARIZATION_THRESH: <= {}'.format(args.threshold))
    log.info('STOPPING_CRITERIA_METRIC: {}'.format(STOPPING_CRITERIA_METRIC))
    log.info('CONV_LAYERS_BEFORE_END_TO_UNFREEZE: {}'.format(args.layers_to_unfreeze))
    #log.info(': {}'.format(args.))


def print_metric_hist(all_metrics, metric='rmse'):
    train, val = [], []
    for v in all_metrics:  # loop across epochs
        train.append(v['train'][metric])
        val.append(v['val'][metric])

    log.info('Train {}: {}'.format(metric, train))
    log.info('Val {}: {}'.format(metric, val))

def print_relevant_metrics(metric_values, args):
    log.info('')
    metrics = get_relevant_metrics(args)
    for metric in metrics:
        print_metric_hist(metric_values, metric)


def run(args, dataloaders, fold):
    """Load the data, train, evaluate, and export the model for serving and
     evaluating.

    Args:
      args: experiment parameters.
      dataloaders: a dict mapping dataset name (e.g. train) to dataloader
      fold: the fold number in a series of k-fold runs
    """
    # Create the model, loss function, and optimizer
    train_len = len(dataloaders['train'])
    shape = dataloaders['train'].dataset.targets.shape
    if len(shape) == 1:
        out_features = 1
    else:
        out_features = shape[1]
    cnn_model, criterion, optimizer, scheduler = model.create(args, train_len, out_features)
    torch.autograd.set_detect_anomaly(True)

    if fold == 0:
        model.print_model_stats(cnn_model, args)

    # Init metrics
    best_model_wts = copy.deepcopy(cnn_model.state_dict())
    best_metric_val = -np.inf
    best_epoch = 0
    all_metrics, metrics_for_epoch, layer_magnitudes = [], {}, {}

    # Train / Validate the model
    log.info('')
    log.info('Begin training')
    training_start_time = time.time()
    for epoch in range(0, args.num_epochs):
        log.info('\n\nK-fold {}/{}\tEpoch {}/{}\n{}'.format(fold+1, args.kfolds, epoch+1, args.num_epochs, '-' * 11))
        epoch_t0 = time.time()
        metrics_for_epoch = {'train': {}, 'val': {}}

        # zero the parameter gradients
#        optimizer.zero_grad(set_to_none=True)

        metrics_for_epoch['train'] = train(cnn_model, dataloaders['train'], criterion, scheduler, optimizer, args.device)
        metrics_for_epoch['val'] = evaluate(cnn_model, dataloaders['validation'], criterion, scheduler, epoch, args.device, report_metric=True)

        log.info("Current learning rate after epoch {} is {}".format(epoch, optimizer.param_groups[0]['lr']))

        if metrics_for_epoch['val'][STOPPING_CRITERIA_METRIC] > best_metric_val:
            best_metric_val = metrics_for_epoch['val'][STOPPING_CRITERIA_METRIC]
            best_model_wts = copy.deepcopy(cnn_model.state_dict())
            best_epoch = epoch

        all_metrics.append(metrics_for_epoch)

#        print_layer_magnitudes(cnn_model, layer_magnitudes)
        print_relevant_metrics(all_metrics, args)
        log.info("Total seconds taken for epoch: %2.4f" % (time.time() - epoch_t0))

    log.info('End training')
    time_elapsed = time.time() - training_start_time
    log.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if 'test' in dataloaders:
        # Evaluate the model
        cnn_model.load_state_dict(best_model_wts)
        cnn_model.eval()
        log.info('')
        log.info("Test the model using the holdout test dataset, and  weights from epoch: {}".format(best_epoch))
        test_metrics = evaluate(cnn_model, dataloaders['test'], criterion, None, args.num_epochs, args.device,
                                        report_metric=False, save_names=True)
        for metric in test_metrics:
            if metric not in ['image_names', 'yhat', 'y', 'preds', 'targets']:
                log.info('{}: {}'.format(metric, test_metrics[metric]))
        write_msgpack(test_metrics, args.job_dir / ('test_metrics_' + str(fold) + '.msgpack'))

    # Export the trained model
    model_path = args.job_dir / (args.model_name + '_f' + str(fold) + 'e' + str(best_epoch))
    torch.save(best_model_wts, model_path)
    log.info('Model saved to: {}'.format(model_path))

    return all_metrics


# Keep in mind that K-fold results are not the best result of k runs, but combines the
# validation results of each fold (a mean for each metric weighted by validation set sizes)
def compile_kfolds_results(results, prediction_labels):
    # Create a dataframe of STOPPING_CRITERIA_METRIC values for each folds vs epochs
    folds = []
    for fold, fold_results in results.items():
        epochs = []
        for epoch_results in fold_results:
            epochs.append(epoch_results['val'][STOPPING_CRITERIA_METRIC])
        folds.append(epochs)
    metric_across_folds_epochs = pd.DataFrame(folds, columns=range(len(folds[0])))

    # Find the best epoch for early stopping across all folds
    # Across all folds which epoch scored highest?
    # This doesn't weight them by fold size but that is probably a more honest
    best_epoch = metric_across_folds_epochs.sum().idxmax()
    fold_metrics = metric_across_folds_epochs[best_epoch]

    # Combine the validation results from each fold
    y, yhat = [], []
    for fold, fold_results in results.items():
        # the best overall epoch
        y.append(fold_results[best_epoch]['val']['y'])
        yhat.append(fold_results[best_epoch]['val']['yhat'])

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    results = assess_performance(y, yhat, prediction_labels)
    results['best_epoch'] = best_epoch
    results['fold_metrics'] = list(fold_metrics)
    results['fold_stop_metric_mean'] = fold_metrics.mean()
    results['fold_stop_metric_stddev'] = fold_metrics.std()
    results['y'] = y
    results['yhat'] = yhat
    results['labels'] = prediction_labels
    return results