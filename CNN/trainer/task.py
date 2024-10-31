import argparse
import logging
from os import environ
from pathlib import Path
import shutil
import sys
import time
import torch

from trainer import experiment
from trainer import inputs

log = logging.getLogger(__name__)


def get_args():
    """Define the task arguments with the default values.

    Returns:
        experiment parameters
    """
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--target-name',
        help='Name of the label variable',
        nargs='?',
        required=True)
    # Saved artifacts arguments
    args_parser.add_argument(
        '--job-dir',
        help='path to export models and metrics',
        default='job_dir',
        type=Path)
    # Data files arguments
    args_parser.add_argument(
        '--data-dir',
        help='GCS bucket with label files',
        nargs='?',
        required=False,
        type = Path)
    args_parser.add_argument(
        '--label-file',
        help='Path to training data',
        nargs='?',
        required=True,
        type=Path)
    args_parser.add_argument(
        '--images-dir',
        help='Path to image files',
        nargs='?',
        required=True,
        type=Path)
    args_parser.add_argument(
        '--seed',
        help='Random seed (default: 42)',
        type=int,
        default=42)
    args_parser.add_argument(
        '--in-cloud',
        action='store_true'
    )

    # Experiment arguments
    zscore_group = args_parser.add_mutually_exclusive_group()
    zscore_group.add_argument(
        '--training-zscore',
        nargs='?',
        type=bool,
        default=False)
    zscore_group.add_argument(
        '--imagenet-zscore',
        nargs='?',
        type=bool,
        default=False)
    zscore_group.add_argument(
        '--zscore-mean',
        nargs='?',
        type=float)
    args_parser.add_argument(
        '--zscore-stddev',
        nargs='?',
        type=float)
    args_parser.add_argument(
        '--model_type',
        help='regression / classification / multiclass',
        nargs='?',
        default='regression')
    args_parser.add_argument(
        '--model_arch',
        help='Model architecture',
        nargs='?',
        default='ResNet18')
    args_parser.add_argument(
        '--pretrain_weights',
        help='Path to pretrained weights',
        type=Path)
    args_parser.add_argument(
        '--layers_to_unfreeze',
        help='Layers before the end to unfreeze',
        type=int,
        default=12)
    data_split_group = args_parser.add_mutually_exclusive_group()
    data_split_group.add_argument(
        '--data_split',
        help="Python dict of categories and split percentages",
        nargs='?',
        default="{'train': 0.7, 'validation': 0.15, 'test': 0.15}")
    data_split_group.add_argument(
        '--data_split_series',
        help="Parquet files containing a Pandas series of 'train/validation/test' values",
        nargs='?',
        type=Path)
    args_parser.add_argument(
        '--kfolds',
        help="If > 1, triggers k-folds (and ignores data_split settings)",
        type=int,
        default=1)
    args_parser.add_argument(  # When omitted, uses the index column of the label file
        '--fold_column',
        help="Column of categories that must be maintained during train/test splits",
        nargs='?',
        default=None)
    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=16)
    args_parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        """,
        default=30,
        type=int)
    args_parser.add_argument(
        '--optimizer',
        help='Optimizer type',
        nargs='?',
        default='Adam')
    args_parser.add_argument(
        '--scheduler',
        help='Scheduler type',
        nargs='?',
        default='OneCycleLR')
    args_parser.add_argument(
        '--learning-rate',
        help='Learning rate value for the optimizers.',
        default=0.001,
        type=float)
    args_parser.add_argument(
        '--model-name',
        help='The name of your saved model',
        default='model.pth')
    args_parser.add_argument(
        '--threshold',
        help='Binarization threshold. Labels at or below this value considered positive case.',
        type=float
    )
    args_parser.add_argument(
        '--weight',
        help='Class balance weight.',
        default=1.0,
        type=float
    )

    return args_parser.parse_args()


def translate_data_split(split_string):
    # TODO add code to check the string for correctness
    return eval(split_string)


def setup_logging(in_cloud):
    # Environmentally specified variables
    loglevel = environ.get("LOGLEVEL", "INFO").upper()
    if in_cloud:
        logging.basicConfig(stream=sys.stdout, level=loglevel, format="%(message)s")
    else:
        logging.basicConfig(level=loglevel, format="%(asctime)s %(levelname)s:\t%(message)s", datefmt='%I:%M:%S%p')


# Write data to msgpack file
def write_msgpack(data, filename):
    def msgpack_encoder(obj):
        if isinstance(obj, Path):
            return {'__Path__': True, 'as_str': str(obj)}
        return obj

    with open(filename, 'wb') as outfile:
        import msgpack
        import msgpack_numpy as m  # encoding/decoding routines to enable the (de)serialization of numpy data types
        m.patch()
        packed = msgpack.packb(data, default=msgpack_encoder)
        outfile.write(packed)


def process_arguments(args):
    # The cloud mounts buckets on the compute node, update to reflect
    if args.in_cloud:
        gcs_mnt = Path('/gcs')
        args.job_dir = gcs_mnt / args.job_dir
        args.data_dir = gcs_mnt / args.data_dir
        args.images_dir = gcs_mnt / args.images_dir
        if args.pretrain_weights:
            args.pretrain_weights = gcs_mnt / args.pretrain_weights
    else:  # Only mkdir if a real filesystem
        if not args.job_dir.exists():
            args.job_dir.mkdir(parents=True)

    # Hold over from when this used several data files and one directory
    args.label_file = args.data_dir / args.label_file

    args.data_split = translate_data_split(args.data_split)

    # Set variables necessary for each model type
    experiment.MODEL_TYPE = args.model_type
    if args.model_type == 'regression':
        experiment.STOPPING_CRITERIA_METRIC = 'R2'
    elif args.model_type == 'classification':
        experiment.STOPPING_CRITERIA_METRIC = 'auc'
    elif args.model_type == 'multiclass':
        experiment.STOPPING_CRITERIA_METRIC = 'acc'

    # Sanity check
    if args.kfolds < 1:
        log.error('Provided a kfolds value of < 1. WTF is that supposed to mean. Quiting in a huff.')
        exit(-1)

    # TODO add more argument sanity checks
    return args


def get_torch_device():
    cuda_availability = torch.cuda.is_available()
    if cuda_availability:
        return cuda_availability, torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        return cuda_availability, 'cpu'


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    args = process_arguments(args)
    setup_logging(args.in_cloud)

    # Copy run settings to directory
    write_msgpack(vars(args), args.job_dir / 'training_params.msgpack')
    shutil.copyfile(args.label_file, args.job_dir / args.label_file.name)

    # Stage the data for the run
    inputs.stage_data(args)

    # Setup
    args.cuda_availability, args.device = get_torch_device()
    experiment.print_run_conditions(args)

    # Need to create a sharable generator to enable parallel dataloader tasks to be repeatable
    g = torch.Generator().manual_seed(args.seed)

    # Get the label dataframe, break into folds/splits
    label_dataframe = inputs.get_label_dataframe(args)
    kfold_datasets = inputs.create_data_splits(args, list(label_dataframe[args.fold_column].unique()), g)

    # Run k times and collect training results
    results = {}
    dataloaders = None
    training_start_time = time.time()
    for fold, dataset in enumerate(kfold_datasets):
        if args.kfolds > 1:
            log.info('\n\nK-fold {}/{}\n{}'.format(fold+1, args.kfolds, '-' * 11))

        # Create our datasets
        dataloaders = inputs.create_dataloaders(args, label_dataframe, dataset, g)

        # Train a model
        results['Fold_' + str(fold)] = experiment.run(args, dataloaders, fold)
    time_elapsed = time.time() - training_start_time
    log.info("Total time taken for all folds: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    # if kfolds, process the results for a total
    if args.kfolds > 1:
        folds = results
        results = {'Folds': folds,
                   'Final': experiment.compile_kfolds_results(folds, dataloaders['train'].dataset.one_hot_labels)}
        # Print results
        for metric in experiment.get_relevant_metrics(args):
            log.info('{}: {}'.format(metric, results['Final'][metric]))
        log.info('Best epoch: {}\tResults across folds: {}={}'.format(results['Final']['best_epoch'],
                                                                      experiment.STOPPING_CRITERIA_METRIC,
                                                                      results['Final']['fold_metrics']))
        log.info('Mean and std dev across folds: {}={}, {}'.format(experiment.STOPPING_CRITERIA_METRIC,
                                                                  results['Final']['fold_stop_metric_mean'],
                                                                  results['Final']['fold_stop_metric_stddev']))

    # save results
    experiment.write_msgpack(results, args.job_dir / 'training_metrics.msgpack')

if __name__ == '__main__':
    main()
