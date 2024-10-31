import logging
import numpy as np
from os import putenv
import pandas as pd
from pathlib import Path
import random
import shutil
import subprocess

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms


log = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STDDEV = [0.229, 0.224, 0.225]
NUM_WORKERS_TO_USE = 8
LOCAL_DIR = Path('/tmp/images')


class ImageDataset(Dataset):
    def __init__(self, args, label_dataframe, transform=None):
        """
        Args:
            args: arguments passed to the python script
            label_dataframe: dataframe with annotations
            transform (callable, optional): Optional transform to be applied to a sample.
        """
        self.target_name = args.target_name
        self.dataframe = label_dataframe[[self.target_name]].dropna().copy()
        self.images_path = LOCAL_DIR
        self.transform = transform
        self.threshold = args.threshold
        self.zscore_mean, self.zscore_stddev = None, None
        if args.zscore_mean and args.zscore_stddev:
            self.zscore_mean, self.zscore_stddev = args.zscore_mean, args.zscore_stddev
        if args.model_type == 'multiclass':
            tmp = pd.get_dummies(self.dataframe)
            self.one_hot_labels = list(tmp.columns)
            self.targets = np.array(tmp)
        elif args.model_type == 'classification':
            self.dataframe[self.target_name] = self.dataframe[self.target_name] <= args.threshold
            self.one_hot_labels = None
            self.targets = np.array(self.dataframe[self.target_name], dtype=int)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_name = self.dataframe.index[idx]
        target = np.array([self.targets[idx]])
        image = np.load(self.images_path / (image_name + '.npy'))
        image = np.broadcast_to(image, (3,) + image.shape).copy()  # create 3 channels

        if self.transform:  # TODO: refactor so transforms are all aggregated then applied
            image = self.transform(image)

        if self.zscore_mean:
            if isinstance(self.zscore_mean, list):
                normalize = transforms.Normalize(mean=self.zscore_mean, std=self.zscore_stddev)
            else:
                normalize = transforms.Normalize(mean=[self.zscore_mean]*3, std=[self.zscore_stddev]*3)
            image = normalize(torch.from_numpy(image).float())
        else:
            image = torch.from_numpy(image).float()

        # Load the data as a tensor
        item = {
            'name': image_name,
            'image': image,
            'target': torch.from_numpy(target).float(),
        }

        return item

    def set_zscore(self, mean, stddev):
        self.zscore_mean = mean
        self.zscore_stddev = stddev


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def random_horizontal_vertical_translation(img, max_horizontal_translation, max_vertical_translation):
    """
    Translates the image horizontally/vertically by a fraction of its width/length.
    To keep the image the same size + scale, we add a background color to fill in any space created.
    TODO: migrate this to a PyTorch transform
    """
    assert max_horizontal_translation >= 0 and max_horizontal_translation <= 1
    assert max_vertical_translation >= 0 and max_vertical_translation <= 1
    if max_horizontal_translation == 0 and max_vertical_translation == 0:
        return img

    img = img.copy()

    assert len(img.shape) == 3
    assert img.shape[0] == 3
    assert img.shape[1] >= img.shape[2]

    height = img.shape[1]
    width = img.shape[2]

    translated_img = img
    horizontal_translation = int((random.random() - .5) * max_horizontal_translation * width)
    vertical_translation = int((random.random() - .5) * max_vertical_translation * height)
    background_color = img[:, -10:, -10:].mean(axis=1).mean(axis=1)

    # first we translate the image.
    if horizontal_translation != 0:
        if horizontal_translation > 0:
            translated_img = translated_img[:, :,
                             horizontal_translation:]  # this cuts off pixels on the left of the image
        else:
            translated_img = translated_img[:, :,
                             :horizontal_translation]  # this cuts off pixels on the right of the image

    if vertical_translation != 0:
        if vertical_translation > 0:
            translated_img = translated_img[:, vertical_translation:, :]  # this cuts off pixels on the top of the image
        else:
            translated_img = translated_img[:, :vertical_translation,
                             :]  # this cuts off pixels on the bottom of the image.

    # then we keep the dimensions the same.
    new_height = translated_img.shape[1]
    new_width = translated_img.shape[2]
    new_image = []
    for i in range(3):  # loop over RGB
        background_square = np.ones([height, width]) * background_color[i]
        if horizontal_translation < 0:
            if vertical_translation < 0:
                # I don't really know if the signs here matter all that much
                # -- it's just whether we're putting the translated images on the left or right.
                background_square[-new_height:, -new_width:] = translated_img[i, :, :]
            else:
                background_square[:new_height, -new_width:] = translated_img[i, :, :]
        else:
            if vertical_translation < 0:
                background_square[-new_height:, :new_width] = translated_img[i, :, :]
            else:
                background_square[:new_height, :new_width] = translated_img[i, :, :]
        new_image.append(background_square)
    new_image = np.array(new_image)

    return new_image


def random_translation(image):
    image = random_horizontal_vertical_translation(image, 0.1, 0.1)
    return image


def check_for_missing_files(expected_files, actual_files):
    # Sanity check, are all the files we expect present?
    missing_files = expected_files - actual_files
    if len(missing_files) > 0:
        log.info('X-ray files missing: {}'.format(missing_files))
        logging.shutdown()
        exit(34)  # Pick a strange number so you know the source


def cloud_happy_mkdir():
    if not LOCAL_DIR.exists():
        log.info("{} doesn't exist, making...".format(LOCAL_DIR))
        LOCAL_DIR.mkdir(parents=True)
        ls_path = shutil.which('ls')
        res = subprocess.run(ls_path + ' -la ' + str(LOCAL_DIR), shell=True, capture_output=True)
        if res.stdout:
            log.info('ls after making dir\n{}'.format(res.stdout.decode('UTF-8')))
        if res.stderr:
            log.info('ls error\n{}'.format(res.stderr.decode('UTF-8')))


def copy_image_files_to_local_storage(args):
    if args.in_cloud:
        gsutil_path = shutil.which('gsutil')
        putenv('CLOUDSDK_PYTHON', shutil.which('python3'))  # eliminate complaint about missing pythonjsonlogger
        res = subprocess.run(gsutil_path + ' -mq cp "gs://' + str(Path(*args.images_dir.parts[2:])) + '/*.npy" ' + str(LOCAL_DIR) + '/', shell=True, capture_output=True)
        if res.returncode != 0:
            log.info('gsutil failed')
            if res.stdout:
                log.info('File copy output\n{}'.format(res.stdout.decode('UTF-8')))
            if  res.stderr:
                log.info('File copy err\n{}'.format(res.stderr.decode('UTF-8')))
    else:  # Local run
        res = subprocess.run('cp ' + str(args.images_dir) + '/*.npy ' + str(LOCAL_DIR) + '/', shell=True, capture_output=True)
        if res.returncode != 0:
            log.info('cp failed')
            if res.stdout:
                log.info('File copy output\n{}'.format(res.stdout.decode('UTF-8')))
            if  res.stderr:
                log.info('File copy err\n{}'.format(res.stderr.decode('UTF-8')))


def get_label_dataframe(args):
    label_dataframe = pd.read_parquet(args.label_file, engine='pyarrow')

    if not args.fold_column:  # Make the index the fold column
        args.fold_column = label_dataframe.index.name
        label_dataframe[args.fold_column] = label_dataframe.index
    return label_dataframe


def create_data_splits(args, vals_to_split, g):
    # Sanity check
    if len(vals_to_split) < args.kfolds:
        log.error('Provided a kfolds value higher than then number of labels to split. WTF is that supposed to mean. '
                  'Quiting in a huff.')
        exit(35)

    if args.kfolds == 1:
        splits = random_split(vals_to_split, args.data_split.values(), g)
        yield {k: list(splits[idx]) for idx, k in enumerate(args.data_split.keys())}
    else:
#        if len(vals_to_split) > args.kfolds:  # k-folds across a column with more groups than k (e.g. patient ID)
        vals_to_split = random_split(vals_to_split, [1 / args.kfolds] * args.kfolds, g)

#        for k in range(args.kfolds):
#            val = vals_to_split[k]
#            if not isinstance(val, list):
#                val = [val]
#            yield {'train': vals_to_split[:k] + vals_to_split[k + 1:], 'validation': val}
        for k in range(args.kfolds):
            # addition on pytorch subsets doesn't return single subset, so do it manually
            indices = [idx for subset in vals_to_split[:k] + vals_to_split[k + 1:] for idx in subset.indices ]
            train = torch.utils.data.Subset(vals_to_split[k].dataset, indices)
            yield {'train': train, 'validation': vals_to_split[k]}


def create_dataloaders(args, label_dataframe, datasets, g):
    log.info('Creating dataloaders')

    # Create the datasets
    for partition_name, values in datasets.items():
        datasets[partition_name] = label_dataframe[label_dataframe[args.fold_column].isin(list(values))].copy(deep=True)
        if partition_name == 'train':
            datasets[partition_name] = ImageDataset(args, datasets[partition_name], random_translation)
        else:
            datasets[partition_name] = ImageDataset(args, datasets[partition_name])

    # Set z-score according to the training data
    if args.training_zscore:
        count = 0
        mean, meansq, std = 0.0, 0.0, 0.0
        for data in datasets['train']:
            images = data['image']
            mean += images[0, :, :].sum()  # only need to do this for one channel
            meansq += (images[0, :, :] ** 2).sum()
            count += np.prod(images.shape) / 3
        total_mean = mean / count
        total_std = torch.sqrt(meansq / count) - (total_mean ** 2)

        # Set for each dataset
        for partition_name, values in datasets.items():
            datasets[partition_name].set_zscore(total_mean, total_std)
    elif args.imagenet_zscore:
        # Set for each dataset
        for partition_name, values in datasets.items():
            datasets[partition_name].set_zscore(IMAGENET_MEAN, IMAGENET_STDDEV)

    # Create dataloaders for each split
    dataloaders = {}
    for partition_name, values in datasets.items():
        dataloaders[partition_name] = DataLoader(datasets[partition_name], batch_size=args.batch_size, shuffle=True,
                                                     pin_memory=True,
                                                     num_workers=NUM_WORKERS_TO_USE, worker_init_fn=seed_worker,
                                                     generator=g)

    log.info('Finished dataloaders')
    log.info('')
    log.info('Data split sizes:')
    log.info('Raw: {}'.format({k:len(v) for k, v in datasets.items()}))
    # Wrong need to drop NA
    log.info('Pct: {}'.format({k: 100*len(v)/len(label_dataframe) for k, v in datasets.items()}))
    return dataloaders


def stage_data(args):
    # Get a list of all needed image files
    label_dataframe = pd.read_parquet(args.label_file)
    xrays_in_label_file = set(label_dataframe.index)
    # Get a list of images files in image dir
    xray_filenames = set([f.stem for f in args.images_dir.glob('*.npy')])
#    xray_filenames = set([f.split('.')[0] for f in listdir(args.images_dir) if f.endswith('.npy')])

    # Sanity check, are all the files we expect present?
    check_for_missing_files(xrays_in_label_file, xray_filenames)

    # Make local image storage dir if needed
    cloud_happy_mkdir()

    copy_image_files_to_local_storage(args)