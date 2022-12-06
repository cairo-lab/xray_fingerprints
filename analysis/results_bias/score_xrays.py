# -*- coding: utf-8 -*-

"""
Created December 02, 2022
"""

import argparse
import glob
import logging
import os
import sys
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor, load as load_torch  # pytorch bug requires top-level import
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import models, transforms

import utils


_log = logging.getLogger(__name__)


class ImageDataset(TorchDataset):
    def __init__(self, oai_source: str) -> None:
        """Custom dataset for available OAI xrays.

        Args:
            oai_source (str): Path to image directory containing OAI xray images.
        """

        self.image_paths = glob.glob(os.path.join(oai_source, "*.npy"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> dict[str, Union[Tensor, str]]:
        if not isinstance(idx, int):
            raise TypeError("index must be int, not {}".format(type(idx).__name__))

        img_path = self.image_paths[idx]

        # image: Tensor = np.load(
        #  img_path
        # )  # Images need to be converted to three channel

        image = Image.fromarray(np.load(img_path)).convert("RGB")
        image = transforms.ToTensor()(image)
        image_id = os.path.basename(img_path).replace(".npy", "")

        _log.debug(f"reading image {img_path} -- {image.shape}")

        return {"image": image, "id": image_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create xray pain scores from images.")
    parser.add_argument(
        "--source", "-s", help="OAI scratch directory", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="output predictions path", type=str, required=True
    )
    parser.add_argument(
        "--weights",
        "-w",
        help="path to model weights to use for image scoring",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch-size", "-b", help="dataloader batch size", type=int, default=8
    )
    parser.add_argument(
        "--shuffle",
        help="whether or not to shuffle the dataloader",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )

    dataset = ImageDataset(args.source or "")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    n_additional_image_features_to_predict = 19
    fully_connected_bias_initialization = 90
    csv_output_path = os.path.join(
        args.output or "", f"predictions_{utils.create_timestring()}.csv"
    )

    device = torch.device("cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(  # type:ignore
        device
    )
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=1 + n_additional_image_features_to_predict,
    )  # reset final fully connected layer and make it so it has a single output.

    nn.init.constant_(
        model.fc.bias.data[:1],  # type:ignore
        fully_connected_bias_initialization,
    )

    model.load_state_dict(load_torch(args.weights or "", map_location=device))

    with torch.no_grad():
        n_batches_loaded = 0

        for batch in dataloader:
            n_batches_loaded += 1

            _log.info(f"reading batch {n_batches_loaded}")

            inputs = batch["image"]
            inputs = Variable(inputs.float().to(device))
            outputs = model(inputs)

            koos_predictions = outputs[:, :-n_additional_image_features_to_predict]

            df = pd.DataFrame(
                data={
                    "koos_prediction": [t[0] for t in koos_predictions.numpy()],
                    "id": batch["id"],
                }
            )

            df["bin_koos"] = df["koos_prediction"].apply(utils.binarize_koos)

            _log.info(f"writing predictions to path {csv_output_path}")

            df.to_csv(
                csv_output_path or "",
                mode="a",
                header=not os.path.exists(csv_output_path),
            )


if __name__ == "__main__":
    main()
