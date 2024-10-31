#import datetime
import logging

#from google.cloud import storage
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from torchinfo import summary
from pathlib import Path

log = logging.getLogger(__name__)


def create_model(args, out_feature=1):
    if args.pretrain_weights:
        model = models.resnet18()
        model.name = 'ResNet18'
        model_weights = torch.load(Path(args.pretrain_weights))
        prior_training_out_features, prior_training_in_features = model_weights['fc.0.weight'].shape
        model.fc = nn.Sequential(
            nn.Linear(prior_training_in_features, prior_training_out_features, bias=True),
        )
        model.load_state_dict(model_weights)
        # Reset to desired number
        model.fc = nn.Sequential(
            nn.Linear(prior_training_in_features, out_feature, bias=True),
        )

    else:  # Use ImageNet
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        model.name = 'ResNet18'

        # reset final fully connected layer and make it  has the desired number of outputs
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, out_feature, bias=True),
            #nn.Sigmoid() # Use for binary classifier
        )

    for params in model.parameters():
        params.requires_grad = True
    param_idx = 0
    all_conv_layers = []

    for name, param in model.named_parameters():
        param_idx += 1
        conv_layer_substring = is_conv_layer(name)
        if conv_layer_substring is not None and conv_layer_substring not in all_conv_layers:
            all_conv_layers.append(conv_layer_substring)

    # now look conv_layers_before_end_to_unfreeze conv layers before the end, and unfreeze all layers after that.
    start_unfreezing = False
    assert args.layers_to_unfreeze <= len(all_conv_layers)
    if args.layers_to_unfreeze > 0:
        conv_layers_to_unfreeze = all_conv_layers[-args.layers_to_unfreeze:]
    else:
        conv_layers_to_unfreeze = []

    for name, param in model.named_parameters():
        conv_layer_substring = is_conv_layer(name)
        if conv_layer_substring in conv_layers_to_unfreeze:
            start_unfreezing = True
        if name in ['fc.0.weight', 'fc.0.bias']:
            # we always unfreeze these layers.
            start_unfreezing = True
        if not start_unfreezing:
            param.requires_grad = False

    return model


# loop over layers from beginning and freeze a couple. First we need to get the layers.
def is_conv_layer(c_name):
    # If a layer is a conv layer, returns a substring which uniquely identifies it. Otherwise, returns None.
    # this logic is probably more complex than it needs to be but it works.
    if c_name[:5] == 'layer':
        sublayer_substring = '.'.join(c_name.split('.')[:3])
        if 'conv' in sublayer_substring:
            return sublayer_substring
    return None


def print_model_stats(model, args):
#    param_idx = 0
#     all_conv_layers = []
#
#     log.info('')
#     log.info('All named parameters:')
#     log.info('Param\tName\t\tSize')
#     for name, param in model.named_parameters():
#         log.info("{}:\t{}\t\t{}".format(param_idx, name, param.data.shape))
#         param_idx += 1
#         conv_layer_substring = is_conv_layer(name)
#         if conv_layer_substring is not None and conv_layer_substring not in all_conv_layers:
#             all_conv_layers.append(conv_layer_substring)
#     log.info('All conv layers:')
#     log.info('{}'.format(all_conv_layers))
#     log.info('')

    log.info(summary(model, input_size=(args.batch_size, 3, 224, 224)))


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum() / weight.sum()


def create(args, data_len, out_features):
    """
    Create the model, loss function, and optimizer to be used for the DNN

    Args:
      args: experiment parameters.
      device: PyTorch device (GPU or CPU)
    """
    log.info('Creating model')
    model = create_model(args, out_features).to(args.device)
    weights = torch.FloatTensor([args.weight]).to(args.device)

    if args.model_type == 'classification':
        #criterion = nn.BCELoss() # only if adding a Sigmoid layer
        criterion = nn.BCEWithLogitsLoss()
    elif args.model_type == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    elif args.model_type == 'regression':
        criterion = nn.MSELoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           betas=(0.9, 0.999), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=data_len,
                                              epochs=args.num_epochs)
    log.info('Model created')

    return model, criterion, optimizer, scheduler
