import time
import os
import sys
import logging
import argparse
import torch
from torch import nn
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore, GarbageCollection
import inferno.extensions.criteria as criteria
import inferno.extensions.metrics as metrics
from inferno.extensions.criteria import Criteria
import neurofire.models as models

from brain_dset import get_loaders

logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def compile_criterion(criterion, **criterion_kwargs):
    if isinstance(criterion, str):
        pr_criterion = getattr(nn, criterion, getattr(criteria, criterion, None))(**criterion_kwargs)
    elif isinstance(criterion, dict):
        cr_list = [getattr(nn, cr, getattr(criteria, cr, None))(**criterion[cr]) for cr in criterion]
        pr_criterion = Criteria(cr_list, **criterion_kwargs)
    else:
        pr_criterion = None
    return pr_criterion


def set_up_training(project_directory, config):
    model_name = config.get('model_name')
    model = getattr(models, model_name)(**config.get('model_kwargs'))
    loss = compile_criterion(config.get('loss'), **config.get('loss_kwargs'))
    metric = getattr(metrics, config.get('metric'))(**config.get('metric_kwargs'))
    if metric is None:
        metric = loss
    logger.info("Building trainer.")
    smoothness = config.get('smoothness', 0)
    trainer = Trainer(model)\
        .set_backprop_every(config.get('backprop_every', 1))\
        .save_every((50, 'iterations'), to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss)\
        .build_validation_criterion(loss)\
        .build_optimizer(**config.get('training_optimizer_kwargs'))\
        .evaluate_metric_every('never')\
        .validate_every((config.get('validate_every', 20), 'iterations'),
                         for_num_iterations=config.get('validate_for', 5))\
        .register_callback(SaveAtBestValidationScore(smoothness=smoothness, verbose=True))\
        .build_metric(metric)\
        .register_callback(AutoLR(factor=0.98,
                                  patience='10 iterations',
                                  monitor='validation_loss_averaged',
                                  monitor_while='validating',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous'))\
        .register_callback(GarbageCollection())

    logger.info("Building logger.")
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'), log_images_every='never').observe_state(
                                    #'validation', observe_while='validating')#.observe_state(
                                    'training', observe_while='training')
    trainer.build_logger(tensorboard,
                         log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


def training(project_directory, train_configuration_file,
             data_configuration_file, max_num_epochs, from_checkpoint):

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get('devices')[0])
    if from_checkpoint:
        trainer = Trainer().load(from_directory=project_directory,
                                 filename='Weights/checkpoint.pytorch')
    else:
        trainer = set_up_training(project_directory, config)

    logger.info("Loading training and validation data loader from %s." % data_configuration_file)
    train_loader, validation_loader = get_loaders(data_configuration_file)

    trainer.set_max_num_epochs(max_num_epochs)

    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train', train_loader).bind_loader(
                        'validate', validation_loader)

    if config.get('devices'):
        trainer.cuda([0])
        trainer.apex_opt_level = config.get('opt_level', "O1")
        trainer.mixed_precision = config.get('mixed_precision', "False")

    trainer.pickle_module = 'dill'
    logger.info("Lift off!")
    start = time.time()
    trainer.fit()
    end = time.time()
    time_diff = end - start
    print("The training took {0} hours {1} minutes".format(time_diff // 3600,
                                                           time_diff % 3600 // 60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('--max_num_epochs', type=int, default=10)
    parser.add_argument('--from_checkpoint', type=int, default=0)

    args = parser.parse_args()

    project_directory = args.project_directory
    assert os.path.exists(project_directory), 'create a project directory with config files!'

    train_config = os.path.join(project_directory, 'train_config.yml')
    data_config = os.path.join(project_directory, 'data_config.yml')

    training(project_directory, train_config, data_config,
             max_num_epochs=args.max_num_epochs, from_checkpoint=args.from_checkpoint)


if __name__ == '__main__':
    main()
