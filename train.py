import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from os import path
from loss.loss_factory import LossFactory
from optimiser.optimiser_factory import OptimiserFactory
from scheduler.scheduler_factory import SchedulerFactory
from dataset.dataset_factory import DatasetFactory
from model.model_factory import ModelFactory
from transformer.transformer_factory import TransformerFactory
from utils.experiment_utils import ExperimentHelper
from utils.custom_bar import CustomBar
from utils.seed_backend import (seed_all, seed_worker)


def train(config, device, auto_aug_policy=None):
    # seed backend if not in augmentation search
    (auto_aug_policy is None) and seed_all(config['seed'])

    # Create pipeline objects
    dataset_factory = DatasetFactory(org_data_dir='./data')

    transformer_factory = TransformerFactory()

    model_factory = ModelFactory()

    writer = SummaryWriter(
        log_dir=path.join(
            'runs', config['experiment_name']
        )
    )

    experiment_helper = ExperimentHelper(
        config['experiment_name'],
        config['validation_frequency'],
        tb_writer=writer,
        overwrite=True,
        publish=config['publish'],
        config=config
    )

    optimiser_factory = OptimiserFactory()

    loss_factory = LossFactory()

    scheduler_factory = SchedulerFactory()

    # ==================== Model training / validation setup ========================

    training_dataset = dataset_factory.get_dataset(
        'train',
        config['train_dataset']['name'],
        transformer_factory.get_transformer(
            height=config['train_dataset']['resize_dims'],
            width=config['train_dataset']['resize_dims'],
            pipe_type=config['train_dataset']['transform'],
            auto_aug_policy=auto_aug_policy
        ),
        config['train_dataset']['fold']
    )

    validation_dataset = dataset_factory.get_dataset(
        'val',
        config['val_dataset']['name'],
        transformer_factory.get_transformer(
            height=config['val_dataset']['resize_dims'],
            width=config['val_dataset']['resize_dims'],
            pipe_type=config['val_dataset']['transform']
        ),
        config['val_dataset']['fold']
    )

    model = model_factory.get_model(
        config['model']['name'],
        config['num_classes'],
        config['model']['pred_type'],
        config['model']['hyper_params'],
        config['model']['tuning_type'],
        config['model']['pre_trained_path'],
        config['model']['weight_type']
    ).to(device)

    optimiser = optimiser_factory.get_optimiser(
        model.parameters(),
        config['optimiser']['name'],
        config['optimiser']['hyper_params']
    )

    scheduler = None
    if config['scheduler']:
        scheduler = scheduler_factory.get_scheduler(
            optimiser,
            config['scheduler']['name'],
            config['scheduler']['hyper_params'],
            epochs=config["epochs"],
            iter_per_epoch=len(training_dataset)/config["batch_size"]
        )

    loss_function = loss_factory.get_loss_function(
        config['loss_function']['name'],
        config['model']['pred_type'],
        config['loss_function']['hyper_params']
    )

    batch_size = config["batch_size"]

    # ===============================================================================

    # =================== Model training / validation loop ==========================

    with CustomBar(config["epochs"], len(training_dataset), batch_size) as progress_bar:

        for i in range(config["epochs"]):
            # progress bar update
            progress_bar.update_epoch_info(i)

            # set model to training mode
            model.train()

            training_loss = 0
            train_output_list = []
            train_target_list = []
            for batch_ndx, sample in enumerate(DataLoader(
                training_dataset,
                batch_size=batch_size,
                worker_init_fn=seed_worker,
                num_workers=4,
                pin_memory=True,
                shuffle=True
            )):

                # progress bar update
                progress_bar.update_batch_info(batch_ndx)

                input, target = sample
                input = input.to(device)
                target = target.to(device)

                # flush accumulators
                optimiser.zero_grad()

                # forward pass
                output = model.forward(input)

                # loss calculation
                loss = loss_function(
                    output,
                    target
                )

                # backward pass
                loss.backward()

                # update
                optimiser.step()

                # update lr using scheduler
                if scheduler:
                    scheduler.step()

                if experiment_helper.should_trigger(i):
                    train_output_list.append(output.detach().cpu())
                    train_target_list.append(target.cpu())
                    training_loss += (loss.item() * input.shape[0])

                # progress bar update
                progress_bar.step()

            # Do a loss check on val set per epoch
            if experiment_helper.should_trigger(i):
                # set model to evaluation mode
                model.eval()

                validation_loss = 0
                val_output_list = []
                val_target_list = []
                for batch_ndx, sample in enumerate(DataLoader(
                    validation_dataset,
                    batch_size=batch_size,
                    num_workers=4,
                    worker_init_fn=seed_worker,
                    pin_memory=True,
                    shuffle=False
                )):

                    with torch.no_grad():
                        input, target = sample
                        input = input.to(device)
                        target = target.to(device)

                        # forward
                        output = model.forward(input)

                        # loss calculation
                        loss = loss_function(
                            output,
                            target
                        )

                        val_output_list.append(output.detach().cpu())
                        val_target_list.append(target.cpu())
                        validation_loss += (loss.item() * input.shape[0])

                train_output_list = torch.cat(train_output_list, dim=0)
                train_target_list = torch.cat(train_target_list, dim=0)
                training_loss /= len(training_dataset)
                val_output_list = torch.cat(val_output_list, dim=0)
                val_target_list = torch.cat(val_target_list, dim=0)
                validation_loss /= len(validation_dataset)

                # validate model
                experiment_helper.validate(
                    config['model']['pred_type'],
                    config['num_classes'],
                    validation_loss,
                    training_loss,
                    val_output_list,
                    val_target_list,
                    train_output_list,
                    train_target_list,
                    i
                )

                # save model weights
                experiment_helper.save_checkpoint(
                    model.state_dict()
                )

    # publish final
    config['publish'] and experiment_helper.publish_final(config)

    return (experiment_helper.best_val_loss, experiment_helper.best_val_kaggle_metric)

    # ===============================================================================
