import torch
from torch import nn
from os import path
import pretrainedmodels
import torchvision.models as models
# from model.efficientnet import EfficientNet
from efficientnet_pytorch import EfficientNet
from model.layer_utils import GeM


class ModelFactory():
    def __init__(self):
        pass

    def get_model(self, model_name, num_classes, pred_type, hyper_params=None, tuning_type='feature-extraction', pre_trained_path=None, weight_type=None):
        if pred_type == 'regression':
            adjusted_num_classes = 1
        elif pred_type == 'mixed':
            adjusted_num_classes = num_classes + 1
        else:
            adjusted_num_classes = num_classes

        model = None

        if model_name == 'efficientnet-b4':
            print("[ Model : Efficientnet B4 ]")
            model = EfficientNet.from_pretrained(
                'efficientnet-b4'
            )
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False

            # changing avg pooling to Generalized Mean Avg
            model._avg_pooling = GeM()

            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(
                nn.Linear(num_ftrs, 1000, bias=True),
                nn.ReLU(),
                nn.Dropout(p=hyper_params['fc_drop_out']),
                nn.Linear(1000, adjusted_num_classes, bias=True)
            )

            # if hyper_params is not None:
            #     model._bn_mom = hyper_params['batch_norm_momentum']

        if model_name == 'efficientnet-b5':
            print("[ Model : Efficientnet B5 ]")

            model = EfficientNet.from_pretrained(
                'efficientnet-b5'
            )
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False

            # changing avg pooling to Generalized Mean Avg
            # model._avg_pooling = GeM()

            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(
                nn.Linear(num_ftrs, 1000, bias=True),
                nn.ReLU(),
                nn.Dropout(p=hyper_params['fc_drop_out']),
                nn.Linear(1000, adjusted_num_classes, bias=True)
            )

            # if hyper_params is not None:
            #     model._bn_mom = hyper_params['batch_norm_momentum']

        if model_name == 'efficientnet-b7':
            print("[ Model : Efficientnet B7 ]")
            model = EfficientNet.from_pretrained(
                'efficientnet-b7'
            )
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False

            # changing avg pooling to Generalized Mean Avg
            model._avg_pooling = GeM()

            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(
                nn.Linear(num_ftrs, 1000, bias=True),
                nn.ReLU(),
                nn.Dropout(p=hyper_params['fc_drop_out']),
                nn.Linear(1000, adjusted_num_classes, bias=True)
            )

            # if hyper_params is not None:
            #     model._bn_mom = hyper_params['batch_norm_momentum']

        if model_name == 'densenet-161':
            print("[ Model : Densenet 161 ]")
            model = models.densenet161(pretrained=True)
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, adjusted_num_classes)
            )

        if model_name == 'resnet-34':
            print("[ Model : Resnet 34 ]")
            model = pretrainedmodels.__dict__[
                'resnet34'](pretrained='imagenet')
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            num_ftrs = model.last_linear.in_features
            model.last_linear = nn.Sequential(
                nn.Linear(num_ftrs, adjusted_num_classes)
            )

        if model_name == 'se-resnet-152':
            print("[ Model : SeResnet 152 ]")
            model = pretrainedmodels.__dict__[
                'se_resnet152'](pretrained='imagenet')
            if tuning_type == 'feature-extraction':
                for param in model.parameters():
                    param.requires_grad = False
            num_ftrs = model.last_linear.in_features
            model._fc = nn.Sequential(
                nn.Linear(num_ftrs, 1000, bias=True),
                nn.ReLU(),
                nn.Dropout(p=hyper_params['fc_drop_out']),
                nn.Linear(1000, adjusted_num_classes, bias=True)
            )

        tuning_type and print("[ Tuning type : ", tuning_type, " ]")
        print("[ Prediction type : ", pred_type, " ]")

        # if model needs to resume from pretrained weights
        if pre_trained_path:
            weight_path = 'weights.pth'
            if weight_type == 'best_val_kaggle_metric':
                weight_path = 'weights_kaggle_metric.pth'
            elif weight_type == 'best_val_loss':
                weight_path = 'weights_loss.pth'
            weight_path = path.join(
                'results', pre_trained_path, weight_path)

            if path.exists(weight_path):
                print("[ Loading checkpoint : ",
                      pre_trained_path, " ]")
                model.load_state_dict(torch.load(
                    weight_path
                    # ,map_location={'cuda:1': 'cuda:0'}
                ))
            else:
                print("[ Provided pretrained weight path is invalid ]")
                exit()

            print(
                "[ Weight type : ", weight_type if weight_type else "Last Epoch", " ]")

        return model
