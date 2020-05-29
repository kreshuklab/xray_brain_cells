import argparse
import os
import sys
import numpy as np
import torch
from inferno.trainers.basic import Trainer
from sklearn import metrics
from brain_dset import get_loaders


CLASS_NAMES = ['pyramidal', 'non pyramidal', 'non neuronal', 'unclassified']


def predict(model, loader, ae):
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for cells, labels in loader:
            if not ae:
                preds = model(cells.cuda()).cpu().numpy().argmax(1)
                all_targets.extend(labels)
            else:
                preds = model.encode(cells.cuda())[1].cpu().numpy().argmax(1)
                all_targets.extend(labels[0])
            all_predictions.extend(preds)
    return np.array(all_predictions), np.array(all_targets)


def print_results(pred_labels, true_labels):
    class_accuracies = []
    for i, name in enumerate(CLASS_NAMES):
        class_accuracy = np.sum(pred_labels[np.where(true_labels == i)] == i) \
                         / np.sum(true_labels == i)
        print("Accuracy for {0} is {1}".format(name, class_accuracy))
        class_accuracies.append(class_accuracy)
    print("Average accuracy (w/o unclassified) is ", np.mean(class_accuracies[:-1]))
    print(metrics.confusion_matrix(true_labels, pred_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict brain cell type')
    parser.add_argument('path', type=str, help='train path with model and configs')
    parser.add_argument('--device', type=str, default='2',
                        choices=[str(i) for i in range(8)], help='GPU to use')
    parser.add_argument('--autoencoder', type=int, default=0,
                        choices=[0, 1], help='Whether the model is autoencoder')
    args = parser.parse_args()
    print("The model is", os.path.split(os.path.normpath(args.path))[-1])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config_file = os.path.join(args.path, 'data_config.yml')
    val_loader = get_loaders(config_file, train=False)
    model_path = os.path.join(args.path, 'Weights')
    best_model = Trainer().load(from_directory=model_path, best=True).model
    predictions, targets = predict(best_model, val_loader, args.autoencoder)
    print_results(predictions, targets)
