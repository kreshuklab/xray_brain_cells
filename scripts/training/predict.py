import os
import sys
import numpy as np
import torch
from inferno.trainers.basic import Trainer
from sklearn import metrics
from brain_dset import get_loaders


CLASS_NAMES = ['pyramidal', 'non pyramidal', 'non neuronal', 'unclassified']


def predict(model, loader):
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for cells, labels in loader:
            preds = model(cells.cuda()).cpu().numpy().squeeze().argmax(1)
            all_targets.extend(labels)
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
    path = sys.argv[1]
    if len(sys.argv) > 2:
        device = str(sys.argv[2])
    else:
        device = '2'
    print("The model is", os.path.split(os.path.normpath(path))[-1])
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    config_file = os.path.join(path, 'data_config.yml')
    val_loader = get_loaders(config_file, train=False)
    model_path = os.path.join(path, 'Weights')
    best_model = Trainer().load(from_directory=model_path, best=True).model
    predictions, targets = predict(best_model, val_loader)
    print_results(predictions, targets)
