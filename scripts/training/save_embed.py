import argparse
import os
import numpy as np
import torch
from inferno.trainers.basic import Trainer
from brain_dset import get_loaders
from torch.utils.tensorboard import SummaryWriter

CLASS_NAMES = ['pyramidal', 'non pyramidal', 'non neuronal', 'unclassified']

def get_embeddings(model, sample):
    sample = model.conv1(sample)
    sample = model.norm1(sample)
    sample = model.relu(sample)
    sample = model.maxpool(sample)
    sample = model.layer1(sample)
    sample = model.layer2(sample)
    sample = model.layer3(sample)
    sample = model.layer4(sample)
    sample = model.avgpool(sample)
    sample = torch.flatten(sample, 1)
    return sample


def predict(model, loader):
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for cells, labels in loader:
            preds = get_embeddings(model, cells.cuda()).cpu().numpy()
            all_predictions.extend(preds)
    return np.array(all_predictions), loader.dataset.annot_cells


def save_embeddings(embedding, labels, path):
    tf_emb_path = os.path.join(path, 'embed')
    if not os.path.exists(tf_emb_path):
        os.makedirs(tf_emb_path)
    classes = [CLASS_NAMES[i] for i in labels[:,1]]
    classes_and_ids = ['{}_{}'.format(CLASS_NAMES[i], str(j)) for j, i in labels]
    writer = SummaryWriter(tf_emb_path)
    writer.add_embedding(embedding, metadata=classes, tag='classes')
    writer.add_embedding(embedding, metadata=classes_and_ids, tag='classes_ids')
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save embeddings')
    parser.add_argument('path', type=str, help='train path with model and configs')
    parser.add_argument('--device', type=str, default='2',
                        choices=[str(i) for i in range(8)], help='GPU to use')
    args = parser.parse_args()
    print("The model is", os.path.split(os.path.normpath(args.path))[-1])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config_file = os.path.join(args.path, 'data_config.yml')
    val_loader = get_loaders(config_file, train=False)
    model_path = os.path.join(args.path, 'Weights')
    best_model = Trainer().load(from_directory=model_path, best=True).model
    predictions, targets = predict(best_model, val_loader)
    save_embeddings(predictions, targets, args.path)
