import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.nn import functional as F


class HandClassifier(nn.Module):
    # NOTE: This neural network is designed specifically for 28*28*1 images as input.
    # Alter the network structure to fit images with different dimentions.
    def __init__(self, num_classes):
        super(HandClassifier, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=45, kernel_size=3, stride=1, padding=1)
        self.batch_norm1= nn.BatchNorm2d(45)
        self.conv2 = nn.Conv2d(in_channels=45, out_channels=55, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm2 = nn.BatchNorm2d(55)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=2695, out_features=num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.batch_norm2(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def gen_dataloader(data, labels, batch_size):
    X_train_val, X_test, y_train_val, y_test= train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=True, stratify=y_train_val)

    X_train = np.array(X_train, dtype=np.float32) / 255
    X_val = np.array(X_val, dtype=np.float32) / 255
    X_test = np.array(X_test, dtype=np.float32) / 255
    X_train = X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2])
    X_val = X_val.reshape(-1, 1, X_val.shape[1], X_val.shape[2])
    X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2])

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=y_val.size)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=y_test.size)

    return train_loader, val_loader, test_loader


def label_idx_converter(labels):
    sorted_unique_labels = sorted(set(labels))
    label_to_idx = {}
    idx_to_label = {}

    for i, label in enumerate(sorted_unique_labels):
        label_to_idx[label] = i
        idx_to_label[i] = label

    return len(sorted_unique_labels), label_to_idx, idx_to_label


def main():
    """Entry point of the prgram."""
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    args_config = dict()
    for arg in vars(args):
        args_config[arg] = getattr(args, arg)
    num_epochs = args_config.get('num_epochs')
    batch_size = args_config.get('batch_size')
    lr = args_config.get('lr')

    ROOT = '.' if os.path.basename(os.getcwd()) == 'my_project' else '..'
    with open(os.path.join(ROOT, 'data/hand_imgs.pkl'), 'rb') as f:
        hand_data = pickle.load(f)

    imgs = np.array(hand_data['imgs'])
    num_classes, label_to_idx, idx_to_label = label_idx_converter(hand_data['labels'])
    labels_idx = np.array([label_to_idx[label] for label in hand_data['labels']])

    # generate data loaders
    train_loader, val_loader, test_loader = gen_dataloader(imgs, labels_idx, batch_size)

    # train model
    criterion = nn.CrossEntropyLoss()
    model = HandClassifier(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for img_batch, label_batch in train_loader:
            pred = model(img_batch)
            loss = criterion(pred, label_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        img_val, label_val = next(iter(val_loader))
        print(f'Epoch {epoch + 1}/{num_epochs}:\n\tValidation accuracy = {accuracy_score(label_val, torch.argmax(model(img_val), dim=1))}')

    img_test, label_test = next(iter(test_loader))
    print(f'Test accuracy = {accuracy_score(label_test, torch.argmax(model(img_test), dim=1))}')

    # save model dict
    torch.save(model.state_dict(), os.path.join(ROOT, 'model/cnn_clf.pth'))


if __name__ == '__main__':
    main()