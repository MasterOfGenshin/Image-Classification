import os
import shutil
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def classes_indices(path):
    classes = [cla for cla in os.listdir(path) if os.path.isdir(os.path.join(path, cla))]

    classes.sort()

    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    return class_indices

def split_images(path, data_dir, dataset_dir):
    _ = classes_indices(path)

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    categories = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    for category in categories:
        category_path = os.path.join(data_dir, category)

        category_train_dir = os.path.join(train_dir, category)
        category_test_dir = os.path.join(test_dir, category)
        os.makedirs(category_train_dir, exist_ok=True)
        os.makedirs(category_test_dir, exist_ok=True)

        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

        random.shuffle(images)

        split_ratio = 0.8
        split_index = int(len(images) * split_ratio)

        train_images = images[:split_index]
        test_images = images[split_index:]

        for img in train_images:
            src_img_path = os.path.join(category_path, img)
            dst_img_path = os.path.join(category_train_dir, img)
            shutil.copy(src_img_path, dst_img_path)

        for img in test_images:
            src_img_path = os.path.join(category_path, img)
            dst_img_path = os.path.join(category_test_dir, img)
            shutil.copy(src_img_path, dst_img_path)

def visualize_training(records, epochs):
    x = np.arange(1, len(records["train_loss"]) + 1)
    plt.figure(figsize=(12, 9))

    plt.subplot(2, 2, 1)
    plt.plot(x, records["train_loss"], "b-", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(x, records["train_acc"], "g-", label="Training Accuracy")
    plt.plot(x, records["test_acc"], "r-", label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('train_result.png')
    plt.show()

def load_data():
    split_images('./Data', 'Data', 'Dataset')

    train_path = "./Dataset/train"
    test_path = "./Dataset/test"

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=train_path, transform=train_transform)
    test_dataset = ImageFolder(root=test_path, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader