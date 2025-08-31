from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import utils
import torch.optim as optim
from torch.optim import lr_scheduler
import json
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Classifier:
    def __init__(self, model_name='model'):
        self.data_path = 'Data'
        self.class_dict = utils.classes_indices(self.data_path)
        self.num_classes = len(self.class_dict)
        self.train_loader, self.test_loader = utils.load_data()
        self.model_name = model_name
        self.json_path='class_indices.json'

    def load_model(self):
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        model = models.resnext50_32x4d(weights=weights)
        num_features = int(model.fc.in_features)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

        return model

    def train_model(self, epochs, lr=1e-3):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = (self.load_model()).to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        save_path = f"./Model/{self.model_name}.pth"
        records = {
            "train_loss": [],
            "train_acc": [],
            "test_acc": []
        }
        best_acc = 0.0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_acc = correct / total

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(images)
                    _, predicted = outputs.max(1)

                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_acc = correct / total

            scheduler.step(test_acc)

            records["train_loss"].append(train_loss)
            records["train_acc"].append(train_acc)
            records["test_acc"].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), save_path)

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)
            print(f"--Train Loss: {train_loss:.4f}  --Train Acc: {train_acc:.4f}  --Val Acc: {test_acc:.4f}")

        utils.visualize_training(records, epochs)

    def predict(self, image_path):
        with open(self.json_path, "r") as f:
            classes = json.load(f)

        model_path = f"./Model/{self.model_name}.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = (self.load_model()).to(device)
        model.eval()
        model.load_state_dict(torch.load(model_path, map_location=device))

        predict_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert('RGB')
        image = predict_transform(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(device)

        with torch.no_grad():
            output = model(image)
            predict = torch.softmax(output, dim=1)
            predict = torch.squeeze(predict)

            predict_idx = torch.argmax(predict).cpu().numpy()
            predict_classes = classes[str(predict_idx)]
            predict_prob = predict[predict_idx].item()
            print(f"predict classes: {predict_classes}")
            print(f"predict probability: {predict_prob:.2f}")

            image = image.cpu().squeeze()
            image = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )(image)
            image = image.permute(1, 2, 0).numpy()
            plt.imshow(image)
            plt.title(f"Classes: {predict_classes}     Prob: {predict_prob:.2f}")
            plt.show()

if __name__ == '__main__':

    classifier = Classifier()
    pattern = int(input("train(1)/test(0): "))

    if pattern == 1:
       classifier.train_model(epochs=10)
    else:
        image_path = os.path.normpath(input("Please intput the path of image: "))
        classifier.predict(image_path)