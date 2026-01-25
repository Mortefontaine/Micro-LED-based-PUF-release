import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import cv2
from tqdm import tqdm
import warnings
import json
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
import math

from scipy.spatial.distance import hamming

warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ClassificationDataset(Dataset):
    def __init__(self, images, labels, image_paths, transform=None):
        self.images = images
        self.labels = torch.LongTensor(labels)
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        path = self.image_paths[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, path


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, 3 * 2)
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.pool1(F.relu(self.conv1(x)))
        xs = self.pool2(F.relu(self.conv2(xs)))
        num_features = xs.size(1) * xs.size(2) * xs.size(3)
        xs = xs.view(-1, num_features)
        theta = self.fc2(F.relu(self.fc1(xs)))
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAM(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.cbam(out) + identity
        out = self.relu(out)
        return out


class BinaryEmbeddingNet(nn.Module):
    def __init__(self, num_classes=9, bitstream_length=256):
        super(BinaryEmbeddingNet, self).__init__()
        self.bitstream_length = bitstream_length
        self.num_classes = num_classes

        self.stn = STN()
        original_resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        self.layer1 = self._make_layer_cbam(BasicBlockCBAM, 64, original_resnet.layer1)
        self.layer2 = self._make_layer_cbam(BasicBlockCBAM, 128, original_resnet.layer2)
        self.layer3 = self._make_layer_cbam(BasicBlockCBAM, 256, original_resnet.layer3)
        self.layer4 = self._make_layer_cbam(BasicBlockCBAM, 512, original_resnet.layer4)
        self.avgpool = original_resnet.avgpool

        self.embedding_layer = nn.Linear(512 * BasicBlockCBAM.expansion, self.bitstream_length)
        self.classification_layer = nn.Linear(self.bitstream_length, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_cbam(self, block, planes, original_layer):
        layers = []
        for i, original_block in enumerate(original_layer):
            downsample = None
            if original_block.downsample is not None:
                downsample = original_block.downsample
            if i == 0:
                layers.append(block(original_block.conv1.in_channels, planes, original_block.conv1.stride[0],
                                    downsample=downsample))
            else:
                layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        continuous_embedding = self.embedding_layer(x)
        binary_probabilities = torch.sigmoid(continuous_embedding)
        hard_binary_bits = (binary_probabilities > 0.5).float()
        bitstream = (hard_binary_bits - binary_probabilities).detach() + binary_probabilities
        class_output = self.classification_layer(continuous_embedding)

        return class_output, continuous_embedding, bitstream

    def extract_embedding(self, x):
        x = self.stn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        continuous_embedding = self.embedding_layer(x)
        return continuous_embedding

    def extract_bitstream(self, x):
        continuous_embedding = self.extract_embedding(x)
        bitstream = (torch.sigmoid(continuous_embedding) > 0.5).float()
        return bitstream


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0, metric='euclidean'):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric

    def forward(self, embeddings, labels):
        if self.metric == 'euclidean':
            dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        elif self.metric == 'cosine':
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            dist_matrix = 1 - torch.mm(embeddings_norm, embeddings_norm.T)
        else:
            raise NotImplementedError("Only 'euclidean' and 'cosine' metrics are supported.")

        loss = 0
        batch_size = embeddings.size(0)
        for i in range(batch_size):
            anchor = embeddings[i]
            label = labels[i]

            positive_indices = (labels == label).nonzero(as_tuple=False).squeeze(1)
            positive_indices = positive_indices[positive_indices != i]
            negative_indices = (labels != label).nonzero(as_tuple=False).squeeze(1)

            if len(positive_indices) > 0 and len(negative_indices) > 0:
                hardest_positive_dist = torch.max(dist_matrix[i, positive_indices])
                hardest_negative_dist = torch.min(dist_matrix[i, negative_indices])

                loss_i = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
                loss += loss_i

        return loss / batch_size if batch_size > 0 else loss


def quantization_loss(embeddings):
    quant_loss = torch.mean((embeddings - torch.round(embeddings)) ** 2)
    return quant_loss


def bit_balance_loss(bitstream):
    bit_means = torch.mean(bitstream, dim=0)
    balance_loss = torch.mean((bit_means - 0.5) ** 2)
    return balance_loss


class DistributionSeparationLoss(nn.Module):
    def __init__(self, margin=0.05):
        super(DistributionSeparationLoss, self).__init__()
        self.margin = margin

    def forward(self, bitstream, labels):
        batch_size, bit_length = bitstream.shape
        distances = torch.sum(bitstream.unsqueeze(1) != bitstream.unsqueeze(0), dim=2).float() / bit_length

        intra_distances = []
        inter_distances = []

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if labels[i] == labels[j]:
                    intra_distances.append(distances[i, j])
                else:
                    inter_distances.append(distances[i, j])

        if not intra_distances or not inter_distances:
            return torch.tensor(0.0, device=bitstream.device, requires_grad=True)

        intra_distances = torch.stack(intra_distances)
        inter_distances = torch.stack(inter_distances)

        max_intra_dist = torch.max(intra_distances)
        min_inter_dist = torch.min(inter_distances)

        loss = F.relu(max_intra_dist - min_inter_dist + self.margin)

        return loss


class LightAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __call__(self, img):
        return self.transform(img)


class MicroLEDPUFClassifier:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = 'binary_embedding_results'
        os.makedirs(self.results_dir, exist_ok=True)
        self.training_history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
        self.model = None
        print(f"Use device: {self.device}")

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"All random seed are set as: {seed}")

    def load_data(self, data_path, max_samples_per_class=None, known_classes_filter=None):
        print(f"Loading data from {data_path} ...")
        images = []
        labels = []
        file_paths = []
        for folder in tqdm(os.listdir(data_path)):
            folder_path = os.path.join(data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            parts = folder.split('_')
            if len(parts) < 2:
                continue
            led_id = parts[0]
            label = led_id
            if known_classes_filter and label not in known_classes_filter:
                continue
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            samples_loaded = 0
            for img_file in image_files:
                if max_samples_per_class and samples_loaded >= max_samples_per_class:
                    break
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (80, 80))
                    images.append(img)
                    labels.append(label)
                    file_paths.append(img_path)
                    samples_loaded += 1
        images = np.array(images)
        labels = np.array(labels)
        file_paths = np.array(file_paths)
        print(f"Data loading finished: Load {len(images)} samples from {data_path} ")
        print(f"Label number: {len(np.unique(labels))}")
        return images, labels, file_paths

    def train_binary_embedding_net(self, train_images, train_labels_encoded, val_images, val_labels_encoded,
                                   num_classes):
        print("\n--- Start training the binary embedding network (multi-loss joint optimization) ---")

        train_transform = LightAugmentation()
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train_dataset = ClassificationDataset(train_images, train_labels_encoded, image_paths=[''] * len(train_images),
                                              transform=train_transform)
        val_dataset = ClassificationDataset(val_images, val_labels_encoded, image_paths=[''] * len(val_images),
                                            transform=val_transform)

        BATCH_SIZE = 64
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        self.model = BinaryEmbeddingNet(num_classes=num_classes).to(self.device)

        criterion_classification = nn.CrossEntropyLoss()
        criterion_triplet = BatchHardTripletLoss(margin=1.0)
        criterion_separation = DistributionSeparationLoss(margin=0.05)

        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        num_epochs = 50
        patience = 5
        best_val_accuracy = 0.0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss_total = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                class_output, continuous_embedding, bitstream = self.model(inputs)

                loss_classification = criterion_classification(class_output, labels)
                loss_triplet = criterion_triplet(continuous_embedding, labels)
                loss_quant = F.mse_loss(torch.sigmoid(continuous_embedding),
                                        torch.round(torch.sigmoid(continuous_embedding)))
                loss_balance = bit_balance_loss(bitstream)
                loss_separation = criterion_separation(bitstream, labels)

                # --- Joint loss weights, adjustable based on performance ---
                lambda_cls = 1.0
                lambda_triplet = 0.5
                lambda_quant = 0.1
                lambda_balance = 0.8
                # Result 1: lambda_balance = 0.2
                lambda_separation = 1.5
                # Result 2: lambda_separation = 1.0

                total_loss = (lambda_cls * loss_classification +
                              lambda_triplet * loss_triplet +
                              lambda_quant * loss_quant +
                              lambda_balance * loss_balance +
                              lambda_separation * loss_separation)

                total_loss.backward()
                optimizer.step()

                train_loss_total += total_loss.item()
                _, predicted = torch.max(class_output.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            avg_train_loss = train_loss_total / len(train_loader)
            train_accuracy = correct_train / total_train

            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1} Val"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    class_output, _, _ = self.model(inputs)
                    loss = criterion_classification(class_output, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(class_output.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_val / total_val
            scheduler.step(avg_val_loss)

            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_accuracy'].append(val_accuracy)

            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, 'best_binary_embedding_model.pth'))
                print("--- Saved best model based on validation accuracy ---")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs. No improvement for {patience} epochs.')
                    break

        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, 'best_binary_embedding_model.pth')))
        print("Binary embedding network training complete and best model loaded.")
        return self.model

    def evaluate_model(self, model, test_images, test_labels_encoded, test_paths, class_names,
                       filename_prefix="test_confusion_matrix"):
        print("\n--- Evaluate model performance ---")
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        test_dataset = ClassificationDataset(test_images, test_labels_encoded, test_paths, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        model.eval()
        all_preds = []
        all_labels = []
        misclassified_images = []
        with torch.no_grad():
            for inputs, labels, paths in tqdm(test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                class_output, _, _ = model(inputs)
                _, predicted = torch.max(class_output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                misclassified_mask = (predicted != labels).cpu().numpy()
                if np.any(misclassified_mask):
                    misclassified_indices = np.where(misclassified_mask)[0]
                    for idx in misclassified_indices:
                        true_label_decoded = self.label_encoder.inverse_transform([labels[idx].cpu().item()])[0]
                        predicted_label_decoded = self.label_encoder.inverse_transform([predicted[idx].cpu().item()])[0]
                        misclassified_images.append({
                            'path': paths[idx],
                            'true_label': true_label_decoded,
                            'predicted_label': predicted_label_decoded
                        })

        all_labels_decoded = self.label_encoder.inverse_transform(all_labels)
        all_preds_decoded = self.label_encoder.inverse_transform(all_preds)

        print("\n--- Classification report ---")
        print(classification_report(all_labels_decoded, all_preds_decoded, zero_division=0))

        # --- Save the origin confusion matrix as .CSV file ---
        cm = confusion_matrix(all_labels_decoded, all_preds_decoded, labels=class_names)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_filename = os.path.join(self.results_dir, f'raw_confusion_matrix_{timestamp}.csv')
        cm_df.to_csv(cm_filename)
        print(f"Save the origin confusion matrix to: {cm_filename}")

        self.plot_and_save_confusion_matrix(
            true_labels=all_labels_decoded,
            predicted_labels=all_preds_decoded,
            class_names=class_names,
            filename_prefix=filename_prefix
        )
        print("The origin confusion matrix has been saved.")

        if misclassified_images:
            print("\n--- Misclassified image information ---")
            for item in misclassified_images:
                print(f"Path: {item['path']}, true label: {item['true_label']}, predicted label: {item['predicted_label']}")
        else:
            print("\n--- Didn't find any misclassified images ---")

    def plot_training_history(self):
        if not self.training_history['train_loss']:
            print("No training history found. Plot training history.")
            return
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.training_history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        ax1.plot(self.training_history['val_loss'], 'g-', linewidth=2, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(self.training_history['train_accuracy'], 'r--', linewidth=2, label='Training Accuracy')
        ax2.plot(self.training_history['val_accuracy'], 'm--', linewidth=2, label='Validation Accuracy')
        ax2.set_ylabel('Accuracy', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')
        plt.title('Binary Embedding Network Training History')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def plot_and_save_confusion_matrix(self, true_labels, predicted_labels, class_names,
                                       filename_prefix="confusion_matrix"):
        cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
        overall_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        print(f"Overall Accuracy (from confusion matrix): {overall_accuracy:.4f}")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Accuracy: {overall_accuracy:.4f})')
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f'{filename_prefix}_{timestamp}.png')
        plt.savefig(filename)
        plt.show()
        plt.close()
        print(f"Save confusion matrix to: {filename}")

    def get_embeddings(self, images, labels_encoded, image_paths):
        if self.model is None:
            print("Error: The model was not loaded. Please train or load the model.")
            return None, None, None

        print("\n--- Extract continuous and binarized feature embeddings ---")
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = ClassificationDataset(images, labels_encoded, image_paths=image_paths, transform=transform)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        self.model.eval()
        all_continuous_embeddings = []
        all_bitstreams = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels, _ in tqdm(data_loader, desc="Extracting Embeddings"):
                inputs = inputs.to(self.device)
                continuous_embedding = self.model.extract_embedding(inputs)
                bitstream = self.model.extract_bitstream(inputs)
                all_continuous_embeddings.append(continuous_embedding.cpu().numpy())
                all_bitstreams.append(bitstream.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        continuous_embeddings = np.vstack(all_continuous_embeddings)
        bitstreams = np.vstack(all_bitstreams)
        print(f"Extract {bitstreams.shape[0]} binary embeddings, with each embedding dimension of {bitstreams.shape[1]}")
        return continuous_embeddings, bitstreams, np.hstack(all_labels)

    def calculate_and_plot_hamming_distances(self, binarized_embeddings, labels_encoded, class_names):
        print("\n--- Calculate and visualize the Hamming distance of binarized embeddings ---")
        intra_distances = []
        inter_distances = []
        decoded_labels = self.label_encoder.inverse_transform(labels_encoded)
        embeddings_by_class = {cls_name: [] for cls_name in class_names}
        for i, label in enumerate(decoded_labels):
            embeddings_by_class[label].append(binarized_embeddings[i])

        print("Calculate intra-class Hamming distance...")
        for cls_name, embeddings_list in embeddings_by_class.items():
            if len(embeddings_list) < 2:
                continue
            for i in range(len(embeddings_list)):
                for j in range(i + 1, len(embeddings_list)):
                    dist = hamming(embeddings_list[i], embeddings_list[j])
                    intra_distances.append(dist)
        intra_distances = np.array(intra_distances)

        print("Calculate inter-class Hamming distance...")
        class_combinations = []
        for i in range(len(class_names)):
            for j in range(i + 1, len(class_names)):
                class_combinations.append((class_names[i], class_names[j]))
        for cls1_name, cls2_name in tqdm(class_combinations, desc="Calculate Hamming distance"):
            embeddings_list_1 = embeddings_by_class[cls1_name]
            embeddings_list_2 = embeddings_by_class[cls2_name]
            if not embeddings_list_1 or not embeddings_list_2:
                continue
            for emb1 in embeddings_list_1:
                for emb2 in embeddings_list_2:
                    dist = hamming(emb1, emb2)
                    inter_distances.append(dist)
        inter_distances = np.array(inter_distances)

        # --- Save origin Hamming distance data as csv. ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(intra_distances) > 0:
            pd.DataFrame(intra_distances, columns=['Intra-class Hamming Distance']).to_csv(
                os.path.join(self.results_dir, f'raw_intra_distances_{timestamp}.csv'), index=False)
            print("Raw intra-class distance data has been saved as CSV.")
        if len(inter_distances) > 0:
            pd.DataFrame(inter_distances, columns=['Inter-class Hamming Distance']).to_csv(
                os.path.join(self.results_dir, f'raw_inter_distances_{timestamp}.csv'), index=False)
            print("Raw inter-class distance data has been saved as CSV.")

        print("\n--- Hamming distance statistics (normalized values) ---")
        if len(intra_distances) > 0:
            print("Intra-class distances:")
            print(f"  Range: [{np.min(intra_distances):.4f}, {np.max(intra_distances):.4f}]")
            print(f"  Central value (mean): {np.mean(intra_distances):.4f}")
            print(f"  Central value (median): {np.median(intra_distances):.4f}")
        else:
            print("Insufficient intra-class distance data.")
        if len(inter_distances) > 0:
            print("Inter-class distances:")
            print(f"  Range: [{np.min(inter_distances):.4f}, {np.max(inter_distances):.4f}]")
            print(f"  Central value (mean): {np.mean(inter_distances):.4f}")
            print(f"  Central value (median): {np.median(inter_distances):.4f}")
        else:
            print("Insufficient inter-class distance data.")

        print("\n--- Distribution overlap analysis ---")
        if len(intra_distances) > 0 and len(inter_distances) > 0:
            intra_max = np.max(intra_distances)
            inter_min = np.min(inter_distances)
            overlap_start = max(np.min(intra_distances), np.min(inter_distances))
            overlap_end = min(np.max(intra_distances), np.max(inter_distances))
            if intra_max > inter_min:
                intra_overlap_count = np.sum(intra_distances > inter_min)
                intra_overlap_percentage = (intra_overlap_count / len(intra_distances)) * 100
                inter_overlap_count = np.sum(inter_distances < intra_max)
                inter_overlap_percentage = (inter_overlap_count / len(inter_distances)) * 100
                print(f"Maximum intra-class distance (Intra_max): {intra_max:.4f}")
                print(f"Minimum inter-class distance (Inter_min): {inter_min:.4f}")
                print("Overlap detected (Intra_max > Inter_min): True")
                print(
                    f"{intra_overlap_percentage:.2f}% of intra-class samples exceed the minimum inter-class distance ({inter_min:.4f})")
                print(
                    f"{inter_overlap_percentage:.2f}% of inter-class samples are below the maximum intra-class distance ({intra_max:.4f})")
                if overlap_end > overlap_start:
                    print(f"Actual overlap interval: [{overlap_start:.4f}, {overlap_end:.4f}]")
                    overlap_length = overlap_end - overlap_start
                    total_span = max(np.max(intra_distances), np.max(inter_distances)) - min(
                        np.min(intra_distances), np.min(inter_distances))
                    if total_span > 0:
                        overlap_ratio_of_span = (overlap_length / total_span) * 100
                        print(f"Overlap interval length as a percentage of total span: {overlap_ratio_of_span:.2f}%")
            else:
                print(
                    "Maximum intra-class distance does not exceed minimum inter-class distance; no significant overlap detected.")
        else:
            print("Insufficient data to compute overlap.")

        plt.figure(figsize=(10, 6))
        if len(intra_distances) > 0:
            sns.histplot(intra_distances, color='blue', kde=True, stat='density', alpha=0.5,
                         label='Intra-class distance')
            plt.axvline(np.mean(intra_distances), color='blue', linestyle='--',
                        label=f'Intra-class mean: {np.mean(intra_distances):.4f}')
        if len(inter_distances) > 0:
            sns.histplot(inter_distances, color='red', kde=True, stat='density', alpha=0.5,
                         label='Inter-class distance')
            plt.axvline(np.mean(inter_distances), color='red', linestyle='--',
                        label=f'Inter-class mean: {np.mean(inter_distances):.4f}')
        plt.title('Hamming Distance Distribution of Binarized Features')
        plt.xlabel('Normalized Hamming Distance')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.xlim(0, 1)
        filename = os.path.join(self.results_dir, f'hamming_distance_distribution_{timestamp}.png')
        plt.savefig(filename)
        plt.show()
        plt.close()
        print(f"Hamming distance distribution plot has been saved to: {filename}")

        def visualize_embeddings_with_tsne(self, continuous_embeddings, labels_encoded, class_names):
            print("\n--- Visualizing continuous feature embeddings using t-SNE ---")
            if continuous_embeddings is None or len(continuous_embeddings) == 0:
                print("Error: No continuous embeddings available for visualization.")
                return
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            print("Running t-SNE, this may take several minutes...")
            embeddings_2d = tsne.fit_transform(continuous_embeddings)

            # --- New: save raw t-SNE reduced data as CSV ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tsne_df = pd.DataFrame(embeddings_2d, columns=['t-SNE_1', 't-SNE_2'])
            tsne_df['label_encoded'] = labels_encoded
            tsne_df['label_decoded'] = self.label_encoder.inverse_transform(labels_encoded)
            tsne_filename = os.path.join(self.results_dir, f'raw_tsne_embeddings_{timestamp}.csv')
            tsne_df.to_csv(tsne_filename, index=False)
            print(f"Raw t-SNE reduced data has been saved to: {tsne_filename}")

            plt.figure(figsize=(12, 10))
            sns.scatterplot(
                x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
                hue=self.label_encoder.inverse_transform(labels_encoded),
                palette=sns.color_palette("hsv", len(class_names)),
                legend="full",
                alpha=0.7
            )
            plt.title('t-SNE Visualization of Continuous Embeddings')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.legend(title='Class')
            plt.grid(True, alpha=0.5)
            filename = os.path.join(self.results_dir, f'tsne_visualization_{timestamp}.png')
            plt.savefig(filename)
            plt.show()
            plt.close()
            print(f"t-SNE visualization has been saved to: {filename}")

        def analyze_results(self, filename='binary_embedding_summary.json'):
            summary = {
                'training_history': self.training_history,
                'device': str(self.device),
                'model_info': "ResNet-34 with STN, CBAM Attention, and Multi-Loss Binary Embedding",
                'num_classes': len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else "N/A",
                'class_names': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder,
                                                                               'classes_') else "N/A",
                'results_directory': os.path.abspath(self.results_dir),
                'timestamp': datetime.now().isoformat()
            }
            with open(os.path.join(self.results_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
            print(f"Analysis summary has been saved to '{os.path.join(self.results_dir, filename)}'")

if __name__ == "__main__":
    DATA_DIR = 'D:/Research/PUF_micro-LED/videos3/frames_selected_by_YOLO_aug'
    KNOWN_CLASSES = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

    classifier_analyzer = MicroLEDPUFClassifier()
    classifier_analyzer.set_seed(42)

    all_images, all_labels_raw, all_paths = classifier_analyzer.load_data(
        DATA_DIR, known_classes_filter=KNOWN_CLASSES
    )
    classifier_analyzer.label_encoder.fit(all_labels_raw)
    all_labels_encoded = classifier_analyzer.label_encoder.transform(all_labels_raw)
    class_names = classifier_analyzer.label_encoder.classes_.tolist()
    num_classes = len(class_names)

    print(f"The model will be trained and evaluated on the following classes: {class_names}")
    print(f"A total of {num_classes} known classes")

    if num_classes != len(KNOWN_CLASSES):
        print(
            f"Warning: The number of loaded classes ({num_classes}) does not match "
            f"the expected number of known classes ({len(KNOWN_CLASSES)}). "
            f"Please check the contents of DATA_DIR."
        )

    train_val_images, test_images, train_val_labels_encoded, test_labels_encoded, train_val_paths, test_paths = train_test_split(
        all_images, all_labels_encoded, all_paths,
        test_size=0.2, stratify=all_labels_encoded, random_state=42
    )
    train_images, val_images, train_labels_encoded, val_labels_encoded, _, _ = train_test_split(
        train_val_images, train_val_labels_encoded, train_val_paths,
        test_size=0.25, stratify=train_val_labels_encoded, random_state=42
    )

    print(f"Number of training samples: {len(train_images)}")
    print(f"Number of validation samples: {len(val_images)}")
    print(f"Number of test samples: {len(test_images)}")

    trained_model = classifier_analyzer.train_binary_embedding_net(
        train_images, train_labels_encoded,
        val_images, val_labels_encoded,
        num_classes
    )

    classifier_analyzer.evaluate_model(
        trained_model,
        test_images,
        test_labels_encoded,
        test_paths,
        class_names
    )
    classifier_analyzer.plot_training_history()
    classifier_analyzer.analyze_results()

    continuous_test_embeddings, binarized_test_embeddings, corresponding_test_labels = classifier_analyzer.get_embeddings(
        test_images, test_labels_encoded, test_paths
    )

    # After get_embeddings:
    np.save(
        os.path.join(classifier_analyzer.results_dir, "binarized_test_embeddings.npy"),
        binarized_test_embeddings
    )
    np.save(
        os.path.join(classifier_analyzer.results_dir, "corresponding_test_labels.npy"),
        corresponding_test_labels
    )
    np.save(
        os.path.join(classifier_analyzer.results_dir, "label_encoder_classes.npy"),
        classifier_analyzer.label_encoder.classes_
    )

    if binarized_test_embeddings is not None:
        print(f"Shape of binarized test embeddings: {binarized_test_embeddings.shape}")
        classifier_analyzer.calculate_and_plot_hamming_distances(
            binarized_test_embeddings,
            corresponding_test_labels,
            class_names
        )
        classifier_analyzer.visualize_embeddings_with_tsne(
            continuous_test_embeddings,
            corresponding_test_labels,
            class_names
        )
    else:
        print("Analysis cannot be performed because no embeddings were generated.")
