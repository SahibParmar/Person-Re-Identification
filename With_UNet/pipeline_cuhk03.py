import kagglehub

# Download latest version
path = kagglehub.dataset_download("priyanagda/cuhk03")

print("Path to dataset files:", path)

import os
os.system('pip install --upgrade torch torchvision torchaudio')
os.system('pip install --upgrade transformers')
os.system('pip install ftfy regex tqdm')
os.system('pip install git+https://github.com/openai/CLIP.git')
os.system('pip install clip')
os.system('pip install openai')
"""#**Evaluation**"""

root_dir=path+'/archive'


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt,image as mpimg
from torch.optim import Adam
from torchvision import transforms as transforms
from torchvision.models import resnet18
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,random_split
import torch.optim as optim
from torchvision.models.resnet import Bottleneck
from PIL import Image
import clip
from matplotlib import pyplot as plt,image as mpimg #Make sure you have imported the necessary modules
from torchvision import transforms
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel


print("Successfully installed all the dependencies!!!!")
torch.multiprocessing.set_sharing_strategy('file_system')




transform= transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize instead of cropping
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])


# Load CLIP Model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")



#clip_model = get_peft_model(clip_model, lora_config)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("CLIP downloaded properly")


class UNetResNet18(nn.Module):
    def __init__(self):
        super(UNetResNet18, self).__init__()
        resnet = resnet18(pretrained=True)
        # Encoder layers
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # (64, 112, 112)
        self.pool0 = resnet.maxpool                                        # (64, 56, 56)
        self.enc1 = resnet.layer1                                          # (64, 56, 56)
        self.enc2 = resnet.layer2                                          # (128, 28, 28)
        self.enc3 = resnet.layer3                                          # (256, 14, 14)
        self.enc4 = resnet.layer4                                          # (512, 7, 7)
        # Decoder layers
        self.up4 = self.up_block(512, 256)
        self.dec4 = self.conv_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.dec3 = self.conv_block(256, 128)
        self.up2 = self.up_block(128, 64)
        self.dec2 = self.conv_block(128, 64)
        self.up1 = self.up_block(64, 64)
        self.dec1 = self.conv_block(128, 64)
        self.up0 = self.up_block(64, 32)
        self.dec0 = self.conv_block(32, 32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_ch, out_ch):
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)        # (64, 112, 112)
        e1 = self.enc1(self.pool0(e0))  # (64, 56, 56)
        e2 = self.enc2(e1)       # (128, 28, 28)
        e3 = self.enc3(e2)       # (256, 14, 14)
        e4 = self.enc4(e3)       # (512, 7, 7)
        # Decoder
        d4 = self.up4(e4)                   # 7 → 14
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        d3 = self.up3(d4)                   # 14 → 28
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)                   # 28 → 56
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d1 = self.up1(d2)                   # 56 → 112
        d1 = self.dec1(torch.cat([d1, e0], dim=1))
        d0 = self.up0(d1)                   # 112 → 224
        d0 = self.dec0(d0)
        out = torch.sigmoid(self.out_conv(d0))  # Binary mask
        return out





class SiameseCLIP(nn.Module):
    def __init__(self, clip_model,segment_model=None):
        super(SiameseCLIP, self).__init__()
        self.segment_model=segment_model
        self.clip_model = clip_model
        self.fc = nn.Linear(768, 512)  # Projection head for similarity learning

    def forward(self, img):
        if self.segment_model is not None:
            mask=self.segment_model(img)
            img=img*mask
        img_features = self.clip_model.get_image_features(img)
        img_proj=self.fc(img_features)
        return img_proj

    def encode_one_image(self, image):
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(image)
        img_proj=self.fc(image_features)
        return img_proj


class Classifier(nn.Module):
    def __init__(self,in_features,num_classes=1501):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)



# Example training loop
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{device} found !")
masker=UNetResNet18()
model = SiameseCLIP(clip_model,segment_model=masker).to(device)

classifier = Classifier(in_features=512).to(device)





print("model defined successfully!")

#total params and trainable params
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
print(f'Percentage:{trainable_params*100/total_params}')

def train_model(dataloader, triplet_loss_fn, classifier_loss_fn, optimizer, epochs=1, th=0.5, model=None, classifier=None, device=None):
    """
    Training loop for the Siamese network with triplet and classification losses.
    """
    model.to(device)
    classifier.to(device)

    for epoch in range(epochs):
        model.train()
        classifier.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for ind, (anchor, pos, neg, anchor_id, pos_id, neg_id) in enumerate(dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            anchor_id, pos_id, neg_id = anchor_id.to(device), pos_id.to(device), neg_id.to(device)

            anchor_feat, pos_feat, neg_feat = model(anchor), model(pos), model(neg)
            torch.cuda.empty_cache()
            # Normalize for triplet loss
            norm_anchor = F.normalize(anchor_feat, p=2, dim=1)
            norm_pos = F.normalize(pos_feat, p=2, dim=1)
            norm_neg = F.normalize(neg_feat, p=2, dim=1)
            # Triplet loss
            triplet_loss = triplet_loss_fn(norm_anchor, norm_pos, norm_neg)
            del norm_anchor, norm_pos, norm_neg
            # Classification logits
            pred_anchor = classifier(anchor_feat)
            pred_pos = classifier(pos_feat)
            pred_neg = classifier(neg_feat)
            # Classification loss
            classifier_loss = (
                classifier_loss_fn(pred_anchor, anchor_id) +
                classifier_loss_fn(pred_pos, pos_id) +
                classifier_loss_fn(pred_neg, neg_id)
            )

            # Total loss
            loss = triplet_loss + classifier_loss

            torch.cuda.empty_cache()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy: top-1 prediction matches
            _, pred_class = torch.max(pred_anchor, dim=1)
            correct_predictions += (pred_class == anchor_id).sum().item()
            total_samples += anchor.size(0)
            if ind%10==0:
               accuracy = 100.0 * correct_predictions / total_samples
               print(f"batch {ind+1} - Triplet Loss: {triplet_loss:.4f} classification_loss {classifier_loss:.4f} Continuous Accuracy: {accuracy:.2f}%")

        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100.0 * correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Classification Accuracy: {accuracy:.2f}%")

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from collections import defaultdict
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np



# ------------------------------------------------------------------
# 1. Load the JSON split file for CUHK03.
# ------------------------------------------------------------------
def load_split(json_file, images_dir):
    """
    Loads the JSON file (which should contain keys: 'train', 'query', 'gallery')
    and prepends the images directory to the image paths.
    """
    with open(json_file, 'r') as f:
        split = json.load(f)
    # print(type(split),len(split))
    split=split[0]
    train, query, gallery = [], [], []
    for rec in split['train']:
        #reading image at path rec[0] and then throw error to stop execution
        # load_and_display_image(os.path.join(images_dir, rec[0][29:]))
        rec[0] = os.path.join(images_dir, rec[0][29:])
        train.append(rec)
    for rec in split['query']:
        rec[0] = os.path.join(images_dir, rec[0][29:])
        query.append(rec)
    for rec in split['gallery']:
        rec[0] = os.path.join(images_dir, rec[0][29:])
        gallery.append(rec)
    print(len(query),len(gallery),len(train))
    return train, query, gallery


# ------------------------------------------------------------------
# 2. Dataset classes for Re-ID and Siamese training
# ------------------------------------------------------------------

class ReIDDataset(Dataset):
    """
    A simple dataset for person re-identification.
    The JSON split is assumed to contain lists for 'train', 'query', and 'gallery'
    with each record having at least the fields 'im_path' and 'pid'. Optionally, 'cam'.
    """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        img_path=record[0]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        pid = record[1]
        cam=record[2]
        return image, pid, cam


class TripletReIDDataset(Dataset):
    """
    Dataset for Triplet Loss + Classification Loss in Person Re-ID.
    Returns: (anchor_img, pos_img, neg_img, anchor_pid, pos_pid, neg_pid)
    """
    def __init__(self, train_data, num_triplets=10000, transform=None):
        self.transform = transform
        self.triplets = []
        self.pid_dict = defaultdict(list)

        for img_path, pid, cam_id in train_data:
            self.pid_dict[pid].append((img_path, pid))  # store pid alongside path

        self.pids = list(self.pid_dict.keys())

        for _ in range(num_triplets):
            # Select anchor and positive
            anchor_pid = random.choice(self.pids)
            anchor_images = self.pid_dict[anchor_pid]
            if len(anchor_images) < 2:
                continue
            (anchor_path, _), (pos_path, _) = random.sample(anchor_images, 2)

            # Select negative
            neg_pid = random.choice([pid for pid in self.pids if pid != anchor_pid])
            neg_path, _ = random.choice(self.pid_dict[neg_pid])

            self.triplets.append(((anchor_path, anchor_pid), (pos_path, anchor_pid), (neg_path, neg_pid)))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        (anchor_path, anchor_pid), (pos_path, pos_pid), (neg_path, neg_pid) = self.triplets[idx]

        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img = Image.open(pos_path).convert("RGB")
        neg_img = Image.open(neg_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        # Convert labels to tensor (CrossEntropy expects LongTensor)
        anchor_pid = torch.tensor(anchor_pid, dtype=torch.long)
        pos_pid = torch.tensor(pos_pid, dtype=torch.long)
        neg_pid = torch.tensor(neg_pid, dtype=torch.long)

        return anchor_img, pos_img, neg_img, anchor_pid, pos_pid, neg_pid




def get_dataloader(train_data, batch_size=32,num_triplets=10000, shuffle=True, num_workers=2):
    dataset = TripletReIDDataset(train_data, num_triplets=num_triplets, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def extract_features(model, dataloader, device):
    """
    Extract features from images using the model's encode_one_image function.
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for imgs, pids, _ in dataloader:
            imgs = imgs.to(device)
            feats = model.encode_one_image(imgs)
            features.append(feats.cpu())
            labels.extend(pids)
    features = torch.cat(features, dim=0)
    return features, labels

def compute_euclidean_distance(query_features, gallery_features):
    """
    Computes the pairwise Euclidean distance matrix between query and gallery features.
    """
    m, n = query_features.size(0), gallery_features.size(0)
    dist = (
        query_features.pow(2).sum(1, keepdim=True).expand(m, n) +
        gallery_features.pow(2).sum(1, keepdim=True).expand(n, m).t()
    )
    dist.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
    return dist.clamp(min=1e-12).sqrt()

def evaluate_rank_map(euclidean_dist, query_labels, gallery_labels):
    """
    Computes Cumulative Matching Characteristic (CMC) and mean Average Precision (mAP).
    """
    query_labels = np.array(query_labels)
    gallery_labels = np.array(gallery_labels)
    num_query = euclidean_dist.size(0)
    CMC = np.zeros(len(gallery_labels))
    APs = []
    skipped_queries = 0
    for i in range(num_query):
        q_label = query_labels[i]
        dist = euclidean_dist[i].numpy()
        indices = np.argsort(dist)  # Lower distance first
        matches = (gallery_labels[indices] == q_label).astype(int)
        if matches.sum() == 0:
            skipped_queries += 1
            continue
        rank_idx = np.where(matches == 1)[0][0]
        CMC[rank_idx:] += 1
        num_rel = matches.sum()
        tmp_cmc = np.cumsum(matches) / (np.arange(len(matches)) + 1)
        AP = (tmp_cmc * matches).sum() / num_rel
        APs.append(AP)
    print(f"Total queries skipped (no relevant match in gallery): {skipped_queries}")
    CMC = CMC / num_query
    mAP = np.mean(APs) if APs else 0.
    return CMC, mAP

def retrieve_by_threshold(euclidean_dist, threshold=1.0):
    """
    Retrieves gallery indices for each query with Euclidean distance below the threshold.
    """
    retrieved = []
    for i in range(euclidean_dist.size(0)):
        indices = torch.where(euclidean_dist[i] < threshold)[0]
        retrieved.append(indices.tolist())
    return retrieved

# ------------------------------------------------------------------
# 3. Main routine to load data, optionally train, and evaluate.
# ------------------------------------------------------------------
def main(protocol='new',train=False,epochs=1,model=model,device=device,lr=1e-5):
    """
    Evaluate the model using the specified protocol.
    protocol: 'new' or 'classic'
    """
    dataset_root=path+'/archive'
    transform_fn = transform
    images_dir = os.path.join(dataset_root, 'images_labeled')

    # Choose the appropriate split JSON file.
    if protocol == 'new':
        json_file = os.path.join(dataset_root, 'splits_new_labeled.json')
    elif protocol == 'classic':
        json_file = os.path.join(dataset_root, 'splits_classic_labeled.json')
    else:
        raise ValueError("Protocol must be 'new' or 'classic'")

    # Load splits from the JSON file (Market1501)
    train_data, query_data, gallery_data = load_split(json_file,images_dir)

    query_dataset = ReIDDataset(query_data, transform=transform)
    gallery_dataset = ReIDDataset(gallery_data, transform=transform)
    query_loader = DataLoader(query_dataset, batch_size=4, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Optionally perform training using ImagePairDataset
    if train:
        dataloader = get_dataloader(train_data, batch_size=4)
        print(len(dataloader), ' batches and ', len(dataloader) * 4, 'image pairs')
        classifier_loss_fn = nn.CrossEntropyLoss()
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for e in range(epochs):
            print(f'---------- Training Epoch {e + 1} ----------')
            train_model(dataloader, triplet_loss_fn, classifier_loss_fn, optimizer, epochs=1, model=model,classifier=classifier, device=device)
            model.eval()
            query_features, query_labels = extract_features(model, query_loader, device)
            gallery_features, gallery_labels = extract_features(model, gallery_loader, device)
            euclidean_dist = compute_euclidean_distance(query_features, gallery_features)
            CMC, mAP = evaluate_rank_map(euclidean_dist, query_labels, gallery_labels)
            print("Evaluation results:")
            print("  Rank-1 Accuracy: {:.2f}%".format(CMC[0] * 100))
            print("  mAP: {:.2f}%".format(mAP * 100))
            torch.save({
            'siamese_clip_state': model.state_dict(),
            'classifier_state': classifier.state_dict(),
            # Optional:
            # 'optimizer_state': optimizer.state_dict(),
            # 'epoch': epoch
            }, 'checkpoint.pth')

            print('weights checkpointed!')

    # Create evaluation datasets and dataloaders


    # Extract features and evaluate
    # model.to(device)
    # model.eval()
    # query_features, query_labels = extract_features(model, query_loader, device)
    # gallery_features, gallery_labels = extract_features(model, gallery_loader, device)
    # euclidean_dist = compute_euclidean_distance(query_features, gallery_features)
    # CMC, mAP = evaluate_rank_map(euclidean_dist, query_labels, gallery_labels)
    # print("Evaluation results:")
    # print("  Rank-1 Accuracy: {:.2f}%".format(CMC[0] * 100))
    # print("  mAP: {:.2f}%".format(mAP * 100))




# ------------------------------------------------------------------------------
# Example usage:
# Assume that you have defined your model (e.g., SiameseCLIP) and set the device.
device = "cuda" if torch.cuda.is_available() else "cpu"
#model = SiameseCLIP(clip_model).to(device)
main(train=True, epochs=5, model=model, device=device,lr=1e-5)
