import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class TinyImageNet(Dataset):
    def __init__(self, data_dir, data_type, transform):
        self.type = data_type
        self.transform = transform

        labels_t = open(f'{data_dir}wnids.txt').read().strip().split('\n')
        labels_map = {label_t: label for label, label_t in enumerate(labels_t)}
        if self.type == 'train':
            self.train_labels = []
            self.train_images = []
            for i, label_t in tqdm(enumerate(labels_t), desc='[Load train images]'):
                txt_path = f'{data_dir}train/{label_t}/{label_t}_boxes.txt'
                image_names = [line.split('\t')[0] for line in open(
                    txt_path).read().strip().split('\n')]
                for image_name in image_names:
                    image_path = f'{data_dir}train/{label_t}/images/{image_name}'
                    img = cv2.imread(image_path)
                    self.train_images.append(img)
                    self.train_labels.append(i)
            self.train_images = np.array(self.train_images)
            self.train_labels = np.array(self.train_labels)
        elif self.type == 'val':
            self.val_images = []
            self.val_labels = []
            with open(f'{data_dir}val/val_annotations.txt') as txt:
                for line in tqdm(txt, desc='[Load val images]'):
                    image_name, label_t = line.strip('\n').split('\t')[:2]
                    image_path = f'{data_dir}val/images/{image_name}'
                    val_label = labels_map[label_t]
                    img = cv2.imread(image_path)
                    self.val_images.append(img)
                    self.val_labels.append(val_label)
            self.val_images = np.array(self.val_images)
            self.val_labels = np.array(self.val_labels)

    def __getitem__(self, index):
        image, label = None, None
        if self.type == 'train':
            label = self.train_labels[index]
            image = self.train_images[index]
        elif self.type == 'val':
            label = self.val_labels[index]
            image = self.val_images[index]
        return self.transform(image), label

    def __len__(self):
        size = None
        if self.type == 'train':
            size = self.train_labels.shape[0]
        elif self.type == 'val':
            size = self.val_labels.shape[0]
        return size


if __name__ == "__main__":
    batch_size = 64
    train_dataset = TinyImageNet(
        './imagenet/tiny-imagenet-200/', 'train', transforms.Compose([transforms.ToTensor()]))
    val_dataset = TinyImageNet(
        './imagenet/tiny-imagenet-200/', 'val', transforms.Compose([transforms.ToTensor()]))
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)
    for batch_image, batch_label in train_dataloader:
        print(batch_image.shape)
        print(batch_label.shape)
        print(np.uni)
        exit()
