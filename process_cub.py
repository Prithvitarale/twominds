import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import glob
from PIL import Image
import torchvision.transforms as transforms
import torch as t
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_data_loader(self, transform=None):
        file_name = "data/CUB_200_2011/images/*"
        if transform is None:
            transform = transforms.Compose([
                transforms.PILToTensor(), transforms.Resize([256, 256])
            ])
        data_x = []
        data_y = []
        max_classes = 0
        for file in glob.glob(file_name):
            for img in glob.glob(os.path.join(file, "*")):
                image = Image.open(img)
                img_tensor = transform(image)
                data_x.append(img_tensor)
                data_y.append(t.tensor(max_classes))
            max_classes+=1
            if max_classes==9:
                break
        data_x = t.stack(data_x)
        data_y = t.stack(data_y)
        indexes = t.randperm(data_x.shape[0])
        data_x = data_x[indexes]
        data_y = data_y[indexes]
        N = len(data_x)
        train_dataset = TensorDataset(data_x[0:int(0.8*N)], data_y[0:int(0.8*N)])
        test_dataset = TensorDataset(data_x[int(0.8*N):N], data_y[int(0.8*N):N])
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        return train_dataloader, test_dataloader

    def get_concepts(self):
        id2path = {}
        path2id = {}
        with open(os.path.join(self.root, 'CUB_200_2011', 'images.txt')) as f:
            lines = f.readlines()
            for line in lines:
                index, path = line.strip().split()
                class_id = int(path[0:3])
                index = int(index)
                path = os.path.join(self.root, 'CUB_200_2011', 'images', path)
                id2path[index] = path
                path2id[path] = (index, class_id)

        file_name = "data/CUB_200_2011/images/*"
        part_locs = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'parts/part_locs.txt'), sep=' ',
                                names=['img_id', 'trait_id', 'x', 'y', 'visible'])

        transform = transforms.Compose([
            transforms.PILToTensor()])

        imager = transforms.ToPILImage()
        window_size = 35
        a = 1
        concepts = [[]]*200
        for file in glob.glob(file_name):
            for img in glob.glob(os.path.join(file, "*")):
                img_id, class_id = path2id[img]
                print(img_id)
                image = Image.open(img)
                img_tensor = transform(image)
                parts = part_locs.iloc[15*(img_id-1):15*(img_id)]
                print(len(parts))
                plt.imshow(imager(img_tensor))
                for part_id in range(len(parts)):
                    if int(parts.iloc[part_id, 4]) == 1:
                        x, y = int(parts.iloc[part_id, 2]), int(parts.iloc[part_id, 3])
                        concept = img_tensor[:, x-window_size:x+window_size, y-window_size:y+window_size]
                        concepts[class_id].append((concept, part_id))
                        print(part_id+1)
                for i, concept in enumerate(concepts[class_id]):
                    c_r, c_id = concept[0], concept[1]
                    # print(c_r)
                    concept_img = imager(c_r)
                    concept_img.save(f'concepts/{class_id}_{c_id+1}.png')
                a-=1
                if a==0:
                    exit()