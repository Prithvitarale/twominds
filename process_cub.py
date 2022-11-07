import os
from torch.utils.data import Dataset
import glob
from PIL import Image
import torchvision.transforms as transforms
import torch as t
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class Cub2011(Dataset):

    def __init__(self, root):
        self.root = os.path.expanduser(root)

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
        cond = 2
        concepts = [[[] for _ in range(15)] for _ in range(201)]
        visibility_vector = [[[] for _ in range(15)] for _ in range(201)]
        for file in glob.glob(file_name):
            for img in glob.glob(os.path.join(file, "*")):
                img_id, class_id = path2id[img]
                image = Image.open(img)
                img_tensor = transform(image)
                parts = part_locs.iloc[15*(img_id-1):15*(img_id)]
                for part_id in range(len(parts)):
                    visibility = int(parts.iloc[part_id, 4])
                    visibility_vector[class_id][part_id].append(visibility)
                    if visibility == 1:
                        x, y = int(parts.iloc[part_id, 2]), int(parts.iloc[part_id, 3])
                        a, b, c, d, invalid = self.__check_constraints__(img_tensor, x, y, window_size)
                        if invalid == 0:
                            concept = img_tensor[:, a:b, c:d]
                            if len(concepts[class_id][part_id]) >= 1:
                                (concept_p, part_id, occurrences, added) = concepts[class_id][part_id][0]
                                if concept_p.shape == concept.shape:
                                    concept = t.divide(t.add(t.multiply(concept_p, added), concept), added+1)
                                    concepts[class_id][part_id][0] = (concept, part_id, occurrences+1, added+1)
                                else:
                                    concepts[class_id][part_id][0] = (concept_p, part_id, occurrences + 1, added)
                            else:
                                concepts[class_id][part_id].append((concept, part_id, 1, 1))

                for part_id_loop in range(len(parts)):
                    if len(concepts[class_id][part_id_loop]) >= 1:
                        c_r, c_id, o, a = concepts[class_id][part_id_loop][0]
                        concept_img = imager(c_r)
                # cond-=1
                # if cond==0:
                #     exit()

        with open("concepts/concepts.pkl", "wb") as fp:
            pickle.dump(concepts, fp)
        with open("concepts/visibility.pkl", "wb") as fp2:
            pickle.dump(visibility_vector, fp2)


    def __check_constraints__(self, img, x, y, window_size):
        max_x = img.shape[1]
        max_y = img.shape[2]
        a = x-window_size
        b = x+window_size
        c = y-window_size
        d = y+window_size
        invalid = 0
        a = max(a, 0)
        c = max(c, 0)
        b = min(b, max_x-1)
        d = min(d, max_y-1)
        if a>b:
            temp = a
            a = b
            b = temp
        if c>d:
            temp = c
            c = d
            d = temp
        if a==b or c==d:
            invalid=1
        return a, b, c, d, invalid