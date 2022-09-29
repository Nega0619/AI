import os
import pandas as pd
import glob
import torch
import torchvision
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

normal_path = '/home/hwi/github/data/VQIS_PoC/normal'
red_path = '/home/hwi/github/data/VQIS_PoC/red'
######################################################################################################

# load and make imgs, lables

def load_all_data(path:str, extension:str):
    return glob.glob(os.path.join(os.path.join(path, '**'), '*'+extension), recursive=True)


normal_path = '/home/hwi/github/data/VQIS_PoC/normal'
red_path = '/home/hwi/github/data/VQIS_PoC/red'

normal = load_all_data(normal_path, '.jpg')         # 149
red = load_all_data(red_path, '.JPG')               # 87

normal += load_all_data(normal_path, '.JPG')         # 149
red += load_all_data(red_path, '.jpg')               # 87

print(len(normal), len(red))

normal_label = ['normal']*len(normal)
red_label    = ['red']*len(red)

images = [Image.open(img) for img in normal]
images += [Image.open(img) for img in red]

labels = normal_label+red_label 
######################################################################################################

# https://wikidocs.net/57165

# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self, images, labels):
    self.x_data = images
    self.y_data = labels

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = self.x_data[idx]
    y = self.y_data[idx]
    
    print(type(x))
    
    tensor_x = transforms.ToTensor()
    tensor_y = transforms.ToTensor()
    
    return tensor_x(x), tensor_y(y)


dataset = CustomDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

next(iter(dataloader))

######################################################################################################

# https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html

# class CustomImageDataset(Dataset):
#     def __init__(self, label_list, img_dir, transform=None, target_transform=None):
#         # self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
#         self.img_labels = label_list
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = self.img_dir
#         image = read_image(img_path)
#         label = self.img_labels
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

######################################################################################################

# https://anweh.tistory.com/10

# trans = transforms.Compose([transforms.Resize((100,100)), 
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
# trans = transforms.Compose([transforms.ToTensor(),
#                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
# trainset = torchvision.datasets.ImageFolder(root = normal_path, transform = trans)

# trainset.__getitem__

######################################################################################################
