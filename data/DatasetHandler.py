import pandas as pd
import glob, os
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def load_all_data(path:str, extension:str):
    return glob.glob(os.path.join(os.path.join(path, '**'), '*'+extension), recursive=True)

def load_all_data_linux(path:str, extension:str):
    output = load_all_data(path, extension.lower())
    output += load_all_data(path, extension.upper())
    return output
    

'''
file list를 받아서 Custom Data 만듬
'''
class CustomDataset_imgPaths(Dataset): 
    def __init__(self, img_paths, labels, transform=None):
        self.x_data = img_paths
        self.y_data = labels
        self.transform = transform

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        img_path = self.x_data[idx]
        label = self.y_data[idx]
        # print('label type', type(label))

        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.ToTensor()
            img = transform(img)
            
        # print('img type', type(img))
        # print('label type', type(label))
        
        return img, label

'''
PILImage받은애들
'''
class CustomDataset_PILImages(Dataset): 
    def __init__(self, img_paths, labels, transform=None):
        self.x_data = img_paths
        self.y_data = labels
        self.transform = transform

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        img_path = self.x_data[idx]
        label = self.y_data[idx]
        # print('label type', type(label))
        # print(img_path)

        if self.transform is not None:
            img = self.transform(img_path)
        else:
            transform = transforms.ToTensor()
            img = transform(img_path)
        
        return img, label
    
'''
file list를 받아서 Custom Data 만듬
'''
class CustomDataset_csv(Dataset): 
    def __init__(self, sourcePath, csv_file, transform=None):
        self.x_data = sourcePath
        self.y_data = pd.read_csv(csv_file)
        self.transform = transform

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.y_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        img_path = os.path.join(self.x_data, self.y_data.iloc[idx, 0])
        label = self.y_data.iloc[idx, 1]
        # print('label type', type(label))

        img = Image.open(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.ToTensor()
            img = transform(img)
            
        # print('img type', type(img))
        # print('label type', type(label))
        
        return img, label