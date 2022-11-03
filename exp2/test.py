import cv2
import numpy as np
import torch
import torch.nn as nn
import warnings

from PIL import Image
from torchvision.models import resnet50
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

class CNNModel_ResNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.cnn_encoder = resnet50(pretrained=True)
        self.cnn_encoder.fc = nn.Linear(2048,n)
    def forward(self, x):
        return self.cnn_encoder(x)

def transform(img):
    trans = transforms.Compose([
    transforms.RandomCrop((400,400)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ])
    img = trans(img)
    return img

if __name__=="__main__":
    model_path="param/model_param.cpt"
    model=CNNModel_ResNet(n=2) #AI or NAI
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    while True:
        sample=[]
        n=5
        p=np.array([0.0,0.0])
        img_path=input("入力画像: ")
        img=cv2.imread(img_path)
        #n回ランダムにクロップした画像の予測結果を返す
        for _ in range(n):
            img2=transform(Image.fromarray(img)) 
            img2=img2[None]
            result = model(img2.to(device))
            soft=nn.functional.softmax(result)
            _, predicted = torch.max(result, 1)
            predicted=predicted.to("cpu").item()
            sample.append(predicted)
            p+=soft.to("cpu").detach().numpy()[0]

        if sample.count(0)>sample.count(1):
            print("AIイラストでない : ",p[0]/n)
        else:
            print("AIイラスト : ",p[1]/n)