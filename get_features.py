import cv2
import torch
from torchvision import transforms
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import time
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
class Res18Feature(nn.Module):
    def __init__(self, pretrained, num_classes = 7):
        super(Res18Feature, self).__init__()
        resnet  = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x,save_path):
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x_npy = x.data.cpu().numpy()
        np.save(save_path, x_npy)
        # print(save_path)

        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out

# 模型存储路径
model_save_path = "./models/mytrain/origin/epoch30_acc0.8739.pth"#修改为你自己保存下来的模型文件
folder_path = "/home/DataBase2/sunpeiwen/dataset/crop_faces_zero_pad"#待测试照片位置
feature_path = folder_path+"-feature"

# ------------------------ 加载数据 --------------------------- #

preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
res18 = Res18Feature(pretrained = False)
checkpoint = torch.load(model_save_path)
res18.load_state_dict(checkpoint['model_state_dict'])
res18.cuda()
res18.eval()

print("0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral")
for root,dirs,files in os.walk(folder_path):
    for file in tqdm(files):
        if(file[-3:] in ["jpg","png"]):
            image = cv2.imread(root+os.sep+file)
            image = image[:, :, ::-1] # BGR to RGB
            image_tensor = preprocess_transform(image)
            #print(image_tensor.shape)
            tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float(), requires_grad=False)

            #print(tensor.shape) #[1,3, 224, 224]
            tensor=tensor.cuda()
            #print(tensor.shape)
            folder=feature_path+root.split(folder_path)[1]+os.sep
            # print(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            _, outputs = res18(tensor,folder+file[:-3]+"npy")
            _, predicts = torch.max(outputs, 1)
            
            # print(predicts)
