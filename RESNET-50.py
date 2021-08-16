#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np

print("PyTorch Version: ",torch.__version__)

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places,eps=1e-05,momentum=0.1,affine=True,track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places,eps=1e-05,momentum=0.1,affine=True,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places,eps=1e-05,momentum=0.1,affine=True,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion,eps=1e-05,momentum=0.1,affine=True,track_running_stats=False),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion,eps=1e-05,momentum=0.1,affine=True,track_running_stats=False)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__=='__main__':
    model = ResNet50()
    #model.load_state_dict(torch.load(r'E:\resnet50-0676ba61.pth'))
    #print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)


# In[2]:


import paddle
import paddle.nn as nn
import numpy as np

print("Paddle Version: ",paddle.__version__)


# In[3]:


class BatchNorm2D(paddle.nn.BatchNorm2D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)


# In[4]:


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2D(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias_attr=False),
        BatchNorm2D(places,eps=1e-05,momentum=0.1,affine=False,track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
    )


# In[5]:


class Bottleneck(nn.Layer):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2D(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias_attr=False),
            BatchNorm2D(places,eps=1e-05,momentum=0.1,affine=False,track_running_stats=False),
            nn.ReLU(),
            nn.Conv2D(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias_attr=False),
            BatchNorm2D(places,eps=1e-05,momentum=0.1,affine=False,track_running_stats=False),
            nn.ReLU(),
            nn.Conv2D(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias_attr=False),
            BatchNorm2D(places*self.expansion,eps=1e-05,momentum=0.1,affine=False,track_running_stats=False),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias_attr=False),
                BatchNorm2D(places*self.expansion,eps=1e-05,momentum=0.1,affine=False,track_running_stats=False),
            )
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# In[6]:


class ResNet(nn.Layer):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))#nn.AvgPool2D(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)
                
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n=m.weight.shape[0]*m.weight.shape[1]*m.weight.shape[2]
                v=np.random.normal(loc=0.,scale=np.sqrt(2./n),size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.weight.shape).astype('float32'))

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x,1)
        x = self.fc(x)
        return x
    
def ResNet50():
    return ResNet([3, 4, 6, 3])


# In[7]:


if __name__=='__main__':
    model_paddle = ResNet50()
    #model.load_state_dict(torch.load(r'E:\resnet50-0676ba61.pth'))
    #print(model)

    a = np.random.random((1,3,224,224))
    input=paddle.to_tensor(a, dtype='float32')
    out = model_paddle(input)
    print(out.shape)


# In[8]:


i=0
for name,parameter in model_paddle.named_parameters():
    i=i+1
    print(i,"-",name,"--",parameter.shape)


# In[9]:


torch_param=model.state_dict()
paddle_dict={}
for key in torch_param:
    weight=torch_param[key].detach().cpu().numpy()
    if key=="fc.weight":
        weight=weight.transpose()
    paddle_dict[key]=weight
paddle.save(paddle_dict,r'E:\RESNET50_paddle.pdparams')


# In[10]:


paddle_checkpoint = paddle.load(r'E:\resnet50_paddle.pdparams')
model_paddle.set_state_dict(paddle_checkpoint)


# In[11]:


np.random.seed(1)
a=np.random.random((1,3,224,224))
x1=torch.tensor(a,dtype=torch.float32)
x2=paddle.to_tensor(a, dtype='float32')
y1=model(x1).detach().cpu().numpy()
y2=model_paddle(x2).detach().cpu().numpy()
print(np.allclose(y1,y2,rtol=1e-09, atol=1e-04))


# In[12]:


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#print(rel_error(y1,y2))
print(y1-y2)


# In[ ]:




