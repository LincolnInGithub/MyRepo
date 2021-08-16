# MyRepo
My first repository
It is a test repo.  
2021/8/16 Update RESNET-50.py    
Pytorch框架下的RESNET50网络转写成paddle框架  
并将权重参数也进行了转换  
需注意的是BatchNorm2D层的参数：在train和test时前向计算的不同以及是否跟踪训练过程中的统计特性，均会导致输出的差异  
最后两个框架下网络输出的比较  
![捕获](https://user-images.githubusercontent.com/88335850/129529091-607c5c70-da92-41a2-9e37-ddd7a94ecda0.JPG)
