# 一、简介
官方网站：http://cocodataset.org/  
全称：Microsoft Common Objects in Context （MS COCO）  
支持任务：Detection、Keypoints、Stuff、Panoptic、Captions  
说明：COCO数据集目前有三个版本，即2014、2015和2017，其中2015版只有测试集，其他两个有训练集、验证集和测试集。  

# 二、数据集下载  
直接官网下载（需要FQ）  

# 三、数据集说明  
COCO数据集包括两大部分：Images和Annotations  
Images：“任务+版本”命名的文件夹（例如：train2014），里面为xxx.jpg的图像文件；  
Annotations：文件夹，里面为xxx.json格式的文本文件（例如：instances_train2014.json）；  
使用COCO数据集的核心就在于xxx.json文件的读取操作，下面详细介绍annotation文件的组织结构和使用方法。  
## 3.1 通用字段  
  COCO有五种注释类型对应五种任务:目标检测、关键点检测、实物分割、全景分割和图像描述。注释使用JSON文件存储。每个xxx.json的内容整体为一个字典，key为“info”、“images“、”annotations“和”licenses“，如下所示：  
![image](https://user-images.githubusercontent.com/88335850/129649949-dc3bc6c2-f2ec-4f7f-949f-2949719a01ab.png)
value为对应的数据类型，其中，info是一个字典，images是一个list，annotations是一个list，licenses是一个list。除annotation外，每部分的内容定义如下：  
![image](https://user-images.githubusercontent.com/88335850/129650255-f9fa25f5-aea4-4b82-99d7-bb88e47f7b7b.png)  
key为”annotation“的value对应不同的xxx.json略有不同，但表示内容含义是一样的，即对图片和实例的描述。同时除了annotation外，还有一个key为”categories“表示类别。以下分别对不同任务的annotation和categories进行说明。  


