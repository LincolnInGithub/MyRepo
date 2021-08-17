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
  {
	"info"			:info,	
	"images"		:[image],
	"annotations"	:[annotation],
	"licenses"		:[license],
}

