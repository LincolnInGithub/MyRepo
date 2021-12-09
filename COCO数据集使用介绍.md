# 一、简介
关于MS COCO数据集的使用说明，转载自 https://blog.csdn.net/qq_29051413/article/details/103448318?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0.base  
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
## 3.2 非通用字段  
### 3.2.1 Object Detection（目标检测）  
  以检测任务为例，对于每一张图片，至少包含一个对象，COCO数据集对每一个对象进行描述，而不是对一张图片。每个对象都包含一系列字段，包括对象的类别id和mask码，mask码的分割格式取决于图像里的对象数目，当一张图像里就一个对象时（iscrowd=0），mask码用RLE格式，当大于一个对象时（iscrowd=1），采用polyhon格式。  
  #### 需要注意的是，annotations里的image_id就是前面images中的id ！以此将image与annotation对应上。
![image](https://user-images.githubusercontent.com/88335850/129651079-d2c5a6f2-9852-4ed5-a20b-0f03c3f6145e.png)  
### 3.2.2 Keypoint Detection（关键点检测）  
  与检测任务一样，一个图像包干若干对象，一个对象对应一个keypoint注释，一个keypoint注释包含对象注释的所有数据（包括id、bbox等）和两个附加字段。  
  首先，key为”keypoints“的value是一个长度为3k的数组，其中k是类别定义的关键点总数（例如人体姿态关键点的k为17）.每个关键点都有一个0索引的位置x、y和可见性标志v（v=0表示未标记，此时x=y=0；v=1时表示标记，但不可见，不可见的原因在于被遮挡了；v=2时表示标记且可见），如果一个关键点落在对象段内，则认为是可见的。  
![image](https://user-images.githubusercontent.com/88335850/129651184-5183c480-f0ef-4ff7-9e4c-99c2d4abcd35.png)  
其中，[cloned]表示从上面定义的Object Detection注释中复制的字段。因为keypoint的json文件包含detection任务所需的字段。  
### 3.2.3 Stuff Segmentation（实例分割）
  分割任务的对象注释格式与上面的Object Detection相同且完全兼容（除了iscrowd是不必要的，默认值为0），分割任务主要字段是“segmentation”

### 3.2.4 Panoptic Segmentation（全景分割）
  对于全景分割任务，每个注释结构是每个图像的注释，而不是每个对象的注释，与上面三个有区别。每个图像的注释有两个部分：1）存储与类无关的图像分割的PNG；2）存储每个图像段的语义信息的JSON结构。  

1、要将注释与图像匹配，使用image_id字段（即：annotation.image_id==image.id）；  
2、对于每个注释，每个像素段的id都存储为一个单独的PNG，PNG位于与JSON同名的文件夹中。每个分割都有唯一的id，未标记的像素为0；  
3、对于每个注释，每个语义信息都存储在annotation.segments_info. segment_info.id，该存储段存储唯一的id，并用于从PNG检索相应的掩码（ids==segment_info.id）。iscrowd表示段内包含一组对象。bbox和area字段表示附加信息。  
![image](https://user-images.githubusercontent.com/88335850/129651493-eb19e542-4f8f-47e0-b3aa-04d88e6297e5.png)    
（⬆⬆⬆这一部分描述不太全面⬆⬆⬆）  
### 3.2.5 Image Captioning（图像字幕）  
  图像字幕任务的注释用于存储图像标题，每个标题描述指定的图像，每个图像至少有5个标题。  
![image](https://user-images.githubusercontent.com/88335850/129651916-d66f5490-ade5-4097-848a-8db2cb004697.png)   
# 四、数据集的使用（Python）  
## 4.1 COCOAPI  
  通过上面的介绍可知COCO数据集的标签有一定复杂度，需要通过各种文件读取来获取注释，为了让用户更好地使用 COCO 数据集, COCO 提供了各种 API，即下面要介绍的cocoapi。  

## 4.2 API安装  
首先安装依赖包  
![image](https://user-images.githubusercontent.com/88335850/129652037-a66cce79-9931-42a4-8672-0ece651ccd81.png)  
 git下载地址：https://github.com/cocodataset/cocoapi.git  
下载后进入到PythonAPI目录下  
![image](https://user-images.githubusercontent.com/88335850/129652099-c386a6e3-0971-47db-87d9-24f6ea8b7456.png)  

## 4.3 COCO API使用（官方例程）  
   安装完在site-packages文件夹可以看到pycocotools包，该包是COCO数据集的Python API，帮助加载、解析和可视化COCO中的注释。使用API的方法是直接使用API提供的函数加载注释文件和读取Python字典。API函数定义如下：  

1、COCO：加载COCO注释文件并准备数据结构的COCO api类。  
2、decodeMask：通过运行长度编码解码二进制掩码M。  
3、encodeMask：使用运行长度编码对二进制掩码M进行编码。  
4、getAnnIds：得到满足给定过滤条件的annotation的id。  
5、getCatIds：获得满足给定过滤条件的category的id。  
6、getImgIds：得到满足给定过滤条件的imgage的id。  
7、loadAnns：使用指定的id加载annotation。  
8、loadCats：使用指定的id加载category。  
9、loadImgs：使用指定的id加载imgage。  
10、annToMask：将注释中的segmentation转换为二进制mask。  
11、showAnns：显示指定的annotation。  
12、loadRes：加载算法结果并创建访问它们的API。  
13、download：从mscoco.org服务器下载COCO图像。  
   下面展示了数据加载、解析和可视化注释等内容，步骤如下：  
### 1、首先导入必要的包  
![image](https://user-images.githubusercontent.com/88335850/129652288-5d1e5b78-2942-41ea-932a-f34d6b9d0a11.png)  
### 2、定义annotation文件路径（以“instances_val2014.json”为例）  
![image](https://user-images.githubusercontent.com/88335850/129652355-267fc728-c582-41dd-9a8c-b16451797d36.png)  
### 3、读取instances_val2014.json文件到COCO类  
![image](https://user-images.githubusercontent.com/88335850/129652454-f4872995-4cba-4bf3-b5ac-cd98d4c1e3bb.png)    
#### 输出如下：  
loading annotations into memory…  
Done (t=4.19s)  
creating index…  
index created!  
### 4、COCO图像类别的读取  
![image](https://user-images.githubusercontent.com/88335850/129652516-9a1a46bb-f877-4e88-ac18-917980be7c40.png)  
#### 输出如下：
![image](https://user-images.githubusercontent.com/88335850/129652570-4f795a75-742d-41a4-9b22-e4e2456ee95e.png)  
### 5、COCO原始图像读取  
![image](https://user-images.githubusercontent.com/88335850/129652634-1c9f856f-7ac4-44d1-adf0-85debdae1a22.png)  
#### 输出如下：  
![image](https://user-images.githubusercontent.com/88335850/129652670-3573fd46-1b57-416a-8b1c-0e46a784a50c.png)  
### 6、加载并显示annotations    
![image](https://user-images.githubusercontent.com/88335850/129652750-37156785-07ae-4129-b0ab-8461760f99e7.png)  

#### 输出如下：  
![image](https://user-images.githubusercontent.com/88335850/129652784-8e1a7973-c7ef-4985-b749-7d1326ebb230.png)  
### 7、加载并显示person_keypoints_2014.json的annotations  
![image](https://user-images.githubusercontent.com/88335850/129652827-fe7832e0-4a8b-43b6-af48-8dddfcadf61b.png)  

#### 输出如下：  
![image](https://user-images.githubusercontent.com/88335850/129652867-6d92dee7-57fb-408b-bd20-b2126a1b5bd7.png)  
### 8、加载并显示captions_2014.json.json的annotations  
![image](https://user-images.githubusercontent.com/88335850/129652933-de185e95-355f-4ddc-a2e2-5e3d3f7ddbac.png)  

#### 输出如下：   
![image](https://user-images.githubusercontent.com/88335850/129653082-26c4ee47-0ef3-4996-9fab-184f2466417a.png)  
 
# 五、COCO数据集的评估  
## 5.1 IOU值计算  
![image](https://user-images.githubusercontent.com/88335850/129653224-9e2d867b-a60f-4fc1-b427-8d84e30663f6.png)  

## 5.2 COCO评估指标    
以下为官方公布的指标定义：  
![image](https://user-images.githubusercontent.com/88335850/129653287-1166f219-895a-4c2c-b387-67190c25db11.png)     
1、除非另有说明，否则AP和AR在多个交汇点（IoU）值上取平均值，使用0.50到0.95共10个IOU阈值下的mAP求平均，结果就是COCO数据集定义的AP，与只用一个IOU=0.50下计算的AP相比，是一个突破；  
2、AP是所有类别的平均值。传统上，这被称为“平均准确度”（mAP，mean average precision）。官方没有区分AP和mAP（同样是AR和mAR），并假定从上下文中可以清楚地看出差异。  
3、AP（所有10个IoU阈值和所有80个类别的平均值）将决定赢家。在考虑COCO性能时，这应该被认为是最重要的一个指标。  
4、在COCO中，比大物体相比有更多的小物体。具体地说，大约41％的物体很小（area<322），34％是中等（322 < area < 962)），24％大（area > 962）。测量的面积（area）是分割掩码（segmentation mask）中的像素数量。  
5、AR是在每个图像中检测到固定数量的最大召回（recall），在类别和IoU上平均。AR与proposal evaluation中使用的同名度量相关，但是按类别计算。  
6、所有度量标准允许每个图像（在所有类别中）最多100个最高得分检测进行计算。  
7、除了IoU计算（分别在框（box）或掩码（mask）上执行）之外，用边界框和分割掩码检测的评估度量在所有方面是相同的。  
## 5.3 COCO结果文件统一格式  
### Object Detection  
   对于边界框的检测，请使用以下格式:  
![image](https://user-images.githubusercontent.com/88335850/129653555-605bd98f-630a-4f1f-97c6-00f379049b83.png)  
框坐标是从图像左上角测量的浮点数(并且是0索引的)。官方建议将坐标舍入到最接近十分之一像素的位置，以减少JSON文件的大小。    
   对于对象segments的检测(实例分割)，请使用以下格式:    
  ![image](https://user-images.githubusercontent.com/88335850/129653919-dbdd65fc-33f8-4902-a28d-4272479858b1.png)  
### Keypoint Detection  
![image](https://user-images.githubusercontent.com/88335850/129654034-62a66dc2-dd18-4d57-afe8-ebc2654064c0.png)  
关键点坐标是从左上角图像角测量的浮点数(并且是0索引的)。官方建议四舍五入坐标到最近的像素，以减少文件大小。还请注意，目前还没有使用vi的可视性标志(除了控制可视化之外)，官方建议简单地设置vi=1。  
### Stuff Segmentation  
![image](https://user-images.githubusercontent.com/88335850/129654085-3dd80465-98e1-466e-8b18-8f22a6cf8766.png)  
除了不需要score字段外，Stuff 分割格式与Object分割格式相同。注意:官方建议用单个二进制掩码对图像中出现的每个标签进行编码。二进制掩码应该使用MaskApi函数encode()通过RLE进行编码。例如，参见cocostuffhelper.py中的segmentationToCocoResult()。为了方便，官方还提供了JSON和png格式之间的转换脚本。  
### Panoptic Segmentation  
![image](https://user-images.githubusercontent.com/88335850/129654154-f051d8fc-1baa-478b-bf5e-9bee9f5bc653.png)  
### Image Captioning  
![image](https://user-images.githubusercontent.com/88335850/129654205-580bcc4e-e67e-4c6d-b6d9-16a2af56cc20.png)  

## 5.4 COCOEVAL API使用（官方例程）  
   COCO还提供了一个计算评估指标的API，即当自己的模型按照官方定义的格式输出后，可以使用API进行快速评估模型的一系列指标。下面是  
### 1、导入必要的包  
![image](https://user-images.githubusercontent.com/88335850/129654256-fe7bc4f5-da5a-4251-ab29-7d59a9f4484e.png)  
### 2、选择任务  
![image](https://user-images.githubusercontent.com/88335850/129654292-6c03bc16-caa0-4b83-bf63-af0b93c4a83a.png)  
### 输出如下：  
Running demo for bbox results.  
### 3、加载json注释文件（即：Ground Truth）  
![image](https://user-images.githubusercontent.com/88335850/129654374-6764cd3c-91b7-4112-a42f-87b8c09f553e.png)  
### 输出如下：  
loading annotations into memory…  
Done (t=3.16s)  
creating index…  
index created!  
### 4、加载result文件（即：Predict）  
   COCO.loadRes(resFile)返回的也是一个COCO类，与COCO(annFile)不同的是，前者加载官方规定格式的result文件，后者加载官方提供的json文件。  
   ![image](https://user-images.githubusercontent.com/88335850/129654461-abdb50c3-5116-462c-9f34-0ca691dd4f1d.png)  
### 输出如下：  
Loading and preparing results…  
DONE (t=0.03s)  
creating index…  
index created!    
### 5、使用测试集当中的100张图片进行评估  
![image](https://user-images.githubusercontent.com/88335850/129654523-d5ddc61c-24b3-4867-a520-bdf2b84f784e.png)  
### 6、执行评估  
![image](https://user-images.githubusercontent.com/88335850/129654743-51b7d9e7-bf36-463c-b9b5-87b7df9522de.png)  
### 输出如下：  
Running per image evaluation…  
Evaluate annotation type bbox  
DONE (t=0.21s).  
Accumulating evaluation results…  
DONE (t=0.25s).  
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.505  
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.697  
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=100 ] = 0.573  
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.586  
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519  
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501  
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 1 ] = 0.387  
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 10 ] = 0.594  
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.595  
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.640  
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.566  
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564  

