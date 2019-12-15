import torch
import torchvision
from PIL import Image, ImageDraw
import numpy as np

###Convolutional Layers###

img_tensor = torch.zeros((1,3,800,800)).float()
print(img_tensor.shape)

#梯度值
img_var = torch.autograd.Variable(img_tensor)
#vgg16作为模型
model = torchvision.models.vgg16(pretrained=False)
fe = list(model.features)
print(fe)

#生成一个列表
req_features = []

k = img_var.clone()

#这里的i(k)不太懂0.0
for i in fe:
    print(i)
    k = i(k)
    print(k.data.shape)
    if k.size()[2] < 800//16:
        break
    req_features.append(i)
    out_channels = k.size()[1]
print(len(req_features))
print(out_channels)

#打印各层情况
for f in req_features:
    print(f)

#"*"是像把名单(列表)传给这个函数一样
#"**"则是字典
#torch.nn.Sequential是按时序将你所给的层串成一个model(也就是文中的特征提取器)
faster_rcnn_fe_extractor = torch.nn.Sequential(*req_features)
out_map = faster_rcnn_fe_extractor(img_var)
print(out_map.size())
#输出特征图谱尺寸torch.Size([1, 512, 50, 50])
#特征提取步骤中只有MaxPooling Layer将图片尺寸变为原来1/2，Conv和ReLu不改变图片大小

#一个特征对应原图片中16*16个像素点区域
fe_size = (800//16)
# ctr_x , ctr_y 每个特征点对应原图片区域的右下方坐标
ctr_x = np.arange(16, (fe_size+1) * 16, 16)
ctr_y = np.arange(16, (fe_size+1) * 16, 16)
print(len(ctr_x)) #共50 * 50个特征点，将原图片分割成50*50=2500个区域

index = 0
# ctr:每个特征点对应的原图片的中心点
ctr = dict()
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index] = [-1, -1]
        ctr[index][1] = ctr_x[x] - 8
        ctr[index][0] = ctr_y[y] - 8
        index += 1
print(len(ctr))
# 2500个对应原图的坐标点

###生成anchor框###

#用于生成anchor框的一些参数
ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]
sub_sample = 16

#初始化:每个区域有9个anchors候选框，每个候选框坐标(y1,x1,y2,x2)
anchors = np.zeros(((fe_size * fe_size * 9), 4))
print(anchors.shape)
index = 0

#将候选框的坐标赋值到anchors
for c in ctr:
    ctr_y, ctr_x = ctr[c]#不懂这里
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            #anchor scales是针对特征图的，所以需要乘以下采样"sub_sample"
            h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
            w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratios[i])
            anchors[index, 0] = ctr_y - h / 2.
            anchors[index, 1] = ctr_x - w / 2.
            anchors[index, 2] = ctr_y + h / 2.
            anchors[index, 3] = ctr_x + w / 2.
            index += 1
#(22500 , 4)
print(anchors.shape)

#有点懵
img_npy = img_tensor.numpy()
img_npy = np.transpose(img_npy[0], (1, 2, 0)).astype(np.float32)
img = Image.fromarray(np.uint8(img_npy))
draw = ImageDraw.Draw(img)

#假设 图片中的两个目标框"ground-truth"
#[y1, x1, y2, x2]format
#asarray省内存
bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32)
draw.rectangle([(30, 20), (500, 400)], outline=(100, 255, 0))
draw.rectangle([(400, 300), (600, 500)], outline=(100, 255, 0))

#假设 图片中两个目标框分别对应的标签
# 0 represents background
labels = np.asarray([6, 8], dtype=np.int8)

# 去除坐标出界的边框，保留图片内的框——图片内框

valid_anchor_index = np.where(
       (anchors[:, 0] >= 0) &
       (anchors[:, 1] >= 0) &
       (anchors[:, 2] <= 800) &
       (anchors[:, 3] <= 800)
   )[0]  # 该函数返回数组中满足条件的index
print(valid_anchor_index.shape)  # (8940,)，表明有8940个框满足条件

valid_anchor_boxes = anchors[valid_anchor_index]
print(valid_anchor_boxes.shape)

#计算有效anchor框"valid_anchor_boxes"与目标框"bbox"
#这里的2表示的是因为我们之前只是假设有两个目标框
ious = np.empty((len(valid_anchor_boxes), 2), dtype = np.float32)
ious.fill(0)
print(bbox)

for num1, i in enumerate(valid_anchor_boxes):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)#anchor框面积
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)#目标框面积
        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        # [x1, y1]左下点
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        # [x2, y2]右上点
        if((inter_x1 < inter_x2) and (inter_y1 < inter_y2)):
            iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0.

        ious[num1, num2] = iou
        #两两之间的iou
#所以此处的ious是8940个候选框和2个目标框的两两计算出的iou的矩阵
print(ious.shape)
#找出每个目标框最大IOU的anchor框的index，共2个
gt_argmax_ious = ious.argmax(axis=0)
#获取每个目标框的最大IOU值
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
#找出每个anchor框最大IOU的目标框index，共8940个
argmax_ious = ious.argmax(axis=1)
# 获取每个anchor框的最大IOU值，与argmax_ious对应
max_ious = ious[np.arange(len(valid_anchor_index)), argmax_ious]

gt_argmax_ious = np.where(ious == gt_max_ious)[0]
print(gt_argmax_ious.shape)

pos_iou_threshold = 0.7
neg_iou_threshold = 0.3
label = np.empty((len(valid_anchor_index), ), dtype=np.int32)
label.fill(-1)
print(label.shape)
#某个点的9个框中的最大的iou都比负阙值小，则舍弃该点
label[max_ious < neg_iou_threshold] = 0
#全局极大必为1
label[gt_argmax_ious] = 1
#某个点的9个框的最大IOU比正阙值大，则保留
label[max_ious >= pos_iou_threshold] = 1


#??????????????????????????????
pos_ratio = 0.5
n_sample = 256
n_pos = pos_ratio * n_sample

#随机获取n_pos个正例
#懵??????????????????????
pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size=len(pos_index) - n_pos, replace=False)
    label[disable_index] = -1

n_neg = n_sample - np.sum(label == 1)
neg_index = np.where(label == 0)[0]

if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size=len(neg_index) - n_neg, replace=False)
    label[disable_index] = -1
print(np.sum(label == 1))
print(np.sum(label == 0))























