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
    ctr_y, ctr_x = ctr[c]
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
# 找出每个目标框最大IOU的anchor框的index，共2个
gt_argmax_ious = ious.argmax(axis=0)
# 获取每个目标框的最大IOU值
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
# 找出每个anchor框最大IOU的目标框index，共8940个
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


#先假设有一半正例，一半反例
pos_ratio = 0.5
n_sample = 256
n_pos = pos_ratio * n_sample

#随机获取n_pos个正例
pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size=len(pos_index) - n_pos, replace=False)
    label[disable_index] = -1

n_neg = n_sample - np.sum(label == 1)
neg_index = np.where(label == 0)[0]

if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size=len(neg_index) - n_neg, replace=False)
    label[disable_index] = -1#这里是不是要改成+1呢?
print(np.sum(label == 1))
print(np.sum(label == 0))

# 现在让我们用具有最大iou的ground truth对象为每个anchor box分配位置
# 注意我们将为所有有效的anchor box分配anchor locs，而不考虑其标签，稍后计算损失时，我们可用简单的过滤器删除他们。

# 为每个点都找到其9个框中iou最高的一个框
max_iou_bbox = bbox[argmax_ious]
print(max_iou_bbox)
print(max_iou_bbox.shape)
# (8940, 4)

# 有效的anchor的中心点和宽高
height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
ctr_y = valid_anchor_boxes[:, 0] + 0.5 * height
ctr_x = valid_anchor_boxes[:, 1] + 0.5 * width
# 有效anchor对应目标框的中心点和宽高
base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

# 有效anchor转为目标框的系数(dy, dx为平移系数；dh, dw是放缩系数)
eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
print(anchor_locs.shape)

anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[valid_anchor_index] = label

# 懵逼
# anchor_locations:每个有效anchor框转换为目标框的系数
anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[valid_anchor_index, :] = anchor_locs

###Region Proposal Network###

import torch.nn as nn
mid_channels = 512
in_channels = 512
# number of anchors at each location
n_anchor = 9
conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

# conv sliding layer
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

# Regression layer
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

# classification layer
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()

# out_map is 由CNN得到的特征图谱
x = conv1(out_map)
#回归层，计算有效anchor转为目标框的四个系数
pred_anchor_locs = reg_layer(x)
#分类层，判断该anchor是否可以捕获目标
pred_cls_scores = cls_layer(x)
# ((1L, 18L, 50L, 50L), (1L, 36L, 50L, 50L))
print(pred_cls_scores.shape, pred_anchor_locs.shape)

# permute用来改变张量的维度顺序，简而言之，使张量变形
# contiguous重新开辟一块空间来存储底层的一维数组，使用此函数后才可以调用view()否则报错
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
print(pred_anchor_locs.shape)
# Out: torch.Size([1, 22500, 4])

pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
print(pred_cls_scores.shape)
# Out torch.Size([1, 50, 50, 18])

objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
print(objectness_score.shape)
# Out torch.Size([1, 22500])

pred_cls_scores = pred_cls_scores.view(1, -1, 2)
print(pred_cls_scores.shape)
# Out torch.size([1, 22500, 2])

# Generating proposals to feed Fast R-CNN network
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

# 转换anchor格式从 y1, x1, y2, x2 到 ctr_x, ctr_y, h, w
anc_height = anchors[:, 2] - anchors[:, 0]
anc_width = anchors[:, 3] - anchors[:, 1]
anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

# 懵逼!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 懵逼!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 懵逼!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 根据预测的四个系数，将anchor框通过平移和缩放转化为预测的目标框
pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()
objectness_score_numpy = objectness_score[0].data.numpy()
# python的双冒号和matlab很像，起始:终止:步长
# 把预测的anchor矩阵中的存储的各个坐标抽出来
# []
dy = pred_anchor_locs_numpy[:, 0::4]
dx = pred_anchor_locs_numpy[:, 1::4]
dh = pred_anchor_locs_numpy[:, 2::4]
dw = pred_anchor_locs_numpy[:, 3::4]

# anc_height.shape (22500,)
# ctr_y.shape (22500,)

# np.newaxis在原来数据基础上增加一个维度
ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
h = np.exp(dh) * anc_height[:, np.newaxis]
w = np.exp(dw) * anc_width[:, np.newaxis]



# ROI : region of interest
# 将预测目标框转换为[y1, x1, y2, x2]格式
# roi是个大矩阵[22500, 4]
roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=pred_anchor_locs_numpy.dtype)
roi[:, 0::4] = ctr_y - 0.5 * h
roi[:, 1::4] = ctr_x - 0.5 * w
roi[:, 2::4] = ctr_y + 0.5 * h
roi[:, 3::4] = ctr_x + 0.5 * w

# 剪辑预测框到图像上
img_size = (800, 800)
roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])
print(roi.shape)

# 去除高度和宽度小于threshold的预测框
# 为什么要去除这些预测框？
hs = roi[:, 2] - roi[:, 0]
ws = roi[:, 3] - roi[:, 1]
keep = np.where((hs >= min_size) & (ws >= min_size))[0]
roi = roi[keep, :]
score = objectness_score_numpy[keep]

order = score.ravel().argsort()[::-1]
print(order.shape)
# (22500, )

# 取前几个预测框pre_nms_topN(如训练时12000，测试时300)
order = order[:n_train_pre_nms]
roi = roi[order, :]
print(roi.shape)
# (12000, 4)

# nms(非极大抑制)计算
# 去除和极大值anchor框IOU大于70%的框
# 去除相交的框，保留score大，且基本无相交的框
nms_thresh = 0.7
y1 = roi[:, 0]
x1 = roi[:, 1]
y2 = roi[:, 2]
x2 = roi[:, 3]
# y1, x1, y2, x2现在代表原图上ROI的左下和右上的坐标列向量

areas = (x2 - x1 + 1) * (y2 - y1 + 1)
# areas是ROI的面积列向量
score = score[order]
order = score.argsort()[::-1]
print(order)
keep = []

while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.maximum(x2[i], x2[order[1:]])
    yy2 = np.maximum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    ovr = inter / (areas[i] + areas[order[1:]] - inter)

    inds = np.where(ovr <= nms_thresh)[0]
    order = order[inds + 1]

keep = keep[:n_train_post_nms]
roi = roi[keep]
print(roi.shape)


###Proposal targets###

n_sample = 128
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5
neg_iou_thresh_lo = 0.0

# 找到每个ground-truth目标(真实目标框)与region proposal(预测目标框)的IOU
ious = np.empty((len(roi), 2), dtype=np.float32)
ious.fill(0)
for num1, i in enumerate(roi):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])

        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0.

        ious[num1, num2] = iou
print(ious.shape)

gt_assignment = ious.argmax(axis=1)
max_iou = ious.max(axis=1)
print(gt_assignment)
print(max_iou)

# 为每个proposal分配标签:
gt_roi_label = labels[gt_assignment]
print(gt_roi_label)

# 希望只保留n_sample*pos_ratio（128*0.25=32）个前景样本，因此如果只得到少于32个正样本，保持原状。
# 如果得到多余32个前景目标，从中采样32个样本
# 前景目标可理解成背景的反义词，是有清晰语义的目标
pos_roi_per_image = 32
pos_index = np.where(max_iou >= pos_iou_thresh)[0]
pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
# 这个if是干嘛用的
if pos_index.size > 0:
    pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
print(pos_index)
# 针对负[背景]region proposal进行相似处理
neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
neg_roi_per_this_image = n_sample - pos_roi_per_this_image
neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
if neg_index.size > 0:
   neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
# print(neg_roi_per_this_image)
print(neg_index)

#有点懵
keep_index = np.append(pos_index, neg_index)
gt_roi_labels = gt_roi_label[keep_index]
# 将negtive labels全部置为0
gt_roi_labels[pos_roi_per_this_image:] = 0
sample_roi = roi[keep_index]
print(sample_roi.shape)
# 目标框
bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]
print(bbox_for_sampled_roi.shape)

height = sample_roi[:, 2] - sample_roi[:, 0]
width = sample_roi[:, 3] - sample_roi[:, 1]
ctr_y = sample_roi[:, 0] + 0.5 * height
ctr_x = sample_roi[:, 1] + 0.5 * width
base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
base_ctr_y = bbox_for_sampled_roi[:, 0] + 0.5 * base_height
base_ctr_x = bbox_for_sampled_roi[:, 1] + 0.5 * base_width


eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)

# 为啥，不应该是训练的吗？咋还直接算出来了？
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

# v-vertical vstack()把矩阵垂直摞起来.
gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()
print(gt_roi_locs.shape)

rois = torch.from_numpy(sample_roi).float()
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()
print(rois.shape, roi_indices.shape)

indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
indices_and_rois = xy_indices_and_rois.contiguous()
print(xy_indices_and_rois.shape)



###ROI Pooling###


size = (7, 7)
adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(size[0], size[1])
output = []
rois = indices_and_rois.float()
rois[:, 1:].mul_(1/16.0)
rois = rois.long()
num_rois = rois.size(0)
for i in range(num_rois):
    roi = rois[i]
    im_idx = roi[0]
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
    output.append(adaptive_max_pool(im)[0].data)
output = torch.stack(output)
print(output.size())

k = output.view(output.size(0), -1)
print(k.shape)


###分类层###
roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096),
                                      nn.Linear(4096, 4096)])
# (VOC 20 classes + 1 background. Each will have 4 co-ordinates)
cls_loc = nn.Linear(4096, 21 * 4)
cls_loc.weight.data.normal_(0, 0.01)

cls_loc.bias.data.zero_()
# (VOC 20 classes + 1 background)
score = nn.Linear(4096, 21)

k = torch.autograd.Variable(k)
k = roi_head_classifier(k)
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)
print(roi_cls_loc.data.shape, roi_cls_score.data.shape)

# Faster RCNN 损失函数
print(pred_anchor_locs.shape)  # torch.Size([1, 22500, 4])  # RPN网络预测的坐标系数
print(pred_cls_scores.shape)  # torch.Size([1, 22500, 2])   # RPN网络预测的类别
print(anchor_locations.shape)  # (22500, 4)  # anchor对应的实际坐标系数
print(anchor_labels.shape)  # (22500,)       # anchor的实际类别

# 将输入输出排成一行
rpn_loc = pred_anchor_locs[0]
rpn_score = pred_cls_scores[0]
gt_rpn_loc = torch.from_numpy(anchor_locations)
gt_rpn_score = torch.from_numpy(anchor_labels)
print(rpn_loc.shape, rpn_score.shape, gt_rpn_loc.shape, gt_rpn_score.shape)

# 对classification用交叉熵损失
gt_rpn_score = torch.autograd.Variable(gt_rpn_score.long())
rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_score, gt_rpn_score, ignore_index=-1)
print(rpn_cls_loss)

# 对于Regression 使用 smooth L1损失
pos = gt_rpn_score.data > 0
mask = pos.unsqueeze(1).expand_as(rpn_loc)
print(mask.shape)  # (22500L, 4L)

# ?????????????????????????????????
# 取有正数标签的边界区域
mask_loc_preds = rpn_loc[mask].view(-1, 4)
mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)
print(mask_loc_preds.shape, mask_loc_targets.shape)

# regression损失应用如下
x = np.abs(mask_loc_targets.numpy() - mask_loc_preds.data.numpy())
print(x.shape)

rpn_loc_loss = ((x < 1) * 0.5 * x**2) + ((x >= 1) * (x-0.5))
rpn_loc_loss = rpn_loc_loss.sum()
print(rpn_loc_loss)

N_reg = (gt_rpn_score > 0).float().sum()
N_reg = np.squeeze(N_reg.data.numpy())

# ??????????????????????????????
print("N_reg: {}, {}".format(N_reg, N_reg.shape))
rpn_loc_loss = rpn_loc_loss / N_reg
rpn_loc_loss = np.float32(rpn_loc_loss)
# rpn_loc_loss = torch.autograd.Variable(torch.from_numpy(rpn_loc_loss))
rpn_lambda = 10.
rpn_cls_loss = np.squeeze(rpn_cls_loss.data.numpy())
print("rpn_cls_loss: {}".format(rpn_cls_loss))  # 0.693146109581
print('rpn_loc_loss: {}'.format(rpn_loc_loss))  # 0.0646051466465
rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
print("rpn_loss: {}".format(rpn_loss))  # 1.33919757605



# Faster R-CNN 损失函数

# 预测
print(roi_cls_loc.shape)  # # torch.Size([128, 84])
print(roi_cls_score.shape)  # torch.Size([128, 21])

# 真实
print(gt_roi_locs.shape)  # (128, 4)
print(gt_roi_labels.shape)  # (128, )

gt_roi_loc = torch.from_numpy(gt_roi_locs)
gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()
print(gt_roi_loc.shape, gt_roi_label.shape)  # torch.Size([128, 4]) torch.Size([128])

# 分类损失
gt_roi_label = torch.autograd.Variable(gt_roi_label)
roi_cls_loss = torch.nn.functional.cross_entropy(roi_cls_score, gt_roi_label, ignore_index=-1)
print(roi_cls_loss)  # Variable containing:  3.0515


# 回归损失
n_sample = roi_cls_loc.shape[0]
roi_loc = roi_cls_loc.view(n_sample, -1, 4)
print(roi_loc.shape)  # (128L, 21L, 4L)

roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]
print(roi_loc.shape)  # torch.Size([128, 4])


# 用计算RPN网络回归损失的方法计算回归损失
# roi_loc_loss = REGLoss(roi_loc, gt_roi_loc)

pos = gt_roi_label.data > 0  # Regression 损失也被应用在有正标签的边界区域中
mask = pos.unsqueeze(1).expand_as(roi_loc)
print(mask.shape)  # (128, 4L)

# 现在取有正数标签的边界区域
mask_loc_preds = roi_loc[mask].view(-1, 4)
mask_loc_targets = gt_roi_loc[mask].view(-1, 4)
print(mask_loc_preds.shape, mask_loc_targets.shape)  # ((19L, 4L), (19L, 4L))


x = np.abs(mask_loc_targets.numpy() - mask_loc_preds.data.numpy())
print(x.shape)  # (19, 4)

roi_loc_loss = ((x < 1) * 0.5 * x**2) + ((x >= 1) * (x-0.5))
print(roi_loc_loss.sum())  # 1.4645805211187053


N_reg = (gt_roi_label > 0).float().sum()
N_reg = np.squeeze(N_reg.data.numpy())
roi_loc_loss = roi_loc_loss.sum() / N_reg
roi_loc_loss = np.float32(roi_loc_loss)
print(roi_loc_loss)  # 0.077294916
# roi_loc_loss = torch.autograd.Variable(torch.from_numpy(roi_loc_loss))


# ROI损失总和
roi_lambda = 10.
roi_cls_loss = np.squeeze(roi_cls_loss.data.numpy())
roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)
print(roi_loss)  # 3.810348778963089


total_loss = rpn_loss + roi_loss

print(total_loss)  # 5.149546355009079





