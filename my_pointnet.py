import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class InputTransform(nn.Module):
    def __init__(self):
        super(InputTransform, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
            
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity_matrix = Variable(torch.from_numpy(np.array
                        ([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype
                        (np.float32))).view(1, 9).repeat(batchsize, 1)
        
        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        x = x + identity_matrix
        x = x.view(-1, 3, 3)
        return x
    
class FeatureTransform(nn.Module):
    def __init__(self, k=64):
        super(FeatureTransform, self).__init__()
        
        self.k = k

        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden_dynamic = torch.eye(self.k).flatten().to(x.device)
        identity_matrix = iden_dynamic.view(1, self.k * self.k).repeat(batchsize, 1)

        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        x = x + identity_matrix
        x = x.view(-1, self.k, self.k)
        return x

class PointNetMain(nn.Module):
    def __init__(self, global_feature = True, input_dims = 4, feature_transfrom = False):
        super(PointNetMain, self).__init__()

        self.Tnet = InputTransform(k = input_dims)
        self.global_feature = global_feature
        self.feature_transform = feature_transfrom

        self.conv1 = torch.nn.Conv1d(input_dims, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.fstn = FeatureTransform(k=64)
    
    def forward(self, x):
        num_pts = x.size()[2]
        trans = self.Tnet(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feature:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_pts)
            return torch.cat ([x, pointfeat], 1), trans, trans_feat
    
class PointNetClassification(nn.Module):
    def __init__(self, k=2, feature_transform = False):
        super(PointNetClassification, self).__init__()
        self.feature_transform = feature_transform
        
        self.feat = PointNetMain(global_feature=True, feature_transform=feature_transform, input_dims=3)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans_feat

class PointNetExtend(nn.Module):
    def __init__(self, category_k, seg_k):
        super(PointNetExtend, self).__init__()
        self.category_k = category_k
        self.seg_k = seg_k
        self.Tnet = InputTransform()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        self.fstn = FeatureTransform(k=128)

        ## Classification Network
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, category_k)
        self.dropout = nn.Dropout(p=0.3)
        self.bnc1 = nn.BatchNorm1d(256)
        self.bnc2 = nn.BatchNorm1d(256)

        ## Segmentation Network
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, seg_k, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):

        batchsize,_ , n_pts = point_cloud.size()
        trans = self.Tnet(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        point_cloud_transformed = torch.bmm(point_cloud, trans)
        point_cloud_transformed = point_cloud_transformed.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud_transformed)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        net = F.relu(self.bnc1(self.fc1(out_max)))
        net = F.relu(self.bnc2(self.dropout(self.fc2(net))))
        net = self.fc3(net) 

        out_max = torch.cat([out_max, label],1)
        expand = out_max.view(-1, 2048+self.category_k, 1).repeat(1, 1, n_pts)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net2 = F.relu(self.bns1(self.convs1(concat)))
        net2 = F.relu(self.bns2(self.convs2(net2)))
        net2 = F.relu(self.bns3(self.convs3(net2)))
        net2 = self.convs4(net2)
        net2 = net2.transpose(2, 1).contiguous()
        net2 = F.log_softmax(net2.view(-1, self.seg_k), dim=-1)
        net2 = net2.view(batchsize, n_pts, self.seg_k) # [B, N 50]

        return net, net2, trans_feat

class PointNetSegmentation(nn.Module):
    def __init__(self,num_class, input_dims=4, feature_transform=False):
        super(PointNetSegmentation, self).__init__()
        self.k = num_class
        self.feat = PointNetMain(global_feat=False,input_dims = input_dims, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


class PointNetLoss(torch.nn.Module):
    def __init__(self, weight=1,mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.weight = weight

    def forward(self, labels_pred, label, seg_pred, seg, trans_feat):
        seg_loss = F.nll_loss(seg_pred, seg)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        label_loss = F.nll_loss(labels_pred, label)

        loss = self.weight * seg_loss + (1-self.weight) * label_loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss, seg_loss, label_loss




        

