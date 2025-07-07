import torch
import timm
import numpy as np
from functools import partial
from timm.models.layers.helpers import to_2tuple
from torch import nn
import math
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    """
        Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input.float(), self.weight.float())
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Separable_block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Separable_block, self).__init__()

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class multi_scales_feature(nn.Module):
    def __init__(self):
        super(multi_scales_feature, self).__init__()
        self.feature_extractor = timm.create_model('xception', pretrained=True, features_only=True, num_classes=0, global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.act1(x)

        x = self.feature_extractor.conv2(x)
        x = self.feature_extractor.bn2(x)
        x = self.feature_extractor.act2(x)

        x = self.feature_extractor.block1(x)
        x_256 = self.feature_extractor.block2(x)

        x = self.feature_extractor.block3(x_256)
        x = self.feature_extractor.block4(x)
        x = self.feature_extractor.block5(x)
        x = self.feature_extractor.block6(x)
        x_728_1 = self.feature_extractor.block7(x)

        x = self.feature_extractor.block8(x_728_1)
        x = self.feature_extractor.block9(x)
        x = self.feature_extractor.block10(x)
        x_728 = self.feature_extractor.block11(x)

        x = self.feature_extractor.block12(x_728)

        x = self.feature_extractor.conv3(x)
        x = self.feature_extractor.bn3(x)
        x_1536 = self.feature_extractor.act3(x)

        x = self.feature_extractor.conv4(x_1536)
        x = self.feature_extractor.bn4(x)
        x_2048 = self.feature_extractor.act4(x)

        x = self.pool(x_2048)
        x_feat = torch.flatten(x, 1)
        return x_728, x_1536, x_2048

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            padding=0,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] + 2 * padding) // patch_size[0], (img_size[1] + 2 * padding) // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=padding, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Myattention_embed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, padding=0, embed_dim=768):
        super(Myattention_embed, self).__init__()
        self.patchembed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, padding=padding)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patchembed.num_patches, embed_dim) * .02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.2)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm_pre = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim)

    def forward(self, x, class_token=None):
        x = self.patchembed(x)
        x = self.pos_drop(x + self.pos_embed)
        if class_token == None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            x = torch.cat((class_token.unsqueeze(1), x), dim=1)
        x = self.norm_pre(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = y.shape

        q = self.to_q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.to_k(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.to_v(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AF_block(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(AF_block, self).__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        # self.attn = Attention(dim=embed_dim, num_heads=num_heads)
        self.cross_atten = CrossAttention(dim=embed_dim, num_heads=num_heads)

    def forward(self, x1, x2):
        x_2 = x2 + self.cross_atten(self.norm1(x1), self.norm2(x2))

        return x_2

class CMF(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(CMF, self).__init__()

        self.embed_728_C = Myattention_embed(img_size=19, patch_size=3, in_chans=embed_dim, embed_dim=embed_dim,
                                               padding=0)
        self.embed_1536_C = Myattention_embed(img_size=10, patch_size=2, in_chans=embed_dim, embed_dim=embed_dim,
                                                padding=1)
        self.embed_2048_C = Myattention_embed(img_size=10, patch_size=2, in_chans=embed_dim, embed_dim=embed_dim,
                                              padding=1)

        self.embed_728_U = Myattention_embed(img_size=19, patch_size=3, in_chans=embed_dim, embed_dim=embed_dim,
                                             padding=0)
        self.embed_1536_U = Myattention_embed(img_size=10, patch_size=2, in_chans=embed_dim, embed_dim=embed_dim,
                                              padding=1)
        self.embed_2048_U = Myattention_embed(img_size=10, patch_size=2, in_chans=embed_dim, embed_dim=embed_dim,
                                              padding=1)

        self.ALL1 = AF_block(embed_dim=embed_dim, num_heads=num_heads)
        self.ALL2 = AF_block(embed_dim=embed_dim, num_heads=num_heads)
        self.ALL3 = AF_block(embed_dim=embed_dim, num_heads=num_heads)


    def forward(self, x1_C, x2_C, x3_C, x1_U, x2_U, x3_U):
        x1_C_emb = self.embed_728_C(x1_C)
        x1_U_emb = self.embed_728_U(x1_U)
        ALL1_cls_token = self.ALL1(x1_C_emb, x1_U_emb)[:, 0, :]
        x2_C_emb = self.embed_1536_C(x2_C, class_token=ALL1_cls_token)
        x2_U_emb = self.embed_1536_U(x2_U, class_token=ALL1_cls_token)
        ALL2_cls_token = self.ALL2(x2_C_emb, x2_U_emb)[:, 0, :]
        x3_C_emb = self.embed_2048_C(x3_C, class_token=ALL2_cls_token)
        x3_U_emb = self.embed_2048_U(x3_U, class_token=ALL2_cls_token)
        ALL3_cls_token = self.ALL3(x3_C_emb, x3_U_emb)

        return ALL3_cls_token

def get_self_pearson(embeddings):
    """
            计算嵌入向量之间的皮尔逊相关系数
            Args:
                embeddings: 形如(batch_size, embed_dim)的张量
            Returns:
                piarwise_distances: 形如(batch_size, batch_size)的张量
        """

    avg_vec = torch.mean(embeddings, dim=-1)
    avg_vec = torch.unsqueeze(avg_vec, 2)
    nomal_embed = embeddings - avg_vec
    dot_product = torch.bmm(nomal_embed, nomal_embed.transpose(1, 2))
    square_norm = torch.diagonal(dot_product, dim1=-2, dim2=-1)
    square_norm = torch.bmm(torch.unsqueeze(square_norm, 2), torch.unsqueeze(square_norm, 1))
    distance = dot_product / torch.sqrt(square_norm)

    return distance

class model_md(nn.Module):
    def __init__(self, class_list):
        super(model_md, self).__init__()
        self.label_list = torch.cumsum(torch.tensor(class_list), dim=0)
        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.num = np.array(class_list).sum()

        self.num = np.array(class_list).sum()
        self.bert_clin = multi_scales_feature()
        self.bert_derm = multi_scales_feature()

        dim_embed = 128
        self.conv2_C = Separable_block(728, dim_embed, 2, 1)
        self.conv3_C = Separable_block(1536, dim_embed, 2, 1)
        self.conv4_C = Separable_block(2048, dim_embed, 2, 1)

        self.conv2_U = Separable_block(728, dim_embed, 2, 1)
        self.conv3_U = Separable_block(1536, dim_embed, 2, 1)
        self.conv4_U = Separable_block(2048, dim_embed, 2, 1)

        self.CMF = CMF(embed_dim=dim_embed, num_heads=4)
        self.class_emb = nn.Linear(dim_embed, self.num)

        self.gc1 = GraphConvolution(37, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        self.avg1 = nn.AdaptiveAvgPool1d(1)

        self.gc3 = GraphConvolution(37, 1024)
        self.gc4 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        self.avg2 = nn.AdaptiveAvgPool1d(1)

        self.task_emb = nn.Linear(37, 8)
        self.gc5 = GraphConvolution(128, 1024)
        self.gc6 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        self.fc_T = nn.Linear(2048, self.num_label)
        self.fc_pn_T = nn.Linear(2048, self.num_pn)
        self.fc_str_T = nn.Linear(2048, self.num_str)
        self.fc_pig_T = nn.Linear(2048, self.num_pig)
        self.fc_rs_T = nn.Linear(2048, self.num_rs)
        self.fc_dag_T = nn.Linear(2048, self.num_dag)
        self.fc_bwv_T = nn.Linear(2048, self.num_bwv)
        self.fc_vs_T = nn.Linear(2048, self.num_vs)

    def forward(self, x_clin, x_derm, adj):
        x_C_728, x_C_1536, x_C_2048 = self.bert_clin(x_clin)
        x_D_728, x_D_1536, x_D_2048 = self.bert_derm(x_derm)

        x_C_1 = self.conv2_C(x_C_728)
        x_C_2 = self.conv3_C(x_C_1536)
        x_C_3 = self.conv4_C(x_C_2048)

        x_D_1 = self.conv2_U(x_D_728)
        x_D_2 = self.conv3_U(x_D_1536)
        x_D_3 = self.conv4_U(x_D_2048)

        feat = self.CMF(x_C_1, x_C_2, x_C_3, x_D_1, x_D_2, x_D_3)

        feat_class = self.class_emb(feat).permute(0, 2, 1)
        feat_S = self.gc1(feat_class, adj)
        feat_S = self.relu(feat_S)
        feat_S = self.gc2(feat_S, adj)
        feat_S_avg = self.avg1(feat_S)
        out_S = torch.flatten(feat_S_avg, start_dim=1)

        adj_D = get_self_pearson(feat_class)
        adj_mask = (adj > 0)
        adj_D = adj_D * adj_mask
        feat_D = self.gc3(feat_class, adj_D)
        feat_D = self.relu(feat_D)
        feat_D = self.gc4(feat_D, adj_D)
        feat_D_avg = self.avg2(feat_D)
        out_D = torch.flatten(feat_D_avg, start_dim=1)

        feat_task = self.task_emb(feat.permute(0, 2, 1)).permute(0, 2, 1)
        adj_T = get_self_pearson(feat_task)
        feat_T = self.gc5(feat_task, adj_T)
        feat_T = self.relu(feat_T)
        feat_T = self.gc6(feat_T, adj_T)

        logit1 = out_S[:, : self.label_list[0]]
        logit_pn1 = out_S[:, self.label_list[0]: self.label_list[1]]
        logit_str1 = out_S[:, self.label_list[1]: self.label_list[2]]
        logit_pig1 = out_S[:, self.label_list[2]: self.label_list[3]]
        logit_rs1 = out_S[:, self.label_list[3]: self.label_list[4]]
        logit_dag1 = out_S[:, self.label_list[4]: self.label_list[5]]
        logit_bwv1 = out_S[:, self.label_list[5]: self.label_list[6]]
        logit_vs1 = out_S[:, self.label_list[6]: self.label_list[7]]

        logit2 = out_D[:, : self.label_list[0]]
        logit_pn2 = out_D[:, self.label_list[0]: self.label_list[1]]
        logit_str2 = out_D[:, self.label_list[1]: self.label_list[2]]
        logit_pig2 = out_D[:, self.label_list[2]: self.label_list[3]]
        logit_rs2 = out_D[:, self.label_list[3]: self.label_list[4]]
        logit_dag2 = out_D[:, self.label_list[4]: self.label_list[5]]
        logit_bwv2 = out_D[:, self.label_list[5]: self.label_list[6]]
        logit_vs2 = out_D[:, self.label_list[6]: self.label_list[7]]

        logit_T = self.fc_T(feat_T[:, 0, :])
        logit_pn_T = self.fc_pn_T(feat_T[:, 1, :])
        logit_str_T = self.fc_str_T(feat_T[:, 2, :])
        logit_pig_T = self.fc_pig_T(feat_T[:, 3, :])
        logit_rs_T = self.fc_rs_T(feat_T[:, 4, :])
        logit_dag_T = self.fc_dag_T(feat_T[:, 5, :])
        logit_bwv_T = self.fc_bwv_T(feat_T[:, 6, :])
        logit_vs_T = self.fc_vs_T(feat_T[:, 7, :])

        diag = logit1 + logit2 + logit_T
        pn = logit_pn1 + logit_pn2 + logit_pn_T
        str = logit_str1 + logit_str2 + logit_str_T
        pig = logit_pig1 + logit_pig2 + logit_pig_T
        rs = logit_rs1 + logit_rs2 + logit_rs_T
        dag = logit_dag1 + logit_dag2 + logit_dag_T
        bwv = logit_bwv1 + logit_bwv2 + logit_bwv_T
        vs = logit_vs1 + logit_vs2 + logit_vs_T

        return (diag, pn, str, pig, rs, dag, bwv, vs)


