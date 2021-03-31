import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from functools import reduce
import numpy as np
from math import ceil
from logging import  error, basicConfig, INFO
from os import path
import json
from math import ceil
import utils.config as config


#torch.backends.cudnn.deterministic=True
torch.set_printoptions(precision=8)

class LayerBase(nn.Module):
    """Some Information about LayerBase"""
    def __init__(self,is_last_layer=False):
        super(LayerBase, self).__init__()
        self.is_last_layer = is_last_layer

    def forward(self, x):

        return x


    def _activation_func(self):
        activation_func_dict={
            'RELU': nn.ReLU,
            'SIGMOID': nn.Sigmoid,
            'TANH': nn.Tanh,
            'ELU': nn.ELU,
            'LEAKYRELU':nn.LeakyReLU,
        }
        return activation_func_dict[config.default_act_fn]()

class ConvLayerBase(LayerBase):
    """Some Information about BaseLayer"""
    def __init__(self,out_channels):
        super(ConvLayerBase, self).__init__()
        self.out_channels=out_channels

    def forward(self, x):

        return x

    def _apply_normalization(self):
        """Provide batch normalization to layer"""
        if config.employ_batch_normalization_conv:
            return nn.BatchNorm2d(num_features=self.out_channels,momentum=config.batch_normalization_momentum)


class Conv2D_Layer(ConvLayerBase):
    """Some Information about Conv2D
    
    """
    def __init__(self,in_channels, out_channels, kernel_size, stride,padding,dilation=(1,1)):
        super(Conv2D_Layer, self).__init__(out_channels)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.same_padding = nn.ReplicationPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding_mode="zeros",padding=0) 
        self.norm1 = self._apply_normalization() if config.employ_batch_normalization_conv else None
        self.act1 = self._activation_func()
        self.drop1 = nn.Dropout2d(p=1-config.dropout_rate) if config.employ_dropout_conv else None

    def forward(self, x):
        x = self.same_padding(x)
        x = self.conv1(x)
        if self.norm1!=None:
            x = self.norm1(x)
        x = self.act1(x)
        if self.drop1!=None:
            x = self.drop1(x)

        return x



class PoolingLayer(nn.Module):
    """Some Information about PoolingLayer"""
    def __init__(self,kind,kernel_size,stride,padding):
        super(PoolingLayer, self).__init__()
        self.padding=padding
        self.same_padding = nn.ReplicationPad2d(padding)
        self.pool = {"AVERAGE":nn.AvgPool2d,"MAX":nn.MaxPool2d}[kind](kernel_size,stride)

    def forward(self, x):
        x = self.same_padding(x)
        x = self.pool(x)
        return x

class ResidualLayerV1(Conv2D_Layer):
    """Some Information about ResidualLayerV1"""
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayerV1, self).__init__(in_channels, out_channels, kernel_size, stride, padding)

        #without stride ... because stride is in second conv 1,1... otherwise big BOOM o.O
        self.same_padding2 = nn.ReplicationPad2d((
            kernel_size[1]//2-1+kernel_size[1]%2,#-(1 if padding[0]%2==0 else 0),
            kernel_size[1]//2,
            kernel_size[0]//2-1+kernel_size[0]%2,#-(1 if padding[1]%2==0 else 0),
            kernel_size[0]//2
            ))

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size,stride=[1,1],padding_mode="zeros",padding=0) 
        self.norm2 = self._apply_normalization() if config.employ_batch_normalization_conv else None
        self.act2 = self._activation_func()
        self.drop2 = nn.Dropout2d(p=1-config.dropout_rate) if  config.employ_dropout_conv else None
        self.stride = stride
        self.pool = PoolingLayer("MAX",kernel_size,stride,padding)


    def forward(self, x):
        residual = x    
        x = self.same_padding(x)  
        x = self.conv1(x)
        if self.norm1!=None:
            x = self.norm1(x)
        x = self.act1(x)
        if self.drop1!=None:
            x = self.drop1(x)
        x = self.same_padding2(x)
        x = self.conv2(x)
        if self.norm2!=None:
            x = self.norm2(x)
        x = self.act2(x)

        residual,x = self.adjust_residual_layer(residual,x)
        x+=residual

        return x

    def adjust_residual_layer(self,residual,x):
        if self.pool!=None:
            residual = self.pool(residual) # 

        dif_channels = abs(self.out_channels-self.in_channels)

        if dif_channels==0:
            return residual,x

        pad=dif_channels//2
        padd_list=[pad,pad]

        if dif_channels%2!=0:
            padd_list[random.randint(0,1)]+=1

        padding = nn.ConstantPad3d((0,0,0,0,padd_list[0],padd_list[1]),0.0) # c_x+pad_x, c_y+pad_y

        if self.in_channels < self.out_channels:
            residual = padding(residual)
        elif self.in_channels > self.out_channels:
            residual = residual[:,dif_channels:]


        return residual,x


class DenseLayerBase(LayerBase):
    """Some Information about DenseLayerBase"""
    def __init__(self,out_channels,is_last_layer):
        super(DenseLayerBase, self).__init__(is_last_layer)
        self.out_channels = out_channels

    def forward(self, x):

        return x

    def _apply_normalization(self):
        """Provide batch normalization to layer"""
        if config.employ_batch_normalization_dense:
            return nn.BatchNorm2d(num_features=self.out_channels,momentum=config.batch_normalization_momentum)
        return None

class DenseLayer(DenseLayerBase):
    """Some Information about Dense_Layer"""
    def __init__(self,prev_layer_type,is_last_layer,in_channels,out_channels):
        super(DenseLayer, self).__init__(out_channels,is_last_layer)
        self.flatten = None if prev_layer_type=="D" else nn.Flatten()
        self.dense = nn.Linear(in_channels,out_channels)
        self.norm = self._apply_normalization()
        self.act = self._activation_func()
        self.drop = nn.Dropout2d(p=1-config.dropout_rate) if config.employ_dropout_dense else None

    def forward(self, x):
        if self.flatten != None:
            x = self.flatten(x)
        x = self.dense(x)
        if not self.is_last_layer:
            if self.norm != None:
                x = self.norm(x)
            x = self.act(x)
            if self.drop != None:
                x = self.drop(x)
        else:
            x = self.act(x)

        return x

class Inception(nn.Module):

    def __init__(self, in_channels, is_pooling, p_k, p_out_channels, is_b1, b1_out_channels, is_b2, b2_k, b2_out_channels, b2_factorization_mode, is_b3, b3_k, b3_out_channels, b3_factorization_mode, stride,in_size):
        super(Inception, self).__init__()
        
        FACTORIZATION = [InceptionBranchFactorization1,InceptionBranchFactorization1Split,InceptionBranchAsynchFactorization]

        self.stride = stride
        padding_b1 = calc_padding([in_size[1],in_size[2]],[1,1],stride)

        self.same_padding_b1 = nn.ReplicationPad2d(padding_b1)

        conv_block = BasicConv2d
        self.is_b1 = is_b1
        if self.is_b1:
            self.branch1_1x1 = conv_block(in_channels, b1_out_channels, kernel_size=1, stride = self.stride)
        
        self.is_b2 = is_b2
        if self.is_b2==1:
            self.branch2 = FACTORIZATION[b2_factorization_mode](in_channels,b2_out_channels,b2_k,stride,in_size) 
            
        self.is_b3 = is_b3

        if self.is_b3==1:
            self.branch3 = FACTORIZATION[b3_factorization_mode](in_channels,b3_out_channels,b3_k,stride,in_size) 
        
        self.is_pooling = is_pooling
        padding_b4 = calc_padding([in_size[1],in_size[2]],p_k,stride)

        self.same_padding_b4 = nn.ReplicationPad2d(padding_b4)

        if self.is_pooling:
            self.branch4_conv = conv_block(in_channels, p_out_channels, kernel_size=1)
            self.p_k = p_k

    def _forward(self, x):
        outputs = []
        if self.is_b1==1:
            branch1 = self.same_padding_b1(x)
            branch1 = self.branch1_1x1(branch1)
            outputs.append(branch1)
        
        if self.is_b2==1:
            branch2 = self.branch2(x)
            outputs.append(branch2)

        if self.is_b3==1:
            branch3 = self.branch3(x)
            outputs.append(branch3)
        
        if self.is_pooling==1:
            branch4_pool = self.same_padding_b4(x)
            branch4_pool = F.avg_pool2d(branch4_pool, kernel_size=self.p_k, stride=self.stride)
            branch4_pool = self.branch4_conv(branch4_pool)
            outputs.append(branch4_pool)
        
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
    
def calc_padding(in_size,kernel_size,stride):
    if (in_size[0] % stride[0] == 0):
        pad_along_height = max(kernel_size[0] - stride[0], 0)
    else:
        pad_along_height = max(kernel_size[0] - (in_size[0] % stride[0]), 0)
    if (in_size[1] % stride[1] == 0):
        pad_along_width = max(kernel_size[1] - stride[1], 0)
    else:
        pad_along_width = max(kernel_size[1] - (in_size[1] % stride[1]), 0)
    
    #Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return [pad_left,pad_right,pad_top,pad_bottom]

class InceptionBranchFactorization1(nn.Module):
    """Some Information about InceptionBranchFactorization1"""
    def __init__(self,in_channels, out_channels, kernel_size, stride, in_size):
        super(InceptionBranchFactorization1, self).__init__()

        conv_block = BasicConv2d
        self.branch_1 = conv_block(in_channels, out_channels, kernel_size=1)
        ks=(ceil(kernel_size[0]/2), ceil(kernel_size[1]/2))
        padding = [
            ks[1]//2-1+ks[1]%2,
            ks[1]//2,
            ks[0]//2-1+ks[0]%2,
            ks[0]//2
        ]
        self.same_padding1 = nn.ReplicationPad2d(padding)

        self.branch_2 = conv_block(out_channels, out_channels, kernel_size=ks)
        padding = calc_padding([in_size[1],in_size[2]],ks,stride)
        self.same_padding2 = nn.ReplicationPad2d(padding)
        self.branch_3 = conv_block(out_channels, out_channels, kernel_size=ks, stride=stride)

    def forward(self, x):
        x = self.branch_1(x)
        x = self.same_padding1(x)
        x = self.branch_2(x)
        x = self.same_padding2(x)
        x = self.branch_3(x)

        return x

class InceptionBranchFactorization1Split(nn.Module):
    """Some Information about InceptionBranchFactorization1Split"""
    def __init__(self,in_channels, out_channels, kernel_size, stride,in_size):
        super(InceptionBranchFactorization1Split, self).__init__()

        conv_block = BasicConv2d
        self.branch_1 = conv_block(in_channels, out_channels, kernel_size=1)
        ks=(ceil(kernel_size[0]/2), ceil(kernel_size[1]/2))
        padding = [
            ks[1]//2-1+ks[1]%2,
            ks[1]//2,
            ks[0]//2-1+ks[0]%2,
            ks[0]//2
        ]
        self.same_padding_1 = nn.ReplicationPad2d(padding)
        self.branch_2 = conv_block(out_channels, out_channels, kernel_size=ks)
        ks=(1, ceil(kernel_size[1]/2))
        padding = calc_padding([in_size[1],in_size[2]],ks,stride)
        self.same_padding_2 = nn.ReplicationPad2d(padding)
        self.branch_3a = conv_block(out_channels, out_channels, kernel_size=ks, stride=stride)
        ks=(ceil(kernel_size[0]/2),1)
        padding = calc_padding([in_size[1],in_size[2]],ks,stride)
        self.same_padding_3 = nn.ReplicationPad2d(padding)
        self.branch_3b = conv_block(out_channels, out_channels, kernel_size=ks, stride=stride)

    def forward(self, x):
        x = self.branch_1(x)
        x = self.same_padding_1(x)
        x = self.branch_2(x)
        xa = self.same_padding_2(x)
        xa = self.branch_3a(xa)
        xb = self.same_padding_3(x)
        xb = self.branch_3b(xb)


        return torch.cat([xa,xb],1)

class InceptionBranchAsynchFactorization(nn.Module):
    """Some Information about InceptionBranchAsynchFactorization"""
    def __init__(self,in_channels, out_channels, kernel_size, stride,in_size):
        super(InceptionBranchAsynchFactorization, self).__init__()
        
        conv_block = BasicConv2d
        self.branch_1 = conv_block(in_channels, out_channels, kernel_size=1)
        ks = (1, kernel_size[1])
        s = (1,stride[1])
        padding = calc_padding([in_size[1],in_size[2]],ks,s)
        
        self.same_padding_1 = nn.ReplicationPad2d(padding)
        self.branch_2 = conv_block(out_channels, out_channels, kernel_size=(1, kernel_size[1]), stride=(1,stride[1]))
        ks = (kernel_size[0],1)
        s = (stride[0],1)
        padding = calc_padding([in_size[1],in_size[2]],ks,s)
        self.same_padding_2 = nn.ReplicationPad2d(padding)
        self.branch_3 = conv_block(out_channels, out_channels, kernel_size=(kernel_size[0], 1), stride=(stride[0],1))

    def forward(self, x):
        x = self.branch_1(x)
        x = self.same_padding_1(x)
        x = self.branch_2(x)
        x = self.same_padding_2(x)
        x = self.branch_3(x)

        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class SubModule(nn.Module):
    """Some Information about CNN"""
    def __init__(self,layers,input_sizes,idx=0):
        super(SubModule, self).__init__()
        self.layers = layers
        self.prev_layer_type = None
        self.is_last_layer = False
        self.channels = input_sizes[0]
        self.input_size = (input_sizes[1],input_sizes[2])
        self.out_sizes = [[self.channels,self.input_size[0],self.input_size[1]]]
        self.sub_model = self._eval_gen()


    def len(self):
        return len(self.sub_model)

    def forward(self, x):
        for l in self.sub_model:
            x = self.sub_model[l](x)

        return x

    def _eval_gen(self):
        model = torch.nn.ModuleDict()
        in_channels = self.channels
        out_channels = self.channels
        in_nodes = 1
        l = None
        base = "layer_"
        for idx,layer in enumerate(self.layers):
            layer_name=base+"{}".format(idx)
            print("CREATE layer: "+layer_name)
            nucleotides = reduce(lambda x,y:x+y,[l.split("_") for l in layer.split(".")])
            if nucleotides[0] == "C":
                out_channels = int(nucleotides[5])
                kernel_size = [int(nucleotides[2]),int(nucleotides[1])]
                stride = [int(nucleotides[4]),int(nucleotides[3])]
                padding = self.calc_padding([self.out_sizes[idx][2],self.out_sizes[idx][1]],kernel_size,stride)
                l = Conv2D_Layer(in_channels,out_channels,kernel_size,stride,padding)
                h_x,h_y = self.calc_output_size([self.out_sizes[idx][2],self.out_sizes[idx][1]],stride=stride)
                self.out_sizes.append([out_channels,h_x,h_y])
                in_channels = out_channels
            elif nucleotides[0] == "R":
                out_channels = int(nucleotides[5])
                kernel_size = [int(nucleotides[2]),int(nucleotides[1])]
                stride = [int(nucleotides[4]),int(nucleotides[3])]
                padding = self.calc_padding([self.out_sizes[idx][2],self.out_sizes[idx][1]],kernel_size,stride)
                l = ResidualLayerV1(in_channels,out_channels,kernel_size,stride,padding)
                h_x,h_y = self.calc_output_size([self.out_sizes[idx][2],self.out_sizes[idx][1]],stride=stride)

                self.out_sizes.append([out_channels,h_x,h_y])
                in_channels = out_channels
            elif nucleotides[0] == "I":
                _,is_pooling,_ = nucleotides[1].split("-")
                is_pooling = int(is_pooling)
                p_k = [int(nucleotides[3]),int(nucleotides[2])]
                stride = [int(nucleotides[5]),int(nucleotides[4])]
                p_out_channels = int(nucleotides[6])

                _,is_b1,_= nucleotides[7].split("-")
                is_b1 = int(is_b1)
                b1_out_channels=int(nucleotides[12])

                _,is_b2,b2_factorization_mode=nucleotides[13].split("-")
                is_b2 = int(is_b2)
                b2_factorization_mode=int(b2_factorization_mode)-1
                b2_k=[int(nucleotides[15]),int(nucleotides[14])]
                b2_out_channels=int(nucleotides[18]) 

                _,is_b3,b3_factorization_mode=nucleotides[19].split("-")
                is_b3 = int(is_b3)
                b3_factorization_mode=int(b3_factorization_mode)-1
                b3_k=[int(nucleotides[21]),int(nucleotides[20])]
                b3_out_channels=int(nucleotides[24])  
                
                in_size = [self.out_sizes[idx][0],self.out_sizes[idx][2],self.out_sizes[idx][1]]
            
                l = Inception(in_channels,is_pooling,p_k,p_out_channels,is_b1,b1_out_channels,is_b2,b2_k,b2_out_channels,b2_factorization_mode,is_b3,b3_k,b3_out_channels,b3_factorization_mode,stride,in_size)
                h_x,h_y = self.calc_output_size([self.out_sizes[idx][2],self.out_sizes[idx][1]],stride=stride)

                out_channels=(p_out_channels if is_pooling else 0)
                out_channels+=(b1_out_channels if is_b1 else 0)
                out_channels+=((b2_out_channels if is_b2 else 0)*(1 if b2_factorization_mode != 1 else 2))
                out_channels+=((b3_out_channels if is_b3 else 0)*(1 if b3_factorization_mode != 1 else 2))

                self.out_sizes.append([out_channels,h_x,h_y])
                in_channels = out_channels
            elif nucleotides[0] == "P":
                kernel_size = [int(nucleotides[1]),int(nucleotides[2])]
                stride = [int(nucleotides[4]),int(nucleotides[3])]
                padding = self.calc_padding([self.out_sizes[idx][2],self.out_sizes[idx][1]],kernel_size,stride)
                l = PoolingLayer("MAX",kernel_size,stride,padding)
                h_x,h_y = self.calc_output_size([self.out_sizes[idx][2],self.out_sizes[idx][1]],stride=stride)
                self.out_sizes.append([out_channels,h_x,h_y])
                in_channels = out_channels
            elif nucleotides[0] == "D":
                prev_layer_type = self.layers[idx-1].split("_")[0] if idx>0 else "In"
                if prev_layer_type!="D":
                    in_nodes= reduce(lambda x,y: x*y,self.out_sizes[idx])
                is_last_layer = True if idx+1==len(self.layers) else False
                out_nodes = int(nucleotides[1])
                l = DenseLayer(prev_layer_type,is_last_layer,in_nodes,out_nodes)
                in_nodes = out_nodes
                self.out_sizes.append([out_nodes])
            model.update({layer_name:l})

        return model
    
    def calc_padding(self,in_size,kernel_size,stride):
        if (in_size[0] % stride[0] == 0):
            pad_along_height = max(kernel_size[0] - stride[0], 0)
        else:
            pad_along_height = max(kernel_size[0] - (in_size[0] % stride[0]), 0)
        if (in_size[1] % stride[1] == 0):
            pad_along_width = max(kernel_size[1] - stride[1], 0)
        else:
            pad_along_width = max(kernel_size[1] - (in_size[1] % stride[1]), 0)
        
        #Finally, the padding on the top, bottom, left and right are:

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return [pad_left,pad_right,pad_top,pad_bottom]

    def calc_output_size(self,in_size, stride):
        h_x = int(ceil(in_size[1] / stride[1]))
        h_y = int(ceil(in_size[0] / stride[0]))
        return h_x,h_y



def build_architecture(arch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layers = arch.split(":")
    
    input_sizes = [int(s) for s in layers[0].split("_")[1:]]
    print("CREATE model...")
    model = SubModule(layers[1:],input_sizes)
    print("MODEL created!")

    print("Ship model to device...")
    model=model.to(device)
    print("MODEL successfully shipped to device!")

    trainable_param_count = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

    return model, trainable_param_count