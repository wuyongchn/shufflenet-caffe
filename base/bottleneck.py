import caffe.proto.caffe_pb2 as caffe_pb2
from .base import Base
from caffe_layers import Slice, Conv2d, DWConv2d, ShuffleChannel, Pooling


class ShuffleBottleneck1(Base):
    expansion = 4

    def __init__(self):
        super(ShuffleBottleneck1, self).__init__()
        self.gconv1x1 = Conv2d(kernel_size=1)
        self.dwconv3x3 = DWConv2d(kernel_size=3)
        self.shuffle = ShuffleChannel()
        self.pool = Pooling()

    def __call__(self, bottom, num_output, stage, out_idx, group, stride=1, branch1=False):
        self.net = caffe_pb2.NetParameter()
        if branch1:
            if stage == 2:
                num_output = (num_output - 24) // 4
            else:
                num_output //= 8
            avgpool, _, _, _ = self.name('a', stage, out_idx, 0, branch=1)
            self.append_single_layer(avgpool, bottom, avgpool, self.pool(3, stride, pool='AVE'))
            bottom1 = avgpool
            impl = self.concat()
            tmp, _, _, _ = self.name('a', stage, out_idx, 0)
            cat_name = 'blk' + tmp[3:tmp.find('_')] + '_concat'
        else:
            num_output //= 4
            bottom1 = bottom
            impl = self.eltwise()
            tmp, _, _, _ = self.name('a', stage, out_idx, 0)
            cat_name = 'blk' + tmp[3:tmp.find('_')] + '_eltwise'
        gconv2a, bn2a, scale2a, relu2a = self.name('g', stage, out_idx, 0)
        self.append_multi_layer(bottom, self.gconv1x1(num_output, group=group), gconv2a, bn2a, scale2a, relu2a)
        shuffle2b, _, _, _ = self.name('s', stage, out_idx, 1)
        self.append_single_layer(shuffle2b, gconv2a, shuffle2b, self.shuffle(group=group))
        dwconv2c, bn2c, scale2c, _ = self.name('dw', stage, out_idx, 2)
        self.append_multi_layer(shuffle2b, self.dwconv3x3(num_output, stride), dwconv2c, bn2c, scale2c)
        gconv2d, bn2d, scale2d, _ = self.name('g', stage, out_idx, 3)
        self.append_multi_layer(
            dwconv2c, self.gconv1x1(num_output * ShuffleBottleneck1.expansion, group=group), gconv2d, bn2d, scale2d)

        self.append_single_layer(cat_name, [bottom1, gconv2d], cat_name, impl)
        relu_name = 'relu_' + cat_name
        self.append_single_layer(relu_name, cat_name, cat_name, self.relu())

        return self.net, cat_name


class ShuffleBottleneck2(Base):
    def __init__(self):
        super(ShuffleBottleneck2, self).__init__()
        self.dwconv3x3 = DWConv2d(kernel_size=3)
        self.shuffle = ShuffleChannel()
        self.slice = Slice()

    def __call__(self, bottom, num_output, stage, out_idx, group, stride=1, branch1=False):
        self.net = caffe_pb2.NetParameter()
        if branch1:
            if stage == 2:
                output1 = 24
            else:
                output1 = num_output // 2
            dwconv1a, bn1a, scale1a, _ = self.name('dw', stage, out_idx, 0, branch=1)
            self.append_multi_layer(bottom, self.dwconv3x3(output1, stride), dwconv1a, bn1a, scale1a)
            conv1b, bn1b, scale1b, _ = self.name('c', stage, out_idx, 1, branch=1)
            self.append_multi_layer(
                dwconv1a, self.conv1x1(num_output//2), conv1b, bn1b, scale1b)
            bottom1 = conv1b
        else:
            tmp, _, _, _ = self.name('a', stage, out_idx, 0)
            split_name = 'blk' + tmp[3:tmp.find('_')] + '_split'
            top_name1 = split_name + '1'
            top_name2 = split_name + '2'
            self.append_single_layer(split_name, bottom, [top_name1, top_name2], self.slice(num_output//2))
            bottom1 = top_name1
            bottom = top_name2
        conv2a, bn2a, scale2a, relu2a = self.name('c', stage, out_idx, 0)
        self.append_multi_layer(bottom, self.conv1x1(num_output//2), conv2a, bn2a, scale2a, relu2a)
        dwconv2b, bn2b, scale2b, _ = self.name('dw', stage, out_idx, 1)
        self.append_multi_layer(conv2a, self.dwconv3x3(num_output//2, stride), dwconv2b, bn2b, scale2b)
        conv2c, bn2c, scale2c, relu2c = self.name('c', stage, out_idx, 2)
        self.append_multi_layer(
            dwconv2b, self.conv1x1(num_output//2), conv2c, bn2c, scale2c, relu2c)

        tmp, _, _, _ = self.name('a', stage, out_idx, 0)
        cat_name = 'blk' + tmp[3:tmp.find('_')] + '_concat'
        self.append_single_layer(cat_name, [bottom1, conv2c], cat_name, self.concat())
        shuffle_name = 'blk' + tmp[3:tmp.find('_')] + '_shuffle'
        self.append_single_layer(shuffle_name, cat_name, shuffle_name, self.shuffle(group=2))
        return self.net, shuffle_name


if __name__ == '__main__':
    pass
