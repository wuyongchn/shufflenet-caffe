import caffe.proto.caffe_pb2 as caffe_pb2
from base import Base
from base import ShuffleBottleneck1, ShuffleBottleneck2
from caffe_layers import IP, Input, Softmax, Pooling


class ShuffleNet(Base):
    def __init__(self, num_classes=1000, layers=(4, 8, 4), data=Input):
        super(ShuffleNet, self).__init__()
        self.num_classes = num_classes
        self.layers = layers
        self.input = data()
        self.ip = IP()
        self.pool = Pooling()
        self.net = caffe_pb2.NetParameter()
        self.stage2_num_output = {
            1: {1: 144, 2: 200, 3: 240, 4: 272, 8: 384},
            2: {0.5: 48, 1: 116, 1.5: 176, 2: 244}}
        self.top = None

    def __call__(self, version, scale_factor=1, group=3):
        self.version = version
        self.block = ShuffleBottleneck1() if version == 1 else ShuffleBottleneck2()
        self._check(version, scale_factor, group)
        if version == 2:
            self.conv5_nout = 2048 if scale_factor == 2 else 1024
        self.append_net_prefix()
        stage2_nout = self._stage2_nout(version, scale_factor, group)
        self.make_layer(2, self.block, stage2_nout * 1, self.layers[0])
        self.make_layer(3, self.block, stage2_nout * 2, self.layers[1])
        self.make_layer(4, self.block, stage2_nout * 4, self.layers[2])
        self.append_net_suffix()
        self.net.name = self.net_name(version, scale_factor, group)

        return self.net

    def make_layer(self, stage, block, planes, blocks, group=3, stride=2):
        if self.version == 2:
            group = 2
        for i in range(0, blocks):
            stride = stride if i == 0 else 1
            with_branch = i == 0
            net, top = block(self.top, planes, stage, i, group=group, stride=stride, branch1=with_branch)
            self.net.MergeFrom(net)
            self.top = top

    def append_net_prefix(self):
        self.append_single_layer('input', None, 'data', self.input())
        self.append_single_layer('conv1', 'data', 'conv1', self.conv3x3(24, stride=2))
        self.append_single_layer('bn_conv1', 'conv1', 'conv1', self.bn())
        self.append_single_layer('scale_conv1', 'conv1', 'conv1', self.scale())
        self.append_single_layer('relu_conv1', 'conv1', 'conv1', self.relu())
        self.append_single_layer('pool1', 'conv1', 'pool1', self.pool(3, 2))
        self.top = 'pool1'

    def append_net_suffix(self):
        if int(self.version) == 2:
            self.append_single_layer('conv5', self.top, 'conv5', self.conv1x1(self.conv5_nout))
            self.append_single_layer('bn_conv5', 'conv5', 'conv5', self.bn())
            self.append_single_layer('scale_conv5', 'conv5', 'conv5', self.scale())
            self.append_single_layer('relu_conv5', 'conv5', 'conv5', self.relu())
            self.top = 'conv5'
        self.append_single_layer('pool5', self.top, 'pool5', self.pool(7, 1, 'AVE'))
        self.append_single_layer('fc1000', 'pool5', 'fc1000', self.ip(self.num_classes))
        self.append_single_layer('prob', 'fc1000', 'prob', Softmax()())
        self.top = 'prob'

    def net_name(self, version, scale_factor, group):
        if version == 1:
            return 'ShuffleNet_v1_{}x_g{}'.format(scale_factor, group)
        else:
            return 'ShuffleNet_v2_{}x'.format(scale_factor)

    def _check(self, version, scale_factor=1, group=3):
        assert int(version) in (1, 2), 'version should be 1 or 2'
        if int(version) == 1:
            assert group in (1, 2, 3, 4, 8), 'shufflenet_v1 group should be 1, 2, 3, 4, 8'
        else:
            assert scale_factor in (0.5, 1, 1.5, 2), 'shufflenet_v2 scale_factor should be 0.5, 1, 1.5, 2'

    def _stage2_nout(self, version, scale_factor, group):
        if version == 1:
            return int(self.stage2_num_output[version][group] * scale_factor)
        else:
            return int(self.stage2_num_output[version][scale_factor])


def make_shufflenet(version, scale, group):
    model = ShuffleNet()(version, scale, group)
    return model


if __name__ == '__main__':
    import google.protobuf.text_format as pb_text
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-v", "--version", type=int, default=1, help="the version of shufflenet")
    parser.add_argument("-g", "--group", type=int, default=3, help="the group of shufflenet")
    parser.add_argument("-s", "--scale", type=int, default=1, help="the scale of shufflenet")
    args = parser.parse_args()

    net = make_shufflenet(args)
    name = net.name.lower() + '-deploy.prototxt'
    print pb_text.MessageToString(net)