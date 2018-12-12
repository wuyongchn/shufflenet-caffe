from caffe import layers as L, params as P


class Input(object):
    def __init__(self, shape=(1, 3, 224, 224)):
        self.shape = shape

    def __call__(self):
        return L.Input(shape=dict(dim=list(self.shape)), ntop=1)


class Data(object):
    def __init__(self, crop_size=0, scale=1.0, backend='LMDB', mean_value=(104, 117, 123)):
        self.db = {'LMDB': P.Data.LMDB, 'LEVELDB': P.Data.LEVELDB}
        self.phase = {'TRAIN': caffe.TRAIN, 'TEST': caffe.TEST}
        self.crop_size = crop_size
        self.scale = scale
        self.backend = self.db[backend]
        self.mean_value = list(mean_value)

    def __call__(self, source, batch_size=1, phase='TRAIN', mirror=False):
        return L.Data(
            source=source, backend=self.backend, batch_size=batch_size, include=dict(phase=self.phase[phase]),ntop=1,
            transform_param=dict(crop_size=self.crop_size, mean_value=self.mean_value, mirror=mirror))


class ImageData(object):
    def __init__(self, crop_size=0, new_height=0, new_width=0, scale=1.0, mean_value=(104, 117, 123)):
        self.phase = {'TRAIN': caffe.TRAIN, 'TEST': caffe.TEST}
        self.crop_size = crop_size
        self.new_height = new_height
        self.new_width = new_width
        self.scale = scale
        self.mean_value = list(mean_value)

    def __call__(self, root_folder, source, batch_size=1, phase='TRAIN', mirror=False):
        return L.ImageData(
            include=dict(phase=self.phase[phase]),
            transform_param=dict(crop_size=self.crop_size, mean_value=self.mean_value, mirror=mirror),
            image_data_param=dict(root_folder=root_folder, source=source, batch_size=batch_size,
                                  new_height=self.new_width, new_width=self.new_width))


class Conv2d(object):
    def __init__(self, kernel_size, pad=None, bias_term=False):
        self.ks = kernel_size
        self.bias_term = bias_term
        if pad is None:
            self.pad = int(kernel_size // 2)
        else:
            self.pad = pad
        if bias_term:
            self.param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
        else:
            self.param = [dict(lr_mult=1, decay_mult=1)]

    def __call__(self, num_output, stride=1, group=1):
        if self.bias_term:
            conv = L.Convolution(
                kernel_size=self.ks, num_output=num_output, stride=stride, pad=self.pad, bias_term=True, group=group,
                param=self.param, weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
        else:
            conv = L.Convolution(
                kernel_size=self.ks, num_output=num_output, stride=stride, pad=self.pad, bias_term=False, group=group,
                param=self.param, weight_filler=dict(type='msra'))
        return conv


class DWConv2d(object):
    def __init__(self, kernel_size, pad=None, bias_term=False):
        self.conv_param = dict()
        self.conv_param['kernel_size'] = kernel_size
        self.conv_param['bias_term'] = bias_term
        self.conv_param['weight_filler'] = dict(type='msra')
        self.bias_term = bias_term
        if pad is None:
            self.conv_param['pad'] = int(kernel_size // 2)
        else:
            self.conv_param['pad'] = pad
        if bias_term:
            self.param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
            self.conv_param['bias_filler'] = dict(type='constant', value=0)
        else:
            self.param = [dict(lr_mult=1, decay_mult=1)]

    def __call__(self, num_output, stride=1):
        self.conv_param['num_output'] = num_output
        self.conv_param['stride'] = stride
        self.conv_param['group'] = num_output

        dwconv = L.DepthwiseConvolution(param=self.param, convolution_param=self.conv_param)

        return dwconv


class ShuffleChannel(object):
    def __call__(self, group=1):
        return L.ShuffleChannel(group=group)


class IP(object):
    def __init__(self, bias_term=True):
        self.bias_term = bias_term
        if bias_term:
            self.conv_param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
        else:
            self.conv_param = [dict(lr_mult=1, decay_mult=1)]

    def __call__(self, num_output):
        if self.bias_term:
            ip = L.InnerProduct(
                num_output=num_output, bias_term=True,
                param=self.conv_param, weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
        else:
            ip = L.InnerProduct(
                num_output=num_output, bias_term=False,
                param=self.conv_param, weight_filler=dict(type='msra'))
        return ip


class BN2d(object):
    def __init__(self):
        self.param = [dict(lr_mult=1), dict(lr_mult=1), dict(lr_mult=1)]

    def __call__(self):
        return L.BatchNorm(param=self.param)


class Scale(object):
    def __init__(self):
        self.param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=1)]

    def __call__(self):
        return L.Scale(param=self.param, scale_param=dict(bias_term=True))


class ReLU(object):
    def __init__(self):
        pass

    def __call__(self):
        return L.ReLU()


class Pooling(object):
    def __init__(self):
        self.method = {
            'MAX': P.Pooling.MAX,
            'AVE': P.Pooling.AVE,
            'STOCHASTIC': P.Pooling.STOCHASTIC}

    def __call__(self, kernel_size=None, stride=1, pool='MAX'):
        if kernel_size is None:
            return L.Pooling(pool=self.method[pool], global_pooling=False)
        else:
            return L.Pooling(kernel_size=kernel_size, stride=stride, pool=self.method[pool])


class Dropout(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, p):
        return L.Dropout(dropout_ratio=self.p)


class Softmax(object):
    def __call__(self):
        return L.Softmax()


class SoftmaxWithLoss(object):
    def __call__(self):
        return L.SoftmaxWithLoss()


class Accuracy(object):
    def __call__(self, top_k=1):
        return L.Accuracy(top_k=top_k)


class Eltwise(object):
    def __init__(self):
        self.op2caffeop = {'SUM': P.Eltwise.SUM, 'PROD': P.Eltwise.PROD, 'MAX': P.Eltwise.MAX}

    def __call__(self, op="SUM"):
        return L.Eltwise(operation=self.op2caffeop[op])


class Concat(object):
    def __call__(self, axis=1):
        return L.Concat(axis=axis)


class Slice(object):
    def __call__(self, slice_point):
        return L.Slice(slice_point=slice_point)


class Sigmoid(object):
    def __call__(self):
        return L.Sigmoid()


if __name__ == '__main__':
    import caffe
    import caffe.proto.caffe_pb2 as caffe_pb2
    import google.protobuf as pb
    import google.protobuf.text_format
    net = caffe_pb2.NetParameter()

    impl = Pooling()
    layer = net.layer.add()
    # layer.CopyFrom(impl(64).to_proto().layer[0])
    layer.CopyFrom(impl(pool='AVE').to_proto().layer[0])
    layer.name = 'name'
    layer.bottom.append('bottom')
    layer.top[0] = 'top'

    print pb.text_format.MessageToString(net)
