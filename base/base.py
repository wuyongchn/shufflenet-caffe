from caffe_layers import Conv2d, BN2d, Scale, ReLU, Eltwise, Concat


class Base(object):
    def __init__(self):
        self.conv3x3 = Conv2d(kernel_size=3)
        self.conv1x1 = Conv2d(kernel_size=1)
        self.bn = BN2d()
        self.scale = Scale()
        self.relu = ReLU()
        self.eltwise = Eltwise()
        self.concat = Concat()
        self.net = None
        self._format = '{}{}{}_branch{}{}'.format
        self._sign2str = {'c': 'conv', 'g': 'gconv', 'dw': 'dwconv', 's': 'shuffle', 'a': 'avg', None: 'res'}

    def append_multi_layer(self, bottom, conv_impl, conv, bn, scale, relu=None):
        self.append_single_layer(conv, bottom, conv, conv_impl)
        self.append_single_layer(bn, conv, conv, self.bn())
        self.append_single_layer(scale, conv, conv, self.scale())
        if relu is not None:
            self.append_single_layer(relu, conv, conv, self.relu())

    def append_single_layer(self, name, bottom, top, impl):
        layer = self.net.layer.add()
        layer.CopyFrom(impl.to_proto().layer[0])
        layer.name = name
        if isinstance(bottom, list):
            for bot in bottom:
                layer.bottom.append(bot)
        elif bottom is not None:
            layer.bottom.append(bottom)
        if isinstance(top, list):
            layer.top[0] = top[0]
            for t in top[1:]:
                layer.top.append(t)
        elif top is not None:
            layer.top[0] = top

    def name(self, sign, stage, out_idx, in_idx, branch=2, cut=False):
        conv = self._format(self._sign2str[sign], stage, self._idx2str(out_idx), branch, self._idx2str(in_idx))
        bn = self._format('bn', stage, self._idx2str(out_idx), branch, self._idx2str(in_idx))
        scale = self._format('scale', stage, self._idx2str(out_idx), branch, self._idx2str(in_idx))
        relu = self._format('relu', stage, self._idx2str(out_idx), branch, self._idx2str(in_idx))
        name_list = [conv, bn, scale, relu]
        if cut:
            offset = len(self._idx2str(in_idx))
            for i in range(len(name_list)):
                name_list[i] = name_list[i][:-offset]
        return name_list

    @staticmethod
    def _idx2str(idx):
        idx += 1
        stk = []
        while idx > 0:
            remainder = idx % 26
            if remainder == 0:
                stk.append('z')
            else:
                stk.append(chr(ord('a') + remainder - 1))
            remainder = remainder if remainder != 0 else 26
            idx = (idx - remainder) / 26
        stk = stk[::-1]
        idx_str = ''
        for ch in stk:
            idx_str += ch
        return idx_str


if __name__ == '__main__':
    test = Base()
    print test.name('dw', 2, 0, 0, 1, True)