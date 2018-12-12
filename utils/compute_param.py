import os
from numpy import prod, sum
from pprint import pprint
import caffe


def print_net_parameters(deploy_file):
    caffe.set_mode_cpu()
    net = caffe.Net(str(deploy_file), caffe.TEST)
    print "Layer-wise parameters: "
    pprint([(k, v[0].data.shape) for k, v in net.params.items()])
    print "Total number of parameters: " + str(sum([prod(v[0].data.shape) for k, v in net.params.items()]))


if __name__ == '__main__':
    folder = 'shufflenet'
    name = 'shufflenet_v2_2x-deploy.prototxt'
    deploy_file = os.path.join(folder, name)
    print_net_parameters(deploy_file)