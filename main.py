import os
os.environ["GLOG_minloglevel"] = '2'
from argparse import ArgumentParser
import google.protobuf.text_format as pb_text
from network import make_shufflenet
from utils import print_net_parameters


def prototxt_generator(args):
    net = make_shufflenet(args.version, args.scale, args.group)
    name = net.name.lower() + '-deploy.prototxt'
    folder = args.network
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, name), 'w') as f:
        f.write(pb_text.MessageToString(net))
    print 'The file is written to %s' % os.path.join(folder, name)
    print_net_parameters(os.path.join(folder, name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", type=int, default=1, help="the version of shufflenet")
    parser.add_argument("-g", "--group", type=int, default=3, help="the group of shufflenet")
    parser.add_argument("-s", "--scale", type=int, default=2, help="the scale of shufflenet")
    args = parser.parse_args()
    prototxt_generator(args)
