import os
import sys
from infer.infer import infer
import train_pspnet as pspnet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))


def main():
    out = os.path.join(pspnet.root_out, 'pspnet')
    os.makedirs(out, exist_ok=True)
    infer(pspnet.nclasses, out, pspnet.inference)


if __name__ == '__main__':
    main()
