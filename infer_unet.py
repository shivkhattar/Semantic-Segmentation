import os
import sys
from infer.infer import infer
import train_unet as unet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))


def main():
    out = os.path.join(unet.root_out, 'unet')
    os.makedirs(out, exist_ok=True)
    infer(unet.nclasses, out, unet.inference)


if __name__ == '__main__':
    main()
