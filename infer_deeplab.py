import os
import sys
from infer.infer import infer
import train_deeplab as deeplab

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))


def main():
    out = os.path.join(deeplab.root_out, 'deeplab')
    os.makedirs(out, exist_ok=True)
    infer(deeplab.nclasses, out, deeplab.inference)


if __name__ == '__main__':
    main()
