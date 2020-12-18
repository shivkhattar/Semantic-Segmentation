import os
import sys
from infer.infer import infer
import train_fpn as fpn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))


def main():
    out = os.path.join(fpn.root_out, 'fpn')
    os.makedirs(out, exist_ok=True)
    infer(fpn.nclasses, out, fpn.inference)


if __name__ == '__main__':
    main()
