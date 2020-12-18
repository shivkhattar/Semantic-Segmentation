import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Test a segmentation model')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    args = parser.parse_args()
    return args
