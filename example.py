import argparse

import cv2

from redact import Redactor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help="Input image path")
    parser.add_argument('-o', '--output', required=True, help="Output image path")
    parser.add_argument('-x', '--xml', default="./haarcascade_frontalface_alt.xml", help="Classifer xml path")
    return parser.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    redactor = Redactor(args.xml)
    redacted_frame = redactor.redact_frame(img)
    cv2.imwrite(args.output, redacted_frame) 


if __name__ == '__main__':
    main()
