#!/usr/bin/env python

import numpy as np
import matplotlib.image as image
import glob
import sys
import os
import argparse


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()

    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input npy."
    )

    parser.add_argument(
        "output_base_name",
        help="base part of output filename (excl. extension and index of array if 2d)"
    )

    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of output images."
    )

    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    img_array = np.load(args.input_file)
    
    if img_array.ndim == 1:
        image.imsave(args.output_base_name+".png",img_array.reshape(image_dims[0],image_dims[1],3))
    elif img_array.ndim == 2:
        for i in range(img_array.shape[0]):
            image.imsave(args.output_base_name+str(i+1)+".png",img_array[i].reshape(image_dims[0],image_dims[1],3))

if __name__ == '__main__':
    main(sys.argv)
