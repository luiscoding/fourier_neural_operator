import os
import sys
import torch
sys.path.append("../code")
from  utilities3 import *
from scipy.io import savemat

def dim_reduction(input_dir, output_dir, step_size):
    r = step_size
    h = int(((421 - 1) / r) + 1)
    s = h

    for filename in os.listdir(input_dir):
        if filename.endswith(".mat")and "3_12" in filename:
            FILE_PATH = os.path.join(input_dir, filename)
            OUT_PATH = os.path.join(output_dir, filename)
            reader = MatReader(FILE_PATH)
            x_train = reader.read_field('coeff')[:, ::r, ::r][:, :s, :s]
            y_train = reader.read_field('sol')[:, ::r, ::r][:, :s, :s]
            savemat(OUT_PATH,
                    {"coeff": x_train.numpy(), 'sol': y_train.numpy()})
            print("finished "+filename)


def read_train_data(input_dir,ntrain):
    count = 0
    x_train = []
    y_train = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".mat"):
            FILE_PATH = os.path.join(input_dir, filename)
            reader = MatReader(FILE_PATH)
            x_train.append(reader.read_field('coeff')[:ntrain])
            y_train.append(reader.read_field('sol')[:ntrain])
            print("finished " + filename)
    x_train_mixed = torch.cat([item for item in x_train], 0)
    y_train_mixed = torch.cat([item for item in y_train], 0)
    return x_train_mixed, y_train_mixed


def main():
    input_dir = "../data/Meta_data_421"
    output_dir = "../data/Meta_data_85"
    # dim_reduction(input_dir, output_dir, 5)
    x_train, y_train = read_train_data(output_dir,1000)
    print(x_train.shape,y_train.shape)
    # filename = "output3_8_train_1000.mat"
    #
    # FILE_PATH = os.path.join(output_dir, filename)
    # reader = MatReader(FILE_PATH)
    # x_train = reader.read_field('coeff')
    # y_train = reader.read_field('sol')


if __name__=="__main__":
    main()