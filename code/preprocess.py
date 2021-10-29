import os
import scipy.io as sio
import numpy as np
input_dir = '../data/Darcy/Meta_data_f'
# if True:
#     if True:
#         for filename in os.listdir(input_dir):
#             if filename.endswith(".mat") and 'f_3' not in filename :
#                 FILE_PATH = os.path.join(input_dir, filename)
#                 a = sio.loadmat(FILE_PATH)
#                 print(a.keys())
#                 new  = {'coeff': a['coeff'],'sol':a['sol']}
#                 np.save(FILE_PATH+".npy",new)
#                 print(a.keys())
#                 print("finished " + filename)


for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        FILE_PATH = os.path.join(input_dir, filename)
        a = np.load(FILE_PATH,allow_pickle=True)
        print(a.item().get('coeff').shape)
