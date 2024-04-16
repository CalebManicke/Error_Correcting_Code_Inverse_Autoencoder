import numpy as np
import os
from pathlib import Path
 
#DATASET_LOCATION = "/home/amey/code/paper_name_not_decided/nd_all"
DATASET_LOCATION = os.getcwd()
 

def path_all_files_in_given_dir(path):
    """
    Return path for all files in a given directory
    """
    return [str(f) for f in path.iterdir() if f.is_file()]


def conv_input_to_array(path):
    """
    Convert CSV to 1D array to make it easier to operate on.
    """
    return np.genfromtxt(path, delimiter=",", dtype=float)[:-1]


files_list = path_all_files_in_given_dir(Path(DATASET_LOCATION))

#H = np.random.choice([0, 1], size=(1024, 1024))
H = np.random.choice([0, 1], size=(1024, 1))

bitstrings_for_each_file_pre = [conv_input_to_array(i) for i in files_list]
bitstrings_for_each_file = np.vectorize(lambda x: np.rint(x).astype(np.int32))(
    bitstrings_for_each_file_pre
)

array_for_caleb = []

for w in bitstrings_for_each_file:
    #Hw = np.mod(np.matmul(H, w), 2)
    Hw = np.concatenate((H, w), axis = 0)
    array_for_caleb.append([w, Hw])

array_for_caleb = np.array(array_for_caleb)

np.save(os.getcwd() + "//random_concatenate.npy", array_for_caleb)

# Have to transpose the dataset otherwise matrix are not compat
big_Hw = np.mod(np.matmul(H, bitstrings_for_each_file.T), 2)
array_for_caleb_two = np.array(big_Hw)

# don't need it i guess but why not
#np.save("big_hW_for_caleb.npy", array_for_caleb_two)