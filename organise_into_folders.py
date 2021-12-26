# original data given was just all files together in one folder and thus re-organisation was required
# Code to group each CT scan with its corresponding files for bone fragments together in one folder
# The folder named dataset should exist with all the files in it which need to be sorted

# imports
import os, shutil
from natsort import natsorted

# functions
def num_digits(filename:str) -> int:
    digs = 0
    i = 0
    while filename[i].isnumeric():
        digs += 1
        i += 1
    return digs

def clean_spaces(filename:str) -> str:
    if ' ' in filename:
        filename = filename.replace(' ', '')
    return filename
        
# main
CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.join(CURRENT_PATH, 'dataset')
files = list(map(clean_spaces, 
                 natsorted([i for i in os.listdir(DATA_PATH) if i[0].isnumeric()])))
filesdir = natsorted(os.listdir(DATA_PATH))
for i in range(len(files)):
    old_path = os.path.join(DATA_PATH, filesdir[i])
    new_path = os.path.join(DATA_PATH, files[i])
    os.rename(old_path, new_path)
digfiles= list(map(num_digits, files))

for i in range(len(files)):
    if os.path.isdir(os.path.join(DATA_PATH, str(files[i][0:digfiles[i]]))):
        shutil.move(os.path.join(DATA_PATH, files[i]), 
                    os.path.join(DATA_PATH, str(files[i][0:digfiles[i]])))
    else:
        os.mkdir(os.path.join(DATA_PATH, str(files[i][0:digfiles[i]])))
        shutil.move(os.path.join(DATA_PATH, files[i]), 
                    os.path.join(DATA_PATH, str(files[i][0:digfiles[i]])))
