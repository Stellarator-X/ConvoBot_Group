# This will probably work better as a notebook

from model import *
import tensorflow as tf 
import os
"""
Loading the Data
"""

def find_files(root_search_path, files_extension):
    files_list = []
    for root, _, files in os.walk(root_search_path):
        files_list.extend([os.path.join(root, file) for file in files if file.endswith(files_extension)])
    return files_list

def clean_label(_str):
        _str = _str.strip()
        _str = _str.lower()
        _str = _str.replace(".", "")
        _str = _str.replace(",", "")
        _str = _str.replace("?", "")
        _str = _str.replace("!", "")
        _str = _str.replace(":", "")
        _str = _str.replace("-", " ")
        _str = _str.replace("_", " ")
        _str = _str.replace("  ", " ")
        return _str

def get_data(path = 'LibriSpeech/'):
    text_files = find_files(path, ".txt")
    data = []
    for text_file in text_files:
        directory = os.path.dirname(text_file)
        with open(text_file, "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                head = line.split(' ')[0]
                if len(head) < 5:
                    # Not a line with a file description
                    break
                audio_file = directory + "/" + head + ".flac"
                if os.path.exists(audio_file):
                    data.append([audio_file, clean_label(line.replace(head, "")), None])
    
    data = np.array(data)
    data = data[:, :-1] # The last index is NoneType
    print(data.shape)
    return data

"""
Data Generation
"""
datagen = None
# First batch with samples in increasing order
firstset = None

"""
Data Augmentation
"""

# Building the model
input_shape = None


# model = DSModel(input_shape)
# model.build()
# model.compile()

# model.summary()

# # Training with sortagrad
# hist1 = model.fit(firstset, epochs = 1)
# hist2 = model.fit(datagen, epochs = 10, validation_data = None)

# m1 = model.getModel()

# m1.save("DSM")