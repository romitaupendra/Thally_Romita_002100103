import os
import json
import pandas as pd
import h5py

from PIL import Image
from nowcast_reader import read_data
from fastapi import FastAPI, Request
from pydantic import BaseModel
import tensorflow as tf


app = FastAPI()

dataset_df = pd.read_csv('catalog_comb.csv')
# Load pretrained nowcasting models
mse_file  = style_file = 'mse_model.h5'
mse_model = tf.keras.models.load_model(mse_file,compile=False,custom_objects={"tf": tf})

def get_filename_index(event_id):
    catlog = pd.read_csv("https://raw.githubusercontent.com/MIT-AI-Accelerator/eie-sevir/master/CATALOG.csv")
    filtered = pd.DataFrame()
    for i in event_id:
        filtered = pd.concat([filtered, catlog[catlog["event_id"] == int(i)]])
        filename = filtered['file_name'].unique()
        fileindex = filtered['file_index'].unique()
    print("We have got the locations, Lets Download the files")
    return filename, fileindex

def One_Sample_HF(directory, fileindex):
    filenames = next(os.walk(directory), (None, None, []))[2]  # [] if no file
    for i in range(len(filenames)):
        print(directory + filenames[i])
        if filenames[i] == '.DS_Store' or filenames[i] == '.gitkeep':
            continue
        with h5py.File(directory + "/" + filenames[i], 'r') as hf:
            image_type = filenames[i].split('_')[1]
            # if image_type == "IR107":
            #     event_id = hf['id'][int(fileindex[1])]
            #     IR107 = hf['ir107'][int(fileindex[1])]
            if image_type == "VIL":
                VIL = hf['vil'][int(fileindex)]
            # if image_type == "IR069":
            #     IR069 = hf['ir069'][int(fileindex[2])]
            # if image_type == "VIS":
            #     VIS = hf['vis'][int(fileindex[0])]
    hf1 = h5py.File('filtered_data.h5', 'w')
    hf1.create_dataset('vil', data=VIL)
    # hf1.create_dataset('vis', data=VIS)
    # hf1.create_dataset('IR107', data=IR107)
    # hf1.create_dataset('IR069', data=IR069)
    print("downloded")

class PredictStormInput(BaseModel):
    event_id: int


@app.post('/predict_storm')
def predict_storm(input_data: PredictStormInput):
    event_id = input_data.event_id
    x_test, _ = read_data('C:\\Users\\17814\\Downloads\\DAMG7245-Assignment-2-Repo-main\\DAMG7245-Assignment-2-Repo-main\\data\\interim\\nowcast_testing.h5',end=1)
    y_pred = mse_model.predict(x_test)
    print(y_pred[0,:,:,0].shape)
    return 'success!'