import torch
import torch.utils.data
import rasterio
import pandas as pd
import numpy as np
import os
from typing import Dict
from torchvision.transforms.functional import rotate

HRFOLDER = "/media/disk/databases/cloudsen12/highprobav/"

def process_metadata(path:str = HRFOLDER)->pd.DataFrame:
    hrfolder = os.path.dirname(path)
    data = pd.read_csv(path)
    data["sensing_time"] = data["sen2"].apply(lambda x: x.split("_")[0])
    data["year"] = data["sensing_time"].apply(lambda x: int(x[:4]))
    #data["path"] = [os.path.join(hrfolder, tup.ROI) for tup in data.itertuples()]

    # TODO remove 2018?
    data_2018 = data[(data.year == 2018) & (data.cloud_coverage == "cloud-free")]

    return data_2018


BAND_NAMES = ["BLUE", "RED", "NIR", "SWIR"]

class TiffDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame, window: int, train: bool):
        self.df = df
        self.window = window
        self.train = train

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        item_path = os.path.join("/media/disk/databases/cloudsen12/highprobav/", self.df["ROI"].iloc[item]) #self.df["path"].iloc[item]
        s1lc_path = os.path.join(item_path, "PVTOA.tif")
        s2l2a_path = os.path.join(item_path, "PVTOC.tif")

        with rasterio.open(s1lc_path) as src1:
            bands = list(src1.descriptions)
            bands_read = [bands.index(b) + 1 for b in BAND_NAMES]
            sl1c = src1.read(indexes=bands_read).astype("float32")

        with rasterio.open(s2l2a_path) as src1:
            bands = list(src1.descriptions)
            bands_read = [bands.index(b) + 1 for b in BAND_NAMES]
            sl2a = src1.read(indexes=bands_read).astype("float32")
        

        #https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands
        
        
        #Sampling for Train-Validation
        if self.train:

            sl1c = torch.tensor(sl1c)
            sl2a = torch.tensor(sl2a)

            #Rotation
            angles = [0, 90, 180, 270]
            rotation_angle = angles[np.random.randint(0, len(angles))]
            sl1c = rotate(sl1c, rotation_angle)
            sl2a = rotate(sl2a, rotation_angle)

            #Window
            sampling_point_x = np.random.randint(0, sl1c.shape[2]-self.window)
            sampling_point_y = np.random.randint(0, sl1c.shape[2]-self.window)
            sl1c = sl1c[:, sampling_point_x:sampling_point_x+self.window, sampling_point_y:sampling_point_y+self.window]
            sl2a = sl2a[:, sampling_point_x:sampling_point_x+self.window, sampling_point_y:sampling_point_y+self.window]

            #Flip
            flip = np.random.randint(0, 100)
            if flip <= 33:
                sl1c = torch.flip(sl1c, [1,2])
                sl2a = torch.flip(sl2a, [1,2])
            elif flip <= 66:
                sl1c = torch.flip(sl1c, [2,1])
                sl2a = torch.flip(sl2a, [2,1])
            else:
                sl1c = sl1c
                sl2a = sl2a

                
            return {"sl1c": sl1c, "sl2a": sl2a}
             
        else:
            sl1c = torch.tensor(sl1c)
            sl2a = torch.tensor(sl2a)
            
            return {"sl1c": sl1c, "sl2a": sl2a}

