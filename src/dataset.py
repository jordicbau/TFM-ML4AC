import torch
import torch.utils.data
import rasterio
import pandas as pd
import numpy as np
import os
from typing import Dict
from torchvision.transforms.functional import rotate

HRFOLDER = "/media/disk/databases/cloudsen12/high/"

def process_metadata(path:str = HRFOLDER)->pd.DataFrame:
    hrfolder = os.path.dirname(path)
    data = pd.read_csv(path)
    data["sensing_time"] = data["sen2"].apply(lambda x: x.split("_")[0])
    data["year"] = data["sensing_time"].apply(lambda x: int(x[:4]))
    data["path"] = [os.path.join(hrfolder, tup.ROI, tup.sen2) for tup in data.itertuples()]

    # TODO remove 2018?
    data_2018 = data[(data.year == 2018) & (data.cloud_coverage == "cloud-free")]

    return data_2018


BANDS_L2A = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
BANDS_L1C = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

aot_min = 0#0.042
aot_max = 0.323

wvp_min = 0#0.012
wvp_max = 7.168
                   

class TiffDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame, window: int, train: bool):
        self.df = df
        self.window = window
        self.train = train

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        item_path = self.df["path"].iloc[item]
        s1lc_path = os.path.join(item_path, "S2L1C.tif")
        s2l2a_path = os.path.join(item_path, "S2L2A.tif")

        with rasterio.open(s1lc_path) as src1:
            bands = list(src1.descriptions)
            bands_read = [bands.index(b) + 1 for b in BANDS_L1C]
            sl1c = src1.read(indexes=bands_read).astype("float32") / 10_000

        with rasterio.open(s2l2a_path) as src1:
            bands = list(src1.descriptions)
            bands_read = [bands.index(b) + 1 for b in BANDS_L2A]
            sl2a = src1.read(indexes=bands_read).astype("float32") / 10_000
            
        with rasterio.open(s2l2a_path) as src1:
            bands = list(src1.descriptions)
            bands_read = bands.index('AOT') + 1
            aot = src1.read(indexes=bands_read).astype("float32") / 1000
            if self.train:
                aot = torch.tensor(aot)
                aot = torch.reshape(aot, (1, aot.shape[0], aot.shape[1]))
                #aot_std = aot.std().cpu().detach().numpy()
                #if aot_std == 0.0:
                #    aot_std = 0.0001
                #aot = (aot - aot.mean()) / aot_std
                #aot = aot * 1.6161999702453613 / 0.318
            

        with rasterio.open(s2l2a_path) as src1:
            bands = list(src1.descriptions)
            bands_read = bands.index('WVP') + 1
            wvp = src1.read(indexes=bands_read).astype("float32") / 1000
            if self.train:
                wvp = torch.tensor(wvp)
                wvp = torch.reshape(wvp, (1, wvp.shape[0], wvp.shape[1]))
                #wvp_std = wvp.std().cpu().detach().numpy()
                #if wvp_std == 0.0:
                #    wvp_std = 0.0001
                #wvp = (wvp - wvp.mean()) / wvp_std
                #wvp = wvp * 1.6161999702453613 / 7.168

        #https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands
        
        
        #Sampling for Train-Validation
        if self.train:

            sl1c = torch.tensor(sl1c)
            sl2a = torch.tensor(sl2a)
            
            sl2a = torch.cat((sl2a, aot, wvp), 0)

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

            #Normalization
            '''
            for idx in range(len(double_std_sl1c)):
                sl2a[idx] = sl2a[idx] / double_std_sl1c[idx]
            for idx in range(len(double_std_sl2a)):
                sl2a[idx] = sl2a[idx] / double_std_sl2a[idx]  ''' 
            sl2a[12] = (sl2a[12] - aot_min) / (aot_max - aot_min)
            sl2a[13] = (sl2a[13] - wvp_min) / (wvp_max - wvp_min)
                
            return {"sl1c": sl1c, "sl2a": sl2a}
             
        else:
            sl1c = torch.tensor(sl1c)
            sl2a = torch.tensor(sl2a)
            aot = torch.tensor(aot)
            wvp = torch.tensor(wvp)

            aot = torch.reshape(aot, (1, aot.shape[0], aot.shape[1]))
            wvp = torch.reshape(wvp, (1, wvp.shape[0], wvp.shape[1]))
            
            sl2a = torch.cat((sl2a, aot, wvp), 0)
            
           # for idx in range(len(mean_sl1c)):
            #    sl1c[idx] = (sl1c[idx] * np.mean(mean_sl1c[:12])) / mean_sl1c[idx]
            
            '''for idx in range(len(double_std_sl1c)):
                sl2a[idx] = sl2a[idx] / double_std_sl1c[idx]
            for idx in range(len(double_std_sl2a)):
                sl2a[idx] = sl2a[idx] / double_std_sl2a[idx]   '''
                
            sl2a[12] = (sl2a[12] - aot_min) / (aot_max - aot_min)
            sl2a[13] = (sl2a[13] - wvp_min) / (wvp_max - wvp_min)
            
            return {"sl1c": sl1c, "sl2a": sl2a}

