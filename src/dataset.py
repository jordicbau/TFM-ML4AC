import torch
import torch.utils.data
import rasterio
import pandas as pd
import os
from typing import Dict

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


class TiffDataset(torch.utils.data.Dataset):
    def __init__(self, df:pd.DataFrame):
        self.df = df

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

        return {"sl1c": torch.tensor(sl1c), "sl2a": torch.tensor(sl2a)}
