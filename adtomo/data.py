import numpy as np
import torch
from torch.utils.data import Dataset

class PhaseDataset(Dataset):
    def __init__(
        self,
        picks,
        events,
        stations,
        batch_size=1000,
        config=None,
    ):
        self.picks = picks
        self.events = events
        self.stations = stations
        self.batch_size = batch_size
        self.config = config

        self.station_index_batch = np.array_split(self.stations.index.values, (len(self.stations) - 1) // self.batch_size + 1)

    def __len__(self):
        return len(self.station_index_batch)
    
    def __getitem__(self, i):

        index_batch = self.station_index_batch[i]

        batch_picks = self.picks.loc[ self.picks["station_index"].isin(index_batch) ]
        # all picks that are in index_batch for those stations

        stations = self.stations.loc[self.stations["station_index"].isin(index_batch) ]
        # all stations chunked for that batch of picks

        events = self.events.loc[self.events["index"].isin(batch_picks["index"].values)]
        # all events chunked for that batch of picks

        return {
            "events": events,
            "stations": stations,
            "picks": batch_picks,
        }
