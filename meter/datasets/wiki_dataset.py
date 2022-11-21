import os
from .base_nlp_dataset import BaseNLPDataset

class WikiDataset(BaseNLPDataset):
    def __init__(self, data_dir, split, max_text_len=512):
        data_dir = os.path.join(data_dir, 'wikipedia')
        super().__init__(data_dir, split, max_text_len)
