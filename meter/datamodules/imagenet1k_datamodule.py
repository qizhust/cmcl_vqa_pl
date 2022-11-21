from .datamodule_base import BaseDataModule
from ..datasets import Imagenet1kDataset


class Imagenet1kDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Imagenet1kDataset

    @property
    def dataset_name(self):
        return "imagenet1k"
