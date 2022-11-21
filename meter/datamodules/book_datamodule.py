from .datamodule_base import BaseDataModule
from ..datasets import BookDataset


class BookDataModule(BaseDataModule):
    def __init__(self, _config):
        super().__init__(_config)
        self.text_data_dir = _config['text_data_root']
        self.max_text_len = _config['max_text_len_nlp']

    @property
    def dataset_cls(self):
        return BookDataset

    @property
    def dataset_name(self):
        return "bookcorpus"

    def setup(self, stage):
        if not self.setup_flag:
            self.train_dataset = self.dataset_cls(self.text_data_dir, 'train', max_text_len=self.max_text_len)
            self.val_dataset = self.dataset_cls(self.text_data_dir, 'val', max_text_len=self.max_text_len)
            self.test_dataset = self.dataset_cls(self.text_data_dir, 'test', max_text_len=self.max_text_len)

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True
