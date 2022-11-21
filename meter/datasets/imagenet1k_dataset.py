from .base_cv_dataset import BaseCVDataset

class Imagenet1kDataset(BaseCVDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"imagenet1k_train_{i}" for i in range(13)]
        elif split == "val":
            names = ["imagenet1k_val_0"]

        super().__init__(*args, **kwargs, names=names, text_column_name="label")

    def __getitem__(self, index):
        return self.get_suite(index)
