import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob
from pdb import set_trace

def path2rest(path, iid2labels):
    split, file_dir, name = path.split("/")[-3:]

    with open(path, "rb") as fp:
        binary = fp.read()  # read the image into binary

    label = iid2labels[file_dir]

    return [
        binary,
        label,
        name,
        split,
    ]


def make_arrow(root, dataset_root):
    with open(os.path.join(root, "train/dataset.json"), "r") as fp:
        label_file = json.load(fp)['labels']  # list of (img_path, label)

    iid2labels = dict()
    for ele in tqdm(label_file):
        iid = ele[0].split("/")[0]  # dir_name
        iid2labels[iid] = ele[1]

    for split in ["val", "train"]:
        paths = list(glob(f"{root}/{split}/*/*"))  # all_img_paths
        random.shuffle(paths)

        sub_len = int(len(paths) // 100000)
        subs = list(range(sub_len + 1))
        for sub in subs:
            sub_paths = paths[sub * 100000: (sub + 1) * 100000]
            bs = [path2rest(path, iid2labels) for path in tqdm(sub_paths)]
            dataframe = pd.DataFrame(
                bs, columns=["image", "label", "image_id", "split"])

            table = pa.Table.from_pandas(dataframe)

            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                f"{dataset_root}/imagenet1k_{split}_{sub}.arrow", "wb"
            ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            del dataframe
            del table
            del bs
            gc.collect()
