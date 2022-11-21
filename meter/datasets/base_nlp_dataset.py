import torch
from torch.utils.data import Dataset, random_split
from datasets import load_from_disk


class BaseNLPDataset(Dataset):
    def __init__(self, data_dir, split, max_text_len=512):
        super().__init__()
        self.data_dir = data_dir
        self.max_text_len = max_text_len

        dataset = load_from_disk(data_dir)
        if 'val' not in dataset and 'validation' not in dataset:
            assert 'test' not in dataset, "No validation data, but found test data"
            len_train = int(0.8*len(dataset['train']))
            len_val = int(0.1*len(dataset['train']))
            len_test = len(dataset['train']) - len_train - len_val
            dataset_splits = random_split(dataset['train'], [len_train, len_val, len_test])

            self.splits_id = {'train': 0, 'val': 1, 'test': 2}
            self.text_data = dataset_splits[self.splits_id[split]]  # return Subset object
        else:
            self.text_data = dataset[split]

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        text = self.text_data.__getitem__(index)['text']
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {"text": (text, encoding)}

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
