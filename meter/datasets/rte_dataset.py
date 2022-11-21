import os
from .base_nlp_dataset import BaseNLPDataset
import torch

class RTEDataset(BaseNLPDataset):
    def __init__(self, data_dir, split, max_text_len=512):
        data_dir = os.path.join(data_dir, 'rte')
        super().__init__(data_dir, split, max_text_len)

    def __getitem__(self, index):
        data = self.text_data.__getitem__(index)
        text = data['sentence1'] + self.tokenizer.sep_token + data['sentence2']
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )

        return {"text": (text, encoding), "label": data['label']}

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)

        dict_batch = {"text": [], "text_ids": [], "labels": [], "text_masks": []}
        for ele in batch:
            dict_batch["text"].append(ele["text"][0])
            dict_batch["text_ids"].append(torch.tensor(ele["text"][1]["input_ids"]).unsqueeze(0))
            dict_batch["labels"].append(torch.tensor(ele["label"]).unsqueeze(0))
            dict_batch["text_masks"].append(torch.tensor(ele["text"][1]["attention_mask"]).unsqueeze(0))

        dict_batch["text_ids"] = torch.cat(dict_batch["text_ids"], 0)
        dict_batch["labels"] = torch.cat(dict_batch["labels"], 0)
        dict_batch["text_labels"] = None
        dict_batch["text_masks"] = torch.cat(dict_batch["text_masks"], 0)

        return dict_batch
