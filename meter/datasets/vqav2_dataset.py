from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqav2_cp_train"]
        elif split == "val":
            names = ["vqav2_cp_val"]
        elif split == "test":
            names = ["vqav2_cp_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)
        text = self.get_text(index)["text"]
        false_image_tensor = self.get_false_image("vqa")

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
        else:
            # answers = list()
            # labels = list()
            # scores = list()
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()

        return {
            "raw_Image": image_tensor['raw_Image'],
            "image": image_tensor["image"],
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
            "false_image": false_image_tensor['false_image_vqa'],
            "raw_false_Image": false_image_tensor['raw_false_Image'],
            "yes_type": int('yes' in answers or 'no' in answers)
        }
