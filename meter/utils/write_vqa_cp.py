import os, json
import pyarrow as pa
import pandas as pd
from collections import defaultdict
import time


def find_iloc(list1, list2):
    return [num for num, ele in enumerate(list2) if ele in list1]

data_dir = "../../datasets/arrows"

t0 = time.time()
table = {}
table['train'] = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{data_dir}/vqav2_train.arrow", "r")).read_all()
table['val'] = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{data_dir}/vqav2_val.arrow", "r")).read_all()

pdt = {}
pdt['train'] = table['train'].to_pandas()
pdt['val'] = table['val'].to_pandas()

print(f"{time.time()-t0} s elapsed...")

for split in ['test', 'train']:
    with open(f'/project/qiz_mm/datasets/vqa_cp/vqacp_v2_{split}_questions.json') as fd:
        ann = json.load(fd)

    im_q_id = {'train': defaultdict(list), 'val': defaultdict(list)}
    for ele in ann:
        if 'train' in ele['coco_split']:
            im_q_id['train'][ele['image_id']].append(ele['question_id'])
        else:
            assert 'val' in ele['coco_split'], "sample not found"
            im_q_id['val'][ele['image_id']].append(ele['question_id'])

    bs = []
    for ss in ['train', 'val']:
        for im_id, qids in im_q_id[ss].items():
            target_row = pdt[ss].loc[pdt[ss]['image_id']==im_id]
            if not target_row['question_id'].empty:
                idx = find_iloc(qids, target_row['question_id'].values[0])
            else:
                continue

            if len(idx) > 0:
                bs.append([target_row['image'].values[0], 
                        target_row['questions'].values[0][idx], 
                        target_row['answers'].values[0][idx],
                        target_row['answer_labels'].values[0][idx],
                        target_row['answer_scores'].values[0][idx],
                        im_id,
                        target_row['question_id'].values[0][idx],
                        split])

    df = pd.DataFrame(bs, columns=pdt['train'].columns)

    table = pa.Table.from_pandas(df)
    with pa.OSFile(f"{data_dir}/vqav2_cp_{split}.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    if split == 'train':
        df1 = df[:1000]
        table = pa.Table.from_pandas(df1)
        with pa.OSFile(f"{data_dir}/vqav2_cp_val.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

# # for check
# test = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{data_dir}/vqav2_cp_test.arrow", "r")).read_all()
# pdt = test.to_pandas()

# gt = {}
# for qid, ans, ans_score in zip(pdt['question_id'], pdt['answers'], pdt['answer_scores']):
#     for x1, x2, x3 in zip(qid, ans, ans_score):
#         gt[str(x1)] = dict(zip(x2, x3))
