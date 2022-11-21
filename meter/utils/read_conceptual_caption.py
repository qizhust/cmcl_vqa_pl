from lmdbdict import lmdbdict  # from ruotian's repo
import pyarrow as pa  # version==0.16.0
import numpy as np
import io
import os
import json
from tqdm import tqdm

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pdb import set_trace

root_path = '/public/data0/MULT/datasets/cc3m'

for split in ['val', 'train']:
    img_path = root_path+'/cc_raw_'+split+'.lmdb'
    cap_path = root_path+'/'+split+'_caption.json'

    cap_file = json.load(open(cap_path))
    cap_dict = dict([(x['file_name'].split('/')[-1],x['caption']) for x in cap_file])
    cap_save = []

    images = lmdbdict(img_path)
    images._keys = [_.decode('ascii') for _ in pa.deserialize(images.db_txn.get(b'__keys__'))]
    images._key_dumps = lambda x: x.encode('ascii')
    images._value_loads = lambda x: x

    # demo_list = ['10492_1755079582', '6223_1541356096']  # for demo use, in val split
    # set_trace()
    cnt = 0
    for img_id in tqdm(images.keys()):
        try:
            image_name = f'{cnt:07d}'
            os.makedirs(root_path+'/images_'+split+'/'+image_name[:4], exist_ok=True)
            image_file_name = f'images_{split}/{image_name[:4]}/{image_name}.jpg'
            rgb_img = Image.open(io.BytesIO(images[img_id]), mode="r").convert('RGB')
            rgb_img.save(os.path.join(root_path, image_file_name))
            cap_save.append((root_path+'/'+image_file_name, cap_dict[img_id]))
            cnt += 1
            # np_img = np.array(rgb_img)
            # print(np_img.shape)
        except:
            print('possibly corrupted image data')

    json.dump(cap_save, open(os.path.join(root_path, split+'_annot.json'), 'w'))
