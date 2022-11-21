import os
import requests
import argparse
from pdb import set_trace

output_directory = 'images_train'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

with open('SBU_captioned_photo_dataset_urls.txt', 'r') as fd:
    lines = fd.readlines()

with open('SBU_captioned_photo_dataset_captions.txt', 'r') as fd:
    captions = fd.readlines()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_pos', type=int, default=0)

    args = parser.parse_args()

    set_trace()
    # start_pos = 0
    for cnt, line in enumerate(lines[args.start_pos:]):
        line = line.strip()
        new_img_name = f"{cnt+args.start_pos:07d}"
        new_subdir = os.path.join(output_directory, new_img_name[:4])
        os.makedirs(new_subdir, exist_ok=True)
        new_img_filename = os.path.join(new_subdir, new_img_name+'.jpg')

        if not os.path.exists(new_img_filename):
            img_data = requests.get(line).content
            with open(new_img_filename, 'wb') as handler:
                handler.write(img_data)
            print(f'{cnt+args.start_pos}/{len(lines)}')