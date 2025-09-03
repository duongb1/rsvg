import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch.utils.data as data
from transformers import AutoTokenizer

from utils.transforms import letterbox

def filelist(root, file_type):
    return [
        os.path.join(directory_path, f)
        for directory_path, _, files in os.walk(root)
        for f in files if f.endswith(file_type)
    ]


class InputExample:
    def __init__(self, unique_id, text_a, text_b=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures:
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(input_line, unique_id):
    line = input_line.strip()
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a, text_b = line, None
    else:
        text_a, text_b = m.group(1), m.group(2)
    return [InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)]


def convert_examples_to_features(examples, seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b) if example.text_b else None

        # truncate if too long
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            tokens_a = tokens_a[: seq_length - 2]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_type_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            input_type_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # pad
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncate sequence pair to max_length total"""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class RSVGDataset(data.Dataset):
    def __init__(self, images_path, anno_path, splits_dir,
                 imsize=640, transform=None, augment=False,
                 split="train", testmode=False, max_query_len=40,
                 bert_model="bert-base-uncased"):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode
        self.query_len = max_query_len
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        # load split indices
        split_file = os.path.join(splits_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            index_list = [int(line.strip()) for line in f.readlines()]

        count = 0
        annotations = filelist(anno_path, ".xml")
        for anno_file in annotations:
            root = ET.parse(anno_file).getroot()
            for member in root.findall("object"):
                if count in index_list:
                    imageFile = os.path.join(images_path, root.find("filename").text)
                    box = np.array([
                        int(member[2][0].text),
                        int(member[2][1].text),
                        int(member[2][2].text),
                        int(member[2][3].text),
                    ], dtype=np.float32)
                    text = member[3].text
                    self.images.append((imageFile, box, text))
                count += 1

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)
        img = cv2.imread(img_path)
        return img, phrase, bbox

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()
        phrase_out = phrase

        h, w = img.shape[:2]
        mask = np.zeros_like(img)

        img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
        bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
        bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh

        if self.transform is not None:
            img = self.transform(img)

        # encode text
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer
        )
        word_id = np.array(features[0].input_ids, dtype=int)
        word_mask = np.array(features[0].input_mask, dtype=int)

        if self.testmode:
            return (
                img, mask, word_id, word_mask,
                np.array(bbox, dtype=np.float32),
                np.array(ratio, dtype=np.float32),
                np.array(dw, dtype=np.float32),
                np.array(dh, dtype=np.float32),
                self.images[idx][0], phrase_out
            )
        else:
            return img, mask, word_id, word_mask, np.array(bbox, dtype=np.float32)
