# dataset.py
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch.utils.data as data

from transformers import AutoTokenizer
from utils.transforms import letterbox
import random

def filelist(root, file_type):
    return [os.path.join(directory_path, f)
            for directory_path, directory_name, files in os.walk(root)
            for f in files if f.endswith(file_type)]

class RSVGDataset(data.Dataset):
    def __init__(self, images_path, anno_path, imsize=640, transform=None, augment=False,
                 split='train', testmode=False, max_query_len=40, bert_model='bert-base-uncased'):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode
        self.query_len = max_query_len  # 40

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)

        file = open('./DIOR_RSVG/' + split + '.txt', "r").readlines()
        Index = [int(index.strip('\n')) for index in file]
        count = 0
        annotations = filelist(anno_path, '.xml')
        for anno_path in annotations:
            root = ET.parse(anno_path).getroot()
            for member in root.findall('object'):
                if count in Index:
                    imageFile = str(images_path) + '/' + root.find("./filename").text
                    box = np.array([
                        int(member[2][0].text),
                        int(member[2][1].text),
                        int(member[2][2].text),
                        int(member[2][3].text)
                    ], dtype=np.float32)
                    text = member[3].text
                    self.images.append((imageFile, box, text))
                count += 1

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=int)  # x1 y1 x2 y2
        img = cv2.imread(img_path)
        # NEW: BGR -> RGB để khớp Normalize(mean/std) kiểu ImageNet
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, phrase, bbox

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        phrase = phrase.lower()
        phrase_out = phrase

        # mask nền: cùng kích thước img, 3 kênh cho tiện copyMakeBorder
        h, w = img.shape[0], img.shape[1]
        mask = np.zeros_like(img)

        img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)  # padding màu RGB
        bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
        bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh

        # torchvision transforms (ToTensor + Normalize) — áp dụng sau letterbox
        if self.transform is not None:
            img = self.transform(img)

        # NEW: dùng tokenizer trực tiếp, padding/truncation chuẩn
        enc = self.tokenizer(
            phrase,
            max_length=self.query_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,    # lấy list để chuyển sang np dễ dàng
            add_special_tokens=True
        )
        # attention_mask: 1=real, 0=pad (chuẩn HF). Ta giữ nguyên ở dataset,
        # phần model sẽ đảo lại sang True=pad theo quy ước DETR.
        word_id = np.array(enc["input_ids"], dtype=int)
        word_mask = np.array(enc["attention_mask"], dtype=int)

        if self.testmode:
            return (img, mask,
                    word_id, word_mask,
                    np.array(bbox, dtype=np.float32),
                    np.array(ratio, dtype=np.float32),
                    np.array(dw, dtype=np.float32),
                    np.array(dh, dtype=np.float32),
                    self.images[idx][0], phrase_out)
        else:
            return img, mask, word_id, word_mask, np.array(bbox, dtype=np.float32)
