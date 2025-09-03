# dataset.py
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.utils.data as data
from transformers import AutoTokenizer


# --------------------------
# Local letterbox (no deps)
# --------------------------
def letterbox_np(img_bgr: np.ndarray,
                 out_size: int):
    """
    Letterbox giữ tỉ lệ: cạnh dài -> out_size, pad cạnh ngắn để thành out_size x out_size.

    Trả về:
      img_out_bgr: np.uint8 (H,W,3) BGR (để sau đó bạn có thể chuyển sang RGB trước transform)
      mask_out:    np.uint8 (H,W,1) với 255 là vùng PAD, 0 là vùng ảnh
      ratio:       float scale áp cho cả W,H
      dw, dh:      int padding left, top
    """
    assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "img phải là BGR HxWx3"
    h0, w0 = img_bgr.shape[:2]
    assert h0 > 0 and w0 > 0, "ảnh rỗng"

    # scale theo cạnh dài
    ratio = out_size / max(w0, h0)
    new_w = int(round(w0 * ratio))
    new_h = int(round(h0 * ratio))

    # resize ảnh
    if (new_w, new_h) != (w0, h0):
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img_bgr

    # mean màu của ảnh resized để pad (đúng theo paper)
    mean_bgr = resized.reshape(-1, 3).mean(axis=0)
    pad_color = tuple(int(round(c)) for c in mean_bgr)
    # Nếu muốn pad màu xám 114 kiểu YOLO, dùng:
    # pad_color = (114, 114, 114)

    # tạo canvas BGR
    canvas = np.full((out_size, out_size, 3), pad_color, dtype=np.uint8)
    dw = (out_size - new_w) // 2
    dh = (out_size - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    # mask: 255 ở vùng pad, 0 ở vùng ảnh
    mask = np.ones((out_size, out_size), dtype=np.uint8) * 255
    mask[dh:dh + new_h, dw:dw + new_w] = 0
    mask = mask[..., None]  # (H,W,1)

    return canvas, mask, ratio, float(dw), float(dh)


def filelist(root, file_type):
    return [
        os.path.join(directory_path, f)
        for directory_path, _, files in os.walk(root)
        for f in files if f.endswith(file_type)
    ]


# --------------------------
# BERT helpers (giữ nguyên)
# --------------------------
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


# --------------------------
# Dataset (DIOR-RSVG)
# --------------------------
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

        # load split indices (mỗi dòng là 1 index)
        split_file = os.path.join(splits_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            index_list = [int(line.strip()) for line in f if line.strip() != ""]

        count = 0
        annotations = filelist(anno_path, ".xml")
        for anno_file in annotations:
            root = ET.parse(anno_file).getroot()
            filename_node = root.find("filename")
            img_name = filename_node.text if filename_node is not None else None
            for member in root.findall("object"):
                if count in index_list:
                    # ảnh
                    imageFile = os.path.join(images_path, img_name)

                    # bbox (an toàn hơn: find theo tag)
                    bnd = member.find("bndbox")
                    if bnd is not None:
                        xmin = int(float(bnd.find("xmin").text))
                        ymin = int(float(bnd.find("ymin").text))
                        xmax = int(float(bnd.find("xmax").text))
                        ymax = int(float(bnd.find("ymax").text))
                    else:
                        # fallback theo index như code cũ
                        xmin = int(member[2][0].text)
                        ymin = int(member[2][1].text)
                        xmax = int(member[2][2].text)
                        ymax = int(member[2][3].text)

                    box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

                    # phrase (an toàn hơn: find theo tag 'text' hoặc 'name')
                    tnode = member.find("text")
                    if tnode is None:
                        tnode = member.find("name")
                    text = (tnode.text if tnode is not None else "").strip()

                    self.images.append((imageFile, box, text))
                count += 1

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        bbox = np.array(bbox, dtype=np.float32)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        return img, phrase, bbox, img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bgr, phrase, bbox, img_path = self.pull_item(idx)
        phrase = (phrase or "").lower()
        phrase_out = phrase  # giữ nguyên để trả về khi test

        # Letterbox (BGR), mask pad 255
        img_lb_bgr, mask_np, ratio, dw, dh = letterbox_np(img_bgr, self.imsize)

        # Scale bbox sang canvas
        bbox = bbox.copy()
        bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
        bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh

        # Chuyển BGR -> RGB trước transform
        img_rgb = cv2.cvtColor(img_lb_bgr, cv2.COLOR_BGR2RGB)

        # Apply transform (ToTensor, Normalize)
        if self.transform is not None:
            img_t = self.transform(img_rgb)  # (3,H,W), float
        else:
            # fallback
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # mask -> torch.uint8 (H,W,1)
        if mask_np.ndim == 2:
            mask_np = mask_np[..., None]
        mask_t = torch.from_numpy(mask_np.astype(np.uint8))  # (H,W,1)

        # encode text
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer
        )
        word_id = np.array(features[0].input_ids, dtype=np.int64)
        word_mask = np.array(features[0].input_mask, dtype=np.int64)

        if self.testmode:
            return (
                img_t, mask_t, word_id, word_mask,
                np.array(bbox, dtype=np.float32),
                np.array(ratio, dtype=np.float32),
                np.array(dw, dtype=np.float32),
                np.array(dh, dtype=np.float32),
                img_path, phrase_out
            )
        else:
            return img_t, mask_t, word_id, word_mask, np.array(bbox, dtype=np.float32)
