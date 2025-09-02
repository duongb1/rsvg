# dataset.py
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _read_lines(fp: str) -> List[str]:
    with open(fp, 'r', encoding='utf-8') as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def _safe_int(x):
    try:
        return int(round(float(x)))
    except Exception:
        return int(x)


def _parse_voc_xml(xml_path: str) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Parse file VOC XML:
      returns (width, height, objects)
      objects: list of dict { 'bbox':(xmin,ymin,xmax,ymax), 'name':str, 'phrase':str }
    Trong DIOR-RSVG, có thể có trường 'phrase' (hoặc 'text') tuỳ bộ chuyển đổi.
    Nếu không có, sẽ fallback dùng 'name' làm phrase.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    if size is not None:
        width = _safe_int(size.find('width').text)
        height = _safe_int(size.find('height').text)
    else:
        # Fallback nếu không có <size>
        # Một số VOC không có size, khi đó đọc từ ảnh khi load
        width, height = -1, -1

    objs = []
    for obj in root.findall('object'):
        name_node = obj.find('name')
        name = name_node.text.strip() if name_node is not None else 'object'
        # phrase có thể nằm ở <phrase> hoặc <text> hoặc <attribute><phrase> tuỳ dataset
        phrase_node = obj.find('phrase') or obj.find('text')
        if phrase_node is None:
            # thử thêm một số biến thể
            attr = obj.find('attribute')
            if attr is not None:
                phrase_node = attr.find('phrase') or attr.find('text')
        phrase = (phrase_node.text.strip() if phrase_node is not None else name)

        bnd = obj.find('bndbox')
        if bnd is None:
            continue
        xmin = _safe_int(bnd.find('xmin').text)
        ymin = _safe_int(bnd.find('ymin').text)
        xmax = _safe_int(bnd.find('xmax').text)
        ymax = _safe_int(bnd.find('ymax').text)

        # Bảo vệ bbox hợp lệ
        if xmax <= xmin or ymax <= ymin:
            continue

        objs.append({
            'name': name,
            'phrase': phrase,
            'bbox': (xmin, ymin, xmax, ymax)
        })
    return width, height, objs


def letterbox(img: Image.Image, new_size: int, color=(114, 114, 114)):
    """
    Resize với tỷ lệ giữ nguyên, dán vào canvas vuông new_size x new_size.
    Trả:
      img_resized(PIL), ratio(float), dw(float), dh(float), pad_mask(ndarray HxWx1, uint8; pad=255, valid=0)
    """
    w, h = img.size
    r = min(new_size / w, new_size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new('RGB', (new_size, new_size), color)
    dw = (new_size - new_w) / 2.0
    dh = (new_size - new_h) / 2.0
    canvas.paste(resized, (int(round(dw)), int(round(dh))))

    # mask: vùng pad = 255
    mask = np.zeros((new_size, new_size, 1), dtype=np.uint8)
    if int(round(dw)) > 0:
        mask[:, :int(round(dw)), 0] = 255
        mask[:, new_size - int(round(dw)):, 0] = 255
    if int(round(dh)) > 0:
        mask[:int(round(dh)), :, 0] = 255
        mask[new_size - int(round(dh)):, :, 0] = 255

    return canvas, float(r), float(dw), float(dh), mask


class RSVGDataset(Dataset):
    """
    Dataset cho DIOR-RSVG (VOC-style):
      - Đọc split train/val/test từ {split_root}/{split}.txt
      - Mỗi ảnh có thể có nhiều object/phrase. Ta "bóc" thành các sample (image_id, phrase, bbox).
      - Ảnh được letterbox về (size x size), mask đánh dấu padding=255.
      - Tokenize phrase bằng HuggingFace AutoTokenizer.

    Trả về:
      train/val: (image_t, mask_t, word_id, word_mask, gt_bbox)
      test:      (image_t, mask_t, word_id, word_mask, gt_bbox, ratio, dw, dh, im_id, phrase)
    """

    def __init__(
        self,
        images_path: str,
        anno_path: str,
        splits_dir: str,
        split: str,
        imsize: int,
        transform,
        max_query_len: int,
        bert_model: str,
        testmode: bool = False
    ):
        super().__init__()
        self.images_path = images_path
        self.anno_path = anno_path
        self.split = split
        self.imsize = int(imsize)
        self.transform = transform
        self.max_query_len = int(max_query_len)
        self.testmode = testmode

        # Tokenizer HF (ưu tiên offline)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)

        # Load split list
        split_fp = os.path.join(splits_dir, f"{split}.txt")
        assert os.path.isfile(split_fp), f"Split file not found: {split_fp}"
        ids = _read_lines(split_fp)

        # Build sample list: mỗi object/phrase là 1 sample
        self.samples = []
        for im_id in ids:
            xml_path = os.path.join(self.anno_path, f"{im_id}.xml")
            img_jpg = os.path.join(self.images_path, f"{im_id}.jpg")
            img_png = os.path.join(self.images_path, f"{im_id}.png")
            img_path = img_jpg if os.path.isfile(img_jpg) else img_png
            if not os.path.isfile(xml_path) or not os.path.isfile(img_path):
                # bỏ qua nếu file thiếu
                continue

            try:
                w_xml, h_xml, objs = _parse_voc_xml(xml_path)
            except Exception:
                continue
            if not objs:
                continue

            for obj in objs:
                xmin, ymin, xmax, ymax = obj['bbox']
                phrase = (obj.get('phrase') or obj.get('name') or 'object').strip()
                # đảm bảo phrase không rỗng
                if not phrase:
                    phrase = 'object'
                self.samples.append({
                    'im_id': im_id,
                    'img_path': img_path,
                    'xml_path': xml_path,
                    'phrase': phrase,
                    'bbox': (xmin, ymin, xmax, ymax),
                    'size_xml': (w_xml, h_xml)
                })

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples built for split='{split}'. Check paths and XML structure!")

        print(f"[RSVGDataset] split={split} samples={len(self.samples)} images={len(set([s['im_id'] for s in self.samples]))}")

    def __len__(self):
        return len(self.samples)

    def _load_and_preprocess(self, img_path: str):
        # Load ảnh
        img = Image.open(img_path).convert('RGB')
        # Letterbox về canvas vuông
        img_resized, ratio, dw, dh, pad_mask = letterbox(img, self.imsize)
        # To tensor + normalize theo transform bên ngoài
        img_t = self.transform(img_resized) if self.transform is not None else torch.from_numpy(np.asarray(img_resized)).permute(2, 0, 1).float() / 255.0
        # mask tensor (H,W,1) uint8
        mask_t = torch.from_numpy(pad_mask)  # (H,W,1) uint8
        return img_t, mask_t, ratio, dw, dh

    @staticmethod
    def _resize_bbox(xmin, ymin, xmax, ymax, ratio, dw, dh):
        # scale theo ratio và tịnh tiến theo dw, dh
        xmin_ = xmin * ratio + dw
        xmax_ = xmax * ratio + dw
        ymin_ = ymin * ratio + dh
        ymax_ = ymax * ratio + dh
        # clamp vào canvas [0, size-1] — clamp sẽ thực hiện ở main, nhưng giữ an toàn ở đây
        return xmin_, ymin_, xmax_, ymax_

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        im_id = s['im_id']
        img_path = s['img_path']
        phrase = s['phrase']
        (xmin, ymin, xmax, ymax) = s['bbox']

        # Load ảnh & letterbox
        image_t, mask_t, ratio, dw, dh = self._load_and_preprocess(img_path)

        # Resize bbox vào hệ toạ độ canvas
        rxmin, rymin, rxmax, rymax = self._resize_bbox(xmin, ymin, xmax, ymax, ratio, dw, dh)
        # Đưa về tensor (trước clamp cuối cùng trong main)
        gt_bbox = torch.tensor([rxmin, rymin, rxmax, rymax], dtype=torch.float32)

        # Tokenize phrase
        enc = self.tokenizer(
            text=phrase,
            max_length=self.max_query_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        word_id = enc['input_ids'][0].to(torch.long)        # (L,)
        word_mask = enc['attention_mask'][0].to(torch.long) # (L,)

        if not self.testmode:
            # train/val tuple
            return image_t, mask_t, word_id, word_mask, gt_bbox
        else:
            # test tuple (thêm thông tin de-letterbox nếu cần)
            # Note: main hiện đang giữ metric trên canvas size, nhưng trả về ratio/dw/dh cho mục đích khác
            return image_t, mask_t, word_id, word_mask, gt_bbox, ratio, dw, dh, im_id, phrase
