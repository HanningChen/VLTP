import glob
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

class PrivacyDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self,
                 base_image_dir,
                 tokenizer,
                 vision_tower,
                 samples_per_epoch=500 * 8 * 2 * 10,
                 precision: str = "fp32",
                 image_size: int = 224,
                 num_classes_per_sample: int = 1,
                 exclude_val=False,
                 Privacy_data="Privacy|train",
                 explanatory=-1
                 ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size) #NOTE: Hanning, check later
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        #NOTE: Hanning, LLava reasoning part
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        Privacy_data, splits = Privacy_data.split("|")
        splits = splits.split("_")
        self.images = []
        for split in splits:
            json_path = glob.glob(os.path.join(base_image_dir, "reason_seg", Privacy_data, "*{}*.json".format(split)))
            with open(json_path[0]) as file:
                privacy_data = json.load(file)
            self.image_infos = privacy_data
            Privacy_path = '/home/hanningchen/PrivacyDataset/florence2/combine_with_tag-3/{}'.format(split)
        
        #NOTE: Hanning, construct images and jsons file path
        jsons = []
        for i, image_info in enumerate(self.image_infos):
            image_file_name = image_info['image_name']
            self.images.append(os.path.join(Privacy_path, image_file_name))
            jsons.append(image_info)
        self.Privacy_data = (self.images, jsons)

    def __len__(self):
        #NOTE: Hanning, test for embedding generation
        return self.samples_per_epoch
        #return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, jsons = self.Privacy_data
        #NOTE: Hanning, comment it when generate embedding
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_data = jsons[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        
        #NOTE: construct the mask
        m_final = np.zeros((json_data["height"], json_data["width"])).astype(np.uint8)
        if len(json_data["mask_list"]) > 0:
            for poly in json_data["mask_list"]:
                try:
                    rle = mask_utils.frPyObjects(poly,
                                        json_data["height"], 
                                        json_data["width"])
                except:
                    print(json_data["mask_list"])
                    break
                m = mask_utils.decode(rle)
                if len(m.shape) == 3:
                    m = np.sum(m, axis=2)
                m_final = m_final | m
        m_final = m_final.astype(np.uint8)
        sampled_masks = m_final
        
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image_name = image_path.split("/")[-1]

        sampled_sent = json_data['expressions']
        questions = []
        answers = []
        #NOTE: TODO, Hanning add sentence later
        is_sentence = True
        if isinstance(sampled_sent, str):
            sampled_sents = [sampled_sent]
        else:
            sampled_sents = sampled_sent
        
        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                #NOTE: TODO, add class for train
                sys.exit("RIO only have full sentence")

            if self.explanatory == -1:
                answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i=0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1
        
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        image_name = image_path.split("/")[-1]

        if self.explanatory ==-1:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            masks = masks.unsqueeze(dim=0)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        #NOTE: Hanning, for embedding generation
        #task_id = json_data["task_id"]
        task_id = -1

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
            task_id,
        )