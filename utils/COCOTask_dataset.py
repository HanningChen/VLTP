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
from torchvision.utils import save_image
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import torchvision.datasets as dset

taskid2name = {
      1: "step on", 
      2: "sit comfortably", 
      3: "place flowers", 
      4: "get potatoes out of fire", 
      5: "water plant",
      6: "get lemon out of tea",
      7: "dig hole",
      8: "open bottle of beer",
      9: "open parcel",
      10: "serve wine",
      11: "pour sugar",
      12: "smear butter",
      13: "extinguish fire",
      14: "pound carpet"
}


class COCOTaskdataset(torch.utils.data.Dataset):
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
                 COCOTask_data="COCOTask|train",
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
        self.task_expression_prefix = "you can use the thing to "

        COCOTask_data, splits = COCOTask_data.split("|")
        splits = splits.split("_")
        images = []
        for split in splits:
            json_path = glob.glob(os.path.join(base_image_dir, "reason_seg", COCOTask_data, "*{}*.json".format(split)))
            jsons = []
            img_metas = []
            for i in range(len(json_path)):
                self.image_infos = COCO(json_path[i])
                COCO_Path = "/media/Data_1/COCO2014/{}2014".format(split)
                dataType = '{}2014'.format(split)
                img_ids = self.image_infos.getImgIds()
                temp = json_path[i].split('/')[-1].split('_')
                task_name = f"{temp[0]}_{temp[1]}"
                task_id = int(temp[1])
                for i, img_id in enumerate(img_ids):
                    image_info = self.image_infos.loadImgs(img_id)[0]
                    image_info["expressions"] = self.task_expression_prefix + taskid2name[task_id]
                    image_info['task_id'] = task_id
                    image_file_name = image_info['file_name']
                    images.append(os.path.join(COCO_Path, image_file_name))
                    img_metas.append(image_info)
                    ann_ids = self.image_infos.getAnnIds(imgIds=img_id)
                    anns = self.image_infos.loadAnns(ann_ids)
                    jsons.append(anns)
                self.COCOTask_data = (images, jsons, img_metas)
                # #NOTE: Hanning, for SAM fine tune
                # self.seg_embedding_dir = '/media/Data_2/hanningchen/{}/SEG/{}'.format(COCOTask_data, splits[0])

    def __len__(self):
        #NOTE: Hanning, test for embedding generation
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, jsons, img_metas = self.COCOTask_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        json_data = jsons[idx]
        img_meta = img_metas[idx]
        image = cv2.imread(image_path)
        # print("*"*100)
        # print("Save original image")
        # cv2.imwrite("./prune_test/{}_original.jpg".format(json_data['image_id']), image)
        # print("*"*100)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # NOTE: preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        #NOTE: construct the mask
        m_final = np.zeros((img_meta["height"], img_meta["width"])).astype(np.uint8)
        for json_id in range(len(json_data)):
            if int(json_data[json_id]['category_id']) != 1:
                continue
            poly = json_data[json_id]["segmentation"]
            rle = mask_utils.frPyObjects(poly,
                                img_meta["height"], 
                                img_meta["width"])
            m = mask_utils.decode(rle)
            if len(m.shape) == 3:
                m = np.sum(m, axis=2)
            m_final = m_final | m
        m_final = m_final.astype(np.uint8)
        sampled_masks = m_final

        image = self.transform.apply_image(image)  # NOTE: preprocess image for sam
        
        resize = image.shape[:2]

        sampled_sent = img_meta['expressions']
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
                sys.exit("COCO Tasks only have full sentence")

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
        
        # print("*"*100)
        # print(image.shape)
        # print("Save second preprocess image")
        # save_image(image,"./prune_test/{}_2ndprocess.jpg".format(json_data['image_id']))
        # print("*"*100)

        if self.explanatory ==-1:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            masks = masks.unsqueeze(dim=0)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        #NOTE: Hanning, for embedding generation
        task_id = img_meta["task_id"]

        #NOTE: Hanning, load SEG embedding
        # seg_embedding_path = os.path.join(self.seg_embedding_dir, "seg_{}.pt".format(json_data['task_id']))
        # if os.path.exists(seg_embedding_path):
        #     seg_embedding = torch.load(seg_embedding_path)
        # else:
        #     seg_embedding = None

        # return (
        #     image_path,
        #     image,
        #     image_clip,
        #     conversations,
        #     masks,
        #     label,
        #     resize,
        #     questions,
        #     sampled_sents,
        #     task_id,
        #     False,
        #     seg_embedding
        # )

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
            False,
        )