import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .RIO_dataset import RIODataset
from .Privacy_dataset import PrivacyDataset
from .COCOTask_dataset import COCOTaskdataset     
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .vqa_dataset import VQADataset
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import json

#NOTE: add by Hanning for COCO Tasks
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

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    task_id_list = []

    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        task_id,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        #NOTE: for embedding generation
        task_id_list.append(task_id)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)

            parts[0] += sep
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "task_id_list": task_id_list
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                #NOTE: Hanning, for finetune
                #continue
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                #NOTE: Hanning, for finetune
                #continue
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                #NOTE: Hanning, for finetune
                #NOTE: Hanning, hardcode explanaory for now
                if 'RIO' in reason_seg_data:
                    self.all_datasets.append(
                        RIODataset(
                            base_image_dir,
                            tokenizer,
                            vision_tower,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            reason_seg_data,
                            -1,
                        )
                    )
                elif 'Privacy' in reason_seg_data:
                    self.all_datasets.append(
                        PrivacyDataset(
                            base_image_dir,
                            tokenizer,
                            vision_tower,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            reason_seg_data,
                            -1,
                        )
                    )
                elif 'COCOTask' in reason_seg_data:
                    self.all_datasets.append(
                        COCOTaskdataset(
                            base_image_dir,
                            tokenizer,
                            vision_tower,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            reason_seg_data,
                            -1,
                        )
                    )
                else:
                    self.all_datasets.append(
                        ReasonSegDataset(
                            base_image_dir,
                            tokenizer,
                            vision_tower,
                            samples_per_epoch,
                            precision,
                            image_size,
                            num_classes_per_sample,
                            exclude_val,
                            reason_seg_data,
                            explanatory,
                        )
                    )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        if len(self.all_datasets) == 1:
            data = self.all_datasets[0]
        else:
            data = self.all_datasets[ind]
        #inference = False
        #return *data[0], inference
        return data[0]


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")

        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )

class ValRIODataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 3:
            ds, split, common_type = splits
            json_path = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, "*_{}*.json".format(common_type))
            )
            with open(json_path[0]) as file:
                coco_data = json.load(file)
            self.image_infos = coco_data
            COCO_Path = "/media/Data_1/COCO2014/{}2014".format(split)
            self.dataType = "{}2014".format(split)
            self.seg_embedding_dir = '/media/Data_2/hanningchen/{}/SEG/{}'.format(ds, "{}_{}".format('test', common_type))
        else:
            print("+"*100)
            print("Running evaluation over training dataset")
            print("+"*100)
            ds, split= splits
            json_path = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, "*_{}*.json".format(split))
            )
            with open(json_path[0]) as file:
                coco_data = json.load(file)
            self.image_infos = coco_data
            COCO_Path = "/media/Data_1/COCO2014/{}2014".format(split)
            self.dataType = "{}2014".format(split)
            self.seg_embedding_dir = '/media/Data_2/hanningchen/{}/SEG/{}'.format(ds, split)

        self.images = []
        self.jsons = []
        for i, image_info in enumerate(self.image_infos):
            image_file_name = "COCO_{}_{}.jpg".format(self.dataType,str(image_info['image_id']).zfill(12))
            self.images.append(os.path.join(COCO_Path, image_file_name))
            self.jsons.append(image_info)

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_data = self.jsons[idx]
        #NOTE: construct the mask
        m_final = np.zeros((json_data["height"], json_data["width"])).astype(np.uint8)
        for poly in json_data["mask_list"]:
            rle = mask_utils.frPyObjects(poly,
                                 json_data["height"], 
                                 json_data["width"])
            m = mask_utils.decode(rle)
            if len(m.shape) == 3:
                m = np.sum(m, axis=2)
            m_final = m_final | m
        m_final = m_final.astype(np.uint8)
        sampled_masks = m_final
        
        sampled_sent = json_data['expressions']
        if isinstance(sampled_sent,str):
            sampled_sents = [sampled_sent]
        elif isinstance(sampled_sent, list):
            sampled_sents = sampled_sent
        else:
            print("Intention is wrong!!!")
        is_sentence = True

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        masks = sampled_masks

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        masks = masks.unsqueeze(dim=0)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        #NOTE: Hanning, for embedding generation
        task_id = json_data["task_id"]
        seg_embedding_path = os.path.join(self.seg_embedding_dir, "seg_{}.pt".format(json_data['task_id']))
        seg_embedding = torch.load(seg_embedding_path)
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            task_id,
            inference,
            seg_embedding
        )

class TrainRIODataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        train_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        RIO_data, splits = train_dataset.split("|")
        splits = splits.split("_")
        self.images = []
        for split in splits:
            json_path = glob.glob(os.path.join(base_image_dir, "reason_seg", RIO_data, "*{}*.json".format(split)))
            with open(json_path[0]) as file:
                coco_data = json.load(file)
            self.image_infos = coco_data
            COCO_Path = "/media/Data_1/COCO2014/{}2014".format(split)
            self.dataType = '{}2014'.format(split)
            
        self.images = []
        self.jsons = []
        for i, image_info in enumerate(self.image_infos):
            image_file_name = "COCO_{}_{}.jpg".format(self.dataType,str(image_info['image_id']).zfill(12))
            self.images.append(os.path.join(COCO_Path, image_file_name))
            self.jsons.append(image_info)

        #self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_data = self.jsons[idx]
        #NOTE: construct the mask
        m_final = np.zeros((json_data["height"], json_data["width"])).astype(np.uint8)
        for poly in json_data["mask_list"]:
            rle = mask_utils.frPyObjects(poly,
                                 json_data["height"], 
                                 json_data["width"])
            m = mask_utils.decode(rle)
            if len(m.shape) == 3:
                m = np.sum(m, axis=2)
            m_final = m_final | m
        m_final = m_final.astype(np.uint8)
        sampled_masks = m_final
        
        sampled_sent = json_data['expressions']
        if isinstance(sampled_sent,str):
            sampled_sents = [sampled_sent]
        elif isinstance(sampled_sent, list):
            sampled_sents = sampled_sent
        else:
            print("Intention is wrong!!!")
        is_sentence = True

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        masks = sampled_masks

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        masks = masks.unsqueeze(dim=0)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        #NOTE: Hanning, for embedding generation
        task_id = json_data["task_id"]

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            task_id,
            inference,
        )

class ValPrivacyDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        print("*"*100)
        print(splits)
        ds, split = splits
        json_path = glob.glob(
            os.path.join(self.base_image_dir, "reason_seg", ds, "*{}*.json".format(split))
        )
        print("+"*100)
        print(json_path)
        print("+"*100)
        with open(json_path[0]) as file:
            privacy_data = json.load(file)
        self.image_infos = privacy_data
        Privacy_path = '/home/hanningchen/PrivacyDataset/florence2/combine_with_tag-3/{}'.format(split)
        
        self.images = []
        self.jsons = []
        for i, image_info in enumerate(self.image_infos):
            image_file_name = image_info['image_name']
            self.images.append(os.path.join(Privacy_path, image_file_name))
            self.jsons.append(image_info)

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_data = self.jsons[idx]
        #NOTE: construct the mask
        m_final = np.zeros((json_data["height"], json_data["width"])).astype(np.uint8)
        if len(json_data["mask_list"]) > 0:
            for poly in json_data["mask_list"]:
                rle = mask_utils.frPyObjects(poly,
                                    json_data["height"], 
                                    json_data["width"])
                m = mask_utils.decode(rle)
                if len(m.shape) == 3:
                    m = np.sum(m, axis=2)
                m_final = m_final | m
        m_final = m_final.astype(np.uint8)
        sampled_masks = m_final
        
        sampled_sent = json_data['expressions']
        if isinstance(sampled_sent,str):
            sampled_sents = [sampled_sent]
        elif isinstance(sampled_sent, list):
            sampled_sents = sampled_sent
        else:
            print("Intention is wrong!!!")
        is_sentence = True

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        masks = sampled_masks

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        masks = masks.unsqueeze(dim=0)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        #NOTE: Hanning, for embedding generation
        #task_id = json_data["task_id"]
        task_id = -1
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            task_id,
            inference,
        )

class ValCOCOTaskDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        
        self.images = []
        self.jsons = []
        self.img_metas = []
        self.task_expression_prefix = "you can use the thing to "

        if len(splits) == 3:
            ds, split, task_id = splits
            # json_path = glob.glob(
            #     os.path.join(self.base_image_dir, "reason_seg", ds, "*{}_{}.json".format(task_id, 'test'))
            # )
            json_path = os.path.join(self.base_image_dir, "reason_seg", ds, "task_{}_test.json".format(task_id))
            self.image_infos = COCO(json_path)
            COCO_Path = "/media/Data_1/COCO2014/{}2014".format(split)
            self.dataType = "{}2014".format(split)
            img_ids = self.image_infos.getImgIds()
            temp = json_path.split('/')[-1].split('_')
            for i, img_id in enumerate(img_ids):
                image_info = self.image_infos.loadImgs(img_id)[0]
                image_info["expressions"] = self.task_expression_prefix + taskid2name[int(task_id)]
                image_info['task_id'] = int(task_id)
                image_file_name = image_info['file_name']
                self.images.append(os.path.join(COCO_Path, image_file_name))
                self.img_metas.append(image_info)
                ann_ids = self.image_infos.getAnnIds(imgIds=img_id)
                anns = self.image_infos.loadAnns(ann_ids)
                self.jsons.append(anns)

            # self.seg_embedding_dir = '/media/Data_2/hanningchen/{}/SEG/{}'.format(ds, "{}_{}".format('test', task_id))
        else:
            print("+"*100)
            print("Running evaluation over training dataset")
            print("Not implement yet")
            exit()
            print("+"*100)
            ds, split= splits
            json_path = glob.glob(os.path.join(base_image_dir, "reason_seg", ds, "*{}*.json".format(split)))
            
            COCO_Path = "/media/Data_1/COCO2014/{}2014".format(split)
            self.dataType = "{}2014".format(split)
            # self.seg_embedding_dir = '/media/Data_2/hanningchen/{}/SEG/{}'.format(ds, split)

        # self.images = []
        # self.jsons = []
        # for i, image_info in enumerate(self.image_infos):
        #     image_file_name = "COCO_{}_{}.jpg".format(self.dataType,str(image_info['image_id']).zfill(12))
        #     self.images.append(os.path.join(COCO_Path, image_file_name))
        #     self.jsons.append(image_info)

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_data = self.jsons[idx]
        img_meta = self.img_metas[idx]
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
        
        sampled_sent = img_meta['expressions']
        if isinstance(sampled_sent,str):
            sampled_sents = [sampled_sent]
        elif isinstance(sampled_sent, list):
            sampled_sents = sampled_sent
        else:
            print("Intention is wrong!!!")
        is_sentence = True

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        masks = sampled_masks

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        masks = masks.unsqueeze(dim=0)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        #NOTE: Hanning, for embedding generation
        task_id = img_meta["task_id"]
        # seg_embedding_path = os.path.join(self.seg_embedding_dir, "seg_{}.pt".format(json_data['task_id']))
        # seg_embedding = torch.load(seg_embedding_path)
        
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            task_id,
            inference
        )

def collate_fn_sam(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    task_id_list = []
    seg_embeddings = []

    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        task_id,
        inference,
        seg_embedding
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        #NOTE: for embedding generation
        task_id_list.append(task_id)
        inferences.append(inference)
        seg_embeddings.append(seg_embedding)

    if tokenizer is not None:
        if use_mm_start_end:
            # replace <image> token
            for i in range(len(conversation_list)):
                replace_token = DEFAULT_IMAGE_TOKEN
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
                conversation_list[i] = conversation_list[i].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token
                )
                
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversation_list
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks = input_ids.ne(tokenizer.pad_token_id)

        conv = conversation_lib.default_conversation.copy()
        targets = input_ids.clone()

        if conv_type == "llava_v1":
            sep = conv.sep + conv.roles[1] + ": "
        else:
            sep = "[/INST] "

        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())
            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)

                parts[0] += sep
                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if False:
                z = target.clone()
                z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
                if local_rank == 0:
                    print(
                        "conversation: ",
                        conversation,
                        "tokenizer.decode(z): ",
                        tokenizer.decode(z),
                    )

            if cur_len < tokenizer.model_max_length:
                assert cur_len == total_len

        if inferences[0] == False:
            truncate_len = tokenizer.model_max_length - 255

            if input_ids.shape[1] > truncate_len:
                input_ids = input_ids[:, :truncate_len]
                targets = targets[:, :truncate_len]
                attention_masks = attention_masks[:, :truncate_len]

    if tokenizer is None:
        input_ids = None
        targets = None
        attention_masks = None

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "task_id_list": task_id_list,
        "seg_embeddings": seg_embeddings
    }
