import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,EXPLANATORY_QUESTION_LIST)

import json
from tqdm import tqdm

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    #parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--version",default="xinlai/LISA-7B-v1-explanatory")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--rio_task",default="uncommon",type=str,choices=["common","uncommon"])
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    #NOTE: Hanning, check args member
    print("+"*100)
    args_dict = vars(args)
    for arg_name, arg_value in args_dict.items():
        print(f"{arg_name}: {arg_value}")
    print("+"*100)

    os.makedirs(args.vis_save_path, exist_ok=True)
    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        #NOTE: Hanning, try to use two GPUs
        model = model.bfloat16().cuda()
        # model.model.vision_tower = model.model.vision_tower.bfloat16().to('cuda:0')
        # model.model.visual_model = model.model.visual_model.bfloat16().to('cuda:0')
        # model.model.mm_projector = model.model.mm_projector.bfloat16().to('cuda:0')
        # model.model.text_hidden_fcs = model.model.text_hidden_fcs.bfloat16().to('cuda:0')
        # model.model.layers = model.model.layers.bfloat16().to('cuda:0')
        # model.model.embed_tokens = model.model.bfloat16().to('cuda:0')
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    #NOTE: Hanning, try to use two GPUs
    #vision_tower.to(device=args.local_rank)
    vision_tower.cuda()

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    #NOTE: Hanning, load RIO data
    COCO_TEST = "/media/Data_1/COCO2014/val2014"
    if args.rio_task == "uncommon":
        print("*"*100)
        print("Evaluate uncommon items")
        with open("RIO_uncommon_all_thing.json") as f:
            rio_data = json.load(f)
    else:
        print("*"*100)
        print("Evaluate common items")
        with open("RIO_common_all_thing.json") as f:
            rio_data = json.load(f)

    dataType = 'val2014'
    for index, annot in enumerate(tqdm(rio_data)):
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        #prompt = input("Please input your prompt: ")
        
        #NOTE: Hanning, for RIO
        prompt = "Can you segment the item " + annot["expressions"][0] 
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        #image_path = input("Please input the image path: ")
        #NOTE: Hanning, for RIO
        image_path = os.path.join(COCO_TEST, "COCO_{}_{}.jpg".format(dataType,str(annot['image_id']).zfill(12)))

        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]] #NOTE: (427, 640)

        #NOTE: Hanning by default image_clip is into cuda:0
        #NOTE: [1, 3, 224, 224]
        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )

        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np) #NOTE: (683, 1024, 3)
        resize_list = [image.shape[:2]]
        
        #NOTE: Hanning, SAM process image on cuda:0
        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        ) #NOTE:[1, 3, 1024, 1024]

        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        
        #input_ids = input_ids.unsqueeze(0).cuda()
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))
        print(annot["expressions"][0])

        #NOTE: Hanning
        if index > 10:
            break


if __name__ == "__main__":
    main(sys.argv[1:])