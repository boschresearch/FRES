""" Utility classes and functions related to FRES (ACL 2025).
Copyright (c) 2025 Robert Bosch GmbH
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
butWITHOUT ANYWARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import base64
import json
import torch
from prompts import PROMPT_NT, PROMPT_NT_IMG
from utils import reformat_table, load_image
from PIL import Image
from tqdm import tqdm
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


MIN_PIXELS = 20*28*28
MAX_PIXELS = 35000*28*28


def prepare_prompts(instance, add_table_text, no_image, style, no_question):
    white_place_holder_url = base64.b64encode(
        open("../data/images/white_place_holder.jpg", 'rb').read()).decode('ascii')
    question = instance["question"] if not no_question else ""
    dataset = instance["dataset"] if "dataset" in list(
        instance.keys()) else ""
    id = instance["table_id"] if "table_id" in list(
        instance.keys()) else instance['id']
    id = str(id) if not isinstance(id, str) else id
    table_caption = instance["table_caption"] if "table_caption" in list(
        instance.keys()) else None
    table_string = instance["table_string"] if "table_string" in list(
        instance.keys()) else reformat_table(instance["table"], table_caption, style=style)
    image_path = f"../data/images/{dataset}_{id}.jpg"
    try:
        image_url = base64.b64encode(
            open(image_path, 'rb').read()).decode('ascii')
    except:
        print("Can not find image path:", image_path)
        image_url = None

    if add_table_text:
        text_prompt = PROMPT_NT[style].format(
            question=question, input_seg=table_string)
    else:
        text_prompt = PROMPT_NT_IMG[style].format(
            question=question)

    if not no_image:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image;base64,{image_url}"},
                    {"type": "text", "text": text_prompt}
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image",
                        "image": f"data:image;base64,{white_place_holder_url}"},
                    {"type": "text", "text": text_prompt}
                ],
            }
        ]
    return messages


def run_glm_hf(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda").eval()
    return model, tokenizer


def glm_inference(model, tokenizer, image, text, max_length=8000):
    image = image.convert('RGB')
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": text}],
                                           add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True)  # chat mode
    # print(len(inputs))  # input_ids  attention_mask  position_ids  images

    if inputs["input_ids"].shape[-1] > max_length:
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        inputs["position_ids"] = inputs["position_ids"][:, :max_length]
    inputs = inputs.to("cuda")
    gen_kwargs = {"max_new_tokens": 256, "do_sample": False}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        generated_texts = tokenizer.decode(outputs[0])
    return generated_texts


def run_phi_hf(model_path):
    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='eager'
    )

    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              num_crops=16
                                              )
    return model, processor


def phi_inference(model, tokenizer, image, text, max_length=8000):
    images = [image]
    placeholder = "<|image_1|>\n"

    messages = [
        {"role": "user", "content": placeholder+text},
    ]

    prompt = tokenizer.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, images, max_length=max_length,
                       return_tensors="pt")  # input_ids, attention_mask, pixel_value, image_sizes

    if inputs["input_ids"].shape[-1] > max_length:
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
    inputs = inputs.to("cuda")
    generation_args = {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(**inputs,
                                  eos_token_id=tokenizer.tokenizer.eos_token_id,
                                  **generation_args
                                  )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.batch_decode(generate_ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    return response


def run_pixtral_hf(model_path):

    tokenizer = MistralTokenizer.from_file(f"{model_path}/tekken.json")
    model = Transformer.from_folder(model_path)
    return model, tokenizer


def run_llava_next(model_path):
    processor = LlavaNextProcessor.from_pretrained(model_path)
    llm = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    llm.to("cuda:0")
    return llm, processor


def llava_inference(model, processor, image, text):

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image")
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt,
                       return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return processor.decode(output[0], skip_special_tokens=True)


def run_internvl(model_path):
    # model_name = "OpenGVLab/InternVL2-2B"

    llm = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False)
    return llm, tokenizer


def internvl_inference(model, tokenizer, image_path, text, max_length=8000):
    # max length 8192
    # set the max number of tiles in `max_num`
    pixel_values = load_image(image_path,
                              max_num=1).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=256, do_sample=False)
    question = f'<image>\n{text}'
    model_inputs_len = len(tokenizer.encode(question))
    if model_inputs_len > max_length:
        question = tokenizer.decode(tokenizer.encode(question)[
                                    :max_length])
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


def loading_model(model_path):
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 2},
        # dtype="float16"
    )
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    return llm, processor


def mistral_inference(model, tokenizer, image_url, text):
    completion_request = ChatCompletionRequest(messages=[UserMessage(
        content=[ImageURLChunk(image_url=image_url), TextChunk(text=text)])])
    encoded = tokenizer.encode_chat_completion(completion_request)
    images = encoded.images
    tokens = encoded.tokens
    out_tokens, _ = generate([tokens], model, images=[images], max_tokens=1024,
                             temperature=0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.decode(out_tokens[0])
    if "```json" in result:
        result = "\n".join(result.split("\n")[1:-1])
    return result


def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        data = [json.loads(item) for item in f]
    return data


def main(args):
    if "qwen" in args.model_name:
        llm, processor = loading_model(args.model_path)
    elif "intern" in args.model_name:
        # processor is the tokenizer
        llm, tokenizer = run_internvl(args.model_path)
    elif "mistral" in args.model_name:
        llm, tokenizer = run_pixtral_hf(args.model_path)
    elif "llava" in args.model_name:
        llm, tokenizer = run_llava_next(args.model_path)
    elif "phi" in args.model_name:
        llm, tokenizer = run_phi_hf(args.model_path)
    elif "glm" in args.model_name:
        llm, tokenizer = run_glm_hf(args.model_path)

    dataset = load_dataset(args.dataset_path)
    for i, item in enumerate(dataset):
        item["instance_index"] = i

    messages = [prepare_prompts(item, add_table_text=args.add_table_text,
                                no_image=args.no_image, style=args.style, no_question=args.no_question) for item in dataset]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        seed=42,
        n=1
    )

    with open(args.output_dir+f'/{args.model_name}_add_table_text_{args.add_table_text}_no_image_{args.no_image}_style_{args.style}_no_question_{args.no_question}_output.json', "w") as f:
        for msg, ori in tqdm(zip(messages, dataset)):
            dataset = ori["dataset"] if "dataset" in list(ori.keys()) else ""
            id = ori["table_id"] if "table_id" in list(ori.keys()) else ""
            if not args.no_image:
                try:
                    image_path = f"images/{dataset}_{id}.jpg"
                    img = Image.open(image_path)
                except:
                    img = None

            else:
                image_path = "images/white_place_holder.jpg"
                img = Image.open(image_path)
            encoded_image_url = msg[0]["content"][0]["image"]
            text = msg[0]["content"][1]["text"]
            if "qwen" in args.model_name:
                prompt = processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True)
                # inputs = processor(
                #     text=[prompt], images=[img], padding=True, return_tensors="pt"
                # )
                # input_token_length = inputs["input_ids"].shape[-1]
                mm_data = {"image": process_vision_info(
                    msg)[0]}   # image_input
                llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
                outputs = llm.generate(
                    [llm_inputs], sampling_params=sampling_params)
                generated_texts = outputs[0].outputs[0].text

            elif "mistral" in args.model_name:
                generated_texts = mistral_inference(
                    llm, tokenizer, encoded_image_url, text)
            elif "llava" in args.model_name:
                generated_texts = llava_inference(llm, tokenizer, img, text)
            elif "intern" in args.model_name:
                generated_texts = internvl_inference(
                    llm, tokenizer, image_path, text)
            elif "phi" in args.model_name:
                generated_texts = phi_inference(llm, tokenizer, img, text)
            elif "glm" in args.model_name:
                generated_texts = glm_inference(llm, tokenizer, img, text)

            idx = ori["id"] if "id" in list(ori.keys()) else ""
            table_idx = ori["table_id"] if "table_id" in list(
                ori.keys()) else ""
            question = ori["question"]
            answer = ori["answer"]
            quetsion_complexity = ori["question_complexity"] if "question_complexity" in list(
                ori.keys()) else ""
            size = ori["size"] if "size" in list(ori.keys()) else ""
            table_res = ori["table_res"] if "table_res" in list(
                ori.keys()) else ""
            dataset = ori["dataset"] if "dataset" in list(
                ori.keys()) else ""

            new_instance = {"question": question,
                            "answer": answer, "pred_answer": generated_texts, "id": idx, "table_id": table_idx, "question_complexity": quetsion_complexity, "size": size, "table_res": table_res, "dataset": dataset}
            f.write(json.dumps(new_instance)+"\n")
            f.flush()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="")
    parser.add_argument('--dataset_path', type=str,
                        default="controlled_data.json")
    parser.add_argument('--model_name', type=str,
                        default="qwen7b")
    parser.add_argument('--output_dir', type=str,
                        default='.')
    parser.add_argument('--no_question', action='store_true')
    parser.add_argument('--add_table_text', action='store_true')
    parser.add_argument('--no_image', action='store_true')
    parser.add_argument('--bz', type=int, default=10)
    parser.add_argument('--style', type=int, default=0)

    args = parser.parse_args()
    main(args)
