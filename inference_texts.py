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
import json
import argparse
import math
from utils import reformat_table
from prompts import PROMPT_NT
from vllm import LLM, SamplingParams
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import AutoModelForCausalLM, AutoTokenizer


def mistral_load(model_path):
    tokenizer = MistralTokenizer.from_file(f"{model_path}/tekken.json")
    model = Transformer.from_folder(model_path)
    return model, tokenizer


def mistral_inference(model, tokenizer, prompts):
    results = []
    for prompt in prompts:
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)])

        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate([tokens], model, max_tokens=512, temperature=0,
                                 eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = tokenizer.decode(out_tokens[0])
        results.append(result)

    return results


def glm_load(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto")
    return model, tokenizer


def glm_inference(model, tokenizer, prompts):
    results = []
    for prompt in prompts:
        message = [
            {
                "role": "system",
                "content": "Answer the following question."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = tokenizer.apply_chat_template(
            message,
            return_tensors='pt',
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)

        input_len = inputs['input_ids'].shape[1]
        generate_kwargs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "max_new_tokens": 256,
            "do_sample": False,
        }
        out = model.generate(**generate_kwargs)
        generated_texts = tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True)
        results.append(generated_texts)
    return results


def prepare_prompt(instance, style, no_table, no_question):

    question = instance["question"] if not no_question else ""
    table_caption = instance["table_caption"] if "table_caption" in list(
        instance.keys()) else None
    if "table_string" in list(instance.keys()):
        input_seg = instance["table_string"]
    else:
        input_seg = reformat_table(
            instance["table"], caption=table_caption, style=style)
    if no_table:
        input_seg = ""

    prompt = PROMPT_NT[style].format(
        input_seg=input_seg, question=question)

    return prompt


def load_model(model_path):
    model = LLM(model=model_path, trust_remote_code=True)
    return model


def inference(prompts, model, bz=100):
    decoded = []
    batch = {i: [] for i in range(math.ceil(len(prompts)/bz))}
    for i in list(batch.keys()):
        batch[i] = prompts[i*bz:(i+1)*bz]
    sampling_params = SamplingParams(
        max_tokens=256, temperature=0, n=1, seed=42)
    for k, v in batch.items():
        outputs_all = model.generate(v, sampling_params)
        decoded_bz = [
            item.text for output in outputs_all for item in output.outputs]
        decoded += decoded_bz
    return decoded


def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        data = [json.loads(item) for item in f]
    return data


def main(args):
    data = load_dataset(args.dataset_path)
    if "mistral-nemo" in args.model_path.lower():
        model, tokenizer = mistral_load(args.model_path)
    elif "glm" in args.model_path.lower():
        model, tokenizer = glm_load(args.model_path)
    else:
        model = load_model(args.model_path)
    prompts = [prepare_prompt(
        item, args.style, args.no_table, args.no_question)for item in data]
    if "mistral-nemo" in args.model_path.lower():
        output = mistral_inference(model, tokenizer, prompts)
    elif "glm" in args.model_path.lower():
        output = glm_inference(model, tokenizer, prompts)
    else:
        output = inference(prompts, model, args.bz)
    with open(args.output_dir+f'/{args.model_name}_add_table_text_{args.add_table_text}_style_{args.style}_no_table_{args.no_table}_no_question_{args.no_question}_output.json', "w") as f:
        for out, ori in zip(output, data):
            idx = ori["id"] if "id" in list(ori.keys()) else ""
            table_idx = ori["table_id"] if "table_id" in list(
                ori.keys()) else ""
            question = ori["question"]
            answer = ori["answer"]
            quetsion_complexity = ori["question_complexity"]
            size = ori["size"]
            table_res = ori["table_res"]
            dataset = ori["dataset"]

            new_instance = {"question": question,
                            "answer": answer, "pred_answer": out, "id": idx, "table_id": table_idx, "question_complexity": quetsion_complexity, "size": size, "table_res": table_res, "dataset": dataset}
            f.write(json.dumps(new_instance)+"\n")
            f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default="controlled_data.json")
    parser.add_argument('--model_name', type=str,
                        default="qwen7b")
    parser.add_argument('--model_path', type=str,
                        default="")
    parser.add_argument('--output_dir', type=str,
                        default=".")
    parser.add_argument('--bz', type=int, default=100)
    parser.add_argument('--no_table', action='store_true')
    parser.add_argument('--no_question', action='store_true')
    parser.add_argument('--add_table_text', action='store_true')
    parser.add_argument('--style', type=int, default=0)
    args = parser.parse_args()
    main(args)
