from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List
import re
from tqdm import tqdm
import argparse
import sys
import os
import magic
import time
import json


MAX_TOKENS = 8192


def split_on_chunks(input_file:str, chunk_size: int):
    st_idx = 0
    end_idx = st_idx + chunk_size
    chunks = []

    with open(input_file, 'r', encoding='utf-8') as f:
        data = str(f.read())

    while st_idx < len(data):
        chunk = data[st_idx: end_idx]
        if end_idx < len(data) - 1 and re.search(r'\s$', chunk) is None:
            last_space = re.search(r'\s\S*$', chunk)
            if last_space is None:
                raise Exception('bad text: no spaces detected')
            chunk_end_idx = last_space.span()[0]
            chunk = chunk[:chunk_end_idx]
            end_idx = st_idx + chunk_end_idx
        chunks.append(chunk)
        st_idx = end_idx
        end_idx = min(st_idx + chunk_size, len(data))

    return chunks


def infer_chat(model, tokenizer, chat_template: List[dict], user_query: str):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=MAX_TOKENS)
    messages = chat_template + [{'role': 'user', 'content': user_query}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = model.generate([text], sampling_params)
    assert len(outputs) == 1

    return outputs[0].outputs[0].text


def refactor_doc(file_path: str, few_shot_prompt: List[dict], output: str, model, tokenizer, chunk_size: int):
    def add_chunk(result, chunk):
        if re.search(r'\s$', chunk) is None:
            return result + " " + chunk
        return result + chunk

    # chunk size is equal to max tokens
    print(f'file path={file_path}')
    data_chunks = split_on_chunks(file_path, chunk_size)
    result = ""
    pbar = tqdm(len(data_chunks))

    for chunk in data_chunks:
        filtered_chunk = infer_chat(model, tokenizer, few_shot_prompt, 'refactor this text: ' + chunk)
        result = add_chunk(result, filtered_chunk)
        pbar.update(1)

    with open(output, 'w', encoding='utf-8') as o:
        o.write(result)


def load_model(model_name: str):
    llm = LLM(model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return llm, tokenizer


def validate_args(args):
    if args.chunk_size < 100:
        raise Exception(f'invalid chunk size={args.chunk_size}, should be 100 at least')
    if args.file_path.strip() != "" and not os.path.isfile(args.file_path):  # file is specified but doesn't exist
        raise Exception(f"file - {args.file_path} doesn't exists")
    if args.sys_prompt.strip() == "":
        raise Exception(f'empty system prompt')
    if not os.path.isdir(args.dir_path) and args.file_path.strip() == "":
        raise Exception(f"input dir - {args.dir_path} doesn't exists and no input file is specified")
    if not os.path.isdir(args.output):
        os.mkdir(args.output)


def is_text(file_path: str):
    try:
        return magic.from_file(file_path, mime=True).split('/')[0] == 'text'
    except Exception:
        # issue only on cyrillic file names on windows
        return file_path.split(".")[-1] in ['txt', 'md']


def print_types(files):
    for file in files:
        print(f"{os.path.basename(file)}:{magic.from_file(file, mime=True)}")


def read_json(file_path:str):
    if not os.path.isfile(file_path):
        raise Exception(f"can't read json: no such file - {file_path}")
    with open(file_path, encoding='utf-8') as f:
        data = f.read()
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path on disk')
    parser.add_argument('--file_path', type=str, required=False, default="",
                        help='file to refactor, if not specified, --dir_path will be used')
    parser.add_argument('--output', type=str, required=False, default='output',
                        help='path to generation result directory')
    parser.add_argument('--dir_path', type=str, required=False, default=".",
                        help='directory to get files on refactoring from. this argument is ignored if '
                             '--file_path is specified')
    parser.add_argument('--chunk_size', type=int, default=MAX_TOKENS, required=False,
                        help="chunk size of text splitting for one model inference iteration")
    parser.add_argument('--chat_prompt', type=bool, default=False, required=False,
                        help="weather of using chat template prompt from json config")
    parser.add_argument('--prompt_file', type=str, required=False, default='sys_prompt.txt',
                        help='json file with few-shot prompt')
    try:
        args = parser.parse_args()
        validate_args(args)
    except Exception as e:
        print('Parse args exception: ' + repr(e), file=sys.stderr)
        parser.print_help()
        return
    model_name = args.model_path
    chunk_size = args.chunk_size
    output = args.output
    file_path = args.file_path.strip()
    dir_path = args.dir_path
    prompt_path = args.prompt_file
    few_shot_prompt = read_json(prompt_path)
    st_time = time.time()

    if file_path != "":
        name, ext = os.path.splitext(os.path.basename(file_path))
        if not is_text(file_path):
            print(f"WARNING: {file_path} - is not a text file, so can't be filtered")
            return
        model, tokenizer = load_model(model_name)
        refactor_doc(file_path, few_shot_prompt, os.path.join(output, name + '_filtered' + ext),
                     model, tokenizer, chunk_size)
    else:
        files = os.listdir(dir_path)
        # select only text files:
        files = [f for f in [os.path.join(dir_path, _) for _ in files] if is_text(f)]

        if len(files) == 0:
            print(f'WARNING: no text files exists here - {dir_path}, nothing to filter')
            return
        model, tokenizer = load_model(model_name)

        for file in files:
            name, ext = os.path.splitext(os.path.basename(file))
            refactor_doc(file, few_shot_prompt, os.path.join(output, name + '_filtered' + ext),
                         model, tokenizer, chunk_size)
            print(f'file {name + ext} is refactored', flush=True)
    end_time = time.time()
    print(f'results is successfully wrote to {output}', flush=True)
    delta = end_time - st_time
    h = delta // 3600
    m = (delta - h * 3600) // 60
    s = (delta - h * 3600 - m * 60)
    print(f'total execution time={h}h:{m}m:{s}s')


if __name__ == "__main__":
    main()
