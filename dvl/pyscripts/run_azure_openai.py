import argparse
import asyncio
import base64
import io
import json
import logging
import os
import random
from os.path import dirname, abspath, join
from typing import Dict

import PIL.Image
from PIL import Image
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio

from dvl.multi_temp.run_vllm import prompt_libs


PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
CURRENT_DIR = dirname(abspath(__file__))


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def encode_base64_from_pillow_image(image_object: Image.Image, image_format: str = 'JPEG') -> str:
    buffered = io.BytesIO()
    image_object.save(buffered, format=image_format)
    binary_data = buffered.getvalue()
    base64_bytes = base64.b64encode(binary_data)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def get_messages_from_data(data_dict: Dict, subset: str, model_id: str):
    if subset in [
        "basic_change_choice_qa", "change_speed_choice_qa", "basic_change_report_qa", "change_speed_report_qa",
        "dense_temporal_caption", "regional_caption",
    ]:
        if subset in ["basic_change_choice_qa", "change_speed_choice_qa"]:
            if data_dict["task_type"] == "Single choice":
                prompt_key = "single_choice"
            elif data_dict["task_type"] == "Multiple choice":
                prompt_key = "multi_choice"
            else:
                raise ValueError(f"Unknown task_type: {data_dict['task_type']}")

            prompt_text = prompt_libs[prompt_key].format(prompt=data_dict["prompts"], options_str=data_dict["options_str"])
        else:
            prompt_text = prompt_libs[subset]

        prompt_splits = prompt_text.split("<image>")
        assert len(prompt_splits) == 2, len(prompt_splits)

        image_root = f"{PROJECT_ROOT}/data/test"
        if subset == "regional_caption":
            image_root = image_root + f"/regional_caption/images"

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_splits[0]}]}]
        for image_path, image_year in zip(data_dict["image_list"], data_dict["time_stamps"]):
            image_year = image_year[:4]
            image_path = join(image_root, image_path)
            image_base64 = encode_base64_from_pillow_image(PIL.Image.open(image_path).convert("RGB"))
            image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}

            messages[0]["content"].append({"type": "text", "text": f"\nyear {image_year}:"})
            messages[0]["content"].append(image_content)

        messages[0]["content"].append({"type": "text", "text": prompt_splits[1]})

    elif subset in ["eco_assessment"]:
        prompt_text = prompt_libs[subset].format(prompt=data_dict["prompts"], options_str=data_dict["options_str"])
        image_path = join(f"{PROJECT_ROOT}/data/test", data_dict["image_path"])
        image_base64 = encode_base64_from_pillow_image(PIL.Image.open(image_path).convert("RGB"))
        image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]

    else:
        raise ValueError('Unknown subset {}'.format(subset))

    return messages


async def run_model_by_message(client, messages, data_id, model, gen_args: Dict):
    retry_num, break_flag = 0, False
    response = None
    while not break_flag:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(model=model, messages=messages, **gen_args),
                timeout=300
            )
            try:
                response = response.choices[0].message.content
            except AttributeError:
                logger.error(f"AttributeError occurred for {model}. Return None for the current request.")
                response = None

            break_flag = True

        except (asyncio.TimeoutError, ConnectionError) as e:
            logger.error(f"Network exception occurred for {data_id}: {e}")

        except Exception as e:
            if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                    or "server error" in str(e).lower()
                    or "connection error" in str(e).lower()
            ):
                logger.error(f"Sever error occurred for {data_id}: {e}")
            else:
                logger.error(f"Other error occurred for {data_id}: {e}. skip it")
                break_flag = True

        if not break_flag:
            retry_num = retry_num + 1
            if retry_num > 10:
                logger.info(f"Max Retry reached for {data_id}, skip it")
                break_flag = True
            else:
                wait_time = min(2 ** retry_num, 10)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    return response


async def process_item(client, data_item, args) -> Dict:
    messages = get_messages_from_data(data_item, subset=args.subset, model_id=args.model_id)

    max_tokens = args.max_tokens
    if args.model_id in ["gpt-4o", "o1"]:
        max_tokens = min(args.max_tokens, 16384)
    if args.model_id in ["o3", "o1", "o4-mini"]:
        gen_args = dict(max_completion_tokens=max_tokens)
    else:
        gen_args = dict(max_tokens=max_tokens)

    response = await run_model_by_message(
        client=client, messages=messages, data_id=data_item["id"], model=args.model_id, gen_args=gen_args
    )

    return dict(
        custom_id=data_item["id"],
        request=data_item,
        response=response,
    )


async def process_with_semaphore(client, data_item, args, semaphore):
    async with semaphore:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return await process_item(client, data_item, args)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--n_thread", type=int, default=64)
    args = parser.parse_args()

    reformat_model_id = args.model_id.replace('/', '--')
    result_save_path = join(PROJECT_ROOT, "results", args.subset, reformat_model_id, "finished.jsonl")
    os.makedirs(dirname(result_save_path), exist_ok=True)

    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f.readlines():
                data_dict = json.loads(line.strip())
                if data_dict["response"] is not None and data_dict["response"] != "":
                    finish_ids.append(data_dict["id"])

    if len(finish_ids) > 0:
        print(f"Resume from {len(finish_ids)} finished samples")
    else:
        print(f"Start from scratch")

    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_BASE"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", None),
        api_key=os.environ.get("AZURE_OPENAI_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )

    subset_json = join(f"{PROJECT_ROOT}/data/test", f"{args.subset}.json")
    print("Reading data from {}".format(subset_json))
    with open(subset_json, "r") as f:
        ds = json.load(f)
        ds = [dict(id=f"{args.subset}_{data_idx}", **data_dict) for data_idx, data_dict in enumerate(ds)]
        ds = [data_dict for data_dict in ds if data_dict["id"] not in finish_ids]

    if len(ds) > 0:
        try:
            tasks = []
            semaphore = asyncio.Semaphore(args.n_thread)
            for data_item in ds:
                tasks.append(
                    process_with_semaphore(
                        client=client,
                        data_item=data_item,
                        args=args,
                        semaphore=semaphore
                    )
                )

            with open(result_save_path, "w" if len(finish_ids) == 0 else "a") as f:
                for task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                    ret_dict = await task
                    generated_text = ret_dict["response"]

                    dump_dict = {
                        "id": ret_dict["custom_id"],
                        "response": generated_text,
                        "request": ret_dict["request"],
                    }

                    f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
                    f.flush()

        finally:
            if client and hasattr(client, 'close'):
                await client.close()

            for task in asyncio.all_tasks(asyncio.get_running_loop()):
                if task is not asyncio.current_task() and not task.done():
                    task.cancel()

    with open(result_save_path) as f:
        finished_data = [json.loads(line.strip()) for line in f.readlines()]
        finished_data = [doc for doc in finished_data if doc["response"] is not None and doc["response"] != ""]

    with open(result_save_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(finished_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    asyncio.run(main())
