import argparse
import json
import os
from dataclasses import asdict
from os.path import dirname, abspath, join
from typing import Dict, List

from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm
from transformers import GenerationConfig
from vllm import EngineArgs, LLM, SamplingParams

from dvl.models import build_model_config, ModelConfig


PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))
CURRENT_DIR = dirname(abspath(__file__))


prompt_libs = dict(
    multi_choice="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years. 
Your objective is to unearth the changes and continuities to answer the question below. Review the provided choices and select all that align with your findings.

<image>

Question: 
{prompt}

Choices:
{options_str}

Submit ONLY the capital letter(s) for your confirmed findings.
- If a single finding is confirmed, answer its letter, such as "B".
- If multiple findings are confirmed, list their letters separated by a comma and a space, such as "C, D, E"
No further commentary or annotations are permitted in your answer.""",

    single_choice="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years. 
Your objective is to unearth the changes and continuities to answer the question below. Review the provided choices and select the one that align with your finding.

<image>

Question: 
{prompt}

Choices:
{options_str}

Submit ONLY the capital letter for your confirmed finding. Directly answer its letter, such as "B".
No further commentary or annotations are permitted in your answer.""",

    eco_assessment="""Your task is to interpret the provided image and answer the question below. 
From the potential choices offered, please identify the one that most accurately answer the question.

Question: 
{prompt}

Choices:
{options_str}

Please provide ONLY the capital letter corresponding to your answer choice, such as "C", with no additional explanations.""",

    basic_change_report_qa="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years. 
Your objective is to identify and report basic land cover changes.

<image>

Generate a concise summary (1-3 sentences) that adheres to the following:
1. Time Period: Begin by stating the full observation period (e.g., "Over the YYYY–YYYY period," or "Over the X-year period from YYYY to YYYY,").
2. Key Transitions: Detail the primary land cover transitions observed, focusing on changes between major land cover types, such as vegetated areas, non-vegetated surfaces, and built-up areas/buildings.
3. Quantification: Report these transitions using percentage ranges (e.g., X-Y%) or approximate percentages (e.g., ~X%).""",

    change_speed_report_qa="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to determine and report the building area changes between consecutive image dates, including expansions, shrinkages, and areas with no change.

<image>

Generate a report string that adheres to the following:
1. Calculate the building area change rates between consecutive years (including expansion, shrinkage, and no change periods)
2. Output the results in the following format (choose one of the following for each time period):
"The changes were as follows: X% expansion from [Start Year] to [End Year]" OR "Y% shrinkage from [Start Year] to [End Year]" OR "no significant change from [Start Year] to [End Year]"

Example: "The expansion rates were as follows: 1.09% from 2006 to 2009, 1.1% from 2009 to 2011, 2.34% from 2011 to 2014.""",

    regional_caption="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to densely describe all change events within the red-boxed area over the time period covered by these images.

<image>

Generate a description string that adheres to the following:
1. Segment descriptions by distinct year ranges (e.g., YYYY-YYYY).
2. For each period, detail the initial land cover, specific changes observed (construction, clearing, feature additions/removals), and explicitly note any stability.
3. Use concise, factual language focusing on visible features like vegetation, buildings, infrastructure, and land states, including spatial references where relevant.""",

    dense_temporal_caption="""Your task is to examine a chronological series of remote sensing images over a consistent geographical area, recorded across different years.
Your objective is to describe the dynamics of this area over the time period covered by these images.

<image>

Generate a description string that adheres to the following:
1. Start with a general summary statement outlining the main trend over the entire period.
2. Detail specific land feature changes chronologically, using clear spatial references (e.g., top-left, center-right).
3. Conclude with an optional overall summary of the changes or final state.""",
)


def get_messages_from_data(data_dict: Dict, subset: str, model_id: str, used_prompt_libs=None):
    if used_prompt_libs is None:
        used_prompt_libs = prompt_libs
    images = None
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

            prompt_text = used_prompt_libs[prompt_key].format(prompt=data_dict["prompts"], options_str=data_dict["options_str"])
        else:
            prompt_text = used_prompt_libs[subset]

        prompt_splits = prompt_text.split("<image>")
        assert len(prompt_splits) == 2, len(prompt_splits)

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_splits[0]}]}]

        image_root = f"{PROJECT_ROOT}/data/test"
        if subset == "regional_caption":
            image_root = image_root + f"/regional_caption/images"

        if "llava" in model_id.lower():
            video = [join(image_root, image_path) for image_path in data_dict["image_list"]]
            video_prefix = "These are the images captured in " + ", ".join([image_year[:4] for image_year in data_dict["time_stamps"]]) + ".\n"
            messages[0]["content"].append({"type": "text", "text": video_prefix})
            messages[0]["content"].append({"type": "video", "video": video})

        else:
            images = []
            for image_path, image_year in zip(data_dict["image_list"], data_dict["time_stamps"]):
                image_year = image_year[:4]
                image_path = join(image_root, image_path)
                images.append(image_path)

                messages[0]["content"].append({"type": "text", "text": f"\nyear {image_year}:"})
                messages[0]["content"].append({"type": "image", "image": image_path})

        messages[0]["content"].append({"type": "text", "text": prompt_splits[1]})

    elif subset in ["eco_assessment"]:
        prompt_text = used_prompt_libs[subset].format(prompt=data_dict["prompts"], options_str=data_dict["options_str"])
        image_path = join(f"{PROJECT_ROOT}/data/test", data_dict["image_path"])

        images = [image_path]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]

    else:
        raise ValueError('Unknown subset {}'.format(subset))

    return messages, images


def create_batch_inputs(data_list: List[Dict], model_config: ModelConfig, args):
    for i in range(0, len(data_list), args.batch_size):
        batch_data = data_list[i:i + args.batch_size]
        batch_inputs = []
        batch_metadata = []

        for data_dict in batch_data:
            messages, images = get_messages_from_data(data_dict, args.subset, args.model_id)
            if "llava" in args.model_id.lower() and args.subset != "eco_assessment":
                prompt_text, multimodal_inputs = model_config.get_prompt_from_question(messages)
                inputs = {"prompt": prompt_text, **multimodal_inputs}
            else:
                prompt_text = model_config.get_prompt_from_question(messages)
                images = [Image.open(image_path).convert("RGB") for image_path in images]
                if args.image_size is not None:
                    for img_idx, image in enumerate(images):
                        images[img_idx] = image.resize((args.image_size, args.image_size), resample=Resampling.BICUBIC)
                inputs = {
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": images}
                }

            batch_inputs.append(inputs)
            batch_metadata.append(data_dict)

        yield batch_inputs, batch_metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    reformat_model_id = "/".join(args.model_id.strip("/").split("/")[-2:]).replace('/', '--')
    if args.image_size is not None:
        reformat_model_id = reformat_model_id + f"_{args.image_size}"
    result_save_path = join(PROJECT_ROOT, "results", args.subset, reformat_model_id, "finished.jsonl")
    os.makedirs(dirname(result_save_path), exist_ok=True)

    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f.readlines():
                data_dict = json.loads(line.strip())
                finish_ids.append(data_dict["id"])

    if len(finish_ids) > 0:
        print(f"Resume from {len(finish_ids)} finished samples")
    else:
        print(f"Start from scratch")

    subset_json = join(f"{PROJECT_ROOT}/data/test", f"{args.subset}.json")
    print("Reading data from {}".format(subset_json))
    with open(subset_json, "r") as f:
        ds = json.load(f)
        ds = [dict(id=f"{args.subset}_{data_idx}", **data_dict) for data_idx, data_dict in enumerate(ds)]
        ds = [data_dict for data_dict in ds if data_dict["id"] not in finish_ids]

    if len(ds) > 0:
        model_config = build_model_config(model_id=args.model_id, max_model_len=args.max_model_len, max_tokens=args.max_tokens)
        engine_args = model_config.default_engine_args
        if "model" not in engine_args:
            engine_args["model"] = args.model_id

        if args.subset in [
            "basic_change_choice_qa", "basic_change_report_qa", "change_speed_choice_qa", "change_speed_report_qa",
            "dense_temporal_caption", "regional_caption",
        ]:
            if "llava" in args.model_id.lower():
                engine_args["limit_mm_per_prompt"] = dict(video=1)
            else:
                engine_args["limit_mm_per_prompt"] = dict(image=10)
        elif args.subset in ["eco_assessment"]:
            engine_args["limit_mm_per_prompt"] = dict(image=1)
        else:
            raise ValueError(f"Unknown subset {args.subset}")

        if args.tensor_parallel_size is not None:
            engine_args["tensor_parallel_size"] = args.tensor_parallel_size
        engine_args = asdict(EngineArgs(**engine_args))
        print(f"Engine arguments: {engine_args}")
        vlm_model = LLM(**engine_args)

        sampling_params = SamplingParams(max_tokens=args.max_tokens)
        try:
            default_generation_config = GenerationConfig.from_pretrained(args.model_id).to_diff_dict()
            for key in default_generation_config:
                if hasattr(sampling_params, key):
                    setattr(sampling_params, key, default_generation_config[key])
            sampling_params.update_from_generation_config(default_generation_config)
        except OSError:
            for key in engine_args["override_generation_config"]:
                if hasattr(sampling_params, key):
                    setattr(sampling_params, key, engine_args["override_generation_config"][key])

        sampling_params.temperature = max(sampling_params.temperature, 0.01)
        print(f"Sampling params: {sampling_params}")

        with open(result_save_path, "w" if len(finish_ids) == 0 else "a") as f:
            progress_bar = tqdm(total=len(ds))
            for batch_inputs, batch_metadata in create_batch_inputs(ds, model_config, args):
                outputs = vlm_model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

                for idx, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    assert int(output.request_id) % args.batch_size == idx, (output.request_id, idx)

                    dump_dict = {
                        "id": batch_metadata[idx]["id"],
                        "response": generated_text,
                        "request": batch_metadata[idx],
                    }
                    f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
                    f.flush()
                    progress_bar.update(1)

            progress_bar.close()

    with open(result_save_path) as f:
        finished_data = [json.loads(line.strip()) for line in f.readlines()]

    with open(result_save_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(finished_data, f, indent=4, ensure_ascii=False)
