import argparse
import asyncio
import copy
import glob
import json
import logging
import math
import os
import random
import string
from enum import Enum
from os.path import dirname, basename
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable, HRuleStyle
from scipy import stats

try:
    from openai import AsyncAzureOpenAI
    from tqdm.asyncio import tqdm_asyncio

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Global configurations
CURRENT_DIR = dirname(__file__)
PROJECT_ROOT = dirname(dirname(dirname(dirname(__file__))))
DATA_ROOT = os.environ.get('DATA_ROOT', './data')

# SAM preprocessing constants
sam_pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
sam_pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

# Data configuration
data_num_libs = {
    "basic_change_choice_qa": 2355,
    "change_speed_choice_qa": 1499,
    "eco_assessment": 14071
}

SEMANTIC_NAME_TO_LABEL = {
    'vegetation': 1,
    'non vegetated surface': 2,
    'water': 3,
    'building': 4,
    'playground': 5
}

# Evaluation prompt templates
BASIC_CHANGE_REPORT_PROMPT = """You are an advanced intelligent chatbot specialized in evaluating remote sensing image change detection results for basic land cover changes. 
Your primary task is to meticulously compare the predicted change description with the ground truth description and assess their accuracy. 

To accomplish this, you will evaluate the results across three key dimensions:

1. Land Cover Type Identification Accuracy:
Evaluate how accurately the predicted result identifies the main types of land cover changes, including:
* Correctly identifying the initial and final land cover types involved in the change
* Correctly describing the direction of transitions between different land cover types
* Capturing all major land cover transition patterns mentioned in the ground truth

2. Time Period Accuracy: 
Assess how accurately the predicted result captures the correct time period mentioned in the ground truth result. This measures whether the overall start year and end year are correctly identified.

3. Change Quantification Accuracy:
Assess how well the predicted result quantifies the magnitude of changes, including:
* Accuracy of the reported percentage changes
* Alignment between predicted quantitative changes and ground truth transition percentages
* Completeness of quantitative information compared to ground truth

Please assign a score for each of these three dimensions, using an integer from 0 to 5, where 5 indicates perfect performance and 0 signifies poor performance. Accompany your assessments with brief explanations to clarify your scoring rationale.

<predicted_result>
{predicted_result}
</predicted_result>

<ground_truth_result>
{ground_truth_result}
</ground_truth_result>

### OUTPUT FORMAT(EXAMPLE)
<land_cover_type_identification_accuracy>4</land_cover_type_identification_accuracy>
<land_cover_type_identification_explanation>
The prediction correctly identifies the main land cover types (vegetation, non-vegetated surfaces, built-up areas) and their transition patterns. It accurately captures the conversion from vegetation to built-up areas and non-vegetated surfaces. However, it misses some minor transition patterns mentioned in the ground truth.
</land_cover_type_identification_explanation>

<time_period_accuracy>5</time_period_accuracy>
<time_period_accuracy_explanation>
The predicted time period (2006–2021) exactly matches the ground truth time period (15-year period from 2006 to 2021).
</time_period_accuracy_explanation>

<change_quantification_accuracy>3</change_quantification_accuracy>
<change_quantification_explanation>
The prediction provides reasonable estimates of overall changes (~20% built-up increase, ~20% vegetation decrease) that roughly align with the ground truth's transition percentages. However, it focuses on net changes rather than specific transition flows and lacks the detailed quantification found in the ground truth.
</change_quantification_explanation>"""

CHANGE_SPEED_REPORT_PROMPT = """You are an advanced intelligent chatbot specialized in evaluating remote sensing image change detection results.
Your primary task is to meticulously compare the predicted change detection results with the ground truth results and assess their accuracy. To accomplish this, you will evaluate the results across three key dimensions:

1. Change Rate Precision: 
Evaluate how accurately the percentage value in the predicted result matches the actual change rate described in the ground truth result. This measures whether the predicted rate is correct (without considering other aspects).

2. Time Period Accuracy: 
Assess how accurately the predicted result captures the correct time period mentioned in the ground truth result. This measures whether the start year and end year are correctly identified.

3. Change Pattern Accuracy:
Evaluate how accurately the predicted result describes the specific pattern and nature of changes, including:
* Correctly identifying the type of change (e.g., expansion, decrease, conversion)
* Accurately describing the spatial distribution or pattern of change
* Properly capturing the change dynamics mentioned in the ground truth

Please assign a score for each of these three dimensions, using an integer from 0 to 5, where 5 indicates perfect performance and 0 signifies poor performance. Accompany your assessments with brief explanations to clarify your scoring rationale.

<predicted_result>
{predicted_result}
</predicted_result>

<ground_truth_result>
{ground_truth_result}
</ground_truth_result>

### OUTPUT FORMAT(EXAMPLE)
<change_rate_precision>4</change_rate_precision>
<change_rate_precision_explanation>
The predicted expansion rate of 1.15% is very close to the ground truth rate of 1.09%, with only a small deviation of 0.06%.
</change_rate_precision_explanation>

<time_period_accuracy>5</time_period_accuracy>
<time_period_accuracy_explanation>
The predicted time period (2006 to 2009) exactly matches the ground truth time period.
</time_period_accuracy_explanation>

<change_pattern_accuracy>3</change_pattern_accuracy>
<change_pattern_accuracy_explanation>
The prediction correctly identifies urban expansion as the main change type. However, it provides limited information about the spatial distribution of changes and doesn't fully capture the detailed change dynamics described in the ground truth.
</change_pattern_accuracy_explanation>"""

DENSE_TEMPORAL_CAPTION_PROMPT = """You are an advanced intelligent chatbot specialized in evaluating dense temporal captioning for remote sensing image time series.

Your primary task is to meticulously compare the predicted dense temporal caption with the ground truth caption and assess their accuracy. To accomplish this, you will evaluate the captions across three key dimensions:

1. Temporal Coverage: 
Evaluate how well the predicted caption captures all significant time points and periods of change throughout the entire temporal range. This includes identifying key temporal milestones, maintaining a logical sequence of events, providing appropriate temporal context, and capturing the complete temporal narrative without significant gaps.

2. Spatial Accuracy: 
Assess how accurately and comprehensively the predicted caption describes the spatial aspects of changes. This includes correctly identifying all regions where significant changes occurred, accurately describing the spatial relationships between different areas, using precise spatial referencing, and ensuring comprehensive coverage of all spatially relevant changes in the image.

3. Process Fidelity:
Evaluate how accurately and completely the predicted caption describes the nature and processes of change. This includes correctly identifying initial and final land cover/use states, describing intermediate stages of development, capturing the complexity of multiple change processes, and accurately describing the specific features that changed.

Please assign a score for each of these three dimensions, using an integer from 0 to 5, where 5 indicates perfect performance and 0 signifies poor performance. Accompany your assessments with brief explanations to clarify your scoring rationale.

<predicted_caption>
{predicted_caption}
</predicted_caption>

<ground_truth_caption>
{ground_truth_caption}
</ground_truth_caption>

### OUTPUT FORMAT(EXAMPLE)
<temporal_coverage>4</temporal_coverage>
<temporal_coverage_explanation>
The predicted caption successfully identifies the overall 15-year timeframe and captures key temporal milestones including the development initiation, 2011 halt, 2016 resumption, and 2018-2021 development phase. The chronological order is logical and comprehensive, though it could provide more specific details about the timing of the initial development phase. The caption maintains a clear temporal progression that allows readers to understand the sequence of changes over time.
</temporal_coverage_explanation>

<spatial_accuracy>5</spatial_accuracy>
<spatial_accuracy_explanation>
The caption precisely identifies all significant spatial regions where changes occurred including the top-right, left region, bottom-center, and bottom-right areas with accurate spatial referencing. The spatial coverage is comprehensive, capturing all relevant areas of change mentioned in the ground truth caption. The spatial descriptions effectively communicate the distribution of changes across the image and establish clear relative positions of different development activities.
</spatial_accuracy_explanation>

<process_fidelity>4</process_fidelity>
<process_fidelity_explanation>
The caption accurately describes the transformation from natural and agricultural lands to residential areas with impervious surfaces, buildings, and roads. It captures the phased development process and specifically mentions the office building construction and new water body. The description provides good detail about the conversion processes and infrastructure development. It could provide slightly more detail about the specific characteristics of the residential development in the left region to achieve perfect fidelity.
</process_fidelity_explanation>"""

REGIONAL_CAPTION_PROMPT = """You are an advanced intelligent chatbot specialized in evaluating regional captions for remote sensing images.

Your primary task is to meticulously compare the predicted regional caption with the ground truth regional caption and assess their accuracy. To accomplish this, you will evaluate the captions across four key dimensions:

1. Temporal Coverage: 
Evaluate how well the predicted caption captures all significant time points and periods of change within the specified region. This includes identifying key temporal milestones, maintaining a logical sequence of events, and capturing the complete temporal narrative without significant gaps.

2. Spatial Accuracy: 
Assess how accurately the predicted caption describes the spatial aspects of changes within the region. This includes correctly identifying sub-areas where changes occurred and accurately describing the spatial relationships between different features.

3. Process Fidelity:
Evaluate how accurately the predicted caption describes the nature and processes of change. This includes correctly identifying initial and final land cover/use states, describing intermediate stages of development, and accurately describing the specific features that changed.

4. Region Containment:
Assess whether the caption strictly focuses on changes within the specified region box only, without including irrelevant information about areas outside the designated region. This measures the caption's precision in adhering to the spatial boundaries defined by the region box.

Please assign a score for each of these four dimensions, using an integer from 0 to 5, where 5 indicates perfect performance and 0 signifies poor performance. Accompany your assessments with brief explanations to clarify your scoring rationale.

<predicted_caption>
{predicted_caption}
</predicted_caption>

<ground_truth_caption>
{ground_truth_caption}
</ground_truth_caption>

### OUTPUT FORMAT(EXAMPLE)
<temporal_coverage>4</temporal_coverage>
<temporal_coverage_explanation>
The predicted caption effectively identifies the three key time periods (2005-2019, 2019-2020, 2020-2024) when changes occurred in the region. The temporal progression is logical and provides a clear understanding of the sequence of development events, though it could provide slightly more detail about specific milestones within the 2020-2024 period.
</temporal_coverage_explanation>

<spatial_accuracy>5</spatial_accuracy>
<spatial_accuracy_explanation>
The caption accurately identifies the specific sub-areas within the region where changes occurred, correctly referencing the "top left farmland" and "bottom left farmland." The spatial relationships are clearly described, providing an excellent understanding of where changes took place within the specified region.
</spatial_accuracy_explanation>

<process_fidelity>4</process_fidelity>
<process_fidelity_explanation>
The caption accurately describes the conversion of farmland to residential areas and the development process including ground hardening. It effectively captures the progression from agricultural use to development and construction. The description of the final state of the bottom left farmland could be more detailed to achieve perfect fidelity.
</process_fidelity_explanation>

<region_containment>5</region_containment>
<region_containment_explanation>
The caption focuses exclusively on the changes occurring within the specified region box, with no references to areas outside the designated boundaries. All described features and changes are properly contained within the region of interest, demonstrating perfect adherence to the spatial constraints.
</region_containment_explanation>"""


# Utility classes
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)
        return fmtstr.format(**self.__dict__)


# Utility functions
def dict_to_table(data_dict, headers=None):
    """Convert dictionary to prettytable"""
    table = PrettyTable()
    table.hrules = HRuleStyle.ALL

    max_list_length = max(len(value) for value in data_dict.values()) if data_dict else 0

    if headers is None:
        headers = ["index"] + [f"value{i + 1}" for i in range(max_list_length)]

    table.field_names = headers

    for key, value_list in data_dict.items():
        row = [key] + value_list + [""] * (max_list_length - len(value_list))
        table.add_row(row)

    return table


def sam_preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - sam_pixel_mean) / sam_pixel_std
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    """Calculate intersection and union on GPU"""
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def accuracy(pred, label, ignore_zero=False):
    """Calculate accuracy"""
    valid = (label >= 0)
    if ignore_zero:
        valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def fast_hist(a, b, n):
    """Fast histogram calculation"""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class):
    """Get histogram for evaluation"""
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    """Calculate kappa score"""
    if hist.sum() == 0:
        return 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            return 0
        else:
            return (po - pe) / (1 - pe)


# Async processing functions
async def process_item(client, data_item, model_name) -> Dict:
    """Process a single item with retry logic"""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package not available")

    data_item = copy.deepcopy(data_item)
    retry_num, break_flag = 0, False
    response = None

    while not break_flag:
        try:
            completion_outputs = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=data_item["request"]["messages"],
                    max_tokens=data_item["request"]["max_tokens"],
                ),
                timeout=300
            )
            try:
                response = completion_outputs.choices[0].message.content
            except AttributeError:
                response = None
            break_flag = True

        except (asyncio.TimeoutError, ConnectionError) as e:
            logging.error(f"Network exception occurred: {e}")

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in
                   ["rate limit", "too many requests", "server error", "connection error"]):
                logging.error(f"Server error occurred: {e}")
            else:
                logging.error(f"Other error occurred: {e}. skip it")
                break_flag = True

        if not break_flag:
            retry_num += 1
            if retry_num > 10:
                logging.info("Max retry reached, skipping")
                break_flag = True
            else:
                wait_time = min(2 ** retry_num, 10)
                logging.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    return dict(
        id=data_item["custom_id"],
        response=response,
    )


async def process_with_semaphore(client, data_item, model_name, semaphore):
    """Process item with semaphore control"""
    async with semaphore:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return await process_item(client, data_item, model_name)


async def async_batch_process(
        batched_requests: List[Dict],
        response_save_path: str,
        model_name: str = "gpt-4.1-mini",
        n_thread: int = 64,
        overwrite: bool = False,
):
    """Main async processing function"""
    if not HAS_OPENAI:
        raise ImportError("OpenAI package not available")

    n_thread = min(n_thread, len(batched_requests))

    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_BASE"),
        api_key=os.environ.get("AZURE_OPENAI_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )

    try:
        finished_ids = []
        if not overwrite and os.path.exists(response_save_path):
            with open(response_save_path, "r") as f:
                for line in f.readlines():
                    data_dict = json.loads(line.strip())
                    if data_dict["response"] is not None:
                        finished_ids.append(data_dict["id"])

        batched_requests = [data_dict for data_dict in batched_requests if data_dict["custom_id"] not in finished_ids]

        tasks = []
        semaphore = asyncio.Semaphore(n_thread)
        for data_item in batched_requests:
            tasks.append(
                process_with_semaphore(
                    client=client,
                    data_item=data_item,
                    model_name=model_name,
                    semaphore=semaphore
                )
            )

        with open(response_save_path, "a") as f:
            for task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                parsed_response = await task
                f.write(json.dumps(parsed_response) + "\n")
                f.flush()

    finally:
        if client and hasattr(client, 'close'):
            await client.close()

        for task in asyncio.all_tasks(asyncio.get_running_loop()):
            if task is not asyncio.current_task() and not task.done():
                task.cancel()


# Evaluation classes
class BaseEvaluator:
    """Base evaluator class"""

    def __init__(self):
        pass

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class MambaSCDEvaluator(BaseEvaluator):
    """Semantic Change Detection Evaluator"""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.reset()

    def update(self, label_trues, label_preds, index_name):
        if not isinstance(label_trues, list):
            label_trues = [label_trues]
        if not isinstance(label_preds, list):
            label_preds = [label_preds]
        if not isinstance(index_name, list):
            index_name = [index_name]

        for i, (lt, lp) in enumerate(zip(label_trues, label_preds)):
            if isinstance(lt, torch.Tensor):
                lt = lt.cpu().numpy()
            if isinstance(lp, torch.Tensor):
                lp = lp.cpu().numpy()

            acc, _ = accuracy(pred=lp, label=lt)
            self.acc_meter.update(acc)

            index_hist = get_hist(image=lp, label=lt, num_class=self.n_classes)
            self.hist += index_hist
            self.index_results[index_name[i]] = self.compute(hist=index_hist)[0]

    def compute(self, hist=None) -> (Dict, Dict):
        if hist is None:
            hist = self.hist

        hist_fg = hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

        pixel_sum = hist.sum()
        change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - hist.sum(0)[0].sum()
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / (change_pred_sum + 1e-10)
        SC_Recall = SC_TP / (change_label_sum + 1e-10)
        Fscd = stats.hmean([SC_Precision, SC_Recall]) if SC_Precision > 0 and SC_Recall > 0 else 0

        results_dict = {
            "kappa_n0": kappa_n0,
            "Fscd": Fscd,
            "IoU_mean": IoU_mean,
            "IoU_fg": IoU_fg,
            "Sek": Sek,
            "Acc": self.acc_meter.avg,
            "Precision": SC_Precision,
            "Recall": SC_Recall,
        }
        return results_dict, self.index_results

    def reset(self):
        self.hist = np.zeros((self.n_classes, self.n_classes))
        self.acc_meter = AverageMeter()
        self.index_results = {}


# Main processing functions
def process_choice_qa_results(model_list, args):
    """Process choice QA results"""
    result_libs = {}

    for model in model_list:
        model = model.replace("/", "--")
        for task in ["basic_change_choice_qa", "change_speed_choice_qa", "eco_assessment"]:
            result_json_path_list = glob.glob(f"{PROJECT_ROOT}/results/multi_temp/{task}/{model}/*.json")

            for result_json_path in result_json_path_list:
                json_marker = result_json_path.split('.')[-2]
                if len(json_marker.split('-')) == 2:
                    continue

                prompt_ver = basename(result_json_path).replace(".json", "")
                if prompt_ver == "finished":
                    prompt_ver = "v0"

                if args.v0_only and prompt_ver != "v0":
                    continue

                result_name = f"{model}---{prompt_ver}"
                if result_name not in result_libs:
                    result_libs[result_name] = []

                with open(result_json_path) as f:
                    result_data = json.load(f)

                if task in ["basic_change_choice_qa", "change_speed_choice_qa"]:
                    splits = ["Single choice", "Multiple choice"]
                else:
                    splits = ["Single choice"]

                for split in splits:
                    acc_list = []
                    for doc in result_data:
                        response = doc["response"]
                        task_type = split

                        if "request" in doc:
                            metadata = doc["request"]
                        elif "metadata" in doc:
                            metadata = doc["metadata"]
                        else:
                            raise RuntimeError("Missing metadata")

                        if task in ["basic_change_choice_qa", "change_speed_choice_qa"] and metadata[
                            "task_type"] != split:
                            continue

                        gt = metadata["ground_truth_option"].strip(".")
                        pred = response.strip(".")

                        if task_type == "Single choice":
                            pred = pred.strip().lower()
                            gt = gt.strip().lower()
                            acc_list.append(1 if pred == gt else 0)

                        elif task_type == "Multiple choice":
                            pred = pred.strip().lower().split(",")
                            pred = [item.strip().strip(".") for item in pred]
                            pred = sorted([item for item in pred if item in string.ascii_lowercase])

                            gt = gt.strip().lower().split(",")
                            gt = sorted([item.strip().strip(".") for item in gt])

                            if len(pred) == len(gt) and all([pred_i == gt_i for pred_i, gt_i in zip(pred, gt)]):
                                acc_list.append(1)
                            else:
                                acc_list.append(0)

                    if len(acc_list) > 0:
                        accuracy = sum(acc_list) / len(acc_list)
                        result_libs[result_name].append(accuracy)
                    else:
                        result_libs[result_name].append(np.nan)

    return result_libs


def process_report_qa_results(model_list, args):
    """Process report QA results"""
    result_libs = {}

    for model in model_list:
        model = model.replace("/", "--")

        for task in ["basic_change_report_qa", "change_speed_report_qa", "regional_caption", "dense_temporal_caption"]:
            if task == "basic_change_report_qa":
                score_types = ["time_period_accuracy", "land_cover_type_identification_accuracy",
                               "change_quantification_accuracy"]
            elif task == "change_speed_report_qa":
                score_types = ["change_rate_precision", "time_period_accuracy", "change_pattern_accuracy"]
            elif task == "regional_caption":
                score_types = ["temporal_coverage", "spatial_accuracy", "process_fidelity", "region_containment"]
            elif task == "dense_temporal_caption":
                score_types = ["temporal_coverage", "spatial_accuracy", "process_fidelity"]
            else:
                raise NotImplementedError

            result_jsonl_path_list = glob.glob(f"{PROJECT_ROOT}/results/multi_temp/{task}/{model}/*.gpt-4.1-mini.jsonl")
            for result_jsonl_path in result_jsonl_path_list:
                prompt_ver = basename(result_jsonl_path).replace(".gpt-4.1-mini.jsonl", "")
                if prompt_ver == "finished":
                    prompt_ver = "v0"

                if args.v0_only and prompt_ver != "v0":
                    continue

                result_name = f"{model}---{prompt_ver}"
                if result_name not in result_libs:
                    result_libs[result_name] = []

                with open(result_jsonl_path) as f:
                    result_data = [json.loads(line.strip()) for line in f.readlines()]

                score_dict = {}
                for doc in result_data:
                    response = doc["response"]
                    if response is None:
                        print(result_name, response)
                        continue

                    for key in score_types:
                        prefix_tag = f"<{key}>"
                        suffix_tag = f"</{key}>"
                        key_score = None
                        if prefix_tag in response and suffix_tag in response:
                            key_score = response.split(prefix_tag)[1].split(suffix_tag)[0].strip()
                        elif prefix_tag in response:
                            key_score = response.split(prefix_tag)[1][0].strip()
                        elif suffix_tag in response:
                            key_score = response.split(suffix_tag)[0][-1].strip()
                        try:
                            key_score = int(key_score)
                        except:
                            key_score = 0

                        if key not in score_dict:
                            score_dict[key] = []
                        score_dict[key].append(key_score)

                key_avg_scores = {}
                for key in score_types:
                    try:
                        key_avg_scores[key] = sum(score_dict[key]) / len(score_dict[key])
                    except:
                        print(result_name, key, score_dict[key])
                        key_avg_scores[key] = None

                try:
                    total_avg_score = sum(key_avg_scores.values()) / len(key_avg_scores)
                except:
                    total_avg_score = None

                result_libs[result_name].append("error" if total_avg_score is None else f"{total_avg_score:.2f}")
                for key in score_types:
                    result_libs[result_name].append(
                        "error" if key_avg_scores[key] is None else f"{key_avg_scores[key]:.2f}"
                    )

                if task != "dense_temporal_caption":
                    result_libs[result_name].append("")

    return result_libs


def main():
    """Main function to orchestrate different evaluation tasks"""
    parser = argparse.ArgumentParser(description="Multi-task evaluation script for remote sensing")
    parser.add_argument('--task', type=str, choices=['choice_qa', 'report_qa', 'segmentation', 'eval_batch'],
                        default='choice_qa', help='Task to run')
    parser.add_argument('--api_models', action='store_true', help='Use API models')
    parser.add_argument('--v0_only', action='store_true', help='Only process v0 prompts')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--data_type', type=str, default='Optical', help='Data type for evaluation')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Define model list (remove sensitive information)
    model_list = [
        # model names to be evaluate
    ]

    if args.task == 'choice_qa':
        result_libs = process_choice_qa_results(model_list, args)

        total_score_dict = {}
        for result_name in result_libs:
            model, prompt_ver = result_name.split("---")
            total_score_dict[result_name] = [model, prompt_ver]
            for acc_value in result_libs[result_name]:
                if isinstance(acc_value, float):
                    total_score_dict[result_name].append(f"{acc_value:.1%}")
                else:
                    total_score_dict[result_name].append(acc_value)

        custom_headers = ["index", "method", "prompt_ver", "BCA (single)", "BCA (multi)", "CSE (single)", "CSE (multi)",
                          "EA"]
        total_score_dict = {idx: value for idx, value in enumerate(total_score_dict.values())}
        table = dict_to_table(total_score_dict, custom_headers)
        print(table)

        with open(f"{CURRENT_DIR}/accuracy_table.txt", "w") as f:
            f.write(str(table))

    elif args.task == 'report_qa':
        result_libs = process_report_qa_results(model_list, args)

        total_score_dict = {}
        for result_name in result_libs:
            modal, prompt_ver = result_name.split("---")
            total_score_dict[result_name] = [modal, prompt_ver, *result_libs[result_name]]

        custom_headers = [
            "index", "modal", "prompt_ver",
            "BCA-AVG", "BCA-TPA", "BCA-LCTIA", "BCA-CQA",
            "|",
            "CSE-AVG", "CSE-CRP", "CSE-TPA", "CSE-CPA",
            "||",
            "RegCap-AVG", "RegCap-TC", "RegCap-SA", "RegCap-PF", "RegCap-RC",
            "|||",
            "DTC-AVG", "DTC-TC", "DTC-SA", "DTC-PF",
        ]
        total_score_dict = {idx: value for idx, value in enumerate(total_score_dict.values())}
        table = dict_to_table(total_score_dict, custom_headers)
        print(table)

        with open(f"{CURRENT_DIR}/generation_table.txt", "w") as f:
            f.write(str(table))

    else:
        print(f"Task {args.task} not implemented in this version")


if __name__ == '__main__':
    main()
