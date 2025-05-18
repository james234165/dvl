import argparse
import asyncio
import glob
import json
import logging
import os
import random
import string
from os.path import dirname, basename, abspath

from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm_asyncio

from prettytable import PrettyTable, HRuleStyle

import copy
from enum import Enum
from typing import Union, List, Any, Dict

import math
import torch
import numpy as np

from scipy import stats


class AggregationMode(Enum):
    SKIP = 0  # skip aggregation
    MEAN = 1  # calculate mean value
    TOTAL = 2  # calculate total sum
    QUANTITY = 3  # count elements


class MetricAccumulator(object):

    def __init__(self, metric_name, display_format=":f", aggregate_mode=AggregationMode.MEAN):
        self.metric_name = metric_name
        self.display_format = display_format
        self.aggregate_mode = aggregate_mode
        self.initialize()

    def initialize(self):
        # Clear all stored values
        self.current_value = 0
        self.mean_value = 0
        self.accumulated_sum = 0
        self.sample_count = 0

    def record(self, new_value, sample_size=1):
        # Update current value and accumulate statistics
        self.current_value = new_value
        self.accumulated_sum += new_value * sample_size  # weighted sum
        self.sample_count += sample_size
        # Compute running average on-the-fly
        if self.sample_count > 0:
            self.mean_value = self.accumulated_sum / self.sample_count

    def __str__(self):
        # Custom string representation
        format_template = "{metric_name} {current_value" + self.display_format + "} ({mean_value" + self.display_format + "})"
        return format_template.format(**self.__dict__)

    def get_summary(self):
        # Generate summary based on aggregation mode
        output_format = ""
        if self.aggregate_mode is AggregationMode.SKIP:
            output_format = ""  # no summary
        elif self.aggregate_mode is AggregationMode.MEAN:
            output_format = "{metric_name} {mean_value:.3f}"
        elif self.aggregate_mode is AggregationMode.TOTAL:
            output_format = "{metric_name} {accumulated_sum:.3f}"
        elif self.aggregate_mode is AggregationMode.QUANTITY:
            output_format = "{metric_name} {sample_count:.3f}"
        else:
            raise ValueError("Unsupported aggregation mode: %r" % self.aggregate_mode)

        return output_format.format(**self.__dict__)


def compute_iou_metrics_cuda(predictions, ground_truth, num_categories, excluded_label=255):
    assert predictions.dim() in [1, 2, 3]
    assert predictions.shape == ground_truth.shape, f"Shape mismatch: {predictions.shape} vs {ground_truth.shape}"
    
    # Flatten tensors for processing
    pred_flattened = predictions.view(-1)
    gt_flattened = ground_truth.view(-1)
    
    # Apply exclusion mask - ignore pixels with excluded_label
    pred_flattened[gt_flattened == excluded_label] = excluded_label
    
    # Find matching predictions (intersection calculation)
    matching_pixels = pred_flattened[pred_flattened == gt_flattened]
    
    # Generate histograms for each category
    intersection_hist = torch.histc(matching_pixels, bins=num_categories, min=0, max=num_categories - 1)
    pred_hist = torch.histc(pred_flattened, bins=num_categories, min=0, max=num_categories - 1)  
    gt_hist = torch.histc(gt_flattened, bins=num_categories, min=0, max=num_categories - 1)
    
    # Calculate union: pred + gt - intersection
    union_hist = pred_hist + gt_hist - intersection_hist
    
    return intersection_hist, union_hist, gt_hist


def build_confusion_matrix_fast(pred_array, truth_array, class_count):
    valid_mask = (pred_array >= 0) & (pred_array < class_count)
    # Convert to linear indexing: truth * class_count + pred for 2D histogram
    linear_indices = class_count * pred_array[valid_mask].astype(int) + truth_array[valid_mask]
    return np.bincount(linear_indices, minlength=class_count ** 2).reshape(class_count, class_count)


def generate_histogram_matrix(predicted_img, true_label, category_num):
    confusion_matrix = np.zeros((category_num, category_num))
    # Flatten arrays and build histogram
    flat_pred = predicted_img.flatten()
    flat_truth = true_label.flatten()
    confusion_matrix += build_confusion_matrix_fast(flat_pred, flat_truth, category_num)
    return confusion_matrix


def compute_cohen_kappa(confusion_hist):
    total_samples = confusion_hist.sum()
    if total_samples == 0:
        observed_agreement = 0
        expected_agreement = 1
        cohen_kappa = 0
    else:
        # Calculate observed agreement (diagonal sum / total)
        observed_agreement = np.diag(confusion_hist).sum() / total_samples
        # Calculate expected agreement based on marginal distributions
        row_marginals = confusion_hist.sum(1)  # sum along columns
        col_marginals = confusion_hist.sum(0)  # sum along rows  
        expected_agreement = np.matmul(row_marginals, col_marginals.T) / (total_samples ** 2)
        
        # Compute Cohen's kappa coefficient
        if expected_agreement == 1:
            cohen_kappa = 0  # perfect expected agreement
        else:
            cohen_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    
    return cohen_kappa


def calculate_prediction_accuracy(predictions, ground_truth_labels, exclude_zeros=False):
    valid_pixels_mask = (ground_truth_labels >= 0)
    if exclude_zeros: 
        # Additional constraint: exclude zero labels
        valid_pixels_mask = (ground_truth_labels > 0)
    
    # Count correct predictions among valid pixels
    correct_predictions = (valid_pixels_mask * (predictions == ground_truth_labels)).sum()
    total_valid_pixels = valid_pixels_mask.sum()
    
    # Calculate accuracy with numerical stability
    accuracy_score = float(correct_predictions) / (total_valid_pixels + 1e-10)
    
    return accuracy_score, total_valid_pixels


def standardize_prediction_labels(
        prediction_data: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
        reference_data: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]]
) -> (List[np.ndarray], List[np.ndarray]):
    
    def normalize_tensor_arrays(input_arrays):
        processed_arrays = copy.deepcopy(input_arrays)
        
        # Convert PyTorch tensors to NumPy arrays
        if isinstance(processed_arrays, torch.Tensor):
            processed_arrays = processed_arrays.numpy()
            
        # Handle different input formats
        if not isinstance(processed_arrays, list):
            array_shape = processed_arrays.shape
            # Convert to list based on dimensionality
            if len(array_shape) in [1, 2]:
                processed_arrays = [processed_arrays]  # single array
            elif len(array_shape) == 3:
                # Split 3D array into list of 2D arrays
                processed_arrays = [array_slice for array_slice in processed_arrays]
            else:
                raise RuntimeError(f"Unsupported tensor shape: {array_shape}")

        # Standardize each array in the list
        for idx in range(len(processed_arrays)):
            # Ensure NumPy format
            if not isinstance(processed_arrays[idx], np.ndarray):
                processed_arrays[idx] = processed_arrays[idx].numpy()
            # Flatten and convert to integer type
            processed_arrays[idx] = processed_arrays[idx].reshape(-1).astype(int)
            
        return processed_arrays

    # Process both prediction and reference data
    processed_predictions = normalize_tensor_arrays(prediction_data)
    processed_references = normalize_tensor_arrays(reference_data)
    
    # Validate array lengths match
    for pred_array, ref_array in zip(processed_predictions, processed_references):
        if len(pred_array) != len(ref_array):
            raise RuntimeError(
                f"Array length mismatch detected! "
                f"Prediction array length: {len(pred_array)}, "
                f"Reference array length: {len(ref_array)}"
            )
    
    return processed_predictions, processed_references


class ChangeDetectionMetricsCalculator:
    def __init__(self, category_count: int):
        self.category_count = category_count
        self.initialize_metrics()

    def process_batch(
            self,
            reference_labels: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
            predicted_labels: Union[torch.Tensor, np.ndarray, List[torch.Tensor or np.ndarray]],
            sample_identifiers: Union[List[Any], Any]
    ):
        # Standardize input data format
        pred_arrays, ref_arrays = standardize_prediction_labels(predicted_labels, reference_labels)
        
        # Ensure sample_identifiers is a list
        if not isinstance(sample_identifiers, List):
            sample_identifiers = [sample_identifiers]
            
        # Process each sample pair
        for sample_idx, (ref_data, pred_data) in enumerate(zip(ref_arrays, pred_arrays)):
            # Calculate accuracy for this sample
            sample_accuracy, _ = calculate_prediction_accuracy(pred=pred_data, ground_truth_labels=ref_data)
            self.accuracy_tracker.record(sample_accuracy)

            # Generate confusion matrix for this sample
            sample_confusion = generate_histogram_matrix(predicted_img=pred_data, true_label=ref_data, category_num=self.category_count)
            self.global_confusion_matrix += sample_confusion
            # Store individual sample results
            self.sample_metrics[sample_identifiers[sample_idx]] = self.calculate_metrics(hist=sample_confusion)[0]

    def calculate_metrics(self, hist=None) -> (Dict, Dict):
        # Use global confusion matrix if none provided
        if hist is None:
            hist = self.global_confusion_matrix

        # Extract foreground classes (excluding background class 0)
        foreground_hist = hist[1:, 1:]
        
        # Create binary confusion matrix (background vs change)
        binary_confusion = np.zeros((2, 2))
        binary_confusion[0][0] = hist[0][0]  # true negative (no change)
        binary_confusion[0][1] = hist.sum(1)[0] - hist[0][0]  # false positive
        binary_confusion[1][0] = hist.sum(0)[0] - hist[0][0]  # false negative  
        binary_confusion[1][1] = foreground_hist.sum()  # true positive (change)
        
        # Compute kappa excluding background class
        hist_no_background = hist.copy()
        hist_no_background[0][0] = 0  # remove true negatives
        kappa_coefficient = compute_cohen_kappa(hist_no_background)
        
        # Calculate IoU for binary classification
        intersection_over_union = np.diag(binary_confusion) / (binary_confusion.sum(1) + binary_confusion.sum(0) - np.diag(binary_confusion))
        foreground_iou = intersection_over_union[1]  # IoU for change class
        mean_iou = (intersection_over_union[0] + intersection_over_union[1]) / 2
        
        # Compute Sek score (kappa * exp(IoU) / e)
        sek_score = (kappa_coefficient * math.exp(foreground_iou)) / math.e

        # Calculate change detection metrics
        total_pixels = hist.sum()
        predicted_change_pixels = total_pixels - hist.sum(1)[0].sum()  # exclude background predictions
        actual_change_pixels = total_pixels - hist.sum(0)[0].sum()  # exclude background truth
        change_proportion = actual_change_pixels / total_pixels
        
        # True positives for change detection
        change_true_positives = np.diag(hist[1:, 1:]).sum()
        
        # Precision and recall for change detection
        change_precision = change_true_positives / (predicted_change_pixels + 1e-10)
        change_recall = change_true_positives / (actual_change_pixels + 1e-10)  # Fixed: should use actual_change_pixels
        
        # F-score using harmonic mean
        f_score_change = stats.hmean([change_precision, change_recall])

        # Compile results dictionary
        metrics_results = {
            "kappa_n0": kappa_coefficient,
            "Fscd": f_score_change,
            "IoU_mean": mean_iou,
            "IoU_fg": foreground_iou,
            "Sek": sek_score,
            "Acc": self.accuracy_tracker.mean_value,
            "Precision": change_precision,
            "Recall": change_recall,
        }
        
        return metrics_results, self.sample_metrics

    def initialize_metrics(self):
        # Initialize confusion matrix and accuracy tracker
        self.global_confusion_matrix = np.zeros((self.category_count, self.category_count))
        self.accuracy_tracker = MetricAccumulator()  # Updated class name
        self.sample_metrics = {}
        self.synchronization_flag = False


def dict_to_table(data_dict, headers):
    table = PrettyTable()
    table.hrules = HRuleStyle.ALL

    max_list_length = max(len(value) for value in data_dict.values())

    table.field_names = headers

    for key, value_list in data_dict.items():
        row = [key] + value_list + [""] * (max_list_length - len(value_list))
        table.add_row(row)

    return table


CURRENT_DIR = dirname(__file__)
PROJECT_ROOT = dirname(dirname(dirname(dirname(abspath(__file__)))))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Data counts for validation
data_num_libs = {
    "basic_change_choice_qa": 2355,
    "change_speed_choice_qa": 1499,
    "eco_assessment": 14071
}

# Model lists
API_MODEL_LIST = [
    # the api models to be evaluated
]

LOCAL_MODEL_LIST = [
    # the local models to be evaluated
]

# Azure OpenAI configuration placeholders
AZURE_CONFIG = {
    "AZURE_OPENAI_API_VERSION": "YOUR_API_VERSION",
    "AZURE_OPENAI_BASE": "YOUR_AZURE_ENDPOINT",
    "AZURE_OPENAI_KEY": "YOUR_AZURE_KEY"
}

GPT_PROMPTS = {
    "basic_change_report_qa": """You are an advanced intelligent chatbot specialized in evaluating remote sensing image change detection results for basic land cover changes. 
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
</change_quantification_explanation>""",

    "change_speed_report_qa": """You are an advanced intelligent chatbot specialized in evaluating remote sensing image change detection results.
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
</change_pattern_accuracy_explanation>""",

    "regional_caption": """You are an advanced intelligent chatbot specialized in evaluating regional captions for remote sensing images.

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
</region_containment_explanation>""",

    "dense_temporal_caption": """You are an advanced intelligent chatbot specialized in evaluating dense temporal captioning for remote sensing image time series.

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
}


async def process_item(client, data_item, model_name) -> Dict:
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
            logger.error(f"Network exception occurred: {e}")

        except Exception as e:
            if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                    or "server error" in str(e).lower()
                    or "connection error" in str(e).lower()
            ):
                logger.error(f"Server error occurred for: {e}")
            else:
                logger.error(f"Other error occurred for: {e}. skip it")
                break_flag = True

        if not break_flag:
            retry_num = retry_num + 1
            if retry_num > 10:
                logger.info(f"Max Retry reached for, skip it")
                break_flag = True
            else:
                wait_time = min(2 ** retry_num, 10)
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    return dict(
        id=data_item["custom_id"],
        response=response,
    )


async def process_with_semaphore(client, data_item, model_name, semaphore):
    async with semaphore:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return await process_item(client, data_item, model_name)


async def run_gpt_eval(
        batched_requests: List[Dict],
        response_save_path: str,
        model_name: str = "gpt-4.1-mini",
        n_thread: int = 128,
        overwrite: bool = False,
):
    n_thread = min(n_thread, len(batched_requests))

    # Set Azure OpenAI environment variables
    os.environ.update(AZURE_CONFIG)

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

        # Clean up unfinished tasks
        for task in asyncio.all_tasks(asyncio.get_running_loop()):
            if task is not asyncio.current_task() and not task.done():
                task.cancel()


def score_accuracy_tasks(model_list, v0_only=False):
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

                if v0_only and prompt_ver != "v0":
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
                            raise RuntimeError

                        if task in ["basic_change_choice_qa", "change_speed_choice_qa"] and metadata[
                            "task_type"] != split:
                            continue

                        gt = metadata["ground_truth_option"].strip(".")
                        pred = response.strip(".")

                        if task_type == "Single choice":
                            pred = pred.strip().lower()
                            gt = gt.strip().lower()
                            if pred == gt:
                                acc_list.append(1)
                            else:
                                acc_list.append(0)

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

                        else:
                            raise ValueError("Unknown task type")

                    if len(acc_list) > 0:
                        accuracy = sum(acc_list) / len(acc_list)
                        result_libs[result_name].append(accuracy)
                    else:
                        result_libs[result_name].append(np.nan)

    # Format and print results
    total_score_dict = {}
    for result_name in result_libs:
        model, prompt_ver = result_name.split("---")
        total_score_dict[result_name] = [model, prompt_ver]
        for acc_value in result_libs[result_name]:
            if isinstance(acc_value, float):
                total_score_dict[result_name].append(f"{acc_value:.1%}")
            else:
                total_score_dict[result_name].append(acc_value)

    custom_headers = [
        "index", "method", "prompt_ver", "BCA (single)", "BCA (multi)", "CSE (single)", "CSE (multi)", "EA"
    ]
    total_score_dict = {idx: value for idx, value in enumerate(total_score_dict.values())}
    table = dict_to_table(total_score_dict, custom_headers)
    print("\n=== Accuracy Task Scores ===")
    print(table)

    with open(f"{CURRENT_DIR}/accuracy_table.txt", "w") as f:
        f.write(str(table))


def score_generation_tasks(model_list, v0_only=False):
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

                if v0_only and prompt_ver != "v0":
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

    # Format and print results
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
    print("\n=== Generation Task Scores ===")
    print(table)

    with open(f"{CURRENT_DIR}/generation_table.txt", "w") as f:
        f.write(str(table))


def run_gpt_evaluation(task: str, model_list: List[str], v0_only: bool = False,
                       debug: bool = False, overwrite: bool = False, n_thread: int = 128):
    if task not in GPT_PROMPTS:
        raise ValueError(f"Unknown task for GPT evaluation: {task}")

    prompt_template = GPT_PROMPTS[task]

    for model in model_list:
        model = model.replace("/", "--")
        print(f"Processing {model} for task {task}")

        prompt_versions = ["finished", "v0"] if v0_only else ["finished", "v0", "v1", "v2", "v3", "v4"]

        for prompt_ver in prompt_versions:
            result_json_path = f"{PROJECT_ROOT}/results/multi_temp/{task}/{model}/{prompt_ver}.json"
            if os.path.exists(result_json_path):
                with open(result_json_path) as f:
                    result_data = json.load(f)

                batched_requests = []
                for doc_idx, doc in enumerate(result_data):
                    pred = doc["response"]
                    if "request" in doc:
                        gt = doc["request"]["ground_truth"]
                    elif "metadata" in doc:
                        gt = doc["metadata"]["ground_truth"]
                    else:
                        raise RuntimeError

                    # Prepare the prompt based on task type
                    if task in ["basic_change_report_qa", "change_speed_report_qa"]:
                        content = prompt_template.format(predicted_result=pred, ground_truth_result=gt)
                    else:  # regional_caption, dense_temporal_caption
                        content = prompt_template.format(predicted_caption=pred, ground_truth_caption=gt)

                    request_dict = dict(
                        custom_id=doc["id"] if "id" in doc else doc_idx,
                        request=dict(
                            messages=[
                                {
                                    "role": "user",
                                    "content": content,
                                }
                            ],
                            max_tokens=4096
                        )
                    )
                    batched_requests.append(request_dict)

                response_save_path = result_json_path.replace(".json", ".gpt-4.1-mini.jsonl")
                asyncio.run(run_gpt_eval(
                    batched_requests=batched_requests,
                    response_save_path=response_save_path,
                    n_thread=1 if debug else n_thread,
                    overwrite=overwrite
                ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['score', 'gpt_eval'], required=True)
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--api_models', action='store_true')
    parser.add_argument('--v0_only', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--n_thread', type=int, default=128)
    args = parser.parse_args()

    # Get model list
    model_list = API_MODEL_LIST if args.api_models else LOCAL_MODEL_LIST

    if args.mode == 'score':
        # Run scoring logic
        if args.task == 'all' or args.task == 'acc':
            score_accuracy_tasks(model_list, args.v0_only)
        if args.task == 'all' or args.task == 'gen':
            score_generation_tasks(model_list, args.v0_only)

    elif args.mode == 'gpt_eval':
        # Run GPT evaluation
        gen_tasks = ["basic_change_report_qa", "change_speed_report_qa", "regional_caption", "dense_temporal_caption"]

        if args.task == 'all':
            tasks_to_run = gen_tasks
        elif args.task in gen_tasks:
            tasks_to_run = [args.task]
        else:
            raise ValueError(f"Unknown task: {args.task}. Available tasks: {', '.join(gen_tasks)}")

        for task in tasks_to_run:
            run_gpt_evaluation(
                task=task,
                model_list=model_list,
                v0_only=args.v0_only,
                debug=args.debug,
                overwrite=args.overwrite,
                n_thread=args.n_thread
            )


if __name__ == '__main__':
    main()
