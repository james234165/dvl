# Environment
```
conda create -n dvl python=3.10 -y && conda activate dvl
(dvl): pip install -e .
(dvl): pip install qwen-vl-utils[decord]==0.0.10
(dvl): pip install flash-attn==2.7.3 --no-build-isolation
```


# Run
Please use the following command to run the zeroshot inference of popular open-source models:
```
(dvl): cd dvl_release
(dvl): python dvl/pyscripts/run_vllm.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --subset basic_change_choice_qa
```
