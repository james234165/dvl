## DyanmicVL
Please run this code for benchmarking the DVL-Bench dataset. 

 
## For open-source models:
```
cd dvl
python dvl/pyscripts/run_vllm.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --subset basic_change_choice_qa
```

```
cd dvl
python dvl/pyscripts/run_vllm.py --model_id OpenGVLab/InternVL3-78B --subset change_speed_choice_qa
```

## For commercial models like GPT
We use Azure here, but it could be changed to other ways like the official OpenAI api:
```
# Please config your identity by AZURE_OPENAI_BASE, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_KEY, and AZURE_OPENAI_API_VERSION
cd dvl
python dvl/pyscripts/run_azure_openai.py --model_id gpt-4o --subset dense_temporal_caption
```