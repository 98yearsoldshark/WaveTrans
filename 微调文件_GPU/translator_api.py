from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import json
import uvicorn

app = FastAPI()

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-July-hf', cache_dir='./')

# 定义模型路径
path = './IEITYuan/Yuan2-2B-Mars-hf'
lora_path = './output/Yuan2.0-2B_lora_bf16/checkpoint-135'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10
# torch_dtype = torch.float16 # P100

# 定义一个函数，用于获取模型和tokenizer
@app.on_event("startup")
async def load_model():
    global tokenizer, model
    print("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>',
                          '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>',
                          '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)

    print("Creating model...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()
    model = PeftModel.from_pretrained(model, model_id=lora_path)

# 定义输入模型
class TranslationRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]

# 定义翻译逻辑
@app.post("/v1/chat/completions")
async def translate(request: TranslationRequest):
    if request.model != "Yuan2.0":
        raise HTTPException(status_code=400, detail="Unsupported model")

    template = """
# 角色
你是一个资深的 AI 翻译助理，具备卓越的语言能力，能够精准、流畅且地道地翻译英语文本。

# 技能
1. 当用户提供英语，你输出中文

# 回复示例
输入： Previously we obtained recombinant soluble human rhIFN-λ1 from Pichia pastoris.
输出: 此前，我们已经从毕赤酵母表达获得了可溶性重组人干扰素-λ1。

# 限制:
- 仅专注于翻译相关的任务，坚决拒绝回答与翻译无关的任何问题。
- 译文要清晰易懂、自然流畅，避免使用生僻或过于复杂的中文表述。

query
"""

    query = request.messages[-1]["content"]
    prompt = template.replace('query', query).strip() + "<sep>"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=1024)
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '')

    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "Yuan2.0",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(inputs[0]),
            "completion_tokens": len(outputs[0]) - len(inputs[0]),
            "total_tokens": len(outputs[0])
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)