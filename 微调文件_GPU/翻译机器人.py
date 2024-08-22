# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from peft import PeftModel
import json
import pandas as pd

# 创建一个标题和一个副标题
st.title("💬 Yuan2.0 翻译助手")

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
@st.cache_resource
def get_model():
    print("Creat tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

    print("Creat model...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    return tokenizer, model

# 加载model和tokenizer
tokenizer, model = get_model()

template = '''
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

'''




# 初次运行时，session_state中没有"messages"，需要创建一个空列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 每次对话时，都需要遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if query := st.chat_input():
    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": query})

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(query)

    # 调用模型
    prompt = template.replace('query', query).strip() + "<sep>" # 拼接对话历史
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=1024) # 设置解码方式和最大生成长度
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '')

    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)


