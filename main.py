from peft import PeftModel
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import pandas as pd
from unsloth import to_sharegpt, standardize_sharegpt, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TextStreamer
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from unsloth import apply_chat_template


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


def get_model(model_name):
    """
    https://huggingface.co/unsloth
    """

    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    _model = FastLanguageModel.get_peft_model(
        _model,
        r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj",
                "lm_head", "embed_tokens"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    return _model, _tokenizer


def get_dataset(data_file):
    df = pd.read_json(data_file)
    _dataset = Dataset.from_pandas(df)

    return _dataset


def format_dataset(_dataset):
    _dataset = to_sharegpt(
        _dataset,
        merged_prompt = "{instruction}{input}",
        output_column_name = "output",
        conversation_extension = 3, # Select more to handle longer conversations
    )
    _dataset = standardize_sharegpt(_dataset)

    return _dataset


def apply_template(_dataset, _tokenizer):
    chat_template = """<｜begin▁of▁sentence｜><｜User｜>{INPUT}<｜Assistant｜>{OUTPUT}<｜end▁of▁sentence｜><｜User｜>Explain more!<｜Assistant｜>"""
    _dataset = apply_chat_template(
    _dataset,
    tokenizer=_tokenizer,
    chat_template=chat_template,
    default_system_message="你需模仿《甄嬛传》中甄嬛的说话风格：用词典雅含蓄，善用隐喻，常引诗词，句式多为短句且带反问。"
    )

    return _dataset


def train(_model, _tokenizer, _dataset):
    trainer = SFTTrainer(
        model = _model,  # 需要微调的 LLM（大语言模型）
        tokenizer = _tokenizer,  # 对应的 tokenizer
        train_dataset = _dataset,  # 训练数据集
        dataset_text_field = "text",  # 训练文本所在列
        max_seq_length = max_seq_length,  # 最大序列长度
        dataset_num_proc = 2,  # 多进程数据加载，提高数据预处理速度
        packing = False,  # 是否启用 packing（合并短文本，减少 padding，提高效率）
        args = TrainingArguments(
            per_device_train_batch_size = 2,  # 每个 GPU 的 batch size
            gradient_accumulation_steps = 4,  # 梯度累积步数
            warmup_steps = 5,  # 预热步数，避免初始学习率过大
            # max_steps = 2,  # 限制训练步数为 60（用于快速测试）
            num_train_epochs = 10,  # 设为 1 可进行完整训练
            learning_rate = 2e-4,  # 学习率
            fp16 = not is_bfloat16_supported(),  # 如果不支持 bfloat16，则使用 float16
            bf16 = is_bfloat16_supported(),  # 如果支持 bfloat16，则启用 bfloat16（减少显存占用）
            logging_steps = 1,  # 每 1 步记录日志
            optim = "adamw_8bit",  # 使用 8-bit AdamW 优化器（减少显存占用）
            weight_decay = 0.01,  # 权重衰减，防止过拟合
            lr_scheduler_type = "linear",  # 线性学习率调度
            seed = 3407,  # 设置随机种子，保证可复现
            output_dir = "outputs",  # 模型保存路径
            report_to = "none",  # 关闭 WandB 日志（如果使用 WandB，可改为 `"wandb"`）
        )
    )
    trainer_stats = trainer.train()

    return trainer_stats


def inference(model, tokenizer):
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "user",      "content": "Continue the fibonacci sequence! Your input is 1, 1, 2, 3, 5, 8"},
        {"role": "assistant", "content": "The fibonacci sequence continues as 13, 21, 34, 55 and 89."},
        {"role": "user",      "content": "What is France's tallest tower called?"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = tokenizer.eos_token_id)


def save_model(model, tokenizer):
    model.save_pretrained("lora_model") # Local saving
    tokenizer.save_pretrained("lora_model")
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

    
def load_lora_model():
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    # 加载LoRA适配器并合并权重
    _model = PeftModel.from_pretrained(_model, "lora_model")
    _model = _model.merge_and_unload()  # 合并到基础模型
    

    return _model, _tokenizer


def load_lora_model_with_huggingface():
    _model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model",
        load_in_4bit = load_in_4bit,
    )
    _tokenizer = AutoTokenizer.from_pretrained("lora_model")

    return _model, _tokenizer


def model2gguf(model, tokenizer):
    # Save to 8bit Q8_0
    # model.push_to_hub_gguf("hf/model", tokenizer, token = "")
    model.save_pretrained_gguf("model", tokenizer,)

    # Save to 16bit GGUF
    # model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
    generate_ollama_model_file(tokenizer)
    # model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

    # Save to q4_k_m GGUF
    # model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    # model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

    # Save to multiple GGUF options - much faster if you want multiple!
    # model.push_to_hub_gguf(
    #     "hf/model", # Change hf to your username!
    #     tokenizer,
    #     quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
    #     token = "", # Get a token at https://huggingface.co/settings/tokens
    # )


def generate_ollama_model_file(tokenizer):
    with open("Modelfile", "w") as f:
        f.write(tokenizer._ollama_modelfile)


def load_ollama_model():
    """
    ollama create unsloth_model -f ./model/Modelfile
    """
    pass


def after_model2gguf():
    model, tokenizer = load_lora_model()
    dataset = get_dataset('./dataset/huanhuan.json')
    dataset = format_dataset(dataset)
    dataset = apply_template(dataset, tokenizer)
    model2gguf(model, tokenizer)


def main():
    model, tokenizer = get_model("DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit")
    dataset = get_dataset('./dataset/huanhuan.json')
    dataset = format_dataset(dataset)
    dataset = apply_template(dataset, tokenizer)
    train(model, tokenizer, dataset)
    save_model(model, tokenizer)
    model2gguf(model, tokenizer)


if __name__ == "__main__":
   main()
   

