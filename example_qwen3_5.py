import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3.5-0.8B")
    tokenizer = AutoTokenizer.from_pretrained(path)

    llm = LLM(
        path,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        num_gdn_slots=2,
        gpu_memory_utilization=0.7,
    )

    question = "用一句话介绍你自己。"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.6, max_tokens=64),
    )

    print(f"\nQuestion: {question}")
    print(f"Completion: {outputs[0]['text']!r}")


if __name__ == "__main__":
    main()
