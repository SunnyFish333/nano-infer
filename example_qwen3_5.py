import argparse
import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 text, image and local video generation")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3.5-0.8B"))
    parser.add_argument("--mode", choices=("text", "image", "video"), default="text")
    parser.add_argument("--image", action="append", help="Local image path; may be repeated")
    parser.add_argument("--video", help="Local video path")
    parser.add_argument("--question", default="请简要介绍你自己或描述输入内容。")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.mode == "image" and not args.image:
        parser.error("--mode image requires --image")
    if args.mode == "video" and not args.video:
        parser.error("--mode video requires --video")
    path = args.model
    tokenizer = AutoTokenizer.from_pretrained(path)

    llm = LLM(
        path,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=2048,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        num_gdn_slots=2,
        gpu_memory_utilization=0.7,
        max_image_pixels=65536,
    )

    question = args.question
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if args.mode != "text":
        content = ([{"type":"image", "image":image} for image in args.image]
                   if args.mode == "image" else [{"type":"video", "video":args.video}])
        content.append({"type":"text", "text":question})
        prompt = {"messages":[{"role":"user", "content":content}]}

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.6, max_tokens=args.max_tokens),
    )

    print(f"\nQuestion: {question}")
    print(f"Completion: {outputs[0]['text']!r}")


if __name__ == "__main__":
    main()
