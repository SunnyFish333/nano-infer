import argparse
import json
from pathlib import Path
import urllib.request
from benchmark.common.io import digest, write, append, now

SOURCES = (
    ("HuggingFaceH4/mt_bench_prompts", "e3a795c5e9a82ee40611c416b8a7786c73198991", "raw/question.jsonl", "questions.jsonl"),
    ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1/test-00000-of-00001.parquet", "wiki.parquet"),
)


def chat_ids(tokenizer, text):
    value = tokenizer.apply_chat_template([dict(role="user", content=text)], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    value = value if isinstance(value, list) else value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one chat")
        value = value[0]
    return value


def prefix_samples(corpus, prefix, count=64):
    """Distinct suffix first blocks guarantee exactly P complete shared blocks."""
    p = corpus[-2048:-2048+prefix]
    used, result = set(), []
    cursor = 8192
    while len(result) < count:
        suffix = corpus[cursor:cursor+2048-prefix]
        if len(suffix) != 2048-prefix:
            raise ValueError("Corpus exhausted while choosing unique suffix blocks")
        key = tuple(suffix[:256])
        if key not in used:
            used.add(key)
            result.append(dict(id=f"prefix{prefix}:{len(result)}", token_ids=p+suffix,
                               prefix_tokens=prefix, suffix_offset=cursor))
        cursor += 521
    return result, p+[corpus[-1]]


def main():
    parser = argparse.ArgumentParser(description="Download pinned small datasets and save deterministic token workloads")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="benchmark/datasets/v1")
    parser.add_argument("--resume", action="store_true", help="Reuse complete source files from interrupted preparation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    if (output/"manifest.json").exists():
        raise ValueError("Complete dataset already exists; choose a new directory")
    sources = []
    for repo, revision, remote, filename in SOURCES:
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}"
        path = output/filename
        if not (args.resume and path.exists()):
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nanoinfer-benchmark/1"}), timeout=60) as response:
                payload = response.read()
            path.write_bytes(payload)
        sources.append(dict(repo=repo, revision=revision, url=url, local=filename, sha256=digest(path)))
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = "\n".join(x for x in pq.read_table(output/"wiki.parquet")["text"].to_pylist() if x.strip())
    corpus = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    prepared = []
    def save(name, items):
        path = output/(name+".jsonl")
        # Prepared files can be regenerated only while there is no final manifest.
        path.write_text("", encoding="utf-8")
        append(path, items)
        prepared.append(path)
    for length in (256, 2048):
        items = []
        for i in range(256):
            offset = i*257
            ids = corpus[offset:offset+length]
            if len(ids) != length:
                raise ValueError("Not enough corpus tokens")
            items.append(dict(id=f"wiki{length}:{i}", token_ids=ids, offset=offset))
        save(f"wiki{length}", items)
    for prefix in (512, 1024, 1792):
        items, seed = prefix_samples(corpus, prefix)
        save(f"prefix{prefix}", items)
        path = output/f"prefix{prefix}-seed.json"
        write(path, seed)
        prepared.append(path)
    questions = [json.loads(line) for line in (output/"questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(questions) != 80:
        raise ValueError("Expected all 80 MT-Bench prompts")
    save("mtbench", [dict(id=f"mtbench:{q['question_id']}", category=q["category"],
        token_ids=chat_ids(tokenizer, q.get("turns", q.get("prompt"))[0])) for q in questions])
    token_files = {p.name: digest(p) for p in Path(args.model).iterdir() if p.is_file()
                   and ("tokenizer" in p.name or p.name in ("vocab.json", "merges.txt", "chat_template.jinja"))}
    write(output/"manifest.json", dict(protocol_version=1, at=now(), sources=sources,
        tokenizer_sha256=token_files, chat_template=dict(add_generation_prompt=True, enable_thinking=False),
        prepared_sha256={p.name: digest(p) for p in prepared}, corpus_tokens=len(corpus),
        selection="Wiki test nonempty rows concatenated; stride257 windows; prefix suffixes stride521; MT-Bench all first turns"))
    print(output/"manifest.json")


if __name__ == "__main__":
    main()
