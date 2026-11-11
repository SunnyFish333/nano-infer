import atexit
from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.utils.image_processor import ImageProcessor
from nanovllm.models.qwen3_5 import build_position_ids
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        self.image_processor = None
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        atexit.unregister(self.exit)
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        seq, items = self.prepare_request(prompt, sampling_params)
        if items:
            self.model_runner.register_visual(seq.seq_id, items)
        return self.enqueue_request(seq)

    def prepare_request(self, prompt, sampling_params):
        items = []
        if isinstance(prompt, dict):
            if getattr(self, "image_processor", None) is None:
                self.image_processor = ImageProcessor(self.config)
            prepared = self.image_processor.process(prompt)
            prompt, items = prepared.token_ids, prepared.visual_items
        elif isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        config = getattr(self, "config", None) or self.model_runner.config
        seq = Sequence(prompt, sampling_params)
        seq.visual_spans = tuple((item.start, item.end, item.modality, item.content_digest)
                                 for item in items)
        if items:
            seq.position_ids, seq.rope_delta = build_position_ids(
                prompt, items, config.vision_config.spatial_merge_size)
        return seq, items

    def enqueue_request(self, seq):
        self.scheduler.add(seq)
        return seq.seq_id

    def abort_requests(self, seqs):
        for seq in seqs:
            self.scheduler.abort(seq)
        self.model_runner.release_visual([seq.seq_id for seq in seqs])

    def step(self):
        try:
            seqs, is_prefill = self.scheduler.schedule()
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
            plan = getattr(self.scheduler, "gdn_plan", None)
            token_ids = (self.model_runner.call("run", seqs, is_prefill, plan) if plan is not None
                         else self.model_runner.call("run", seqs, is_prefill))
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            if getattr(self.model_runner, "visual_requests", None):
                self.model_runner.release_visual([seq.seq_id for seq in seqs if seq.is_finished])
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            return outputs, num_tokens
        except BaseException:
            affected = getattr(self.scheduler, "_step_seqs", [])
            self.scheduler.rollback_step()
            if getattr(self.model_runner, "visual_requests", None):
                self.model_runner.release_visual([seq.seq_id for seq in affected])
            raise

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str | list[int] | dict],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        prepared = [self.prepare_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        for seq, items in prepared:
            if items:
                self.model_runner.register_visual(seq.seq_id, items)
        for seq, _ in prepared:
            self.enqueue_request(seq)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
