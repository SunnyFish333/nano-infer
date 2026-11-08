import atexit
from dataclasses import fields
from time import perf_counter, perf_counter_ns
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.observer import emit


class LLMEngine:

    def __init__(self, model, observer=None, **kwargs):
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.observer = observer
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
        self.scheduler = Scheduler(config, observer=observer)
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
        if not isinstance(sampling_params, SamplingParams):
            raise ValueError("sampling_params must be SamplingParams")
        sampling_params.validate()  # SamplingParams is mutable after construction.
        items = []
        if isinstance(prompt, dict):
            if self.config.vision_config is None or self.config.tensor_parallel_size != 1:
                raise ValueError("Visual requests require a single-GPU multimodal checkpoint")
            from nanovllm.utils.image_processor import ImageProcessor
            if getattr(self, "image_processor", None) is None:
                self.image_processor = ImageProcessor(self.config)
            prepared = self.image_processor.process(prompt)
            prompt, items = prepared.token_ids, prepared.visual_items
        elif isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not isinstance(prompt, list) or not prompt or any(type(token) is not int for token in prompt):
            raise ValueError("Prompt must contain a nonempty list of integer token IDs")
        config = getattr(self, "config", None) or self.model_runner.config
        vocab_size = config.text_config.vocab_size
        if any(token < 0 or token >= vocab_size for token in prompt):
            raise ValueError("Token ID outside model vocabulary")
        if len(prompt) + sampling_params.max_tokens > config.max_model_len:
            raise ValueError("Expanded prompt and output budget exceed max_model_len")
        if not items and getattr(config, "vision_config", None) is not None and any(
                token in (config.image_token_id, config.video_token_id) for token in prompt):
            raise ValueError("Visual placeholders require matching image/video data")
        seq = Sequence(prompt, sampling_params)
        if items:
            from nanovllm.models.qwen3_5 import build_position_ids
            seq.position_ids, seq.rope_delta = build_position_ids(
                prompt, items, config.vision_config.spatial_merge_size)
        return seq, items

    def enqueue_request(self, seq):
        observer = getattr(self, "observer", None)
        submitted_ns = perf_counter_ns() if observer is not None else None
        self.scheduler.add(seq)
        if observer is not None:
            emit(observer, "request_enqueued", seq_id=seq.seq_id,
                 submitted_ns=submitted_ns, prompt_tokens=seq.num_prompt_tokens,
                 max_tokens=seq.max_tokens, waiting=len(self.scheduler.waiting),
                 running=len(self.scheduler.running))
        return seq.seq_id

    def abort_requests(self, seqs):
        for seq in seqs:
            self.scheduler.abort(seq)
        self.model_runner.release_visual([seq.seq_id for seq in seqs])

    def set_observer(self, observer):
        self.observer = observer
        self.scheduler.observer = observer

    def step(self):
        observer = getattr(self, "observer", None)
        if observer is not None:
            emit(observer, "step_start")
        try:
            seqs, is_prefill = self.scheduler.schedule()
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
            if observer is not None:
                emit(observer, "scheduled", phase="prefill" if is_prefill else "decode",
                     waiting=len(self.scheduler.waiting), running=len(self.scheduler.running),
                     sequences=[dict(seq_id=s.seq_id, cached_tokens=s.num_cached_tokens,
                                     scheduled_tokens=s.num_scheduled_tokens) for s in seqs])
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            if getattr(self.model_runner, "visual_requests", None):
                self.model_runner.release_visual([seq.seq_id for seq in seqs if seq.is_finished])
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            if observer is not None:
                emit(observer, "step_end", waiting=len(self.scheduler.waiting),
                     running=len(self.scheduler.running))
            return outputs, num_tokens
        except BaseException as error:
            if "seqs" in locals() and getattr(self.model_runner, "visual_requests", None):
                self.abort_requests(seqs)
            if observer is not None:
                emit(observer, "step_error", error=type(error).__name__ + ": " + str(error))
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
        if len(sampling_params) != len(prompts):
            raise ValueError("sampling_params must match the number of prompts")
        # Validate the entire batch before publishing requests to the scheduler.
        prepared = [self.prepare_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        try:
            for seq, items in prepared:
                if items:
                    self.model_runner.register_visual(seq.seq_id, items)
        except BaseException:
            self.model_runner.release_visual([seq.seq_id for seq, _ in prepared])
            raise
        for seq, _ in prepared:
            self.enqueue_request(seq)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            try:
                output, num_tokens = self.step()
            except BaseException:
                self.abort_requests([seq for seq, _ in prepared])
                pbar.close()
                raise
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
