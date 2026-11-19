from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Case:
    group: str
    workload: str
    input_tokens: int | None
    output_tokens: int
    batch: int = 1
    mode: str = "graph"
    prefix: int = 0
    cache: str = "off"
    k: int = 0
    continuous: bool = False
    ignore_eos: bool = True
    samples: int = 64
    repeat: int = 0
    profile: bool = False

    def __post_init__(self):
        if len(self.group) != 1 or self.group not in "ABCDE" or self.mode not in ("eager", "graph"):
            raise ValueError("Unknown group or execution mode")
        if self.batch < 1 or self.samples < self.batch or self.output_tokens < 2:
            raise ValueError("Invalid batch/sample/output size")
        if not 0 <= self.k <= 4 or self.repeat < 0:
            raise ValueError("Invalid speculative window or repeat")
        if self.cache not in ("off", "hot") or (self.k and self.cache != "off"):
            raise ValueError("MTP and prefix reuse cannot be combined")
        if self.prefix and (self.prefix % 256 or not self.input_tokens or self.prefix >= self.input_tokens):
            raise ValueError("Prefix must end at a complete block before the input tail")
        if not self.continuous and self.samples % self.batch:
            raise ValueError("Fixed batches must be complete")

    @property
    def id(self):
        return (f"{self.group}_{self.workload}_o{self.output_tokens}_b{self.batch}_{self.mode}_p{self.prefix}_{self.cache}_k{self.k}"
                f"_r{self.repeat}" + ("_profile" if self.profile else ""))

    def to_dict(self):
        return asdict(self)

    def with_repeat(self, repeat):
        return replace(self, repeat=repeat)

    def engine_options(self):
        maximum_batch = self.batch if self.group == "E" else (1 if self.group == "C" else 4)
        return dict(max_model_len=2304 if self.group == "E" else 4608,
            max_num_batched_tokens=256, max_num_seqs=maximum_batch,
            gpu_memory_utilization=.8, tensor_parallel_size=1, kvcache_block_size=256,
            num_gdn_slots=None if self.group == "E" else 12,
            adapt_cache_to_workload=False, adaptive_speculation=False,
            enforce_eager=self.mode == "eager", enable_gdn_cuda_graph=self.mode == "graph",
            cuda_graph_max_batch_size=maximum_batch, enable_gdn_prefix_cache=self.cache == "hot",
            enable_mtp=self.k > 0, num_speculative_tokens=self.k,
            speculative_max_batch_size=1, profile_mtp=self.profile)


def repeat_plan(cases, repeats=3):
    """Rotate order without changing paired inputs."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return [case.with_repeat(r) for r in range(repeats)
            for case in cases[r % len(cases):] + cases[:r % len(cases)]]
