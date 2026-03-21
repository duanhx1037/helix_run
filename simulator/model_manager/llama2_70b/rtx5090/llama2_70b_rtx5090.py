import os

from typing import Dict

from simulator.model_manager.base_classes import ModelOnMachine
from simulator.model_manager.llama2_70b.helper import llama70b_workload_ratio, llama70b_typical_statistics
from simulator.event_simulator.model import MachineProfile
from simulator.event_simulator.compute_node import InferenceSettings
from simulator.event_simulator.utils import load_profile_csv
from simulator.event_simulator.utils import VLLM_BLOCK_SIZE, MAX_INPUT_LEN, DECODE_PER_TOKEN_MAX_CONTEXT


class LLaMa70BonRTX5090(ModelOnMachine):
    def __init__(self, num_machines_dict: Dict[str, int], typical_layers_dict: Dict[str, int],
                 normalized_perf_dict: Dict[str, float]):
        """
        LLaMa2-70B + RTX5090.
        Latency tables: load from this directory (same folder as this file):
          prompt_bs2time.csv, decode_bs2time.csv
        Copy profiling outputs here after running helix-rtx5090-profiling.
        """
        machine_name: str = "RTX5090"
        # Assumes ~32GB VRAM (consumer RTX 5090). max_num_layers = max transformer layers placed on ONE
        # such GPU in your Helix cluster (not full 70B). Tune if your placement / TP differs.
        max_num_layers: int = 10

        # vllm_num_blocks_dict: Helix uses these like the official profiles (see llama2_70b_a100 / l4).
        # Below is a HEURISTIC, not paper-measured: linear blend L4(24GB)↔A100(40GB) at α=(32-24)/(40-24)=0.5
        # for layers 1–7; layers 8–10 use A100 values × (32/40) (no L4 row to blend).
        # For production, re-measure with vLLM on your 5090 (gpu_memory_utilization, dtype, TP) and replace.
        vllm_num_blocks_dict: Dict[int, int] = {
            1: 382236,
            2: 178562,
            3: 109998,
            4: 75970,
            5: 55554,
            6: 41943,
            7: 32221,
            8: 31554,
            9: 25726,
            10: 21065,
        }
        # L4 uses 1 concurrent prompt req; A100 uses 3 — midpoint 2 for 32GB class.
        prompt_max_requests_dict: Dict[int, int] = {i: 2 for i in range(1, max_num_layers + 1)}
        # decode_max_tokens: blend L4 (380, …, 280@7) and A100 (480, …) for 1–7; 8–10 scale A100 by 32/40.
        decode_max_tokens_dict: Dict[int, int] = {
            1: 430,
            2: 430,
            3: 430,
            4: 430,
            5: 430,
            6: 430,
            7: 380,
            8: 384,
            9: 384,
            10: 360,
        }

        self.machine_name: str = machine_name
        self.num_machines_dict: Dict[str, int] = num_machines_dict

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.prompt_bs2time: Dict[int, float] = load_profile_csv(
            file_name=os.path.join(dir_path, "prompt_bs2time.csv")
        )
        self.prompt_bs2vram: Dict[int, float] = {bs: 0 for bs in self.prompt_bs2time}
        self.decode_bs2time: Dict[int, float] = load_profile_csv(
            file_name=os.path.join(dir_path, "decode_bs2time.csv")
        )
        self.decode_bs2vram: Dict[int, float] = {bs: 0 for bs in self.decode_bs2time}

        self.max_num_layers: int = max_num_layers
        self.kv_cache_capacity: Dict[int, int] = {
            _num_layers: VLLM_BLOCK_SIZE * _num_blocks * _num_layers for _num_layers, _num_blocks in
            vllm_num_blocks_dict.items()
        }
        self.activation_backup_capacity: Dict[int, int] = {
            _num_layers: 0 for _num_layers in self.kv_cache_capacity
        }

        self.num_layers_to_inference_settings: Dict[int, InferenceSettings] = {}
        for cur_num_layers in range(1, self.max_num_layers + 1):
            cur_workload_ratio = llama70b_workload_ratio(
                target_machine_name=machine_name,
                target_num_layers=cur_num_layers,
                num_machines_dict=num_machines_dict,
                typical_layers_dict=typical_layers_dict,
                normalized_perf_dict=normalized_perf_dict
            )
            prompt_typical_requests, prompt_typical_tokens, decode_typical_tokens = llama70b_typical_statistics(
                workload_ratio=cur_workload_ratio,
                num_kv_cache_entries=self.kv_cache_capacity[cur_num_layers],
                num_layers_on_node=cur_num_layers
            )
            assert prompt_typical_requests <= 1, "Typical requests should be less than 1!"
            self.num_layers_to_inference_settings[cur_num_layers] = InferenceSettings(
                prompt_max_requests=prompt_max_requests_dict[cur_num_layers],
                prompt_max_tokens=prompt_max_requests_dict[cur_num_layers] * MAX_INPUT_LEN,
                prompt_typical_requests=prompt_typical_requests,
                prompt_typical_tokens=prompt_typical_tokens,
                decode_max_context=decode_max_tokens_dict[cur_num_layers] * DECODE_PER_TOKEN_MAX_CONTEXT,
                decode_max_tokens=decode_max_tokens_dict[cur_num_layers],
                decode_typical_tokens=decode_typical_tokens
            )

    def get_profiling_results(self) -> MachineProfile:
        machine_profile = MachineProfile(prompt_bs2time=self.prompt_bs2time, prompt_bs2vram=self.prompt_bs2vram,
                                         decode_bs2time=self.decode_bs2time, decode_bs2vram=self.decode_bs2vram)
        return machine_profile

    def get_max_num_layers(self) -> int:
        return self.max_num_layers

    def get_inference_settings(self, num_on_node_layers: int) -> InferenceSettings:
        assert 0 < num_on_node_layers <= self.max_num_layers, "Bad number of layers on node!"
        return self.num_layers_to_inference_settings[num_on_node_layers]

    def get_typical_token_throughput(self, num_on_node_layers: int) -> float:
        inference_settings = self.get_inference_settings(num_on_node_layers=num_on_node_layers)
        prompt_typical_requests = inference_settings.prompt_typical_requests
        prompt_typical_tokens = inference_settings.prompt_typical_tokens
        decode_typical_tokens = inference_settings.decode_typical_tokens

        from simulator.event_simulator.utils import linear_interpolate

        def _get_prompt_time(prompt_num_tokens: int) -> float:
            prompt_left, prompt_right = -1, 1000 * 1000
            for prompt_point in self.prompt_bs2time:
                if prompt_left < prompt_point <= prompt_num_tokens:
                    prompt_left = prompt_point
                if prompt_num_tokens <= prompt_point < prompt_right:
                    prompt_right = prompt_point
            return linear_interpolate(x_0=prompt_left, y_0=self.prompt_bs2time[prompt_left],
                                      x_1=prompt_right, y_1=self.prompt_bs2time[prompt_right],
                                      x_target=prompt_num_tokens)

        def _get_decode_time(decode_num_tokens: int) -> float:
            decode_left, decode_right = -1, 1000 * 1000
            for decode_point in self.decode_bs2time:
                if decode_left < decode_point <= decode_num_tokens:
                    decode_left = decode_point
                if decode_num_tokens <= decode_point < decode_right:
                    decode_right = decode_point
            return linear_interpolate(x_0=decode_left, y_0=self.decode_bs2time[decode_left],
                                      x_1=decode_right, y_1=self.decode_bs2time[decode_right],
                                      x_target=decode_num_tokens)

        if prompt_typical_requests >= 1:
            total_tokens = prompt_typical_tokens + decode_typical_tokens
            layer_prompt_time = _get_prompt_time(prompt_num_tokens=prompt_typical_tokens)
            layer_decode_time = _get_decode_time(decode_num_tokens=decode_typical_tokens)
            total_time = num_on_node_layers * (layer_prompt_time + layer_decode_time)
            return total_tokens / total_time
        else:
            rescaling = 1 / prompt_typical_requests
            total_tokens = rescaling * (prompt_typical_tokens + decode_typical_tokens)
            layer_prompt_time = _get_prompt_time(prompt_num_tokens=int(prompt_typical_tokens * rescaling))
            layer_decode_time = _get_decode_time(decode_num_tokens=decode_typical_tokens) * rescaling
            total_time = num_on_node_layers * (layer_prompt_time + layer_decode_time)
            return total_tokens / total_time

    def get_kv_cache_capacity(self, num_on_node_layers: int) -> int:
        return self.kv_cache_capacity[num_on_node_layers]

    def get_activation_backup_capacity(self, num_on_node_layers: int) -> int:
        return self.activation_backup_capacity[num_on_node_layers]
