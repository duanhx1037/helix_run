# 2024.10.28 Yixuan Mei
import os
import sys

# Repo root (parent of baseline/) must be on sys.path when running: python baseline/step1_gen_cluster.py
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec
from simulator.initial_layout.fake_cluster_generator import PartitionedClusterGenerator


def generate_partitioned(file_name: str):
    """
    This function shows how to generate a cluster configuration file to represent a partitioned cluster.
    """
    generator = PartitionedClusterGenerator()

    # TODO set the statistics of the cluster
    generator.add_partition(nodes_list=["A100"] * 1)
    generator.add_partition(nodes_list=["RTX5090"] * 1)
    generator.add_partition(nodes_list=["RTX5090"] * 1)
    # TODO set the network statistics
    # If each partition only has one machine, the in_partition network statistics will not be used.
    generator.set_network_statistics(
        in_partition_avg_bandwidth=1.25 * gbps, in_partition_var_bandwidth=125 * mbps,
        in_partition_avg_latency=1 * MilliSec, in_partition_var_latency=0,
        cross_partition_avg_bandwidth=12.5 * mbps, cross_partition_var_bandwidth=2.5 * mbps,
        cross_partition_avg_latency=50 * MilliSec, cross_partition_var_latency=10 * MilliSec
    )

    # generate the cluster
    generator.generator_fake_cluster(file_name=file_name, seed=0, create_separate=False)


def main():
    """
    We provide two automatic ways to generate the cluster configuration file. Refer to FakeClusterGenerator
    and PartitionedClusterGenerator for more details. If you have a specific cluster configuration, you can
    also write your own script to generate the cluster configuration file.

    Note: currently, the simulator only supports machines with {"A100", "V100", "L4", "L4x2", "T4", "T4x2",
    "T4x4", "RTX5090"} GPUs. You can add more machines by profiling them and add the data to
    simulator/model_manager.
    """
    out_dir = os.path.join(_repo_root, "baseline", "config")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_partition_cluster.ini")
    generate_partitioned(file_name=out_path)
    print(f"Partitioned cluster configuration file is generated to {out_path}")


if __name__ == '__main__':
    main()
