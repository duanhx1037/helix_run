# 2024.10.29 Yixuan Mei
import os
import shutil
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from simulator.initial_layout.layout_synthesizer import LayoutMethod, LayoutSynthesizer, ModelName
from simulator.event_simulator.utils import kbps, mbps, gbps, KB, MB, GB, Sec, MilliSec

_BASELINE_DIR = os.path.join(_repo_root, "baseline")
# Cluster from baseline/step1_gen_cluster.py; machine profile from examples (has RTX5090, etc.)
_CLUSTER_INI = os.path.join(_BASELINE_DIR, "config", "cross_partition_cluster.ini")
_MACHINE_PROFILE = os.path.join(_repo_root, "examples", "simulation", "config", "machine_profile.ini")
_ILP_WORKSPACE = os.path.join(_BASELINE_DIR, "layouts", "ilp")


def _reset_ilp_workspace() -> None:
    """ILP synthesize() requires an empty workspace when use_existing_sol is False."""
    if os.path.isdir(_ILP_WORKSPACE):
        shutil.rmtree(_ILP_WORKSPACE)
    os.makedirs(_ILP_WORKSPACE, exist_ok=True)


def ilp_layout():
    _reset_ilp_workspace()
    # initialize the layout synthesizer
    layout_synthesizer = LayoutSynthesizer(
        complete_cluster_file_name=_CLUSTER_INI,
        machine_profile_name=_MACHINE_PROFILE,
        model_name=ModelName.LLaMa70B,
        workspace_path=_ILP_WORKSPACE,
        layout_method=LayoutMethod.ILP,
        machine_num_dict={"A100": 1, "RTX5090": 2},
    )

    # setting arguments for ILP layout synthesis
    # see simulator.initial_layout.layout_synthesizer.synthesize for more details about the arguments
    ilp_args = {
        # pruning
        # pruning removes some edges in the graph to reduce the problem size
        "enable_pruning": False,
        "min_keep": 12,
        "max_keep": 12,
        "keep_bandwidth_threshold": 1 * mbps,
        # ILP
        # if "use_existing_sol" is True, the synthesizer will only load an existing solution and verify it
        # if you want to continue optimize from an existing solution, you can use "start_from_heuristic" below
        # here, if "use_existing_sol" is True, the "existing_sol_path" should be the ilp_solution.sol you
        # want to verify
        "use_existing_sol": False,
        "allow_partial_inference": False,
        "remove_redundant": True,
        "max_run_time": 36000,
        "early_stop_time": 100,
        "early_stop_threshold": 0.95,
        "existing_sol_path": "path/to/existing/ilp_solution.sol",
        # heuristic (needs petals_sol.ini); off for minimal baseline — turn on after running Petals elsewhere
        "start_from_heuristic": False,
        "heuristic_sol_path": os.path.join(_repo_root, "examples", "simulation", "layouts", "petals", "petals_sol.ini"),
    }

    # run the ILP layout synthesis
    layout_synthesizer.synthesize(args=ilp_args)


def main():
    """
    The second step is to find a model placement for the cluster. The model placement specifies which
    layers each machine holds.
    Helix simulator supports four layout synthesis methods:
     1. ILP: the MILP-based layout method in Helix
     2. Petals: our implementation of Petals' layout method
     3. Swarm: our implementation of Swarm's layout method
     4. Homogeneous: similar to Orca, each pipeline contains the same type of machines
    """
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <layout_method> (ilp/swarm/petals/homogeneous)"
    layout_method = sys.argv[1]

    if layout_method == "ilp":
        # ILP layout synthesis
        # Note: We set the max running time to 10 hours. However, you can stop the process at any time (ctrl + c ONCE)
        # and the best solution found so far will be saved. In this example, we early stop at around 10 minutes.
        # Depending on the random seed, the running time and model placement found may vary.
        ilp_layout()
        print(f"ILP layout synthesis is done! (Results in {_ILP_WORKSPACE})")
    else:
        raise ValueError(f"Unknown layout method: {layout_method}")


if __name__ == '__main__':
    main()
