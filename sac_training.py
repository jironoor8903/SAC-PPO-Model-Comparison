# edit from remote mac
import os

from ray.rllib.algorithms.sac.sac import SAC, SACConfig

from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.algorithm import Algorithm
from sumo_multi_agent_env import SumoConfig  # , SumoEnv
from sumo_multi_agent_env_hierarchical import SumoEnvHierarchical
from sumo_centralized_env import SumoEnvHierarchicalCentralized

# from models.combined_planner_module import MultiAgentSharedPlanner
from models.hierarchical_planner_module import HierarchicalMultiAgentSharedPlanner
from models.car_following import IDM, IDMConfig

# from models.speed_planner import PiecewiseConstantSpeedPlanner

from utils.i24_utils import get_main_road_west_edges


def policy_mapping_function(agent_id: str, episode, worker, **kwargs):
    return "hierarchical_policy"


if __name__ == "__main__":
    # print("running from remote")

    scn_list = [

    "edge_flows_interval_8400_taz_reduced_single_lane_periodic_warmup_240_high_30s_low_8400s_single_lane_av_10_percent.sumocfg",
    "edge_flows_interval_8400_taz_reduced_periodic_warmup_240_high_30s_low_8400s_av_10_percent.sumocfg",
    "edge_flows_interval_8400_taz_reduced_single_lane_periodic_warmup_240_high_30s_low_8400s_single_lane_av_50_percent.sumocfg",
    "edge_flows_interval_8400_taz_reduced_periodic_warmup_240_high_30s_low_8400s_av_50_percent.sumocfg",
    "edge_flows_interval_8400_taz_reduced_single_lane_periodic_warmup_240_high_30s_low_8400s_single_lane_av_100_percent.sumocfg",
    "edge_flows_interval_8400_taz_reduced_periodic_warmup_240_high_30s_low_8400s_av_100_percent.sumocfg"

    ]

    for sumo_config_file in scn_list:
        print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(sumo_config_file)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        checkpoint_path: str = None  # Specify checkpoint to load algorithm

        if checkpoint_path is not None:
            algo = Algorithm.from_checkpoint(str(checkpoint_path))

        else:
            centralized = True
            # For centralized only:
            num_simulation_steps_per_step = 40

            # Set seed to an integer for deterministic training process. Set to None for
            # default behavior.
            rl_train_seed = None

            # Set seed to an integer for deterministic simulation. Set to None for
            # default behavior.
            sumo_seed = None
            #
            # scenario_dir = "scenarios/reduced_junctions"
            # sumo_config_file = (
            #     "edge_flows_interval_8400"
            #     + "_taz_reduced"
            #     + "_flow_scale_0.8"
            #     # + "_custom"
            #     + "_av_10_percent"
            #     + "_highway_profile"
            #     + "_priority"
            #     + "_short_merge_lane"
            #     + ".sumocfg"
            # )

            scenario_dir = "scenarios/single_junction/test_single_lane_high_flow"
            seconds_per_step = 0.5

            speed_profile_detector_file = "add_highway_profile_detectors.xml"
            if "short_merge_lane" in sumo_config_file:
                speed_profile_detector_file = (
                    "short_merge_lane_" + speed_profile_detector_file
                )

            sumo_config = SumoConfig(
                scenario_dir=scenario_dir,
                sumo_config_file=sumo_config_file,
                seconds_per_step=seconds_per_step,
                show_gui=False,
                # show_gui=True,
                speed_profile_detector_file=speed_profile_detector_file,
                no_warnings=True,
                seed=sumo_seed,
                use_libsumo=True,
            )

            # eval_config = EvaluationConfig()
            eval_config = None

            simulation_time = 3000  # 8400
            num_simulation_steps = int(simulation_time / seconds_per_step)
            warm_up_time = 0  # 1200
            num_eval_episodes = 1

            scenario_start_edge = "977008893.1043"
            control_start_edge = "977008892"
            scenario_end_edge = "635078551"

            highway_edges = get_main_road_west_edges(with_internal=True)
            scenario_edges = highway_edges[
                highway_edges.index(scenario_start_edge) : highway_edges.index(
                    scenario_end_edge
                )
                + 1
            ]
            # Start control at edge "977008892". This is the edge after "977008893.1043"
            control_edges_list = highway_edges[
                highway_edges.index(control_start_edge) : highway_edges.index(
                    scenario_end_edge
                )
                + 1
            ]
            control_edges = {edge: {"start": 0, "end": -1} for edge in control_edges_list}
            simulation_time = 1400  # 8400
            num_simulation_steps = int(simulation_time / seconds_per_step)
            warm_up_time = 240  # 1200
            num_eval_episodes = 1


            # eval_config = EvaluationConfig()
            eval_config = None

            highway_edges = get_main_road_west_edges(with_internal=True)
            scenario_edges = highway_edges[
                             highway_edges.index(scenario_start_edge): highway_edges.index(
                                 scenario_end_edge
                             )
                                                                       + 1
                             ]
            # Start control at edge "977008892". This is the edge after "977008893.1043"
            control_edges_list = highway_edges[
                                 highway_edges.index(control_start_edge): highway_edges.index(
                                     scenario_end_edge
                                 )
                                                                          + 1
                                 ]
            control_edges = {edge: {"start": 0, "end": -1} for edge in control_edges_list}

            env_config = dict(
                simulation_time=simulation_time,
                sumo_config=sumo_config,
                warm_up_time=warm_up_time,
                control_edges=control_edges,
                eval_config=eval_config,
                name_postfix="test_env",
                highway_sorted_road_edges=scenario_edges,
                # Comment car following model to use SUMO built-in models
                # car_following_model_class=IDM,
                # car_following_model_config=IDMConfig(numpy_input=True),
                normalize_car_following_obs=False,
                use_outflow_reward=False,
                use_time_delay_reward=True,
                state_merge_edges=["277208926"],
            )

            if centralized:
                env_config.update(
                    dict(num_simulation_steps_per_step=num_simulation_steps_per_step)
                )
            # If False, uses environment steps to count rollout and batch steps. Each
            # environment step can include many agent (autonomous vehicle) steps.
            count_steps_by_agent = False
            # If 0, uses the default worker for rollouts. If larger than 0, creates
            # separate worker instances for each rollout worker
            num_rollout_workers = 5
            def_batch_size = 4000
            rollout_fragment_length = (
                int(def_batch_size / max(1, num_rollout_workers))
                if count_steps_by_agent
                else num_simulation_steps
            )
            if centralized:
                rollout_fragment_length = int(
                    rollout_fragment_length / num_simulation_steps_per_step
                )
            # A training step is run on a batch of experience.
            #
            # For PPO specifically: In each training step the weights are updated
            # multiple time on mini batches of 128 agent steps by default.
            #
            # Training seems to work even if the episode is not done. The episode is
            # done only when `truncated["__all__"]` or `terminated["__all__"]` are
            # `True`.
            #
            # However, evaluation metrics are computed only for episodes that are done.
            # See the methods step, `_process_observations`, `_handle_done_episode`, and
            # `_get_rollout_metrics` in ray.rllib.evaluation.env_runner_v2.py. Only when
            # the episode is done, the environment runner `step` function will return a
            # `RolloutMetrics` object. Only then it is put into the metrics queue of the
            # sampler object calling the env runner (see the get_data method in
            # ray.rllib.evaluation.sampler.py).
            #
            # Sampling is done by each rollout worker in its `sample` method (see
            # `RolloutWorker` class in ray.rllib.evaluation.rollout_worker.py). It then
            # gets a batch from a sampler (`SamplerInput`) using its `next` function,
            # which calls the `get_data` method.
            train_batch_size = rollout_fragment_length * max(1, num_rollout_workers)

            env_cls = SumoEnvHierarchicalCentralized if centralized else SumoEnvHierarchical
            speed_planner_cls = (
                "CentralizedSpeedPlanner"
                if centralized
                else "PiecewiseConstantSpeedPlanner"
            )
            alg_config = (
                SACConfig()
                .environment(env=env_cls, env_config=env_config)
                .rollouts(
                    num_rollout_workers=num_rollout_workers,
                    rollout_fragment_length=rollout_fragment_length,
                    create_env_on_local_worker=True,
                )
                # SAC-specific training parameters
                .training(
                    train_batch_size=train_batch_size,
                    # Additional SAC-specific configurations can go here
                )
                # If using multi-agent, you can configure multi-agent settings similarly to PPO
                .multi_agent(
                    policy_mapping_fn=policy_mapping_function,
                    policies={
                        "hierarchical_policy": (None, env_cls(env_config).observation_space, env_cls(env_config).action_space, {}),
                    }
                )
                .debugging(seed=rl_train_seed)
            )
            alg_config._disable_preprocessor_api = True

            algo = SAC(alg_config)

        num_training_steps = 66
        checkpoint_dir = "checkpoints"
        for train_step in range(num_training_steps):
            training_progress = algo.train()
            print(f"Completed training step #{train_step}")
            # print(training_progress)

            # Call `save()` to create a checkpoint.
            save_result = algo.save(
                checkpoint_dir=os.path.join(algo.logdir, f"checkpoint_{train_step}")
            )
            path_to_checkpoint = save_result.checkpoint.path
            print(f"{save_result = }")
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )

        # If evaluation_duration_unit is "episodes", the algorithm runs
        # evaluation_duration episodes with rollout_fragment_length steps. If
        # count_steps_by is "env_steps", counts each environment step as a single
        # step, and it does not matter how many agents are in the environment. If
        # count_steps_by is "agent_steps", each agent step is counted as a single
        # step.
        eval = algo.evaluate()
        print(eval)

        # Terminate the algo
        algo.stop()