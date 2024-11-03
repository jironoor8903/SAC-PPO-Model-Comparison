import os
from pathlib import Path
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.algorithm import Algorithm
from sumo_multi_agent_env import SumoConfig  # , SumoEnv
from sumo_multi_agent_env_hierarchical import SumoEnvHierarchical
from sumo_centralized_env import SumoEnvHierarchicalCentralized

# from models.combined_planner_module import MultiAgentSharedPlanner
from models.hierarchical_planner_module import HierarchicalMultiAgentSharedPlanner
# from models.car_following import IDM, IDMConfig

# from models.speed_planner import PiecewiseConstantSpeedPlanner
from utils.sumo_utils import extract_highway_profile_detector_file_name
from utils.i24_utils import get_main_road_west_edges


def policy_mapping_function(agent_id: str, episode, worker, **kwargs):
    return "hierarchical_policy"


if __name__ == "__main__":
    # Specify checkpoint to load algorithm
    checkpoint_path: str = None
    # checkpoint_path: str = (
    #     "/home/nz45jg/ray_results/"
    #     "PPO_SumoEnvHierarchicalCentralized_2024-06-23_12-29-09uwabdlev/"
    #     "checkpoint_499"
    # )
    checkpoint_num = 0
    if checkpoint_path is not None:
        algo = Algorithm.from_checkpoint(str(checkpoint_path))
        checkpoint_num = int(str(checkpoint_path).split("checkpoint_")[-1]) + 1

    else:
        # Specify scenario directory
        # scenario_dir = "scenarios/single_junction/test_single_lane_high_flow"
        scenario_dir = "scenarios/single_junction/test_calibrated"

        # List of SUMO config scenario files to use. Runs training for each of
        # the scenarios separately.
        scn_list = [
            (
                "edge_flows_interval_8400_taz_reduced_single_lane_2_sec"
                "_periodic_warmup_200_high_30s_low_8400s"
                "_single_lane_av_100_percent.sumocfg"
            )
        ]

        for sumo_config_file in scn_list:
            # Whether to use centralized environment, using global state data
            # instead of per-vehicle data.
            centralized = True

            # For centralized only:
            num_simulation_steps_per_step = 5  # 40

            # Set seed to an integer for deterministic training process. Set to None for
            # default behavior.
            rl_train_seed = 1234#None

            # Set seed to an integer for deterministic simulation. Set to None for
            # default behavior.
            sumo_seed = None

            seconds_per_step = 0.5

            speed_profile_detector_file = extract_highway_profile_detector_file_name(
                Path(scenario_dir) / sumo_config_file
            )

            sumo_config = SumoConfig(
                scenario_dir=scenario_dir,
                sumo_config_file=sumo_config_file,
                seconds_per_step=seconds_per_step,
                show_gui=False, #True,
                # show_gui=True,
                speed_profile_detector_file=speed_profile_detector_file,
                no_warnings=True,
                seed=sumo_seed,
                use_libsumo=True,#False,
            )

            # eval_config = EvaluationConfig()
            eval_config = None

            # Single junction
            state_start_edge = "992666043"

            # End after junction
            state_end_edge = "634155175.210"

            # Start control from the second edge
            # control_start_edge = "992666043.117"

            # Start control from the first edge
            control_start_edge = "992666043"

            # End control before merge
            control_end_edge = "992666042"

            # Two junctions
            # state_start_edge = "977008893.1043"
            # Final edge of i-24 network
            # state_end_edge = "635078551"

            # Start control from edge "977008892". This is the edge after "977008893.1043"
            # control_start_edge = "977008892"
            # control_end_edge = state_end_edge

            simulation_time = 500  # 700  # 1400  # 8400
            num_simulation_steps = int(simulation_time / seconds_per_step)
            warm_up_time = 100  # 150  # 240  # 1200
            num_eval_episodes = 1

            # eval_config = EvaluationConfig()
            eval_config = None

            if centralized:
                highway_edges = get_main_road_west_edges(with_internal=False)
            else:
                highway_edges = get_main_road_west_edges(with_internal=True)

            scenario_edges = highway_edges[
                highway_edges.index(state_start_edge) : highway_edges.index(
                    state_end_edge
                )
                + 1
            ]

            control_edges_list = highway_edges[
                highway_edges.index(control_start_edge) : highway_edges.index(
                    control_end_edge
                )
                + 1
            ]
            control_edges = {
                edge: {"start": 0, "end": -1} for edge in control_edges_list
            }

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
                include_tse_pos_in_obs=False,
                include_av_frac_in_obs=False,
            )

            if centralized:
                env_config.update(
                    dict(num_simulation_steps_per_step=num_simulation_steps_per_step)
                )
            # If False, uses environment steps to count rollout and batch steps. Each
            # environment step can include many agent (autonomous vehicle) steps.
            count_steps_by_agent = False

            # If 0, uses the default worker for rollouts. If larger than 0, creates
            # separate worker instances for each rollout worker.
            num_rollout_workers = 10#5#7#0#5  # For debug, more convenient to use 0
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

            env_cls = (
                SumoEnvHierarchicalCentralized if centralized else SumoEnvHierarchical
            )
            speed_planner_cls = (
                "CentralizedSpeedPlanner"
                if centralized
                else "PiecewiseConstantSpeedPlanner"
            )
            alg_config = (
                PPOConfig()
                .environment(env=env_cls, env_config=env_config)
                .rollouts(
                    num_rollout_workers=num_rollout_workers,
                    rollout_fragment_length=rollout_fragment_length,
                    create_env_on_local_worker=True,
                )
                # SAC-specific training parameters
                .training(
                    #enable_learner_api=True,
                    #train_batch_size=train_batch_size,
                    #lambda_=0.97,
                    #gamma=0.99,
                    #clip_param=0.2,
                    #num_sgd_iter=10,
                    #use_gae=True,
                    #kl_target=0.2,
                    #entropy_coeff=0.001,

                )
                # If using multi-agent, you can configure multi-agent settings similarly to PPO
                .multi_agent(
                    policy_mapping_fn=policy_mapping_function,
                    policies={
                        "hierarchical_policy": (
                            None,
                            env_cls(env_config).observation_space,
                            env_cls(env_config).action_space,
                            {},
                        ),
                    },
                )
                .debugging(
                    seed=rl_train_seed,
                )
            )
            alg_config._disable_preprocessor_api = True

            algo = PPO(alg_config)

        num_training_steps = 2000
        num_steps_between_saves = 50
        checkpoint_dir = "checkpoints"
        for train_step in range(num_training_steps):
            training_progress = algo.train()
            print(f"Completed training step #{train_step}")
            # print(training_progress)

            # Call `save()` to create a checkpoint.
            if (
                checkpoint_num + train_step
            ) % num_steps_between_saves == 0 or train_step == num_training_steps - 1:
                save_result = algo.save(
                    checkpoint_dir=os.path.join(
                        algo.logdir, f"checkpoint_{checkpoint_num + train_step}"
                    )
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

        # eval = algo.evaluate()
        # print(eval)

        # Terminate the algo
        algo.stop()