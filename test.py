agents:
  decision_model_config:
    hidden_dims: []
  do_include_fruit: true
  do_include_id_fruit: true
  do_include_novelty_hunger: true
  do_include_value_fruit: true
  do_include_value_global: false
  do_include_values_other_fruits: false
  do_use_decision_model: false
  do_use_different_model_fruit: false
  do_use_model_fruit: false
  factor_normalization_table_value_fruits: 1
  hp_initial:
    strength_mutation: 0.05
  list_channels_visual_field:
  - plants
  - agents
  - fruits_0
  - fruits_1
  - fruits_2
  - fruits_3
  log_dir: logs
  log_weights: true
  metrics:
    aggregators_lifespan:
    - class_string: ecojax.metrics.aggregators:AggregatorLifespanCumulative
      config:
        keys_measures:
        - reward
        log_final: true
        n_agents: 1500
        prefix_metric: life_cum
    - class_string: ecojax.metrics.aggregators:AggregatorLifespanAverage
      config:
        keys_measures:
        - reward
        - hp
        - params_reward_model
        - weights_agents
        - loss_q
        - q_values_max
        - q_values_min
        - q_values_mean
        - q_values_median
        - target
        - gradients weights
        - gradients bias
        - table_value_fruits
        n_agents: 1500
        prefix_metric: life_avg
    aggregators_population:
    - class_string: ecojax.metrics.aggregators:AggregatorPopulationMean
      config:
        keys_measures:
        - reward
        - hp
        - params_reward_model
        - weights_agents
        - loss_q
        - q_values_max
        - q_values_min
        - q_values_mean
        - q_values_median
        - target
        - gradients weights
        - gradients bias
        - table_value_fruits
        keys_measures_prefix:
        - life
        - weights
        - params_reward_model
        - hp
        n_agents: 1500
        prefix_metric: pop_mean
    measures:
      behavior: []
      global:
      - params_agents
      immediate:
      - reward
      state:
      - hp
      - params_reward_model
      - weights_agents
      - loss_q
      - q_values_max
      - q_values_min
      - q_values_mean
      - q_values_median
      - target
      - gradients weights
      - gradients bias
      - table_value_fruits
  mode_weights_transmission: none
  n_weights_log: 30
  name: AdaptiveAgents
  reward_model:
    dict_reward:
      age: 0
      energy: 0
      n_childrens: 0
    func_weight: hardcoded
  run_name: run_20250427_185919-bench_p-region/omega_0/seed_31631
benchmark_name: bench_p
do_cli: false
do_csv: false
do_global_log: false
do_jax_prof: false
do_jit: true
do_render: true
do_snakeviz: true
do_tb: false
do_tqdm: true
do_wandb: true
env:
  age_max: 200
  dim_appearance: 0
  do_fruits: true
  do_normalize_fruits: false
  do_plant_grow_in_fruit_clusters: false
  duration: 200000
  e_fruit_0_abs_max: 40
  e_fruit_T_abs_max: 30
  e_fruit_abs_max_ref: 10
  energy_cost_reprod: 60
  energy_fruit_min: -10
  energy_initial: 50
  energy_max: 100
  energy_plant: 20
  energy_req_reprod: 100
  energy_thr_death: 0
  factor_plant_asphyxia: 0
  factor_plant_reproduction: 0
  factor_sun_effect: 0
  factors_fruits:
  - 1
  - 0.5
  - -0.5
  - -1
  height: 110
  is_terminal: true
  list_actions:
  - forward
  - left
  - backward
  - right
  - eat
  list_channels_visual_field:
  - plants
  - agents
  - fruits_0
  - fruits_1
  - fruits_2
  - fruits_3
  list_death_events:
  - age
  - energy
  list_observations:
  - visual_field
  - energy
  - age
  - novelty_hunger
  - n_childrens
  method_sun: none
  metrics:
    aggregators_lifespan:
    - class_string: ecojax.metrics.aggregators:AggregatorLifespanCumulative
      config:
        keys_measures:
        - do_action_eat
        - do_action_transfer
        - do_action_reproduce
        - do_action_forward
        - do_action_nothing
        - amount_food_eaten
        - amount_children
        - died
        - density_plants_observed
        - density_agents_observed
        - density_fruits_observed
        log_final: true
        n_agents: 1500
        prefix_metric: life_cum
    - class_string: ecojax.metrics.aggregators:AggregatorLifespanAverage
      config:
        keys_measures:
        - do_action_eat
        - do_action_transfer
        - do_action_reproduce
        - do_action_forward
        - do_action_nothing
        - amount_food_eaten
        - amount_children
        - died
        - density_plants_observed
        - density_agents_observed
        - density_fruits_observed
        - energy
        - age
        - novelty_hunger
        - n_childrens
        - x
        - appearance
        - eating
        - moving
        n_agents: 1500
        prefix_metric: life_avg
    - class_string: ecojax.metrics.aggregators:AggregatorLifespanVariation
      config:
        keys_measures:
        - energy
        - age
        - novelty_hunger
        - n_childrens
        - x
        - appearance
        - eating
        - moving
        keys_measures_prefix:
        - eating
        - moving
        n_agents: 1500
        prefix_metric: life_var
    aggregators_population:
    - class_string: ecojax.metrics.aggregators:AggregatorPopulationMean
      config:
        keys_measures:
        - do_action_eat
        - do_action_transfer
        - do_action_reproduce
        - do_action_forward
        - do_action_nothing
        - amount_food_eaten
        - amount_children
        - died
        - density_plants_observed
        - density_agents_observed
        - density_fruits_observed
        - energy
        - age
        - novelty_hunger
        - n_childrens
        - x
        - appearance
        - eating
        - moving
        keys_measures_prefix:
        - life
        - last_final
        - novelty_hunger
        - eating
        - moving
        n_agents: 1500
        prefix_metric: pop_mean
    - class_string: ecojax.metrics.aggregators:AggregatorPopulationStd
      config:
        keys_measures:
        - energy
        - age
        - novelty_hunger
        - n_childrens
        - x
        - appearance
        - eating
        - moving
        keys_measures_prefix:
        - life
        - last_final
        n_agents: 1500
        prefix_metric: pop_std
    behavior_measures_on_render:
    - appetite
    - eating_behavior
    - moving_behavior
    config_video:
      color_background: gray
      dict_name_channel_to_color_tag:
        agents: blue
        fruits_0: red
        fruits_1: orange
        fruits_2: purple
        fruits_3: pink
        plants: green
        sun: yellow
      dir_videos: ./logs/run_20250427_185919-bench_p-region/omega_0/seed_31631/videos
      do_agent_video: false
      do_video: true
      fps_video: 20
      height_max_video: 500
      n_steps_min_between_videos: 5000
      n_steps_per_video: 500
      width_max_video: 500
    measures:
      behavior: []
      environmental:
      - n_agents
      - n_plants
      - net_energy_transfer_per_capita
      - values_fruits
      - map_scaling_factors
      - energy_fruit_max_abs
      - entropy_fruits
      immediate:
      - do_action_eat
      - do_action_transfer
      - do_action_reproduce
      - do_action_forward
      - do_action_nothing
      - amount_food_eaten
      - amount_children
      - died
      - density_plants_observed
      - density_agents_observed
      - density_fruits_observed
      state:
      - energy
      - age
      - novelty_hunger
      - n_childrens
      - x
      - appearance
      - eating
      - moving
  min_density_fruits: 0.4
  mode_variability_fruits: space_diffusion
  name: Gridworld
  omega: 0
  p_base_fruit_death: 0.0005
  p_base_fruit_growth: 0.0065
  p_base_plant_death: 0
  p_base_plant_growth: 0
  period_fruits: null
  period_sun: 300
  prop_empty_clusters: 0
  proportion_fruit_initial: 0.4
  proportion_plant_initial: 0
  radius_plant_asphyxia: 20
  radius_plant_reproduction: 10
  radius_sun_effect: 10
  radius_sun_perception: 40
  range_cluster_fruits: 5
  run_name: run_20250427_185919-bench_p-region/omega_0/seed_31631
  side_cluster_fruits: 11
  sum_energy_map_ref: 275
  t_fruit_T: 50000
  threshold_n_agents: 300
  variability_fruits:
  - 0
  - 0.1
  - 0.5
  - 1
  vision_range_agent: 7
  width: 110
log_dir: logs
model:
  factor_normalization_table_value_fruits: 1
  mlp_config:
    hidden_dims: []
    n_output_features: 5
  mlp_region_config:
    hidden_dims:
    - 12
    - 8
    n_output_features: 1
  name: RegionalModel
  weighting_method: uniform
n_actions: 5
n_agents_initial: 400
n_agents_max: 1500
n_timesteps: 200000
period_eval: 500
run_name: run_20250427_185919-bench_p-region/omega_0/seed_31631
seed: 31631
wandb_config:
  project: EcoJAX
