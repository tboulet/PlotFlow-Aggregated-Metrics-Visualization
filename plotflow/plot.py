import cProfile
import datetime
import enum
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tbutils
from tbutils.seed import set_seed, try_get_seed
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Project imports
from plotflow.functions import dict_agg_functions, dict_deltas_functions


# Import path constants for config management
from plotflow.utils_config import (
    DIR_CONFIGS,
    NAME_CONFIG_DEFAULT,
    assert_config_dir_exists,
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[PlotFlow] %(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Kwarg imports for eval
KWARGS_IMPORTS = {"np": np, "pd": pd, "re": re}

# Representation functions
join = lambda *args: ", ".join(args)
join_par = lambda *args: f"({join(*args)})" if len(args) > 1 else join(*args)
drop_prefix = lambda s: re.sub(r"\b\w+\.(\w+)\b", r"\1", s)


# Enum classes
class EvalArg(enum.Enum):
    """Enum class for the arguments that can be passed to the eval function."""

    RUN_NAME = "run_name"
    CONFIG = "config"
    X_VALUES = "x_values"
    METRICS = "metrics"
    METRIC = "metric"
    METRIC_NAME = "metric_name"
    METRIC_EXPRESSION = "metric_expression"
    GROUP_KEY = "group_key"


class EvalCase(enum.Enum):
    """Enum class for the cases of arguments to be passed to the eval function.
    This is helpful to determine when to apply filters or groupings based on the arguments, to avoid unnecessary computations.
    """

    # For args not containing field specific to numerical values, only config or run_name
    BASED_ON_CONFIG = 1
    # For args containing numerical fields but common to all metrics : x_values, metrics
    BASED_ON_ALL_METRICS = 2
    # For args also containing numerical fields specific to a metric : metric, metric_expression, metric_name
    BASED_ON_CURRENT_METRIC = 3

    @classmethod
    def get_case(cls, args: Set[EvalArg]) -> "EvalCase":
        """Get the case of the arguments."""
        # If numerical fields, furthermore specific to a metric run
        if any(
            arg in args
            for arg in [EvalArg.METRIC, EvalArg.METRIC_EXPRESSION, EvalArg.METRIC_NAME]
        ):
            return EvalCase.BASED_ON_CURRENT_METRIC
        # Elif numerical fields, but common to all metrics
        elif any(arg in args for arg in [EvalArg.X_VALUES, EvalArg.METRICS]):
            return EvalCase.BASED_ON_ALL_METRICS
        # Else : only config or run_name
        else:
            return EvalCase.BASED_ON_CONFIG


class RunMetricData:
    """This class contains the data of one particular metric for one particular run."""

    def __init__(
        self,
        run_name: str,
        config: Dict[str, Any],
        metric_values: np.ndarray,
        x_values: np.ndarray,
        metric_expression: str,
        metric_name: str,
        group_key: Tuple = (),
    ):
        """Initializes the RunMetricData class.

        Args:
            run_name (str): the name of the run.
            config (Dict[str, Any]): the configuration of the run.
            metric_values (pd.Series): the values of the metric considered.
            x_values (pd.Series): the x values of the run.
            metric_expression (str): the expression of the metric considered.
            metric_name (str): the name of the metric considered.
            group_key (Tuple, optional): the key of the group. Defaults to ().
        """
        self.run_name = run_name
        self.config = config
        self.metric_values = metric_values
        self.x_values = x_values
        self.metric_expression = metric_expression
        self.metric_name = metric_name
        self.group_key = group_key


class GroupIndexer:
    """A class that manages the indexes of each group in the scope of the all plot."""

    def __init__(self, groups: List[Dict[str, Any]]):
        self.groups = groups
        self.expression_group_to_index_mapping = {
            dict_group["expression"]: {} for dict_group in groups
        }

    def add_group(self, group_key: Tuple):
        """Add a group to the index mapping."""
        for i, (group_subkey, dict_group) in enumerate(zip(group_key, self.groups)):
            expression_group = dict_group["expression"]
            if (
                group_subkey
                not in self.expression_group_to_index_mapping[expression_group]
            ):
                self.expression_group_to_index_mapping[expression_group][
                    group_subkey
                ] = len(self.expression_group_to_index_mapping[expression_group])

    def get_group_index(self, group_key: Tuple) -> Tuple[int]:
        """Get the index of a group."""
        return tuple(
            self.expression_group_to_index_mapping[dict_group["expression"]][
                group_subkey
            ]
            for group_subkey, dict_group in zip(group_key, self.groups)
        )

    def get_max_indexes(self) -> Tuple[int]:
        """Get the maximum index for each group."""
        return tuple(
            len(self.expression_group_to_index_mapping[dict_group["expression"]])
            for dict_group in self.groups
        )

    def get_group_kwargs_plot(self, group_key) -> Dict:
        """Get the kwargs_plot for a group, from the args_plot specified in the groups if any."""
        kwargs_plot = {}
        for group_subkey, dict_group in zip(group_key, self.groups):
            if "key_plot" in dict_group and "args_plot" in dict_group:
                key_plot = dict_group["key_plot"]
                args_plot = dict_group["args_plot"]
                if len(args_plot) == 0:
                    continue
                idx_subgroup = self.expression_group_to_index_mapping[
                    dict_group["expression"]
                ][group_subkey]
                idx_subgroup_modulo = idx_subgroup % len(args_plot)
                kwargs_plot[key_plot] = args_plot[idx_subgroup_modulo]
        return kwargs_plot

    def get_x_and_width(self, group_key: Tuple, x_min: float, x_max: float) -> Tuple:
        """Get the x values and the width for a group."""
        return self.get_x_and_width_from_index(
            self.get_group_index(group_key), self.get_max_indexes(), x_min, x_max
        )

    def get_x_and_width_from_index(
        self, group_index: Tuple, group_index_max: Tuple, x_min: float, x_max: float
    ) -> Tuple:
        print(group_index, group_index_max, x_min, x_max)
        """Get the x values and the width for a group."""
        assert len(group_index) == len(group_index_max), "Invalid group index."
        if len(group_index) == 0:
            raise ValueError("Empty group index.")
        elif len(group_index) == 1:
            index = group_index[0]
            index_max = group_index_max[0]
            n_indexes = index_max + 1
            percentage_width = 0.8
            width = (x_max - x_min) * percentage_width / n_indexes
            x = x_min + (1 - percentage_width) / 2 + width * index
            return x, width
        else:
            index_considered = group_index[0]
            index_max_considered = group_index_max[0]
            n_indexes_considered = index_max_considered + 1
            width_allocated = (x_max - x_min) / n_indexes_considered
            x_min_next = x_min + width_allocated * index_considered
            x_max_next = x_min + width_allocated * (index_considered + 1)
            return self.get_x_and_width_from_index(
                group_index[1:], group_index_max[1:], x_min_next, x_max_next
            )


class PlotFlowPlotter:

    def __init__(self, config: DictConfig):
        """Initializes the Plotter class with configuration parameters.

        Args:
            config (DictConfig): Hydra configuration object.
        """
        self.config = config
        # Set logger and seed
        self.logger = logger
        seed = try_get_seed(self.config, warn_if_unvalid=False)
        set_seed(seed)
        self.logger.info(f"Seed set to {seed}.")
        # Date
        self.date_plot = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Paths
        self.runs_path = self.config.get("runs_path", "data/exports")
        self.file_scalars: str = self.config.get("file_scalars", "scalars.csv")
        self.file_config: str = self.config.get("file_config", "config.yaml")
        # Metric expressions
        self.metrics_expressions: List[Dict] = self.config.get(
            "metrics_expressions", []
        )
        assert (
            len(self.metrics_expressions) > 0
        ), "No metric(s) specified. You must specify your metrics in the 'metrics_expressions' field."
        for i, dict_metric_expr in enumerate(self.metrics_expressions):
            if isinstance(dict_metric_expr, str):
                self.metrics_expressions[i] = {
                    "expression": dict_metric_expr,
                    "name": dict_metric_expr,
                }
        self.metrics_expressions_repr = drop_prefix(
            join(
                *(
                    dict_metric_expr["name"]
                    for dict_metric_expr in self.metrics_expressions
                )
            )
        )
        # Filter expressions
        self.filters: List[Dict] = self.config.get("filters", [])
        for i, dict_filter in enumerate(self.filters):
            if isinstance(dict_filter, str):
                self.filters[i] = {"expression": dict_filter}
        filters_expressions: List[str] = [
            filter["expression"] for filter in self.filters
        ]
        self.filters_args: Set[EvalArg] = {
            eval_arg
            for eval_arg in EvalArg
            if eval_arg.value in " ".join(filters_expressions)
        }
        # Grouping expressions
        self.groups: List[Dict] = self.config.get("groups", [])
        for i, dict_group in enumerate(self.groups):
            if isinstance(dict_group, str):
                self.groups[i] = {"expression": dict_group}
        groups_expressions: List[str] = [group["expression"] for group in self.groups]
        self.groups_args: Set[EvalArg] = {
            eval_arg
            for eval_arg in EvalArg
            if eval_arg.value in " ".join(groups_expressions)
        }
        self.groups_repr = drop_prefix(join_par(*groups_expressions))
        self.group_indexer = GroupIndexer(self.groups)
        # Aggregation options
        self.method_aggregate: str = self.config.get("method_aggregate", "mean")
        assert (
            self.method_aggregate in dict_agg_functions
        ), f"Invalid method_aggregate: {self.method_aggregate}. Must be in {list(dict_agg_functions.keys())}"
        self.fn_agg = dict_agg_functions[self.method_aggregate]
        self.method_error: str = self.config.get("method_error", "std")
        assert (
            self.method_error in dict_deltas_functions
        ), f"Invalid method_error: {self.method_error}. Must be in {list(dict_deltas_functions.keys())}"
        self.fn_error = dict_deltas_functions[self.method_error]
        self.n_samples_shown: int = self.config.get("n_samples_shown", 0)
        self.max_n_runs: int = self.config.get("max_n_runs", np.inf)
        self.max_n_curves: int = self.config.get("max_n_curves", np.inf)
        if self.max_n_runs in [None, "None", "null", "inf", "np.inf"]:
            self.max_n_runs = np.inf
        # Plotting interface options
        self.x_axis: str = self.config.get("x_axis", "_step")
        self.x_lim = self.config.get("x_lim", None)
        if self.x_lim is None:
            self.x_lim = [None, None]
        self.y_lim = self.config.get("y_lim", None)
        if self.y_lim is None:
            self.y_lim = [None, None]
        self.do_try_include_y0: bool = self.config.get("do_try_include_y0", False)
        self.ratio_near_y0: str = self.config.get("ratio_near_y0", 0.5)
        self.x_label: str = self.config.get("x_label", self.x_axis)
        self.alpha_shaded: float = self.config.get("alpha_shaded", 0.2)
        self.max_legend_length: int = self.config.get("max_legend_length", 10)
        self.kwargs_plot: Dict = self.config.get("kwargs_plot", {})
        self.do_show_one_by_one: bool = self.config.get("do_show_one_by_one", False)
        self.figsize: List[int] = self.config.get("figsize", [10, 6])
        # Show/save options
        self.do_show: bool = self.config.get("do_show", True)
        self.do_save: bool = self.config.get("do_save", False)
        self.path_save: str = self.config.get(
            "path_save", "f'plots/plot_{metric}_{date}.png'"
        )

    def load_grouped_data(self, run_dirs: List[str]) -> Dict[str, List[RunMetricData]]:
        grouped_data: Dict[str, List[RunMetricData]] = {}
        n_run_loaded = 0
        n_curves_showed = 0
        for run_path in tqdm(run_dirs, desc="[PlotFlow] Filtering ..."):
            if n_run_loaded >= self.max_n_runs:
                break
            # Get run name from run_path
            run_name = os.path.basename(run_path)
            # Load run config
            config_path = os.path.join(run_path, self.file_config)
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    run_config = yaml.safe_load(f)
            else:
                run_config = {}
                self.logger.warning(
                    f"[Run Loading] Could not find config file {self.file_config} in {run_path}. Using empty config for this run."
                )
            run_config = OmegaConf.create(run_config)
            # Determine in which case we are for the filters and grouping
            case_filters = EvalCase.get_case(self.filters_args)
            case_groups = EvalCase.get_case(self.groups_args)
            # Filter runs here if based on config/other arguments
            if case_filters == EvalCase.BASED_ON_CONFIG:
                if not self.apply_filters(run_name=run_name, run_config=run_config):
                    continue
            # Group runs here if based on config/other arguments
            if case_groups == EvalCase.BASED_ON_CONFIG:
                group_key = self.get_group_key(run_name=run_name, run_config=run_config)
            # Load scalars file
            scalars_path = os.path.join(run_path, self.file_scalars)
            if not os.path.exists(scalars_path):
                self.logger.error(
                    f"Could not find scalars file {self.file_scalars} in {run_path}. Skipping run."
                )
                continue
            file_scalars = open(scalars_path, "r")
            metrics = pd.read_csv(file_scalars)
            file_scalars.close()
            # Extract x values
            if not self.x_axis in metrics.columns:
                self.logger.error(
                    f"Could not find x-axis column '{self.x_axis}' in run {run_name}. Skipping run."
                )
                continue
            x_values = metrics[self.x_axis]
            # Filter runs here if based on all metrics
            if case_filters == EvalCase.BASED_ON_ALL_METRICS:
                if not self.apply_filters(
                    run_name=run_name,
                    run_config=run_config,
                    metrics=metrics,
                    x_values=x_values,
                ):
                    continue
            # Group runs here if based on all metrics
            if case_groups == EvalCase.BASED_ON_ALL_METRICS:
                group_key = self.get_group_key(
                    run_name=run_name,
                    run_config=run_config,
                    metrics=metrics,
                    x_values=x_values,
                )
            # Iterate on metrics
            is_curve_added = False
            for dict_metric_expr in self.metrics_expressions:
                # Break if max number of runs reached
                if n_curves_showed >= self.max_n_curves:
                    break
                metric_name = dict_metric_expr["name"]
                metric_expression = dict_metric_expr["expression"]
                # Evaluate metric expression
                try:
                    metric_values = self.eval_expression(
                        dict_metric_expr,
                        {
                            EvalArg.CONFIG.value: run_config,
                            EvalArg.RUN_NAME.value: run_name,
                            EvalArg.X_VALUES.value: x_values,
                            EvalArg.METRICS.value: metrics,
                            **KWARGS_IMPORTS,
                        },
                    )
                except Exception as e:
                    self.logger.error(
                        f"Could not evaluate metric expression '{metric_expression}' for run {run_name}, skipping this expression - {e}"
                    )
                    continue
                # Assert obtained metric values are of a valid type
                valid_types = [pd.Series, np.float64]
                type_metric_values = self.get_general_type(metric_values)
                if not type_metric_values in valid_types:
                    self.logger.error(
                        f"Invalid type for metric {metric_expression}: {type(metric_values)}'s general type is not in {valid_types}. Skipping this expression."
                    )
                    continue
                # Filter runs here if based on current metric
                if case_filters == EvalCase.BASED_ON_CURRENT_METRIC:
                    if not self.apply_filters(
                        run_name=run_name,
                        run_config=run_config,
                        metric_expression=metric_expression,
                        metric_name=metric_name,
                        metric_values=metric_values,
                        x_values=x_values,
                        metrics=metrics,
                    ):
                        continue
                # Filter runs here if based on current metric
                if case_groups == EvalCase.BASED_ON_CURRENT_METRIC:
                    group_key = self.get_group_key(
                        run_name=run_name,
                        run_config=run_config,
                        metric_expression=metric_expression,
                        metric_name=metric_name,
                        metric_values=metric_values,
                        x_values=x_values,
                        metrics=metrics,
                    )
                # Create RunMetricData object
                run_metric_data = RunMetricData(
                    run_name=run_name,
                    config=run_config,
                    metric_values=metric_values,
                    x_values=x_values,
                    metric_expression=metric_expression,
                    metric_name=metric_name,
                    group_key=group_key,
                )
                if not group_key in grouped_data:
                    grouped_data[group_key] = []
                grouped_data[group_key].append(run_metric_data)
                # Add group to the group indexer
                self.group_indexer.add_group(group_key)
                # Increment number of curves showed and run loaded
                n_curves_showed += 1
                is_curve_added = True
            # Increment number of run loaded
            if is_curve_added:
                n_run_loaded += 1
        lengths_groups = [len(v) for v in grouped_data.values()]
        grouped_data_keys = list(grouped_data.keys())
        if len(grouped_data_keys) > 0:
            self.logger.info(
                f"After filtering and grouping : {n_run_loaded} runs loaded, {n_curves_showed} curves showed."
            )
            if len(self.groups) > 0:
                self.logger.info(
                    f"Obtained {len(grouped_data_keys)} groups by grouping by {self.groups_repr} : {grouped_data_keys if len(grouped_data_keys) < 10 else grouped_data_keys[:10] + ['...']}, of sizes {lengths_groups if len(lengths_groups) < 10 else lengths_groups[:10] + ['...']} (average {np.mean(lengths_groups):.2f} ± {np.std(lengths_groups):.2f}, runs/group, min {np.min(lengths_groups)}, max {np.max(lengths_groups)})"
                )
        else:
            self.logger.error("No runs were loaded.")
        return grouped_data

    def apply_filters(
        self,
        run_name: str,
        run_config: Dict[str, Any],
        x_values: Optional[np.ndarray] = None,
        metrics: Optional[pd.DataFrame] = None,
        metric_expression: Optional[str] = "no metric (no-metric condition)",
        metric_name: Optional[str] = "no metric (no-metric condition)",
        metric_values: Optional[np.ndarray] = None,
    ):
        """Applies filters on a RunMetricData components to determine if it should be included in the data.
        A RunMetricData corresponds to a specific metric for a specific run.
        RunMetricData components involves the config of the run, the "metric" the y_values of the metric on this run, "x" the x_values of the run, and can also involve metrics (the whole run dataframe).
        All these fields except the run_name and config are optional for this method, but they must appear if they appear in the filters.

        Args:
            run_config (Dict[str, Any]): The config of the run.
            run_name (str): The name of the run.
            x_values (Optional[np.ndarray]): The x_values of the run.
            metrics (Optional[pd.DataFrame]): The run dataframe.
            metric_expression (str): The name of the metric.
            metric_name (str): The name of the metric.
            metric (Optional[np.ndarray]): The values of the metric on this run.

        Returns:
            bool: True if run satisfies all filters, False otherwise.
        """
        for dict_filter in self.filters:
            try:
                if not self.eval_expression(
                    dict_filter,
                    {
                        EvalArg.CONFIG.value: run_config,
                        EvalArg.RUN_NAME.value: run_name,
                        EvalArg.X_VALUES.value: x_values,
                        EvalArg.METRICS.value: metrics,
                        EvalArg.METRIC_EXPRESSION.value: metric_expression,
                        EvalArg.METRIC_NAME.value: metric_name,
                        EvalArg.METRIC.value: metric_values,
                        **KWARGS_IMPORTS,
                    },
                ):
                    return False
            except Exception as e:
                self.logger.warning(
                    f"[Filtering] Could not evaluate filter condition '{dict_filter}' for run {run_name} metric {metric_expression}, skipping - {e}"
                )
                return False
        return True

    def get_group_key(
        self,
        run_name: str,
        run_config: Dict[str, Any],
        x_values: Optional[np.ndarray] = None,
        metrics: Optional[pd.DataFrame] = None,
        metric_expression: Optional[str] = "no metric (no-metric condition)",
        metric_name: Optional[str] = "no metric (no-metric condition)",
        metric_values: Optional[np.ndarray] = None,
    ) -> Tuple:
        """Constructs a group key based on a RunMetricData components.
        A RunMetricData corresponds to a specific metric for a specific run.
        RunMetricData components involves the config of the run, the "metric" the y_values of the metric on this run, "x" the x_values of the run, and can also involve metrics (the whole run dataframe).
        All these fields except the run_name and config and run_name are optional for this method, but they must appear if they appear in the grouping.

        Args:
            run_name (str): The name of the run.
            run_config (Dict[str, Any]): The config of the run.
            metrics (Optional[pd.DataFrame]): The run dataframe.
            metric_expression (str): The expression of the metric.
            metric (Optional[np.ndarray]): The values of the metric on this run.
            x_values (Optional[np.ndarray]): The x_values of the run.

        Returns:
            Tuple: The group key.
        """
        if len(self.groups) == 0:
            return (run_name,)
        group_key = []
        for dict_group in self.groups:
            expression_group = dict_group["expression"]
            try:
                value = self.eval_expression(
                    dict_group,
                    {
                        EvalArg.RUN_NAME.value: run_name,
                        EvalArg.CONFIG.value: run_config,
                        EvalArg.X_VALUES.value: x_values,
                        EvalArg.METRICS.value: metrics,
                        EvalArg.METRIC_EXPRESSION.value: metric_expression,
                        EvalArg.METRIC_NAME.value: metric_name,
                        EvalArg.METRIC.value: metric_values,
                        **KWARGS_IMPORTS,
                    },
                )

            except Exception as e:
                self.logger.warning(
                    f"[Grouping] Could not evaluate grouping expression '{expression_group}' for run {run_name} and metric {metric_name}, skipping curve - {e}"
                )
                value = "None"  # assign 'None' if field evaluation fails
            group_key.append(value)
        group_key = tuple(group_key)  # use tuple for hashability
        return group_key

    def eval_expression(
        self, dict_expression: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> Any:
        """Evaluates an expression with the given arguments.

        Args:
            dict_expression (Dict[str, Any]): a dictionnary containing 'expression' and eventually 'aggregators'.
            kwargs (Dict[str, Any]): the arguments to pass to the expression.

        Returns:
            Any: the result of the evaluation : eval(dict_expression['expression'], kwargs)
        """
        expression: str = dict_expression["expression"]
        if "aggregators" in dict_expression and len(dict_expression["aggregators"]) > 0:
            aggregators: List[Dict[str, Any]] = dict_expression["aggregators"]
            agg_first = aggregators[0]
            # From the expression, get the list of expressions with the aggregator key replaced by each value
            key = f"@{agg_first['key']}"
            values = agg_first["values"]
            list_expressions = [expression.replace(key, str(val)) for val in values]
            # Evaluate each expression
            results = [
                self.eval_expression(
                    {"expression": expr, "aggregators": aggregators[1:]}, kwargs
                )
                for expr in list_expressions
            ]
            # Transform the results into a DataFrame
            merged_df = self.get_merged_df(results)
            # Apply the operation on the results
            operation = agg_first["operation"]
            assert (
                operation in dict_agg_functions
            ), f"Unsupported aggregation operation: {operation}. Must be in {list(dict_agg_functions.keys())}"
            fn_operation = dict_agg_functions[operation]
            return fn_operation(merged_df)

        else:
            return eval(expression, kwargs)

    def try_include_y0(
        self, y_lim: Optional[List[float]], y_min: float, y_max: float
    ) -> List[float]:
        """Tries to include y=0 in the y-axis limits if 0 is close to the points relatively to the range |y_max - y_min|.

        Args:
            y_lim (Optional[List[float]]): Y-axis limits.
            merged_df (pd.DataFrame): DataFrame containing all metric values.

        Returns:
            List[float]: Updated y-axis limits.
        """
        if y_min <= 0 <= y_max:  # negative and positive values (already includes 0)
            return y_lim
        range_values = y_max - y_min
        if y_min <= y_max <= 0:  # negative values only
            if range_values > self.ratio_near_y0 * -y_min and y_lim[1] is None:
                return [y_min, 0]
            else:
                return y_lim
        elif 0 <= y_min <= y_max:  # positive values only
            if range_values > self.ratio_near_y0 * y_max and y_lim[0] is None:
                return [0, y_max]
            else:
                return y_lim
        else:
            raise ValueError("Invalid y-axis limits.")

    def sanitize_filepath(self, filepath: str, replacement="_"):
        """
        Sanitize a file path by replacing invalid characters in the filename
        while optionally replacing directory separators.

        Args:
            filepath (str): The original file path.
            replacement (str, optional): Character to replace invalid ones. Defaults to "_".

        Returns:
            str: A sanitized file path safe for saving.
        """
        # Define invalid characters for filenames (excluding / and \ for now)
        invalid_chars = r'[<>:"|?*]'

        # Separate the directory path and the filename
        directory, filename = os.path.split(filepath)

        # Sanitize the filename
        sanitized_filename = re.sub(invalid_chars, replacement, filename)
        sanitized_filename = re.sub(
            rf"{re.escape(replacement)}+", replacement, sanitized_filename
        )
        sanitized_filename = sanitized_filename.strip(replacement)

        # Reconstruct the full path
        sanitized_path = os.path.join(directory, sanitized_filename)

        # Only takes first 255 characters
        if len(sanitized_path) > 255:
            path, extension = os.path.splitext(sanitized_path)
            path = path[:255]
            sanitized_path = path + extension
            self.logger.warning(
                f"Saving path is too long, only taking the first 255 characters."
            )
        return sanitized_path

    def treat_grouped_data(self, grouped_data: Dict[Any, List[RunMetricData]]):
        """Plots metrics with mean and standard error for each group.
        Also defines y_min and y_max based on the metric values.
        """

        # Define label function
        def label_fn(group_key: Tuple):
            group_key_pretty_list = []
            for elem in group_key:
                if isinstance(elem, str):
                    elem = drop_prefix(elem)
                else:
                    elem = str(elem)
                group_key_pretty_list.append(elem)
            group_key_pretty = join(*group_key_pretty_list)
            return group_key_pretty

        # Iterate on the groups
        plt.figure(figsize=self.figsize)
        y_min, y_max = np.inf, -np.inf
        x_max = max(
            max(
                run_metric_data.x_values.max()
                for run_metric_data in list_runs_metric_data
            )
            for list_runs_metric_data in grouped_data.values()
        )
        x_min = min(
            min(
                run_metric_data.x_values.min()
                for run_metric_data in list_runs_metric_data
            )
            for list_runs_metric_data in grouped_data.values()
        )
        bar_ticks = []
        bar_labels = []
        for group_key, list_runs_metric_data in grouped_data.items():
            # Check if non empty group
            if not list_runs_metric_data:
                self.logger.warning(
                    f"[Aggregating] Skipping group {group_key} due to missing data."
                )
                continue
            n_curve_plotted = 0
            # Get the merged DataFrame for all runs in the group
            list_metric_values: List[pd.Series] = []
            x_values_global: pd.Series = pd.Series()
            max_t = -np.inf
            for run_metric_data in list_runs_metric_data:
                # Get the metric values
                metric_values = run_metric_data.metric_values
                list_metric_values.append(metric_values)
                # Get the x-axis values global
                x_values = run_metric_data.x_values
                if max_t < x_values.max():
                    max_t = x_values.max()
                    x_values_global = x_values
            if len(list_metric_values) == 0:  # Skip group if no valid data
                self.logger.warning(
                    f"[Aggregating] Skipping group {group_key} due to missing data."
                )
                continue
            merged_df = self.get_merged_df(list_metric_values)
            if merged_df is None:
                self.logger.error(
                    f"[Aggregating] Could not merge data for group {group_key}. Skipping this group."
                )
                continue

            type_metric_values = self.get_general_type(list_metric_values[0])

            # Get plot kwargs
            kwargs_plot_group = self.kwargs_plot.copy()
            indexes_group = self.group_indexer.get_group_index(group_key)
            for num_group_category, (group_subkey, dict_group) in enumerate(
                zip(group_key, self.groups)
            ):
                if "key_plot" in dict_group:
                    if (
                        "kwargs_plot" not in dict_group
                        and "args_plot" not in dict_group
                    ):
                        self.logger.warning(
                            f"Group {expression_group} has a key_plot but no args_plot/kwargs_plot. Skipping the label."
                        )
                        continue
                else:
                    continue
                key_plot = dict_group["key_plot"]
                expression_group = dict_group["expression"]
                group_subkey_repr = str(group_subkey)
                group_subkey_idx = indexes_group[num_group_category]
                if "kwargs_plot" in dict_group:
                    kwargs_plot = dict_group["kwargs_plot"]
                    if group_subkey_repr not in kwargs_plot:
                        self.logger.warning(
                            f"Category {group_subkey_repr} not in kwargs_plot for group {expression_group}. Skipping the label/using args_plot."
                        )
                        if "args_plot" in dict_group:
                            args_plot = dict_group["args_plot"]
                            kwargs_plot_group[key_plot] = args_plot[
                                group_subkey_idx % len(args_plot)
                            ]
                        else:
                            continue
                    else:
                        kwargs_plot_group[key_plot] = kwargs_plot[group_subkey_repr]
                elif "args_plot" in dict_group:
                    args_plot = dict_group["args_plot"]
                    kwargs_plot_group[key_plot] = args_plot[
                        group_subkey_idx % len(args_plot)
                    ]

            # Get group label
            label = (
                label_fn(group_key)
                if n_curve_plotted < self.max_legend_length
                else None
            )

            # Compute aggregated and error values
            mean_values = merged_df.mean(axis=1, skipna=True)
            values_aggregated = self.fn_agg(merged_df)
            delta_low, delta_high = self.fn_error(merged_df, values_aggregated)

            # Plot aggregated values (curves)
            if type_metric_values == pd.Series:
                plt.plot(
                    x_values_global,
                    values_aggregated,
                    label=label,
                    **kwargs_plot_group,
                )
                n_curve_plotted += 1
                if delta_low is not None and delta_high is not None:
                    plt.fill_between(
                        x_values_global,
                        mean_values - delta_low,
                        mean_values + delta_high,
                        alpha=self.alpha_shaded,
                        **kwargs_plot_group,
                    )

                # Plot samples
                n = merged_df.shape[1]
                sampled_indices = np.random.choice(
                    np.arange(n), size=min(self.n_samples_shown, n), replace=False
                )
                for i in sampled_indices:
                    plt.plot(
                        x_values_global,
                        merged_df.iloc[:, i],
                        alpha=self.alpha_shaded,
                        **kwargs_plot_group,
                    )

            elif type_metric_values == np.float64:
                # Plot the bars for aggregated values
                x, width = self.group_indexer.get_x_and_width(group_key, x_min, x_max)
                plt.bar(
                    [x],
                    values_aggregated[0],
                    label=label,
                    width=width,
                    **kwargs_plot_group,
                )
                n_curve_plotted += 1
                if delta_low is not None and delta_high is not None:
                    plt.errorbar(
                        x,
                        y=mean_values[0],
                        yerr=[[delta_low[0]], [delta_high[0]]],
                        fmt="o",
                        **kwargs_plot_group,
                    )
                bar_ticks.append(x)
                bar_labels.append(label)
                n = merged_df.shape[1]
                sampled_indices = np.random.choice(
                    np.arange(n), size=min(self.n_samples_shown, n), replace=False
                )
                for i in sampled_indices:
                    plt.hlines(
                        merged_df.iloc[:, i],
                        x - width / 2 * 1.2,
                        x + width / 2 * 1.2,
                        alpha=self.alpha_shaded,
                        **kwargs_plot_group,
                    )
            # Set xticks
            if type_metric_values == np.float64:
                plt.xticks(bar_ticks, bar_labels, rotation=30)

            # Update y_min and y_max based on the group values
            if delta_low is not None and delta_high is not None:
                y_min = min(values_aggregated.min() - delta_low.min(), y_min)
                y_max = max(values_aggregated.max() + delta_high.max(), y_max)
            else:
                y_min = min(values_aggregated.min(), y_min)
                y_max = max(values_aggregated.max(), y_max)
        # Don't plot if no data
        if y_min == np.inf and y_max == -np.inf:
            self.logger.error("[Plotting] No data to plot.")
            return

        self.logger.info(
            f"[Plotting] Building plot for {len(grouped_data)} aggregated curves."
        )

        # ======= Plot settings =======
        plt.xlabel(self.x_label)
        plt.ylabel(self.metrics_expressions_repr)
        # Set x and y limits
        plt.xlim(self.x_lim)
        if self.do_try_include_y0:
            y_lim = self.try_include_y0(self.y_lim, y_min=y_min, y_max=y_max)
        else:
            y_lim = self.y_lim
        plt.ylim(y_lim)
        # Legend and grid
        if len(self.groups) > 0:
            if n_curve_plotted > self.max_legend_length:
                title_legend = f"Groups (first {self.max_legend_length} shown / {n_curve_plotted} total)"
            else:
                title_legend = "Groups"
            plt.legend(title=title_legend, **self.config.get("kwargs_legend", {}))
        plt.grid(**self.config.get("kwargs_grid", {}))
        # Title
        if self.method_error == "none":
            string_methods_agg_error = f"{self.method_aggregate}"
        else:
            string_methods_agg_error = f"{self.method_aggregate}, σ={self.method_error}"
        if len(self.groups) == 0:
            title = f"{self.metrics_expressions_repr} ({string_methods_agg_error})"
        else:
            title = f"{self.metrics_expressions_repr} (grouped by {self.groups_repr} : {string_methods_agg_error})"
        plt.title(title, **self.config.get("kwargs_title", {}))

        # Save plot
        if self.do_save:
            metric_repr_file_compatible = self.metrics_expressions_repr.replace(
                "/", "_per_"
            )
            try:
                path_save = eval(
                    self.path_save,
                    {"metric": metric_repr_file_compatible, "date": self.date_plot},
                )
            except Exception as e:
                path_save = (
                    f"plots/plot_{metric_repr_file_compatible}_{self.date_plot}.png"
                )
                self.logger.error(
                    f"Could not evaluate path_save expression '{self.path_save}', using default path '{path_save}' instead - {e}"
                )
            path_save = self.sanitize_filepath(path_save)
            os.makedirs(os.path.dirname(path_save), exist_ok=True)
            plt.savefig(path_save, **self.config.get("kwargs_savefig", {}))
            self.logger.info(
                f"""Saved plot for metrics "{self.metrics_expressions_repr}" to {path_save}"""
            )

        # Show plot
        if self.do_show:
            plt.show()

    # ======= Utility functions =======

    def get_merged_df(self, list_metric_values: List[Any]) -> pd.DataFrame:
        # Check type of each element of list_metric_values and apply adapted operation
        # Return a (n_run_metric, n_x_value) DataFrame
        type_metric_values = self.get_general_type(list_metric_values[0])
        if not all(
            self.get_general_type(elem) == type_metric_values
            for elem in list_metric_values
        ):
            self.logger.error(
                f"[Aggregating] Inconsistent types in list_metric_values: {[type(elem) for elem in list_metric_values]}."
            )
            return None
        if type_metric_values == pd.Series:
            merged_df = pd.concat(list_metric_values, axis=1, join="outer")
        elif type_metric_values == np.float64:
            merged_df = pd.DataFrame(list_metric_values).T
        else:
            self.logger.error(
                f"[Aggregating] Invalid type for metric values: {type_metric_values}."
            )
            return None
        return merged_df

    def get_general_type(self, value: Any) -> Type:
        """Returns the general type of a value."""
        if np.isscalar(value):
            return np.float64
        elif isinstance(value, pd.Series):
            return pd.Series
        else:
            return type(value)

    # ======= Main function =======
    def run(self):
        """Executes the full pipeline: loading, filtering, grouping, and plotting."""
        run_dirs = [
            os.path.join(self.runs_path, d)
            for d in os.listdir(self.runs_path)
            if os.path.isdir(os.path.join(self.runs_path, d))
        ]
        self.logger.info(f"Found {len(run_dirs)} runs in {self.runs_path}.")

        # Load and filter run data
        grouped_data = self.load_grouped_data(run_dirs=run_dirs)
        if sum(len(v) for v in grouped_data.values()) == 0:
            self.logger.error("No data to plot. Exiting.")
            return

        # Plot grouped data
        self.treat_grouped_data(grouped_data=grouped_data)
        self.logger.info("End.")


@hydra.main(
    config_path=str(Path.cwd() / DIR_CONFIGS),
    config_name=NAME_CONFIG_DEFAULT,
    version_base="1.3.2",
)
def main(config: DictConfig):
    assert_config_dir_exists()
    # Maybe logging.basicConfig(level=logging.INFO) here
    with cProfile.Profile() as pr:
        plotter = PlotFlowPlotter(config)
        plotter.run()
    log_file_cprofile = "logs/profile_stats.prof"
    os.makedirs("logs", exist_ok=True)
    pr.dump_stats(log_file_cprofile)
    logger.info(
        f"Profile stats dumped to {log_file_cprofile}. You can visualize it using 'snakeviz {log_file_cprofile}'"
    )
    logger.info("Export complete.")


if __name__ == "__main__":
    main()
