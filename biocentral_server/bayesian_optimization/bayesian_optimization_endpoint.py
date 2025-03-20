import hashlib
import json
from flask import current_app

from flask import request, jsonify, Blueprint

from biotrainer.protocols import Protocol
from biotrainer.config import Configurator, ConfigurationException
import yaml

from .bayesian_optimization_task import BayesTask

from ..server_management import (
    TaskManager,
    UserManager,
    FileManager,
    StorageFileType,
    TaskStatus,
)

"""
for this functionality:
    1. take uploaded dataset from the file manager
    2. embed the sequences
    3. launch BO model training process and obtain a ranking based on score
    4. return the data back to client
break into APIs:
    1. upload api: upload dataset to backend
    2. train & inference api: launch training and inference task
    3. query API: obtaining status
    4. fetch result api: fetch results
train & inference api:
    1. take uploaded dataset from the file manager
    2. embed the sequences
    3. launch BO model training process, that run training, inference and 
        save the output to somewhere in file manager when finished
    4. return a handle to query and fetch result for current inference run 
fetch result api: fetch the result according to handle
"""
bayesian_optimization_service_route = Blueprint(
    "bayesian_optimization_service", __name__
)


def verify_config(config_dict: dict):
    """
    ### Configuration Requirements
    - `databsase_hash :: str`
    - `model_type = "gaussian_process"`
    - `coefficient :: float` in `[0, 1]`
    #### Configurating optimization target
    - `discrete :: bool`
    - When `discrete = true`
        - `discrete_labels :: list`
        - `discrete_targets :: list`
        - targets should be true subset of labels
    - When `discrete` = false
        - `target_interval_lb target_interval_ub :: floats` (-inf, inf)
        - representing upper and lower bound of interval
        - `value_preference :: str` enum can take
            - `maximize` that prioritize larger value,
            - `minimize` that prioritize smaller value, or
            - `neutral` when value is not considered in final score
    """
    database_hash: str = config_dict.get("database_hash")
    model_type: str = config_dict.get("model_type").lower()
    coefficient: float = config_dict.get("coefficient")
    if (
        not isinstance(database_hash, str)
        or not isinstance(model_type, str)
        or not isinstance(coefficient, float)
    ):
        raise TypeError(
            "[verify_config]: Config need to include: database_hash :: str, model_type :: str and coefficient :: float"
        )
    if model_type not in BayesTask.SUPPORTED_MODELS:
        raise ValueError(
            f"[verify_config]: unsupported model type. Valid model types: {BayesTask.SUPPORTED_MODELS}"
        )
    if coefficient > 1 or coefficient < 0:
        raise ValueError("[verify_config]: Coefficient should fall in range [0, 1]")
    verify_optim_target(config_dict)


def verify_optim_target(config_dict: dict):
    is_discrete: bool = config_dict.get("discrete")
    if is_discrete is None:
        raise KeyError("[verify_config]: Config need to include: is_discrete :: bool")
    if is_discrete:
        labels = config_dict.get("discrete_labels")
        targets = config_dict.get("discrete_targets")
        if not (labels and targets):
            raise KeyError(
                "[verify_config]: Config for discrete target need to include discrete_labels and discrete_targets field"
            )
        sl, st = set(labels), set(targets)
        if not (sl.issuperset(st) and len(sl) > len(st)):
            raise ValueError("[verify_config]: targets should be true subset of labels")
    else:
        lb = config_dict.get("target_interval_lb")
        ub = config_dict.get("target_interval_ub")
        if not (lb and ub):
            raise KeyError(
                "[verify_config]: Config for continuous target need to include target_interval_lb and target_interval_ub field"
            )
        if lb >= ub:
            raise ValueError(
                "[verify_config]: target_interval_lb should < target_interval_ub"
            )
        value_preference = config_dict.get("value_preference")
        if not value_preference or value_preference not in {
            "maximize",
            "minimize",
            "neutral",
        }:
            raise KeyError(
                "[verify_config]: Config for continuous target need to include value_preference that allow maximize, minimize, or neutral strategy"
            )


@bayesian_optimization_service_route.route(
    "/bayesian_optimization_service/training", methods=["POST"]
)
def train_and_inference():
    # verify configuration dict
    config_dict: dict = request.get_json()
    try:
        verify_config(config_dict)
    except Exception as e:
        return jsonify({"error": str(e)})
    database_hash = config_dict.get("database_hash")
    # fetch util classes
    user_id = UserManager.get_user_id_from_request(req=request)
    file_manager = FileManager(user_id=user_id)
    task_manager = TaskManager()
    # get task id that's unique w.r.t config dict
    task_hash = hashlib.md5(
        json.dumps(config_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()
    task_id = f"biocentral-bayesian_optimization-{task_hash}"
    # get output path
    output_dir = file_manager.get_biotrainer_model_path(
        database_hash=database_hash, model_hash=task_id
    )
    print(f"output_path:{output_dir}")
    # idempotence
    # prepare more training configurations
    config_dict["output_dir"] = str(output_dir.absolute())
    try:
        files = file_manager.get_file_paths_for_biotrainer(
            database_hash=database_hash,
            embedder_name="one_hot_encoding",
            protocol=Protocol.sequence_to_class,
        )
        config_dict["sequence_file"] = files[
            0
        ]  # currently raw fasta file that contain everything
    except FileNotFoundError as e:
        return jsonify({"error": str(e)})
    # launch process
    bo_process = BayesTask(
        config_dict,
        database_instance=current_app.config["EMBEDDINGS_DATABASE"],
    )
    task_manager.add_task(task=bo_process, task_id=task_id)
    return jsonify({"task_id": task_id})


def verify_request(req_body: dict):
    database_hash = req_body.get("database_hash")
    task_id = req_body.get("task_id")
    if (
        database_hash is None
        or task_id is None
        or not isinstance(database_hash, str)
        or not isinstance(task_id, str)
    ):
        raise KeyError(
            "model_results require database_hash :: str and tasl_id :: str in request"
        )


@bayesian_optimization_service_route.route(
    "/bayesian_optimization_service/model_results/", methods=["POST"]
)
def model_results():
    req_body: dict = request.get_json()
    try:
        verify_request(req_body)
    except Exception as e:
        return jsonify({"error": str(e)})
    database_hash = req_body.get("database_hash")
    task_id = req_body.get("task_id")
    task_manager = TaskManager()
    # error could occur if is_task_finished executed before is_task_running
    if not task_manager.is_task_running(task_id) and not task_manager.is_task_finished(
        task_id
    ):
        return jsonify({"error": "Invalid task_id"})
    task_status = task_manager.get_task_status(task_id)
    # print(TaskManager().get_task_dto(task_id))
    # make distinction
    if task_status != TaskStatus.FINISHED:
        return jsonify(
            {"error": "Trying to retrieve model files before task has finished!"}
        )
    user_id = UserManager.get_user_id_from_request(request)
    file_manager = FileManager(user_id=user_id)
    # read from storage/{user_id}/{database_hash}/models/{model_hash}/out.yml
    out_file = (
        file_manager.get_biotrainer_model_path(
            database_hash=database_hash, model_hash=task_id
        )
        / "out.yml"
    )
    if not out_file.exists():
        return jsonify(
            {
                "error": f"Server error: task finished but result file {out_file} not found"
            }
        )
    with out_file.open("r") as f:
        results_data = yaml.load(f, Loader=yaml.FullLoader)
        return jsonify(results_data)
    # model_file_dict = file_manager.get_biotrainer_result_files(
    #     database_hash=database_hash, model_hash=task_id
    # )
    # return jsonify(model_file_dict)
