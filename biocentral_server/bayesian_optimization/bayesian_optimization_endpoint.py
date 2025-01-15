import asyncio
import threading

import yaml
import flask

from flask import request, jsonify, Blueprint

from biotrainer.protocols import Protocol
from biotrainer.config import Configurator, ConfigurationException

from ..prediction_models import BiotrainerProcess

from ..server_management import ProcessManager, UserManager, FileManager, \
    StorageFileType, TaskStatus

bayesian_optimization_service_route = Blueprint("bayesian_optimization_service", __name__)

@bayesian_optimization_service_route.route('/bayesian_optimization_service/inference/<model_hash>/<ee_coeff>', methods=['GET'])
def inference(model_hash, ee_coeff):
    # fetch model by hash
    # a way to get candidates. for now, candidates and training data are in the same dataset
    # inferencing: query model & output sorted result w.r.t ee_coefficient
    return jsonify({model_hash: ee_coeff})
