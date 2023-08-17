import os
from flask import request
from flask_restx import Namespace, Resource, reqparse
from hordeling.flask import cache
from loguru import logger
from hordeling import exceptions as e
from hordeling.civitai import CivitAIModel

api = Namespace('v1', 'API Version 1' )

from hordeling.apis.models.v1 import Models

models = Models(api)

handle_bad_request = api.errorhandler(e.BadRequest)(e.handle_bad_requests)
handle_forbidden = api.errorhandler(e.Forbidden)(e.handle_bad_requests)
handle_unauthorized = api.errorhandler(e.Unauthorized)(e.handle_bad_requests)
handle_not_found = api.errorhandler(e.NotFound)(e.handle_bad_requests)
handle_internal_server_error = api.errorhandler(e.InternalServerError)(e.handle_bad_requests)
handle_service_unavailable = api.errorhandler(e.ServiceUnavailable)(e.handle_bad_requests)

# Used to for the flask limiter, to limit requests per url paths
def get_request_path():
    # logger.info(dir(request))
    return f"{request.remote_addr}@{request.method}@{request.path}"


class Embedding(Resource):
    get_parser = reqparse.RequestParser()
    get_parser.add_argument("Client-Agent", default="unknown:0:unknown", type=str, required=False, help="The client name and version.", location="headers")

    @api.expect(get_parser)
    @cache.cached(timeout=10, query_string=True)
    @api.marshal_with(models.response_model_download_url, code=200, description='Download URL', skip_none=True)
    def get(self, model_id: str):
        '''Ensure the download URL for an embedding is a safetensor
        '''
        self.args = self.get_parser.parse_args()
        if not model_id.isdigit():
            raise e.BadRequest("You can only pass CivitAI mdoel IDs")
        model: CivitAIModel = CivitAIModel(model_id)
        if not model.is_valid():
            raise e.ServiceUnavailable("Problem retrieving model info from CivitAI, please try again later.")
        if not model:
            raise e.BadRequest(f"{model.name} has not passed the CivitAI pickle scanner succesfully")
        if model.type != "TextualInversion":
            raise e.BadRequest(f"{model.name} is not an Embedding / Textual Inversion")
        return {"url": model.get_safetensors_download()},200
