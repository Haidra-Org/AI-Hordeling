import hordeling.apis.v1.base as base
from hordeling.apis.v1.base import api

api.add_resource(base.Embedding, "/embedding/<string:model_id>")
