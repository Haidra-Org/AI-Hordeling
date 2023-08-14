import os
import socket

from loguru import logger
from hordeling.flask import APP
from hordeling.routes import *
from hordeling.apis import apiv1
from hordeling.argparser import args
from hordeling.consts import HORDELING_VERSION
import hashlib

APP.register_blueprint(apiv1)


@APP.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, apikey, Client-Agent, X-Fields"
    response.headers["Hordeling-Node"] = f"{socket.gethostname()}:{args.port}:{HORDELING_VERSION}"
    try:
        etag = hashlib.sha1(response.get_data()).hexdigest()
    except RuntimeError:
        etag = "Runtime Error"
    response.headers["ETag"] = etag
    return response
