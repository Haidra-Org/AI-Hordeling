import os
from flask import Flask, redirect
from flask_caching import Cache
from werkzeug.middleware.proxy_fix import ProxyFix
# from flask_sqlalchemy import SQLAlchemy
from loguru import logger
from hordeling.redis_ctrl import is_redis_up, ger_cache_url

cache = None
APP = Flask(__name__)
APP.wsgi_app = ProxyFix(APP.wsgi_app, x_for=1)

# SQLITE_MODE = os.getenv("USE_SQLITE", "0") == "1"

# if SQLITE_MODE:
#     logger.warning("Using SQLite for database")
#     APP.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///hordeling.db"
# else:
#     APP.config["SQLALCHEMY_DATABASE_URI"] = os.getenv('POSTGRES_URI')
#     APP.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
#         "pool_size": 50,
#         "max_overflow": -1,
#     }
# APP.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(APP)
# db.init_app(APP)

# if not SQLITE_MODE:
#     with APP.app_context():
#         logger.debug("pool size = {}".format(db.engine.pool.size()))
# logger.init_ok("Safetensor API Database", status="Started")

# Allow local workstation run
if is_redis_up():
    try:
        cache_config = {
            "CACHE_REDIS_URL": ger_cache_url(),
            "CACHE_TYPE": "RedisCache",  
            "CACHE_DEFAULT_TIMEOUT": 300
        }
        cache = Cache(config=cache_config)
        cache.init_app(APP)
        logger.init_ok("Flask Cache", status="Connected")
    except Exception as e:
        logger.error(f"Flask Cache Failed: {e}")
        pass

# Allow local workstation run
if cache is None:
    cache_config = {
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 300
    }
    cache = Cache(config=cache_config)
    cache.init_app(APP)
    logger.init_warn("Flask Cache", status="SimpleCache")

