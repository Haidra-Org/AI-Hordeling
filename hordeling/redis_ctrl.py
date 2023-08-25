import os
import socket
import redis
import json

from loguru import logger

redis_hostname = os.getenv('REDIS_IP', "localhost")
redis_port = 6379
redis_address = f"redis://{redis_hostname}:{redis_port}"

hordeling_db = 0
limiter_db = 1
cache_db = 2

def is_redis_up() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((redis_hostname, redis_port)) == 0

def is_local_redis_up() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", 6379)) == 0

def ger_limiter_url():
    return(f"{redis_address}/{limiter_db}")

def ger_cache_url():
    return(f"{redis_address}/{cache_db}")

def get_hordeling_db():
    return redis.Redis(
        host=redis_hostname,
        port=redis_port,
        db = hordeling_db,
        decode_responses=True)

def get_local_hordeling_db():
    return redis.Redis(
        host="127.0.0.1",
        port=6379,
        db = 6,
        decode_responses=True)

def get_redis_db_server(server_ip):
    return redis.Redis(
        host=server_ip,
        port=redis_port,
        db = hordeling_db,
        decode_responses=True)

def get_all_redis_db_servers():
    """An array of all the redis servers in the cluster
    We use this to always store the entries in all servers
    This allows redis to transparently failover.
    """
    # Currently only local redis supported
    return [get_hordeling_db()]
    # try:
    #     return [get_redis_db_server(rs) for rs in json.loads(os.getenv("REDIS_SERVERS"))]
    # except:
    #     logger.error(f"Error setting up REDIS_SERVERS array. Falling back to loadbalancer.")
    #     return [get_hordeling_db()]
