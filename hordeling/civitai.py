import requests
from loguru import logger

def retrieve_model_metadata(model_id):
    try:
        civreq = requests.get(f"https://civitai.com/api/v1/models/{model_id}", timeout=3)
        if not civreq.ok:
            logger.error(f"Error when retrieving CivitAI metadata for {model_id}: {civreq.text}")
            return
        return civreq.json()
    except Exception as err:
        logger.error(f"Exception when retrieving CivitAI metadata for {model_id} with error: {err}")

def is_safe(model_metadata):
    files = model_metadata["modelVersions"][0]["files"]
    is_ok = True
    for f in files:
        if f["pickleScanResult"] != "Success":
            is_ok = False
    return is_ok

def get_safetensor(model_metadata):
    files = model_metadata["modelVersions"][0]["files"]
    for f in files:
        if f["metadata"]["format"] == "SafeTensor":
            return f["downloadUrl"]
        logger.debug(f["metadata"]["format"])
    return

def get_pickletensor(model_metadata):
    files = model_metadata["modelVersions"][0]["files"]
    for f in files:
        if f["metadata"]["format"] == "PickleTensor":
            return f["downloadUrl"]
    return
