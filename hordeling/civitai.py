import os
import requests
from loguru import logger
from pathlib import Path
from hordeling.convert_to_safetensors import download_and_convert_pickletensor
from hordeling import r2

class CivitAIModel:

    model_metadata: dict = {}
    type: str = None
    name: str = None
    is_safe: bool = True
    safetensor_url: str = None
    pickletensor_url: str = None
    pickletensor_hash: str = None
    pickletensor_id: str = None
    filename: Path = None
    filepath: Path = None

    def __init__(self, model_id):
        self.model_metadata = self.retrieve_model_metadata(model_id)
        if not self.is_valid():
            return
        self.set_safe()
        self.set_safetensor()
        self.set_pickletensor()
        self.type = self.model_metadata['type']
        self.name = self.model_metadata['name']

    def is_valid(self):
        return self.model_metadata is not None

    def retrieve_model_metadata(self, model_id):
        try:
            civreq = requests.get(f"https://civitai.com/api/v1/models/{model_id}", timeout=5)
            if not civreq.ok:
                logger.error(f"Error when retrieving CivitAI metadata for {model_id}: {civreq.text}")
                return
            return civreq.json()
        except Exception as err:
            logger.error(f"Exception when retrieving CivitAI metadata for {model_id} with error: {err}")

    def set_safe(self):
        files = self.model_metadata["modelVersions"][0]["files"]
        for f in files:
            if f["pickleScanResult"] != "Success":
                self.is_safe = False

    def set_safetensor(self):
        files = self.model_metadata["modelVersions"][0]["files"]
        for f in files:
            if f["metadata"]["format"] == "SafeTensor":
                self.safetensor_url = f["downloadUrl"]
                self.filename = Path(f["name"])
                self.filepath = Path("models/" + f["name"])

    def set_pickletensor(self):
        files = self.model_metadata["modelVersions"][0]["files"]
        for f in files:
            if f["metadata"]["format"] == "PickleTensor":
                self.pickletensor_url = f["downloadUrl"]
                self.pickletensor_hash = f["hashes"]["SHA256"]
                self.filename = Path(f["name"])
                self.filepath = Path("models/" + f["name"])
                self.pickletensor_id = f["id"]

    def get_safetensor_filepath(self):
        # We attach the model filepath id in the filepath to know if it's receiverd a new version
        return f"{self.filepath.parent}/{self.filepath.stem}_{self.pickletensor_id}.safetensors"

    def get_safetensor_filename(self):
        # We attach the model filepath id in the filepath to know if it's receiverd a new version
        return f"{self.filepath.stem}_{self.pickletensor_id}.safetensors"

    def ensure_dir_exists(self):
        os.makedirs(self.filepath.parents[0], exist_ok=True)

    def get_safetensors_download(self):
        if self.safetensor_url is not None:
            return self.safetensor_url
        if self.pickletensor_url:
            if not r2.check_safetensor(self.get_safetensor_filename()):
                download_and_convert_pickletensor(self)
                r2.upload_safetensor(self)
                logger.debug("upload")
            return r2.generate_safetensor_download_url(self.get_safetensor_filename())
