import os
import requests
from loguru import logger
from pathlib import Path
import hashlib
from hordeling.convert_to_safetensors import download_and_convert_pickletensor, download_created_safetensor
from hordeling import r2
from hordeling import hordeling_redis

class CivitAIModel:

    model_id: int = None
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
    _fault_msg:  str = None
    rc: int = 200

    def __init__(self, model_id):
        self.model_id = model_id
        self.model_metadata = self.retrieve_model_metadata(model_id)
        if self.model_metadata is None:
            return
        self.type = self.model_metadata['type']
        self.name = self.model_metadata['name']
        self.set_safe()
        self.set_safetensor()
        self.set_pickletensor()

    def is_valid(self):
        if self.model_metadata is None:
            return False
        if self.pickletensor_url is None and self.safetensor_url is None:
            return False
        return True


    @property
    def fault_msg(self):
        if self._fault_msg is not None:
            return self._fault_msg
        if self.pickletensor_url is None and self.safetensor_url is None:
            return f"The model '{self.name}' is of an unexpected type"

    def retrieve_model_metadata(self, model_id):
        try:
            civreq = requests.get(f"https://civitai.com/api/v1/models/{model_id}", timeout=5)
            if not civreq.ok:
                self.rc = civreq.status_code
                if civreq.status_code == 404:
                    self._fault_msg = f"Model {model_id} does not exist"
                else:
                    self._fault_msg = f"Error {civreq.status_code} when retrieving CivitAI metadata for {model_id}: {civreq.text}"
                logger.error(self._fault_msg)
                return
            return civreq.json()
        except Exception as err:
            self._fault_msg = f"Exception when retrieving CivitAI metadata for {model_id} with error: {err}"
            logger.error(self._fault_msg)

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
            if f["metadata"]["format"] == "Other" and (f["name"].endswith(".pt") or f["name"].endswith(".bin")):
                f["metadata"]["format"] = "PickleTensor"
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
                logger.info(f"Converted and uploaded {self.name}")
            return r2.generate_safetensor_download_url(self.get_safetensor_filename())

    def get_sha256(self):
        if self.safetensor_url is not None:
            return None
        hash = hordeling_redis.hordeling_r_get(self.model_id)
        if hash is None:
            stpath = Path(self.get_safetensor_filepath())
            if not stpath.exists():
                download_created_safetensor(self)
            return self.store_sha256()
        return hash

    def store_sha256(self):
        hash_object = hashlib.sha256()
        with open(self.get_safetensor_filepath(), "rb") as file:
            while chunk := file.read(8192):  # Read the file in chunks of 8KB
                hash_object.update(chunk)
        sha256 = hash_object.hexdigest()
        hordeling_redis.hordeling_r_set(self.model_id,sha256)
        return sha256
