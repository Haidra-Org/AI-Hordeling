from safetensors.torch import load_file, save_file

import os
import requests
import hashlib
from collections import defaultdict
from pathlib import Path
from loguru import logger
import torch

def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def convert_file(
    pt_filename: str,
    sf_filename: str,
):
    loaded = torch.load(pt_filename, map_location="cpu")
    model_tensors = loaded.get('string_to_param').get('*')

    s_model = {
          'emb_params': model_tensors
            }


    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(s_model, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)

    if not torch.equal(model_tensors, reloaded["emb_params"]):
        raise RuntimeError("The output tensors do not match")

def download_and_convert_pickletensor(pt_url: str, model_metadata: dict):
    response = requests.get(pt_url, timeout=5)
    hash_object = hashlib.sha256()
    hash_object.update(response.content)
    sha256 = hash_object.hexdigest()
    for f in model_metadata["modelVersions"][0]["files"]:
        if f["downloadUrl"] == pt_url:
            if f["hashes"]["SHA256"].lower() != sha256.lower():
                logger.debug([f["hashes"]["SHA256"],hash_object])
                raise Exception("Downloaded file does not match hash")
            else:
                filename = Path("models/" + f["name"])
    os.makedirs(filename.parents[0], exist_ok=True)
    with open(filename, "wb") as outfile:
        outfile.write(response.content)

    convert_file(filename, filename.with_suffix('.safetensors'))

