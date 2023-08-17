from safetensors.torch import load_file, save_file

import os
import requests
import hashlib
from collections import defaultdict
from pathlib import Path
from loguru import logger
import torch

from pathlib import Path

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
    input_filename: str,
    safetensors_filename: str,
):
    """Convert a PyTorch model (*.pt or *.bin) to SafeTensors format

    Args:
        input_filename (str): The input file name
        safetensors_filename (str): The file name to save the converted model to

    """
    # Load the model from the input file
    loaded_model = torch.load(input_filename, map_location="cpu")

    # Get the file extension
    extension = Path(input_filename).suffix

    model_to_save = None
    model_tensors = None

    # If the file is a PyTorch .pt file
    if extension == ".pt":
        # Get the model tensors
        model_tensors = loaded_model.get('string_to_param').get('*')

        # Create a dictionary with the embedding parameters
        model_to_save = {
            'emb_params': model_tensors
        }

    # If the file is a fairseq .bin file
    if extension == ".bin":
        # Remove shared weights from the model
        shared = shared_pointers(loaded_model)
        for shared_weights in shared:
            for name in shared_weights[1:]:
                loaded_model.pop(name)
        # Create a dictionary with the model parameters
        model_to_save = {k: v.contiguous() for k, v in loaded_model.items()}
        model_tensors = list(model_to_save.values())[0]

    # Create the output directory if it doesn't exist
    dirname = os.path.dirname(safetensors_filename)
    os.makedirs(dirname, exist_ok=True)
    # Save the model parameters to the output file
    save_file(model_to_save, safetensors_filename, metadata={"format": "pt"})
    # Check that the output file size is not too large
    check_file_size(safetensors_filename, input_filename)
    # Load the saved model parameters to verify that they were saved correctly
    reloaded = load_file(safetensors_filename)

    if "reloaded" in reloaded and not torch.equal(model_tensors, reloaded["emb_params"]):
        raise RuntimeError("The output tensors do not match")
    elif not torch.equal(model_tensors, reloaded.popitem()[1]):
        raise RuntimeError("The output tensors do not match")


def  download_and_convert_pickletensor(civitai_model):
    response = requests.get(civitai_model.pickletensor_url, timeout=5)
    hash_object = hashlib.sha256()
    hash_object.update(response.content)
    sha256 = hash_object.hexdigest()
    if civitai_model.pickletensor_hash.lower() != sha256.lower():
        raise Exception("Downloaded file does not match hash")
    civitai_model.ensure_dir_exists()
    with open(civitai_model.filepath, "wb") as outfile:
    # with open("negative_hand-neg.pt", "wb") as outfile:
        w = outfile.write(response.content)
    convert_file(civitai_model.filepath, civitai_model.get_safetensor_filepath())
