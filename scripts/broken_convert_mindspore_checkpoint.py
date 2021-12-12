import mindspore as ms
import numpy as np
from tqdm import tqdm

import os
import re


def convert_mindspore_checkpoint(
        input_path: str,
        output_path: str,

        input_pattern: re.Pattern = r"filerted_(\d+)\.ckpt",
):
    # load all chunks
    chunks = {}
    for filename in tqdm(os.listdir(input_path)):
        path = os.path.join(input_path, filename)
        if not os.path.isfile(path):
            continue
        m = re.match(input_pattern, filename)
        if not m:
            continue
        chunks[int(m.group(1))] = ms.load_checkpoint(path)

    # concat parameters
    parameters = {}
    for k in tqdm(chunks[0].keys()):
        is_same = (chunks[0][k] == chunks[1][k]).all()
        if is_same:
            parameters[k] = chunks[0][k]
        else:
            parameters[k] = ms.ops.Concat(axis=0)([chunks[idx][k] for idx in range(len(chunks))])

    # dump numpy
    for k, v in parameters.items():
        v_numpy = v.asnumpy()
        print("{}: {}".format(k, v_numpy.shape))
        np.save(os.path.join(output_path, "{}.npy".format(k)), v_numpy)


def main():
    convert_mindspore_checkpoint("data/checkpoint", "data/checkpoint_numpy")


if __name__ == "__main__":
    main()
