import mindspore as ms
import numpy as np

import os


def convert_mindspore_checkpoint(
        input_file: str,
        output_path: str,
):
    parameters = ms.load_checkpoint(input_file)

    # dump numpy
    for k, v in parameters.items():
        v_numpy = v.asnumpy()
        print("{}: {}".format(k, v_numpy.shape))
        np.save(os.path.join(output_path, "{}.npy".format(k)), v_numpy)


def main():
    convert_mindspore_checkpoint("data/Pangu-alpha_2.6B.ckpt", "data/checkpoint_numpy")


if __name__ == "__main__":
    main()
