from model.modeling_gptpangu import GPTPanguForCausalLM
from model.configuration_gptpangu import GPTPanguConfig
from model.tokenization_gptpangu import GPTPanguTokenizer

import os

import torch
import numpy as np


def convert_numpy_to_transformers(
        input_path,
        output_path,

        tokenizer_path = "data/vocab/vocab.model"
):
    # load all numpy parameters
    print("Loading numpy parameters...")
    src_parameters = {}
    for filename in os.listdir(input_path):
        path = os.path.join(input_path, filename)
        if os.path.isfile(path) and path.endswith(".npy"):
            key = filename[:-len(".npy")]
            src_parameters[key] = np.load(path)

            print("{}: {} {}".format(key, src_parameters[key].shape, src_parameters[key].dtype))

    # create model
    print("Creating model...")
    config = GPTPanguConfig()
    model = GPTPanguForCausalLM(config)
    tokenizer = GPTPanguTokenizer(tokenizer_path)

    # load state dict routines
    print("Loading state dict...")
    dst_state_dict = model.state_dict()
    src_loaded_keys = []
    dst_loaded_keys = []

    def load_to(dst_key, src_key, transpose=False):
        # type checks
        assert src_parameters[src_key].dtype == np.float32
        assert dst_state_dict[dst_key].dtype == torch.float32

        # check not loaded already
        assert src_key not in src_loaded_keys
        assert dst_key not in dst_loaded_keys
        src_loaded_keys.append(src_key)
        dst_loaded_keys.append(dst_key)
        # load
        x = torch.from_numpy(src_parameters[src_key])
        if transpose:
            x = x.transpose(0, 1)
        dst_state_dict[dst_key].copy_(x)

    # start load
    load_to("transformer.wte.weight", "backbone.word_embedding.embedding_table")
    load_to("transformer.wpe.weight", "backbone.position_embedding.embedding_table")
    load_to("transformer.wqe.weight", "backbone.top_query_embedding.embedding_table")
    for layer_idx in range(config.num_layers):
        src_prefix = "backbone.blocks.{}.".format(layer_idx)
        if layer_idx == config.num_layers - 1:
            src_prefix = "backbone.top_query_layer."
        dst_prefix = "transformer.h.{}.".format(layer_idx)
        # layer norm
        load_to(dst_prefix + "ln_1.weight", src_prefix + "layernorm1.gamma")
        load_to(dst_prefix + "ln_1.bias", src_prefix + "layernorm1.beta")
        load_to(dst_prefix + "ln_2.weight", src_prefix + "layernorm2.gamma")
        load_to(dst_prefix + "ln_2.bias", src_prefix + "layernorm2.beta")
        # attention
        # Attention QKV layers use nn.Dense, same format as PyTorch
        load_to(dst_prefix + "attn.q_proj.weight", src_prefix + "attention.dense1.weight", transpose=False)
        load_to(dst_prefix + "attn.q_proj.bias", src_prefix + "attention.dense1.bias")

        load_to(dst_prefix + "attn.k_proj.weight", src_prefix + "attention.dense2.weight", transpose=False)
        load_to(dst_prefix + "attn.k_proj.bias", src_prefix + "attention.dense2.bias")

        load_to(dst_prefix + "attn.v_proj.weight", src_prefix + "attention.dense3.weight", transpose=False)
        load_to(dst_prefix + "attn.v_proj.bias", src_prefix + "attention.dense3.bias")

        # Projection layers use Mapping, require transposition
        load_to(dst_prefix + "attn.out_proj.weight", src_prefix + "attention.projection.weight", transpose=True)
        load_to(dst_prefix + "attn.out_proj.bias", src_prefix + "attention.projection.bias")

        # mlp
        # MLP layers use Output (Mapping and Mapping_output inside), require transposition
        load_to(dst_prefix + "mlp.c_proj.weight", src_prefix + "output.projection.weight", transpose=True)
        load_to(dst_prefix + "mlp.c_proj.bias", src_prefix + "output.projection.bias")

        load_to(dst_prefix + "mlp.c_fc.weight", src_prefix + "output.mapping.weight", transpose=True)
        load_to(dst_prefix + "mlp.c_fc.bias", src_prefix + "output.mapping.bias")

    load_to("transformer.ln_f.weight", "backbone.layernorm.gamma")
    load_to("transformer.ln_f.bias", "backbone.layernorm.beta")

    # Special: tie embedding to output
    dst_state_dict["lm_head.weight"].copy_(dst_state_dict["transformer.wte.weight"])
    dst_loaded_keys.append("lm_head.weight")

    # print not loaded
    print("Source not loaded fields: {}".format([k for k in src_parameters.keys() if k not in src_loaded_keys]))
    print("Dest not loaded fields: {}".format([k for k in dst_state_dict.keys() if k not in dst_loaded_keys]))

    # Test inference
    with torch.no_grad():
        prompt = "问：中国的首都是哪里？\n答："
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = [tokenizer.decode(model.generate(input_ids, max_length=20, top_p=0.9, do_sample=True, top_k=0)[0].numpy().tolist())
                  for _ in range(5)]
        print(output)

    # Save model
    model.save_pretrained(output_path)


def main():
    convert_numpy_to_transformers("data/checkpoint_numpy", "data/transformers")


if __name__ == "__main__":
    main()
