from model.modeling_gptpangu import GPTPanguForCausalLM
from model.configuration_gptpangu import GPTPanguConfig
from model.tokenization_gptpangu import GPTPanguTokenizer


if __name__ == "__main__":
    tokenizer = GPTPanguTokenizer("../data/vocab/vocab.model")

    config = GPTPanguConfig()
    model = GPTPanguForCausalLM(config)

    print(model)
