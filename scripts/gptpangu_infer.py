from model.modeling_gptpangu import GPTPanguForCausalLM
from model.configuration_gptpangu import GPTPanguConfig
from model.tokenization_gptpangu import GPTPanguTokenizer


if __name__ == "__main__":
    tokenizer = GPTPanguTokenizer("./data/vocab/vocab.model")
    model = GPTPanguForCausalLM.from_pretrained("data/transformers")

    model(tokenizer("我是"))
