import transformers


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("../data/transformers", trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained("../data/transformers", trust_remote_code=True)

    text_generator = transformers.TextGenerationPipeline(model, tokenizer)
    print(text_generator("中国和美国和日本和法国和加拿大和澳大利亚的首都分别是哪里？", max_length=50))
