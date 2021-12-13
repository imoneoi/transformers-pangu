import transformers


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("../data/transformers", trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained("../data/transformers", trust_remote_code=True)

    print(model)
