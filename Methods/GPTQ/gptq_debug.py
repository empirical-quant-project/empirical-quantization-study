from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model class:", model.__class__.__name__)
print("=" * 60)

prompts = [
    "write a quick sort algorithm in Python.",
    "write a function in Java that checks if a string is a palindrome."
]

for i, prompt in enumerate(prompts):
    print(f"\n{'=' * 60}")
    print(f"PROMPT {i+1}: {prompt}")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nFormatted input:\n{text[:200]}...")

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"Input token count: {model_inputs.input_ids.shape[1]}")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    print(f"Output token count: {generated_ids.shape[1]}")
    print(f"New tokens generated: {generated_ids.shape[1] - model_inputs.input_ids.shape[1]}")

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\nRESPONSE:\n{response}")

    if not response.strip():
        print("⚠️  WARNING: Empty response!")
        # Debug: decode without skipping special tokens
        raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"Raw response (with special tokens): {repr(raw_response)}")