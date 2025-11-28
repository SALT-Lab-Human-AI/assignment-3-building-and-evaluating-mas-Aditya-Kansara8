from openai import OpenAI
import os

# Use environment variable for API key in tests
# Set VLLM_API_KEY environment variable before running
# pragma: allowlist secret
client = OpenAI(
    base_url="https://vllm.salt-lab.org/v1",
    api_key=os.getenv("VLLM_API_KEY", "")  # pragma: allowlist secret
)

result = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain what MXFP4 quantization is."}
    ]
)

print(result.choices[0].message.content)

response = client.responses.create(
    model="openai/gpt-oss-20b",
    instructions="You are a helfpul assistant.",
    input="Explain what MXFP4 quantization is."
)

print(response.output_text)
