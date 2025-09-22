import time
import openai

client = openai.OpenAI(api_key="xxxx")
# ================================= #


def ask_LLM(model, messages, temperature, max_retry):
    for retry in range(max_retry + 1):
        try:
            if temperature > 0:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if retry == max_retry:
                raise
            wait = 2 ** retry
            print(f"[WARN] OpenAI exception: {e}. retrying in {wait}s …")
            time.sleep(wait)
