# Integrations

Wrap each model request in `et.llm_call(model, provider=...)` and pass the response to `call.record(...)`.
Token counts are read from the response; cost comes from your pricing file.

## OpenAI

```python
from openai import OpenAI
import llm_agent_eval_toolkit as et

client = OpenAI()

with et.llm_call("gpt-4o-mini", provider="openai") as call:
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    call.record(resp)  # usage.prompt_tokens / usage.completion_tokens; resp.model is used for pricing
```

The Responses API (`client.responses.create`) works the same way (`usage.input_tokens` / `usage.output_tokens`).

## Anthropic

```python
import anthropic
client = anthropic.Anthropic()

with et.llm_call("claude-sonnet-4", provider="anthropic") as call:
    msg = client.messages.create(model="claude-sonnet-4", max_tokens=1024, messages=messages)
    call.record(msg)
```

## Google Gemini

```python
from google import genai
client = genai.Client()

with et.llm_call("gemini-2.0-flash", provider="gcp.gemini") as call:
    resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    call.record(resp)  # usage_metadata.prompt_token_count / candidates_token_count
```

## Embeddings and other operations

Set `operation` so spans are named and filtered correctly:

```python
with et.llm_call("text-embedding-3-small", provider="openai", operation="embeddings") as call:
    emb = client.embeddings.create(model="text-embedding-3-small", input=chunks)
    call.record(input_tokens=emb.usage.prompt_tokens, output_tokens=0)
```

## Streaming

Record usage once the stream ends. With OpenAI, ask for it with `stream_options={"include_usage": True}`:

```python
with et.llm_call("gpt-4o-mini", provider="openai") as call:
    stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True,
                                            stream_options={"include_usage": True})
    for chunk in stream:
        ...
        if chunk.usage:
            call.record(chunk)
```

Latency is measured until the `with` block exits, so it covers the whole stream.

## Async code and web servers

`traced_agent`, `traced_step` and `traced_llm` work on `async def` functions. In a web server, decorate the
request handler's agent function with `traced_agent` so each request becomes one trace, and use
`et.configure(..., batch=True)` so spans are exported in the background.

## A function that returns the provider response

```python
@et.traced_llm("gpt-4o-mini", provider="openai")
def complete(messages):
    return client.chat.completions.create(model="gpt-4o-mini", messages=messages)
```

## Existing OpenTelemetry setup

If your app already configures a `TracerProvider`, pass it in and the toolkit adds its exporters to it:

```python
et.configure("support-agent", tracer_provider=provider, store="agent_eval.db")
```
