"""
Example: All Supported LLM Providers

Shows how to initialize AgentFabric with every supported LLM provider.
The API is identical regardless of which provider you choose —
only the initialization step differs.
"""

from agentfabric import AgentFabric

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Initialize the LLM provider of your choice
# ─────────────────────────────────────────────────────────────────────────────

# ── OpenAI (default recommended) ────────────────────────────────────────────
from agentfabric.providers import OpenAIProvider
provider = OpenAIProvider(
    api_key="sk-...",          # or set OPENAI_API_KEY env var
    model="gpt-4o",            # or "gpt-4o-mini", "o1", "o3-mini"
)

# ── Anthropic / Claude ───────────────────────────────────────────────────────
# from agentfabric.providers import AnthropicProvider
# provider = AnthropicProvider(
#     api_key="sk-ant-...",          # or set ANTHROPIC_API_KEY env var
#     model="claude-sonnet-4-6",     # or "claude-opus-4-7", "claude-haiku-4-5-20251001"
# )

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
# pip install agentfabric[azure]
# from agentfabric.providers import AzureOpenAIProvider
# provider = AzureOpenAIProvider(
#     azure_endpoint="https://my-resource.openai.azure.com/",
#     azure_deployment="gpt-4o-prod",   # your deployment name in Azure portal
#     api_key="your-azure-key",
#     api_version="2024-02-01",
# )

# ── Google Gemini ────────────────────────────────────────────────────────────
# pip install agentfabric[gemini]
# from agentfabric.providers import GeminiProvider
# provider = GeminiProvider(
#     api_key="AIza...",             # or set GOOGLE_API_KEY env var
#     model="gemini-1.5-pro",        # or "gemini-2.0-flash", "gemini-2.5-pro"
# )

# ── AWS Bedrock ──────────────────────────────────────────────────────────────
# pip install agentfabric[bedrock]
# from agentfabric.providers import BedrockProvider
# provider = BedrockProvider(
#     model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     region_name="us-east-1",
#     # aws_access_key_id and aws_secret_access_key optional — uses credential chain
# )

# ── Ollama (local, no API key needed) ────────────────────────────────────────
# pip install agentfabric[ollama]
# ollama serve && ollama pull llama3.1
# from agentfabric.providers import OllamaProvider
# provider = OllamaProvider(
#     model="llama3.1",              # or "mistral", "phi3", "qwen2.5", "gemma2"
#     host="http://localhost:11434",
# )

# ── HuggingFace Inference API ────────────────────────────────────────────────
# pip install agentfabric[huggingface]
# from agentfabric.providers import HuggingFaceProvider
# provider = HuggingFaceProvider(
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     api_key="hf_...",              # or set HF_TOKEN env var
# )

# ── Any LangChain BaseChatModel ──────────────────────────────────────────────
# pip install langchain-core langchain-mistralai
# from langchain_mistralai import ChatMistralAI
# from agentfabric.providers import LangChainProvider
# provider = LangChainProvider(
#     ChatMistralAI(api_key="...", model="mistral-large-latest")
# )

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Initialize AgentFabric with your provider
# ─────────────────────────────────────────────────────────────────────────────
fabric = AgentFabric(provider)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Synthesize networks and query — same API for every provider
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # You can create multiple networks from one fabric instance
    network = fabric.create("Criminal Defense Law Firm")

    print(network.describe())
    network.visualize(backend="mermaid")

    result = network.query(
        "A client is charged with drug possession. The evidence was seized without a warrant. "
        "What are the strongest legal defenses available?"
    )
    print(result.answer)


if __name__ == "__main__":
    main()
