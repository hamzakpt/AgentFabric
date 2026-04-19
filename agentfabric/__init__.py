"""
AgentFabric — Autonomously synthesize a multi-agent network from a single role description.

Quickstart::

    from agentfabric import AgentFabric
    from agentfabric.providers import OpenAIProvider

    # 1. Initialize your LLM provider
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")

    # 2. Initialize AgentFabric
    fabric = AgentFabric(provider)

    # 3. Synthesize a network
    network = fabric.create("Criminal Defense Law Firm")

    # 4. Visualize and query
    network.visualize()
    result = network.query("Draft a motion to suppress illegally obtained evidence.")
    print(result.answer)
"""

from agentfabric.fabric import AgentFabric, FabricNetwork
from agentfabric.core.agent import Agent
from agentfabric.core.network import AgentNetwork
from agentfabric.core.topology import TopologyType
from agentfabric.providers.base import LLMProvider
from agentfabric.providers import (
    AnthropicProvider,
    OpenAIProvider,
    AzureOpenAIProvider,
    GeminiProvider,
    BedrockProvider,
    OllamaProvider,
    HuggingFaceProvider,
    LangChainProvider,
    get_provider,
)

__all__ = [
    # Core
    "AgentFabric",
    "FabricNetwork",
    "Agent",
    "AgentNetwork",
    "TopologyType",
    # Providers
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "GeminiProvider",
    "BedrockProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "LangChainProvider",
    "get_provider",
]

__version__ = "0.1.0"
