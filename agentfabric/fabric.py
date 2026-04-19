"""
AgentFabric: the top-level public API.

Usage::

    from agentfabric import AgentFabric
    from agentfabric.providers import OpenAIProvider

    # 1. Initialize your chosen LLM provider
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")

    # 2. Initialize AgentFabric with that provider
    fabric = AgentFabric(provider)

    # 3. Synthesize a network from a role description
    network = fabric.create("Criminal Defense Law Firm")

    # 4. Visualize and query
    network.visualize()
    result = network.query("Draft a motion to suppress illegally obtained evidence.")
    print(result.answer)
"""

from __future__ import annotations

import asyncio
from typing import Optional

from agentfabric.core.architect import MetaArchitect, NetworkBlueprint
from agentfabric.core.factory import AgentFactory
from agentfabric.core.network import AgentNetwork, NetworkQueryResult
from agentfabric.providers.base import LLMProvider
from agentfabric.utils.logger import get_logger

logger = get_logger(__name__)


class AgentFabric:
    """
    Initialize AgentFabric with an LLM provider, then synthesize agent networks.

    The provider is the LLM that powers both the Meta-Architect (role
    decomposition) and every individual agent in the synthesized network.
    Initialize the provider class of your choice and pass it in — the rest
    is handled automatically.

    Parameters
    ----------
    provider:
        An initialized ``LLMProvider`` instance. Import and initialize
        whichever provider you need before passing it here.

    Examples::

        # ── OpenAI ──────────────────────────────────────────────────────
        from agentfabric import AgentFabric
        from agentfabric.providers import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")
        fabric = AgentFabric(provider)
        network = fabric.create("Criminal Defense Law Firm")

        # ── Anthropic / Claude ──────────────────────────────────────────
        from agentfabric.providers import AnthropicProvider

        provider = AnthropicProvider(api_key="sk-ant-...", model="claude-opus-4-7")
        fabric = AgentFabric(provider)
        network = fabric.create("Hospital Emergency Department")

        # ── Azure OpenAI ─────────────────────────────────────────────────
        from agentfabric.providers import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            azure_endpoint="https://my-resource.openai.azure.com/",
            azure_deployment="gpt-4o-prod",
            api_key="...",
        )
        fabric = AgentFabric(provider)
        network = fabric.create("Law Firm")

        # ── Google Gemini ────────────────────────────────────────────────
        from agentfabric.providers import GeminiProvider

        provider = GeminiProvider(api_key="AIza...", model="gemini-1.5-pro")
        fabric = AgentFabric(provider)
        network = fabric.create("Research Lab")

        # ── AWS Bedrock ──────────────────────────────────────────────────
        from agentfabric.providers import BedrockProvider

        provider = BedrockProvider(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",
        )
        fabric = AgentFabric(provider)
        network = fabric.create("Healthcare Network")

        # ── Ollama (local, no API key) ───────────────────────────────────
        from agentfabric.providers import OllamaProvider

        provider = OllamaProvider(model="llama3.1")
        fabric = AgentFabric(provider)
        network = fabric.create("Software Team")

        # ── HuggingFace ──────────────────────────────────────────────────
        from agentfabric.providers import HuggingFaceProvider

        provider = HuggingFaceProvider(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            api_key="hf_...",
        )
        fabric = AgentFabric(provider)
        network = fabric.create("Research Lab")

        # ── Any LangChain BaseChatModel ──────────────────────────────────
        from langchain_mistralai import ChatMistralAI
        from agentfabric.providers import LangChainProvider

        provider = LangChainProvider(ChatMistralAI(api_key="..."))
        fabric = AgentFabric(provider)
        network = fabric.create("Software Team")

        # ── Reuse one fabric instance to create multiple networks ─────────
        fabric = AgentFabric(OpenAIProvider(api_key="sk-..."))
        law_firm   = fabric.create("Criminal Defense Law Firm")
        hospital   = fabric.create("Hospital Emergency Department")
        school     = fabric.create("High School Operations")
    """

    def __init__(self, provider: LLMProvider) -> None:
        if not isinstance(provider, LLMProvider):
            raise TypeError(
                f"AgentFabric expects an initialized LLMProvider instance, "
                f"got {type(provider).__name__!r}.\n\n"
                "Example:\n"
                "  from agentfabric.providers import OpenAIProvider\n"
                "  provider = OpenAIProvider(api_key='sk-...')\n"
                "  fabric = AgentFabric(provider)"
            )
        self._provider = provider
        self._architect = MetaArchitect(provider=provider)
        self._factory = AgentFactory(provider=provider)

    @property
    def provider(self) -> LLMProvider:
        """The LLM provider powering this AgentFabric instance."""
        return self._provider

    # ------------------------------------------------------------------
    # Network synthesis
    # ------------------------------------------------------------------

    def create(self, meta_role: str) -> "FabricNetwork":
        """
        Synthesize a multi-agent network from a role description (synchronous).

        Parameters
        ----------
        meta_role:
            A natural-language description of the organization or role to model.
            Examples: ``"Criminal Defense Law Firm"``, ``"E-commerce Platform"``,
            ``"Hospital Emergency Department"``, ``"Software Engineering Team"``.

        Returns
        -------
        FabricNetwork
            The synthesized, ready-to-query agent network.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed into a valid blueprint.

        Example::

            fabric = AgentFabric(OpenAIProvider(api_key="sk-..."))
            network = fabric.create("Criminal Defense Law Firm")
            print(network.describe())
        """
        logger.info(f"Creating network for role: {meta_role!r}")
        blueprint = self._architect.decompose_sync(meta_role)
        agents = self._factory.build(blueprint)
        network = AgentNetwork(blueprint=blueprint, agents=agents)
        return FabricNetwork(network=network, blueprint=blueprint)

    async def create_async(self, meta_role: str) -> "FabricNetwork":
        """
        Async version of :meth:`create`.

        Use this inside an ``async`` function or event loop::

            fabric = AgentFabric(OpenAIProvider(api_key="sk-..."))
            network = await fabric.create_async("Hospital Emergency Department")
            result  = await network.query_async("Patient triage protocol?")
        """
        logger.info(f"Creating network (async) for role: {meta_role!r}")
        blueprint = await self._architect.decompose(meta_role)
        agents = self._factory.build(blueprint)
        network = AgentNetwork(blueprint=blueprint, agents=agents)
        return FabricNetwork(network=network, blueprint=blueprint)

    def __repr__(self) -> str:
        return f"AgentFabric(provider={self._provider!r})"


# ---------------------------------------------------------------------------
# FabricNetwork — the object returned by fabric.create()
# ---------------------------------------------------------------------------

class FabricNetwork:
    """
    A synthesized, ready-to-use multi-agent network.

    Returned by :meth:`AgentFabric.create`. Holds all instantiated agents
    and exposes methods to query and visualize the network.
    """

    def __init__(self, network: AgentNetwork, blueprint: NetworkBlueprint) -> None:
        self._network = network
        self._blueprint = blueprint

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
    ) -> NetworkQueryResult:
        """
        Route a query through the agent network.

        Parameters
        ----------
        user_query:
            The question or task to send to the network.
        entry_agent:
            Name of the agent to handle the query first. If omitted the
            network picks the best entry point automatically (hub for star
            topology, source node for pipeline, etc.).
        broadcast:
            If ``True``, all agents receive the query in parallel and their
            responses are merged into a single result.

        Returns
        -------
        NetworkQueryResult
            Exposes ``.answer`` (primary text), ``.all_responses`` (every
            agent's response), and ``.routed_path`` (list of agent names that
            handled the query).

        Example::

            result = network.query("Draft a motion to suppress evidence.")
            print(result.answer)
            print(result.full_report())   # each agent's individual response
        """
        return self._network.query(user_query, entry_agent=entry_agent, broadcast=broadcast)

    async def query_async(
        self,
        user_query: str,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
    ) -> NetworkQueryResult:
        """Async version of :meth:`query`."""
        return await self._network.query_async(
            user_query, entry_agent=entry_agent, broadcast=broadcast
        )

    def describe(self) -> str:
        """Return a human-readable summary of the network (agents + topology)."""
        return self._network.describe()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        backend: str = "mermaid",
        output_path: Optional[str] = None,
        fmt: str = "svg",
        view: bool = False,
    ) -> str:
        """
        Visualize the agent network.

        Parameters
        ----------
        backend:
            ``"mermaid"`` — text diagram, no extra deps, paste into
            https://mermaid.live for interactive view.
            ``"graphviz"`` — SVG/PNG/PDF export, requires
            ``pip install agentfabric[visualization]``.
        output_path:
            File to save the diagram. Prints to stdout if omitted (mermaid).
        fmt:
            Graphviz output format: ``"svg"``, ``"png"``, or ``"pdf"``.
        view:
            Open the rendered file automatically (graphviz only).

        Returns
        -------
        str
            Diagram string (mermaid) or saved file path (graphviz).
        """
        if backend == "mermaid":
            return self._visualize_mermaid(output_path)
        if backend == "graphviz":
            return self._visualize_graphviz(output_path or "agent_network", fmt, view)
        raise ValueError(
            f"Unknown visualization backend {backend!r}. Use 'mermaid' or 'graphviz'."
        )

    def mermaid(self) -> str:
        """Return the Mermaid diagram string directly (no I/O side effects)."""
        from agentfabric.visualization.mermaid import MermaidRenderer
        return MermaidRenderer().render(self._blueprint)

    def _visualize_mermaid(self, output_path: Optional[str]) -> str:
        from agentfabric.visualization.mermaid import MermaidRenderer
        renderer = MermaidRenderer()
        diagram = renderer.render_to_markdown(self._blueprint)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(diagram)
            print(f"Mermaid diagram saved to: {output_path}")
        else:
            print(diagram)
        return diagram

    def _visualize_graphviz(self, output_path: str, fmt: str, view: bool) -> str:
        from agentfabric.visualization.graphviz import GraphvizRenderer
        return GraphvizRenderer().render_to_file(
            self._blueprint, output_path, fmt=fmt, view=view
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def meta_role(self) -> str:
        return self._blueprint.meta_role

    @property
    def topology(self):
        return self._blueprint.topology_type

    @property
    def blueprint(self) -> NetworkBlueprint:
        return self._blueprint

    @property
    def agents(self):
        return self._network.agents

    @property
    def agent_names(self) -> list[str]:
        return self._network.agent_names

    def get_agent(self, name: str):
        return self._network.get_agent(name)

    def __repr__(self) -> str:
        return (
            f"FabricNetwork("
            f"meta_role={self.meta_role!r}, "
            f"topology={self.topology.value}, "
            f"agents={self.agent_names})"
        )
