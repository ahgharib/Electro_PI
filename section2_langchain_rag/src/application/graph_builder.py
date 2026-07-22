from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.application.graph_nodes import GraphNodes
from src.application.graph_state import RagState
from src.core.models import GateStatus


def build_graph(nodes: GraphNodes):
    """Assembles the RAG graph.

        retrieve -> gate --(PASS)-----> generate -> END
                         `--(else)-----> short_circuit -> END

    The short_circuit branch never calls the LLM -- this is the structural
    guarantee that the model cannot hallucinate an answer when there's no
    relevant context (see NOTES.md, "no relevant context found").
    """
    graph = StateGraph(RagState)

    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("gate", nodes.gate)
    graph.add_node("generate", nodes.generate)
    graph.add_node("short_circuit", nodes.short_circuit)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "gate")
    graph.add_conditional_edges(
        "gate",
        lambda state: "generate" if state["gate_status"] == GateStatus.PASS else "short_circuit",
        {"generate": "generate", "short_circuit": "short_circuit"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("short_circuit", END)

    return graph.compile()
