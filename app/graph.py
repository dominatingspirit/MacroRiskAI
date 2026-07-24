from langgraph.graph import StateGraph, END
from typing import TypedDict, Any

from app.schemas.common import AnalysisContext
from app.agents.macro_agent import run_macro_agent
from app.agents.company_health import run_finance_agent
from app.agents.risk_agent import run_stress_engine, run_risk_agent
from app.agents.strategy_agent import run_strategy_agent


class GraphState(TypedDict):
    context: AnalysisContext


def _wrap(agent_fn):
    def node(state: GraphState) -> GraphState:
        state["context"] = agent_fn(state["context"])
        return state
    return node


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("macro_agent", _wrap(run_macro_agent))
    graph.add_node("finance_agent", _wrap(run_finance_agent))
    graph.add_node("stress_engine", _wrap(run_stress_engine))
    graph.add_node("risk_agent", _wrap(run_risk_agent))
    graph.add_node("strategy_agent", _wrap(run_strategy_agent))

    graph.set_entry_point("macro_agent")
    graph.add_edge("macro_agent", "finance_agent")
    graph.add_edge("finance_agent", "stress_engine")
    graph.add_edge("stress_engine", "risk_agent")
    graph.add_edge("risk_agent", "strategy_agent")
    graph.add_edge("strategy_agent", END)

    return graph.compile()


_compiled_graph = None


def get_pipeline():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(context: AnalysisContext) -> AnalysisContext:
    """Runs the full macro -> finance -> stress -> risk -> strategy pipeline."""
    pipeline = get_pipeline()
    result: GraphState = pipeline.invoke({"context": context})
    return result["context"]
