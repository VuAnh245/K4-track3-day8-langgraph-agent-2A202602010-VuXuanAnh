import os
from typing import Any

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Pydantic model for structured classification output ──────────────
class ClassificationOutput(BaseModel):
    route: str = Field(
        description="The classified route. Must be exactly one of: 'simple', 'tool', 'missing_info', 'risky', 'error'."
    )
    risk_level: str = Field(
        description="The risk level: 'high' for risky actions (refunds, cancellations, deletions, email sending), 'low' otherwise."
    )
    reasoning: str = Field(description="Brief reasoning behind the classification.")


CLASSIFY_PROMPT = """You are a support-ticket routing system for a customer service agent.
Classify the user's query into EXACTLY ONE of the following 5 categories based on intent:

1. 'risky': High-risk actions with side effects like refunding customers, deleting accounts/data, sending emails, cancelling subscriptions.
2. 'tool': Information lookup queries requiring database or external service search (order status, tracking info, user details).
3. 'missing_info': Vague, ambiguous, or incomplete queries lacking actionable context (e.g., "Can you fix it?", "Fix it").
4. 'error': Explicit reports of system/service failures, timeouts, crashes, unrecoverable errors (e.g., "Timeout failure while processing request").
5. 'simple': General questions or how-to queries answerable directly without tools or side-effects (e.g., "How do I reset my password?").

Priority order when overlapping: risky > tool > missing_info > error > simple.

User Query: {query}
"""


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify the query into a route using an LLM structured output."""
    query = state.get("query", "").strip()
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ClassificationOutput)
    result: ClassificationOutput = structured_llm.invoke(CLASSIFY_PROMPT.format(query=query))

    route = result.route.lower().strip()
    valid_routes = {"simple", "tool", "missing_info", "risky", "error"}
    if route not in valid_routes:
        route = "simple"

    risk_level = "high" if route == "risky" or result.risk_level.lower() == "high" else "low"

    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified query as '{route}'",
                route=route,
                risk_level=risk_level,
            )
        ],
    }


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a mock tool call with simulated transient errors."""
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    query = state.get("query", "")

    if route == "error" and attempt < 2:
        result_str = (
            f"ERROR: Transient failure while processing query '{query}' (attempt {attempt})"
        )
    else:
        result_str = f"Tool Result: Data retrieved successfully for '{query}'."

    return {
        "tool_results": [result_str],
        "events": [
            make_event("tool", "executed", f"tool executed (attempt {attempt})", result=result_str)
        ],
    }


def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate tool results to decide retry vs success."""
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    if "ERROR" in latest_result:
        evaluation_result = "needs_retry"
    else:
        evaluation_result = "success"

    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"evaluation result: {evaluation_result}",
                evaluation_result=evaluation_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a final grounded response using an LLM."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval", {})

    context = f"User Query: {query}\n"
    if tool_results:
        context += f"Tool Results: {' | '.join(tool_results)}\n"
    if approval:
        context += f"Approval Decision: {approval}\n"

    prompt = f"""You are a helpful customer support agent. Generate a concise, polite, and grounded answer for the user query based ONLY on the provided context.

Context:
{context}

Response:"""

    try:
        llm = get_llm(temperature=0.2)
        res = llm.invoke(prompt)
        answer_text = str(res.content).strip()
    except Exception:
        answer_text = f"Thank you for contacting support regarding '{query}'. Your request has been processed."

    return {
        "final_answer": answer_text,
        "events": [make_event("answer", "completed", "generated final answer")],
    }


def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    pending_question = (
        f"Could you please provide more details or context regarding your request: '{query}'?"
    )
    return {
        "pending_question": pending_question,
        "final_answer": pending_question,
        "events": [make_event("clarify", "completed", "requested clarification")],
    }


def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed_action = f"Proposed Action: Perform high-risk operation for query '{query}' (requires human approval)."
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event("risky_action", "prepared", f"prepared risky action: {proposed_action}")
        ],
    }


def approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval step."""
    if os.getenv("LANGGRAPH_INTERRUPT") == "true":
        try:
            from langgraph.types import interrupt

            approval_decision = interrupt({"proposed_action": state.get("proposed_action")})
        except ImportError:
            approval_decision = {
                "approved": True,
                "reviewer": "mock-reviewer",
                "comment": "Auto-approved in mock mode",
            }
    else:
        approval_decision = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "Approved by mock reviewer",
        }

    if isinstance(approval_decision, bool):
        approval_decision = {"approved": approval_decision, "reviewer": "reviewer", "comment": ""}

    return {
        "approval": approval_decision,
        "events": [make_event("approval", "completed", f"approval outcome: {approval_decision}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Record a retry attempt and increment counter."""
    current_attempt = state.get("attempt", 0)
    new_attempt = current_attempt + 1
    err_msg = f"Attempt {new_attempt} failed."
    return {
        "attempt": new_attempt,
        "errors": [err_msg],
        "events": [
            make_event(
                "retry", "attempted", f"incremented attempt to {new_attempt}", attempt=new_attempt
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Handle unresolvable failures after max retries exceeded."""
    max_attempts = state.get("max_attempts", 3)
    final_answer = f"System failure: Unable to complete request after {max_attempts} attempt(s). Routed to dead letter queue."
    return {
        "final_answer": final_answer,
        "events": [make_event("dead_letter", "escalated", "request routed to dead letter queue")],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit a final audit event. All routes pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
