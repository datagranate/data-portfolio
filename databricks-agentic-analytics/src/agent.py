import json
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
import warnings

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI, UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

############################################
# Define LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "system.ai.meta-llama-3-3-70b-instruct"

SYSTEM_PROMPT = """You are a marketing analytics assistant for an online supermarket. You answer questions about customer behaviour using the available tools.

GENERAL RULES:
- Always call a tool before answering any question about the data. Never state numbers from memory or invent them.
- The currency is NOT USD.  Do NOT use currency symbols at all
- If no tool can answer the question, say so plainly. Do not guess.
- demographic_profile, channel_mix_by_demographic and deal_affinity_by_demographic take one of exactly these values: Education, Marital_Status, Age_category,
  Income_category.
- The Education levels are Brazilian, and the categories correspond roughly to the following years of education: Basic: 12, Graduation: 16, 2n Cycle:18, Master: 18, PhD: 22 
- Ratios are proportions between 0 and 1 (e.g. 0.35 = 35% of purchases).
- When you report figures, name the source tool and the group it refers to.
- Keep answers under 150 words unless asked for detail. Use plain English, not statistical jargon.

  RFM SEGMENT DEFINITIONS:
  - rfm_customers_in_segment takes one of exactly these eight segment names: Champions, High-potential growth, Loyal, Recents, Critical retention, Low-priority churn, High value at risk, Standard base.
- rfm_customers_by_score_band takes score_component of R, F or M. Valid scores: R = recency score 1-3, F/M = frequency/monetary 1-5.  Higher = better.
- R=3 (recently active), R=2 (cooling), R=1 (lapsed).
- Champions: recent + frequent + high spend. Protect, upsell.
- High-potential growth: recent, strong on one of F/M but not both. Nurture toward Champions.
- Loyal: recent and frequent, lower spend. Increase basket value.
- Recents: recent but low frequency and monetary. Onboard, build habit.
- High value at risk: WAS frequent and high-value, now cooling (R=2). Priority win-back -  relationship still recoverable.
- Critical retention: identical value profile to High value at risk but already lapsed (R=1).  Last-chance offer only; assume recovery cost is higher.
- Standard base: average behaviour across all dimensions. Baseline comms only.
- Low-priority churn: lapsed AND low value. Do not allocate campaign budget.

RFM-SPECIFIC RULES:
- Segments sharing the same F/M bands differ ONLY on recency. Never treat them as interchangeable - the recency gap is the entire commercial distinction.
- When two segments have similar monetary value, prioritise by recency: retaining an actively-cooling customer costs less than reactivating a lapsed one.
 - When comparing segments of similar value, explicitly state your recency-based prioritisation. 
- Never recommend acquisition spend while High value at risk remains untargeted."""


###############################################################################
## Define tools for agent
###############################################################################
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


TOOL_INFOS = []

UC_TOOL_NAMES = ["analytics.marketing.channel_mix_by_rfm_segment", "analytics.marketing.rfm_customers_by_score_band", "analytics.marketing.rfm_customers_in_segment", "analytics.marketing.rfm_segment_summary", "analytics.marketing.demographic_profile", "analytics.marketing.deal_affinity_by_demographic", "analytics.marketing.channel_mix_by_demographic"]

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
uc_function_client = get_uc_function_client()
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))

class ToolCallingAgent(ResponsesAgent):
    """
    Class representing a tool-calling Agent
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.model_serving_client: OpenAI = DatabricksOpenAI(use_ai_gateway=True)
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @staticmethod
    def _merge_consecutive_assistant_messages(cc_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # The canonical OpenAI shape for parallel tool calls is one assistant
        # message whose tool_calls array lists every call in that turn.
        # to_chat_completions_input emits a separate assistant message per
        # function_call, which some providers reject. Collapse consecutive
        # assistant messages so all tool_calls from one turn share a single
        # message.
        merged: list[dict[str, Any]] = []
        for msg in cc_messages:
            if (
                msg.get("role") == "assistant"
                and merged
                and merged[-1].get("role") == "assistant"
            ):
                prev = merged[-1]
                prev_text = prev.get("content") if prev.get("content") not in (None, "", "tool call") else None
                cur_text = msg.get("content") if msg.get("content") not in (None, "", "tool call") else None
                if prev_text and cur_text:
                    prev["content"] = prev_text + "\n" + cur_text
                elif cur_text:
                    prev["content"] = cur_text
                if msg.get("tool_calls"):
                    prev["tool_calls"] = (prev.get("tool_calls") or []) + msg["tool_calls"]
            else:
                merged.append(dict(msg))
        return merged

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        cc_messages = self._merge_consecutive_assistant_messages(to_chat_completions_input(messages))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=cc_messages,
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        try:
            args = json.loads(tool_call.get("arguments"))
        except Exception as e:
            args = {}
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            # The LLM may emit multiple tool calls in a single turn (parallel
            # tool calls). The next LLM request must include a tool_result for
            # every tool_use emitted in the previous turn; missing any one
            # causes the request to be rejected. Before going back to the LLM,
            # execute every function_call whose call_id has no matching
            # function_call_output yet.
            handled = {m["call_id"] for m in messages if m.get("type") == "function_call_output"}
            pending = [m for m in messages if m.get("type") == "function_call" and m["call_id"] not in handled]
            if pending:
                for call in pending:
                    yield self.handle_tool_call(call, messages)
                continue

            last_msg = messages[-1]
            if last_msg.get("type") == "message" and last_msg.get("role") == "assistant":
                return

            yield from output_to_responses_items_stream(
                chunks=self.call_llm(messages), aggregator=messages
            )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)


# Log the model using MLflow
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
mlflow.models.set_model(AGENT)
