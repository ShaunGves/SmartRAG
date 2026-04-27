"""
rag/agent.py

Multi-step reasoning agent with tool calling.

Unlike single-pass RAG (retrieve once → generate), the agent:
  1. THINKS about what it needs to answer the question
  2. CALLS tools (vector search, web search, code executor)
  3. OBSERVES the results
  4. DECIDES whether to act again or produce a final answer
  5. Repeats up to max_iterations

This is called a ReAct (Reason + Act) loop — the standard architecture
used in production AI agents at Anthropic, OpenAI, and Google.

Why agents outperform single-pass RAG:
  - Complex questions need multiple retrieval steps
  - Agent can detect when retrieval fails and try differently
  - Agent can execute code to verify numerical answers
  - Agent can fall back to web search for very recent info

Example (AI Programmer Assistant):
  Q: "What's the difference between asyncio.gather and asyncio.wait,
      and which is faster for 100 concurrent HTTP requests?"

  Agent Step 1: vector_search("asyncio.gather vs asyncio.wait")
  Agent Step 2: vector_search("asyncio performance comparison concurrent")
  Agent Step 3: code_executor("benchmark asyncio.gather vs asyncio.wait")
  Agent Step 4: Synthesize → final answer with benchmark results

Tools available:
  - vector_search   : Query the ChromaDB vector store
  - hybrid_search   : BM25 + dense hybrid retrieval
  - web_search      : Real-time web search (fallback for recent info)
  - code_executor   : Safe Python sandbox execution
  - calculator      : Math expression evaluation
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────
@dataclass
class ToolCall:
    """A single tool invocation by the agent."""
    name: str
    input: str
    output: str = ""
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class AgentStep:
    """One complete reasoning step: thought → tool call → observation."""
    step_number: int
    thought: str
    tool_call: Optional[ToolCall] = None
    observation: str = ""
    is_final: bool = False
    final_answer: str = ""


@dataclass
class AgentResponse:
    """Complete agent response with full reasoning trace."""
    question: str
    final_answer: str
    steps: List[AgentStep] = field(default_factory=list)
    total_latency_ms: float = 0.0
    num_tool_calls: int = 0
    sources: List[str] = field(default_factory=list)


# ─── Tool Definitions ────────────────────────────────────────────
class ToolRegistry:
    """
    Registry of tools available to the agent.
    Each tool is a callable that takes a string input and returns a string output.
    """

    def __init__(self, vectorstore=None, hybrid_retriever=None):
        self._tools: Dict[str, Callable] = {}
        self.vectorstore = vectorstore
        self.hybrid_retriever = hybrid_retriever
        self._register_defaults()

    def _register_defaults(self):
        """Register all built-in tools."""

        # ── Tool 1: Vector Search ─────────────────────────────────
        def vector_search(query: str) -> str:
            """Search the ChromaDB vector store for relevant documents."""
            if self.vectorstore is None:
                return "Vector store not available."
            try:
                from rag.vectorstore import retrieve
                docs = retrieve(query, self.vectorstore, top_k=4)
                if not docs:
                    return "No relevant documents found in the vector store."
                parts = []
                for i, doc in enumerate(docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    parts.append(f"[Doc {i} | {src}]\n{doc.page_content[:400]}")
                return "\n\n".join(parts)
            except Exception as e:
                return f"Vector search failed: {e}"

        # ── Tool 2: Hybrid Search ─────────────────────────────────
        def hybrid_search(query: str) -> str:
            """BM25 + dense hybrid search for better keyword + semantic coverage."""
            if self.hybrid_retriever is None:
                return "Hybrid retriever not available — falling back not possible."
            try:
                chunks = self.hybrid_retriever.retrieve(query, top_k=4)
                if not chunks:
                    return "No results from hybrid search."
                parts = []
                for i, chunk in enumerate(chunks, 1):
                    src = chunk.document.metadata.get("source", "unknown")
                    parts.append(
                        f"[Doc {i} | hybrid_score={chunk.hybrid_score:.3f} | {src}]\n"
                        f"{chunk.document.page_content[:400]}"
                    )
                return "\n\n".join(parts)
            except Exception as e:
                return f"Hybrid search failed: {e}"

        # ── Tool 3: Code Executor (sandboxed) ─────────────────────
        def code_executor(code: str) -> str:
            """
            Execute Python code safely in a restricted sandbox.
            Only math, string ops, and stdlib allowed. No file/network access.
            """
            ALLOWED_BUILTINS = {
                "print": print, "len": len, "range": range,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "str": str, "int": int, "float": float, "bool": bool,
                "sum": sum, "min": min, "max": max, "abs": abs,
                "round": round, "sorted": sorted, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "__import__": None,  # Block imports by default
            }
            # Allow safe stdlib imports only
            safe_imports = {"math", "statistics", "itertools", "functools", "collections"}
            import math, statistics, itertools, functools, collections
            ALLOWED_GLOBALS = {
                "__builtins__": ALLOWED_BUILTINS,
                "math": math,
                "statistics": statistics,
                "itertools": itertools,
            }
            try:
                import io, contextlib
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(code, "<agent_code>", "exec"), ALLOWED_GLOBALS)
                result = output.getvalue()
                return result if result else "Code executed successfully (no output)."
            except Exception as e:
                return f"Code execution error: {type(e).__name__}: {e}"

        # ── Tool 4: Calculator ────────────────────────────────────
        def calculator(expression: str) -> str:
            """Evaluate a mathematical expression safely."""
            import ast
            try:
                tree = ast.parse(expression, mode="eval")
                # Only allow safe node types
                allowed = {
                    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num,
                    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
                    ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
                }
                for node in ast.walk(tree):
                    if type(node) not in allowed:
                        return f"Unsafe expression: {type(node).__name__} not allowed"
                result = eval(compile(tree, "<calc>", "eval"))
                return str(result)
            except Exception as e:
                return f"Calculator error: {e}"

        # ── Tool 5: Web Search (stub → plug in SerpAPI/Tavily) ────
        def web_search(query: str) -> str:
            """
            Real-time web search for information not in the vector store.
            In production: integrate SerpAPI, Tavily, or Brave Search API.
            """
            try:
                # Attempt Tavily if API key is set
                import os
                api_key = os.getenv("TAVILY_API_KEY")
                if api_key:
                    import requests
                    resp = requests.post(
                        "https://api.tavily.com/search",
                        json={"api_key": api_key, "query": query, "max_results": 3},
                        timeout=10,
                    )
                    results = resp.json().get("results", [])
                    if results:
                        parts = [f"[{r['title']}]\n{r['content'][:300]}" for r in results]
                        return "\n\n".join(parts)
                return (
                    f"Web search not configured (set TAVILY_API_KEY env var). "
                    f"Query was: '{query}'"
                )
            except Exception as e:
                return f"Web search failed: {e}"

        self.register("vector_search",  vector_search,  "Search the indexed document vector store")
        self.register("hybrid_search",  hybrid_search,  "BM25 + dense hybrid document search")
        self.register("code_executor",  code_executor,  "Execute Python code in a safe sandbox")
        self.register("calculator",     calculator,     "Evaluate math expressions")
        self.register("web_search",     web_search,     "Search the web for current information")

    def register(self, name: str, fn: Callable, description: str = "") -> None:
        self._tools[name] = {"fn": fn, "description": description}

    def call(self, name: str, input_str: str) -> ToolCall:
        """Execute a tool and return a ToolCall result."""
        if name not in self._tools:
            return ToolCall(name=name, input=input_str, output=f"Unknown tool: {name}", success=False)
        t0 = time.perf_counter()
        try:
            output = self._tools[name]["fn"](input_str)
            return ToolCall(
                name=name,
                input=input_str,
                output=str(output),
                latency_ms=(time.perf_counter() - t0) * 1000,
                success=True,
            )
        except Exception as e:
            return ToolCall(
                name=name,
                input=input_str,
                output="",
                latency_ms=(time.perf_counter() - t0) * 1000,
                success=False,
                error=str(e),
            )

    def describe(self) -> str:
        """Generate tool descriptions for the agent's system prompt."""
        lines = []
        for name, meta in self._tools.items():
            lines.append(f"  - {name}: {meta['description']}")
        return "\n".join(lines)


# ─── ReAct Agent ─────────────────────────────────────────────────
class ReActAgent:
    """
    ReAct (Reason + Act) agent for multi-step question answering.

    Loop:
      Thought → Action (tool call) → Observation → ... → Final Answer

    The LLM generates structured output:
      Thought: <reasoning about what to do next>
      Action: <tool_name>
      Action Input: <tool input>

    Agent parses this, calls the tool, feeds observation back, repeats.

    When LLM outputs "Final Answer: <answer>", the loop terminates.
    """

    SYSTEM_PROMPT = """You are an expert AI assistant for programmers. 
You answer questions by thinking step by step and using tools to find accurate information.

Available tools:
{tools}

Format your response EXACTLY like this:
Thought: <your reasoning about what to do>
Action: <tool_name>
Action Input: <input to the tool>

When you have enough information to answer:
Thought: I now have enough information to answer.
Final Answer: <your complete answer>

Rules:
- Always think before acting
- Use vector_search or hybrid_search first for questions about documents
- Use code_executor to verify or demonstrate code answers
- Use web_search only if the vector store has no relevant results
- Be precise and cite sources when available
- Never hallucinate function names or API details
"""

    def __init__(self, llm_pipeline, tool_registry: ToolRegistry):
        self.llm = llm_pipeline
        self.tools = tool_registry
        self.system_prompt = self.SYSTEM_PROMPT.format(tools=tool_registry.describe())

    def run(self, question: str) -> AgentResponse:
        """
        Execute the full ReAct loop for a question.

        Returns AgentResponse with complete reasoning trace.
        """
        t0 = time.perf_counter()
        steps: List[AgentStep] = []
        sources: List[str] = []
        conversation = f"Question: {question}\n\n"

        for i in range(cfg.agent.max_iterations):
            # ── Generate next thought/action ──────────────────────
            prompt = self.system_prompt + "\n\n" + conversation
            raw_output = self._generate(prompt)

            step = self._parse_step(raw_output, step_number=i + 1)
            steps.append(step)

            # ── Final answer reached ──────────────────────────────
            if step.is_final:
                log.info(f"Agent finished in {i+1} steps")
                break

            # ── Execute tool ──────────────────────────────────────
            if step.tool_call:
                tool_result = self.tools.call(step.tool_call.name, step.tool_call.input)
                step.tool_call.output   = tool_result.output
                step.tool_call.latency_ms = tool_result.latency_ms
                step.tool_call.success  = tool_result.success
                step.observation        = tool_result.output

                # Track sources
                if tool_result.success and "source" in tool_result.output.lower():
                    sources.append(f"Via {step.tool_call.name}: {step.tool_call.input}")

                # Update conversation with observation
                conversation += (
                    f"Thought: {step.thought}\n"
                    f"Action: {step.tool_call.name}\n"
                    f"Action Input: {step.tool_call.input}\n"
                    f"Observation: {step.observation[:800]}\n\n"
                )
            else:
                # No tool call — add thought and continue
                conversation += f"Thought: {step.thought}\n\n"

        final_answer = next(
            (s.final_answer for s in steps if s.is_final),
            "I was unable to find a complete answer within the reasoning limit."
        )

        return AgentResponse(
            question=question,
            final_answer=final_answer,
            steps=steps,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            num_tool_calls=sum(1 for s in steps if s.tool_call),
            sources=list(set(sources)),
        )

    def _generate(self, prompt: str) -> str:
        """Run LLM generation."""
        try:
            output = self.llm(prompt, max_new_tokens=cfg.agent.max_tokens_per_step)
            if isinstance(output, list):
                return output[0].get("generated_text", "")
            return str(output)
        except Exception as e:
            log.error(f"LLM generation failed: {e}")
            return "Final Answer: I encountered an error during reasoning."

    @staticmethod
    def _parse_step(raw: str, step_number: int) -> AgentStep:
        """Parse LLM output into a structured AgentStep."""
        thought = ""
        tool_name = None
        tool_input = ""
        is_final = False
        final_answer = ""

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                tool_name = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                tool_input = line[len("Action Input:"):].strip()
            elif line.startswith("Final Answer:"):
                is_final = True
                final_answer = line[len("Final Answer:"):].strip()

        tool_call = None
        if tool_name and not is_final:
            tool_call = ToolCall(name=tool_name, input=tool_input)

        return AgentStep(
            step_number=step_number,
            thought=thought,
            tool_call=tool_call,
            is_final=is_final,
            final_answer=final_answer,
        )
