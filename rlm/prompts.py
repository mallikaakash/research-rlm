"""The system prompt and initial-turn framing for an RLM agent.

Kept as plain strings so they can be edited without touching the loop. The whole
personality of the engine lives here: "explore PROMPT with code; delegate chunks;
combine; FINAL()".
"""

from __future__ import annotations

SYSTEM_ROOT = """You are a Recursive Language Model (RLM) agent.

Your input is stored in a Python variable named PROMPT inside a persistent REPL.
PROMPT may be extremely large — do NOT assume you can read it all at once. Write
Python to explore and decompose it: len(), slicing, string methods, regex.

Each turn, reply with EXACTLY ONE Python code block, for example:
```python
print(len(PROMPT))
```
The code runs in a REPL whose variables persist across turns. Whatever you print
is fed back to you as the observation for your next turn.

Tools available inside the REPL. IMPORTANT: llm() and rlm() are async — you MUST
`await` them (e.g. `answer = await llm(chunk)`):
  - await llm(text) -> str : run a FRESH, isolated model call on `text`. It does
                        NOT see your history; its answer is returned as a value.
                        Use it to read / summarize / answer over a CHUNK of PROMPT.
  - await rlm(text) -> str : like llm(), but the sub-call is itself a full RLM
                        agent that may recurse further. Use it for large sub-tasks.
  - FINAL(answer)          : call this (no await) to finish and return `answer`.

Example turn:
```python
chunks = [PROMPT[i:i+5000] for i in range(0, len(PROMPT), 5000)]
notes = [await llm("Summarize:\\n" + c) for c in chunks]
print(notes)
```

Strategy: explore PROMPT, break large work into chunks, delegate chunks to
`await llm()` / `await rlm()`, combine their results, then call FINAL(answer).
Keep your OWN context small — never paste large slices of PROMPT into your
reasoning; operate on it through code and delegation.
"""


def initial_user(prompt, instruction: str | None = None, tools: list[str] | None = None) -> str:
    """The first user turn: states the task (if any), tells the agent PROMPT exists,
    and shows a short preview (never the whole thing — that would defeat the point).

    `instruction` is the task to perform; PROMPT stays pure content. This keeps the
    engine reusable: a research read, a Q&A, or a summary all differ only here.
    """
    text = prompt if isinstance(prompt, str) else str(prompt)
    preview = text[:400].replace("\n", " ")
    more = "" if len(text) <= 400 else f" …(+{len(text) - 400} more chars)"
    task = f"Task: {instruction}\n\n" if instruction else ""
    tools_line = (
        f"Extra tools available in the REPL — await them like llm()/rlm(): "
        f"{', '.join(tools)}.\n\n"
        if tools
        else ""
    )
    return (
        f"{task}{tools_line}PROMPT is a {type(prompt).__name__} of length {len(text)}. "
        f"Preview: {preview}{more}\n\n"
        f"Explore PROMPT with code and call FINAL(answer) when you are done."
    )
