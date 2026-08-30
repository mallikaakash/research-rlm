"""tui — a Textual observability dashboard for the RLM engine.

Left panel: the recursion + what context each call receives, streamed live.
Right panel: the REPL namespace (ns) populating with variables as the model works.

The engine is blocking and synchronous; Textual runs an asyncio event loop. So the
engine runs on a background thread and its events are marshalled to the UI thread
as Textual messages — see app.py. The pure event->view logic lives in state.py so
it can be tested without a terminal.
"""

from .state import DashboardModel

__all__ = ["DashboardModel"]
