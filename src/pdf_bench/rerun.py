"""Re-evaluate the saved answers after the evaluation itself was changed, without parsing anything again.

The scorer, the converter, the post-processing, the questions and the judge are plain Python modules. A running app
has already imported them, and Streamlit keeps the scores it computed. ``reevaluate()`` reloads those modules from disk
and empties the score caches, so the next run of a page scores the *same saved answers* with the code as it is now.
Nothing here calls a parser or a model: answers come from ``results/`` and judge verdicts from ``results/judge/``.
"""

from __future__ import annotations

import importlib
import sys

# dependencies first: a module reloaded after what it imports sees the new version
EVALUATION = ("elements", "convert", "locate", "questions", "score", "postprocess", "judge", "view", "recommend")


def reevaluate() -> list[str]:
    """Reload the evaluation modules and clear Streamlit's data cache. Returns the modules that were reloaded."""
    done = []
    for name in EVALUATION:
        mod = sys.modules.get(f"{__package__}.{name}")
        if mod is not None:
            importlib.reload(mod)
            done.append(name)
    try:
        import streamlit as st

        st.cache_data.clear()
    except Exception:  # noqa: BLE001 - outside Streamlit (tests, command line) there is no cache to clear
        pass
    return done
