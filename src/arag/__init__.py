"""Agentic RAG over Indian health insurance policy wordings.

Package boundaries (enforced by tests/test_import_boundaries.py):

    ingest  -> index -> retrieval -> agent -> api
    obs, config: imported by anything; import nothing from the pipeline
    eval: imports the protocols, never a concrete implementation

Only ``arag.agent`` may import LangGraph/LangChain. See DESIGN.md §5.7.
"""

__version__ = "0.1.0"
