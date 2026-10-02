"""Local-model instruments. OFFLINE ONLY - never reachable from the request path.

This package exists so that a provider backed by ``transformers`` and ``torch`` has
somewhere honest to live. It was first written into ``arag.agent.providers`` with the
heavy imports deferred inside ``__init__``, and ``tests/test_import_boundaries.py``
rejected it - correctly. A lazy import still means the dependency has to be present in
whatever bundle ships the module, and ``arag.agent`` is an ONLINE package that DESIGN
§9 caps at a serverless function's size.

The right fix was to move the code, not to relax the rule. ``arag.local`` is absent
from ``ONLINE_PACKAGES``, so the boundary tests leave it alone and the online bundle
never sees it.
"""
