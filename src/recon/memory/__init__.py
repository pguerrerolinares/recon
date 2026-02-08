"""Cross-run knowledge persistence via memvid.

Requires the optional ``memvid-sdk`` package::

    pip install recon-ai[memory]

When memvid-sdk is not installed, ``MemoryStore`` can still be imported
and instantiated. The ``ImportError`` is raised lazily on first use
(when ``_ensure_open()`` calls ``_get_memvid()``), so code that checks
``plan.memory.enabled`` before creating a store never hits the error.
"""
