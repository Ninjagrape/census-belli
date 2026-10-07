"""Read-only inspection of pipeline state, for the Streamlit dashboard.

Everything the dashboard shows comes through this package: SQL in
:mod:`pipeline.inspector.queries`, files in :mod:`pipeline.inspector.files`,
quality gates in :mod:`pipeline.inspector.gates`, and the read-only engine in
:mod:`pipeline.inspector.connection`.
"""
