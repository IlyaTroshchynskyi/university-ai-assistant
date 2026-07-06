"""Shared storage configuration.

`STORE_BACKEND` selects where flow state and conversation history live:
`sqlite` (default, survives restarts) or `memory` (process-only). `APP_DB_PATH`
points at the SQLite file used by both stores.
"""

import os

STORE_BACKEND = os.environ.get('STORE_BACKEND', 'sqlite').lower()
DB_PATH = os.environ.get('APP_DB_PATH', 'app_state.sqlite3')
