from typing import Dict, Any
import uuid


class SessionStore:
    # Use a class-level dictionary to share sessions across all instances
    sessions: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        # No per-instance initialization needed; sessions are stored at class level
        pass

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())

        self.sessions[session_id] = {}

        return session_id

    def get(self, session_id: str):
        """Return the session dict for the given ID or None if missing."""
        return SessionStore.sessions.get(session_id)

    def get_value(self, session_id: str, key: str):
        """Retrieve a value for a key in a session.
        Raises ValueError if session not found, KeyError if key missing.
        """
        session = SessionStore.sessions.get(session_id)
        if session is None:
            raise ValueError("Session not found")
        if key not in session:
            raise KeyError(f"Key '{key}' not found in session")
        return session[key]

    def set(self, session_id: str, key: str, value: Any):
        """Set a key-value pair in a session. Creates session if it does not exist."""
        if session_id not in SessionStore.sessions:
            SessionStore.sessions[session_id] = {}
        SessionStore.sessions[session_id][key] = value

    def delete_session(self, session_id: str):
        """Delete a session if it exists."""
        SessionStore.sessions.pop(session_id, None)
        self.sessions.pop(session_id, None)