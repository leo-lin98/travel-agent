"""Phase 0 acceptance tests for offline isolation.

The default offline pytest configuration must block outbound network access, and the
contracts must be usable without any provider credentials in the environment.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from pytest_socket import SocketBlockedError

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
# Documentation-only address (RFC 5737): never routable, so a failure to block cannot
# reach a real host.
UNROUTABLE_ADDRESS = ("192.0.2.1", 443)
PROVIDER_ENV_PREFIXES = ("OPENAI", "GOOGLE", "TAVILY", "ELEVENLABS", "LANGFUSE", "ANTHROPIC")
SECRET_ENV_MARKERS = ("API_KEY", "SECRET", "TOKEN")

CONTRACT_SMOKE_SCRIPT = """
import sys
from pathlib import Path

from travel_agent.contracts import PlanningRequest

PlanningRequest.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
"""


def is_provider_variable(name: str) -> bool:
    """Return whether an environment variable name looks like provider configuration."""
    upper = name.upper()
    return upper.startswith(PROVIDER_ENV_PREFIXES) or any(
        marker in upper for marker in SECRET_ENV_MARKERS
    )


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_outbound_tcp_socket_is_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_outbound_connection_is_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.create_connection(UNROUTABLE_ADDRESS, timeout=1)


def test_contracts_work_without_provider_credentials() -> None:
    environment = {
        name: value for name, value in os.environ.items() if not is_provider_variable(name)
    }

    completed = subprocess.run(
        [sys.executable, "-c", CONTRACT_SMOKE_SCRIPT, str(FIXTURES_DIR / "planning_request.json")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
