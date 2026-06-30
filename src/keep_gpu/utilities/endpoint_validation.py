"""Shared validation for public HTTP endpoint inputs."""

from __future__ import annotations

import ipaddress
from typing import Any, Tuple

ENDPOINT_HOST_ERROR = "host must be a DNS hostname or IPv4 address"
ENDPOINT_PORT_ERROR = "port must be an integer between 1 and 65535"


def _is_dns_hostname(value: str) -> bool:
    if len(value) > 253 or value.endswith("."):
        return False
    labels = value.split(".")
    if labels[-1].isdigit():
        return False
    for label in labels:
        if not 1 <= len(label) <= 63:
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        if not all(
            char.isascii() and (char.isalnum() or char == "-") for char in label
        ):
            return False
    return True


def validate_endpoint_host(host: Any) -> str:
    """Validate a public service host as a DNS hostname or IPv4 literal."""
    if not isinstance(host, str) or host.strip() != host or not host:
        raise ValueError(ENDPOINT_HOST_ERROR)
    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        if not _is_dns_hostname(host):
            raise ValueError(ENDPOINT_HOST_ERROR) from None
    else:
        if parsed_ip.version != 4:
            raise ValueError(ENDPOINT_HOST_ERROR)
    return host


def validate_endpoint_port(port: Any) -> int:
    """Validate a public service port and return it as an integer."""
    if isinstance(port, bool):
        raise ValueError(ENDPOINT_PORT_ERROR)

    if isinstance(port, int):
        parsed_port = port
    elif isinstance(port, str):
        if not port or port.strip() != port:
            raise ValueError(ENDPOINT_PORT_ERROR)
        try:
            parsed_port = int(port)
        except ValueError:
            raise ValueError(ENDPOINT_PORT_ERROR) from None
    else:
        raise ValueError(ENDPOINT_PORT_ERROR)

    if not 1 <= parsed_port <= 65535:
        raise ValueError(ENDPOINT_PORT_ERROR)
    return parsed_port


def validate_endpoint(host: Any, port: Any) -> Tuple[str, int]:
    """Validate a public service host/port pair."""
    return validate_endpoint_host(host), validate_endpoint_port(port)
