import pytest

from keep_gpu.utilities.endpoint_validation import (
    ENDPOINT_HOST_ERROR,
    ENDPOINT_PORT_ERROR,
    validate_endpoint,
    validate_endpoint_host,
    validate_endpoint_port,
)


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "gpu-box",
        "gpu-box.local",
        "node-01.cluster.local",
        "127.0.0.1",
        "0.0.0.0",
    ],
)
def test_validate_endpoint_host_accepts_dns_names_and_ipv4_literals(host):
    assert validate_endpoint_host(host) == host


@pytest.mark.parametrize(
    "host",
    [
        "",
        " ",
        "bad host",
        "%",
        "%zz",
        "\\host",
        "host\\path",
        "*",
        "-bad",
        "bad-",
        "bad..host",
        "999.999.999.999",
        "256.0.0.1",
        "123",
        "foo.123",
        "http://localhost",
        "localhost:8765",
        "::1",
    ],
)
def test_validate_endpoint_host_rejects_malformed_values(host):
    with pytest.raises(ValueError, match=ENDPOINT_HOST_ERROR):
        validate_endpoint_host(host)


@pytest.mark.parametrize("port", [1, 8765, 65535, "1", "8765", "65535"])
def test_validate_endpoint_port_accepts_ints_and_clean_digit_strings(port):
    assert validate_endpoint_port(port) == int(port)


@pytest.mark.parametrize(
    "port",
    [
        0,
        65536,
        -1,
        True,
        False,
        "",
        " ",
        "abc",
        " 8765",
        "8765 ",
        "+8765",
        "8_765",
        "１２３",
        8765.0,
        None,
    ],
)
def test_validate_endpoint_port_rejects_invalid_values(port):
    with pytest.raises(ValueError, match=ENDPOINT_PORT_ERROR):
        validate_endpoint_port(port)


def test_validate_endpoint_returns_normalized_host_and_port_pair():
    assert validate_endpoint("localhost", "8765") == ("localhost", 8765)


def test_validate_endpoint_rejects_invalid_host_or_port():
    with pytest.raises(ValueError, match=ENDPOINT_HOST_ERROR):
        validate_endpoint("bad host", "8765")
    with pytest.raises(ValueError, match=ENDPOINT_PORT_ERROR):
        validate_endpoint("localhost", "70000")
