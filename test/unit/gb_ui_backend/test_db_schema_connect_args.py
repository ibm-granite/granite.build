#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""_build_connect_args() translates the JSON-serializable connect args crossing the
env-var boundary (GB_UI_DATABASE_CONNECT_ARGS) into what create_async_engine() expects.
An ssl.SSLContext isn't JSON-serializable, so gbserver only ever sends the cert file
path under "sslrootcert_file" — this builds the actual context right before the
engine is created.
"""

import datetime
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from gb_ui_backend.services.db_schema import _build_connect_args


def _self_signed_cert_pem() -> bytes:
    """A minimal self-signed CA cert — ssl.create_default_context(cafile=...) needs
    an actual parseable X.509 cert, not arbitrary bytes."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-root")])
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


class TestBuildConnectArgs:
    def test_empty_input_returns_empty(self):
        assert _build_connect_args({}) == {}

    def test_passthrough_keys_unchanged(self):
        assert _build_connect_args({"timeout": 5}) == {"timeout": 5}

    def test_sslrootcert_file_becomes_ssl_context(self, tmp_path):
        cert_path = tmp_path / "root.pem"
        cert_path.write_bytes(_self_signed_cert_pem())

        result = _build_connect_args({"sslrootcert_file": str(cert_path)})

        assert "sslrootcert_file" not in result
        assert isinstance(result["ssl"], ssl.SSLContext)

    def test_sslrootcert_file_combined_with_other_keys(self, tmp_path):
        cert_path = tmp_path / "root.pem"
        cert_path.write_bytes(_self_signed_cert_pem())

        result = _build_connect_args({"sslrootcert_file": str(cert_path), "timeout": 5})

        assert result["timeout"] == 5
        assert isinstance(result["ssl"], ssl.SSLContext)
