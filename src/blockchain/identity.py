#########----------------------------------------------------------------------#######
#########----------src/blockchain/identity---------------------------------------#######
"""Ed25519 identities for federated clients and validators."""

from __future__ import annotations

from dataclasses import dataclass


def _backend():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        return Ed25519PrivateKey
    except ImportError as exc:
        raise ImportError(
            "Ed25519 support requires the 'cryptography' package"
        ) from exc


@dataclass
class Identity:
    name: str
    private_key: object

    @classmethod
    def generate(cls, name: str) -> "Identity":
        return cls(name, _backend().generate())

    @property
    def public_key(self) -> object:
        return self.private_key.public_key()

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            self.public_key.verify(signature, message)
            return True
        except Exception:
            return False

    @property
    def public_hex(self) -> str:
        return public_hex(self.public_key)

    def private_hex(self) -> str:
        from cryptography.hazmat.primitives import serialization

        return self.private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        ).hex()

    @classmethod
    def from_private_hex(cls, name: str, value: str) -> "Identity":
        return cls(name, _backend().from_private_bytes(bytes.fromhex(value)))


def public_hex(public_key: object) -> str:
    from cryptography.hazmat.primitives import serialization

    return public_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()


def verify_with_public_hex(
    public_key_hex: str, message: bytes, signature_hex: str
) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature_hex), message
        )
        return True
    except Exception:
        return False
