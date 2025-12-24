# Asset Manifest Evidence Encoding

The internal `AssetManifestEvidence` payloads are encoded in a fixed-length binary format to
support deterministic PVGS verification.

## Encoding Layout

All numeric fields are encoded in little-endian order. Digests are 32-byte raw values.

```
version (u32 LE)
created_at_ms (u64 LE)
manifest_digest (32 bytes)
morph_digest (32 bytes)
channel_params_digest (32 bytes)
syn_params_digest (32 bytes)
connectivity_digest (32 bytes)
```

The total payload length is **172 bytes**.

## Digest Binding

PVGS recomputes the manifest digest as:

```
BLAKE3-256("UCF:ASSET:MANIFEST" || payload_bytes_with_zeroed_manifest_digest)
```

`payload_bytes_with_zeroed_manifest_digest` is the 172-byte payload where the 32 bytes of
`manifest_digest` are zeroed before hashing. The recomputed digest must match the
`manifest_digest` embedded in the payload, and the payload commit request must supply the same
digest.
