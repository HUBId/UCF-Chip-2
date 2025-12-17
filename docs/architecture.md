# UCF Chip 2 Regulation Engine Scaffold

## Workspace Overview
The workspace is organized as a Rust workspace rooted at the repository root. All crates forbid unsafe code and expose placeholder APIs only.

- **wire**: Frame IO primitives with placeholders for signing and verification. All trait methods default to `NotImplemented` errors to keep logic stubbed out.
- **rsv**: Regulator State Vector types and storage abstraction. Provides a serializable `RegulatorState` structure and `StateStore` trait for future persistence backends.
- **profiles**: Profile and overlay composition primitives. Offers serializable definitions and a `ProfileComposer` trait with a placeholder implementation that always reports `NotImplemented`.
- **engine**: Update engine surface. Defines the `UpdateEngine` trait, input/output structures, and a default `NotImplemented` application path for future regulation logic.
- **hpa**: Placeholder for the DBM-HPA layer with configuration and snapshot stubs, plus a `PlaceholderHpa` client that returns `NotImplemented`.
- **pvgs_client**: Client placeholder for CBV/HBV retrieval. Includes configuration and snapshot scaffolding alongside a `PlaceholderPvgsClient` that has no real network behavior.
- **app**: Binary crate that validates configuration paths, exercises the placeholder components, and prints a `boot ok` message without performing real regulation.

## Configuration Directory
The `config/` directory contains placeholder YAML files for profiles, overlays, update tables, windowing, class thresholds, and HPA settings. They are referenced by the `app` crate but not parsed yet.

## CI Workflow
The GitHub Actions workflow at `.github/workflows/ci.yml` runs `cargo fmt`, `cargo clippy`, and `cargo test` across the workspace to ensure the scaffold remains buildable and warning-free.
