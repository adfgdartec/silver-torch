# Changelog

## 0.4.0 - 2026-08-22

- Added complete Apache license and project URL metadata to built distributions.
- Validated feed-forward width, sequence length, output size, and dropout before
  constructing transformer modules.
- Added explicit runtime errors for malformed or overlong transformer inputs.
- Rejected missing classification labels during fitting.
- Restored tuple-typed categorical state when loading persisted pipelines.
- Hardened CI and releases with supported-Python testing and tag/version checks.

## 0.3.0 - 2026-08-15

- Coordinated the package release with transformer, GPT-style, and GPU examples.

## 0.2.0 - 2026-08-15

- Added JSON persistence for fitted preprocessing pipelines so serving processes can
  restore training statistics without refitting.
- Added `fit_transform()`, feature-name metadata, and label-mapping accessors.
- Added declarative transformer/GPT-style model blueprints with causal language-model mode.

## 0.1.0

- Initial release of `silver-torch`.
