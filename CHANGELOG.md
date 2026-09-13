# Changelog

User-visible changes are recorded here. New changes belong under Unreleased.

## [Unreleased]

### Added

- `Reference` for Dempster-Hill cells, sampling, localization, and densities.
- `Law` cell mixtures, sampling, density evaluation, and `expect` quadrature.
- Smooth maps with Jacobians, inversion, and pullbacks.
- Transport labels, potentials, cell halfspaces, batched assignments, and `extend`.

### Changed

- Clearer numerical-stability and input-validation behavior.

## [0.1.0] — 2026-08-31

### Added

- Exact candidate-augmented transport with O(n³) fitting.
- Default Dempster-Hill target barycentres.
- Transport target values, ranks, signs, and candidate assignments.
- Custom `target=` barycentres.
