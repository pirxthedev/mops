# FEA Conformance Test Suite

An open-source, comprehensive conformance test suite for finite element analysis codes. The suite provides specifications derived from publicly available benchmarks (NAFEMS, MacNeal-Harder, shell classics, and others), enabling anyone to validate an FEA implementation or build one from scratch using agentic coding tools. See [INDEX.md](INDEX.md) for navigation to all test cases.

## License

This repository is licensed under the MIT License. All specifications are written in original prose. Each test case cites its source. Problems (geometry, loads, physics) are factual and not copyrightable.

## Usage

Each test case in `tests/` is a self-contained markdown file specifying geometry, materials, loads, boundary conditions, and expected results with tolerances. Implement the problem in your FEA code and compare against the reference values.
