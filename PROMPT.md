Study docs/plans/* for architecture specs. Run `bd ready` and `bd list` to understand current state.

Planned structure: mops-core/ (Rust), mops-python/ (Python+PyO3), tests/, examples/.

## Rules

- Use subagents for research
- Create issues immediately when work identified
- Track dependencies between issues
- Before ending: `bd sync && git push`

ULTIMATE GOAL: Production-ready linear static FEA solver with full element library, Python API, and NAFEMS-verified results. Consider missing elements (tet4, tet10, hex8, plane stress/strain, axisymmetric), solver backends (SuiteSparse, hypre), and API features. If a feature spec is missing, author at docs/plans/YYYY-MM-DD FEATURE NAME.md (search before creating). Create implementation issues with bd create.
