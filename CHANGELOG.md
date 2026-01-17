# Changelog

## 2026-01-17 - Project Restructuring

### Added
- **Folder structure**: Organized codebase into `src/`, `docs/`, `results/`, `tests/`
- **Wrapper scripts**: `run.py` and `run_gpu.py` for convenient execution from project root
- **Updated README.md**: New root-level README with project structure overview

### Changed
- **Moved all source code** to `src/` directory
- **Moved all documentation** to `docs/` directory
- **Moved all result files** (.npz, .png) to `results/` directory
- **Moved test files** to `tests/` directory
- **Updated start_overnight.sh**: Now runs from src/ directory

### Migration Guide

**Before:**
```bash
python3 main.py
python3 main_gpu.py
```

**After (from root):**
```bash
python3 run.py
python3 run_gpu.py
```

**Or (from src/):**
```bash
cd src
python3 main.py
python3 main_gpu.py
```

All imports still work within `src/` directory. Wrapper scripts handle path setup for root execution.

---

## 2026-01-17 - Complete Platform Audit

### Added
- **ARCHITECTURE.md**: Complete system architecture documentation
- **TECHNICAL_DEBT.md**: 31 catalogued technical debt items
- **EXTENSION_POINTS.md**: Extensibility analysis
- **REFACTOR_PLAN.md**: Prioritized refactoring roadmap (6-8 weeks)
- **EXTENSION_ARCHITECTURE.md**: Concrete examples for future extensions
- **QUESTIONS.md**: Design clarification questions

### Findings
- Identified hardcoded 2-species architecture as main extensibility blocker
- Discovered GPU neuroevolution differs from CPU implementation
- Documented 6-8 week refactoring path to N-species support
- All existing functionality stable (no critical bugs)

---

## Earlier History

See git log for full commit history.
