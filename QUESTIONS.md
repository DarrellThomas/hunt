# Questions About Design Intent

Questions raised during the audit for clarification on design decisions.

## 1. GPU Neuroevolution vs CPU Implementation

**Question**: Is the GPU version's global network mutation (every 50 steps) intentional, or should it match the CPU version's per-agent genetic inheritance?

**Current behavior**:
- **CPU**: Each agent has individual brain, offspring inherit parent's weights with mutation
- **GPU**: All agents share one network that mutates periodically

**Context**: TECHNICAL_DEBT.md #6

**Options**:
A. GPU behavior is intentional (different algorithm for performance)
B. GPU should match CPU (proper neuroevolution required)

**Recommendation**: If B, this is a critical refactor (see REFACTOR_PLAN.md #1)

---

## 2. Island Behavior CPU/GPU Divergence

**Question**: Is it intentional that island behavior modifiers only affect GPU version currently?

**Current**:
- GPU: Island affects speed, hunger, reproduction
- CPU: Island only provides safety (no flow), no behavior modifiers

**Context**: TECHNICAL_DEBT.md #12

**Recommendation**: Unify implementations (see REFACTOR_PLAN.md #5, 2-day fix)

---

## 3. Extinction Prevention Strategy

**Question**: Should CPU's dual-threshold extinction prevention be ported to GPU, or GPU's simpler approach adopted for both?

**CPU approach**:
1. Emergency respawn when population hits 0 (5 agents)
2. Minimum threshold based on world size (prevents low populations)

**GPU approach**:
1. Emergency respawn only (when population hits 0)

**Impact**: Different evolutionary dynamics between versions

**Recommendation**: Unify to one strategy (see TECHNICAL_DEBT.md #9)

---

## 4. Observation Sampling Asymmetry

**Question**: Is the asymmetric sampling in GPU observations intentional?

**Current**:
- Prey sample up to 100 predators
- Predators sample up to 200 prey

**Context**: TECHNICAL_DEBT.md #14

**Justification**: Unknown (predators need better vision? More prey to choose from?)

**Recommendation**: Document rationale or make symmetric

---

## 5. River Path Parameters

**Question**: What is the rationale for the river generation parameters?

**Current**:
- `num_points = 50`
- Sine waves with periods 4 and 2.3, amplitudes 0.2 and 0.15

**Context**: TECHNICAL_DEBT.md #11

**Needed**: Either document why these values work well, or make them configurable

---

## 6. Configuration Philosophy

**Question**: Is the global constant approach in config.py intentional for simplicity, or should it evolve to a more structured system?

**Trade-off**:
- **Current (global constants)**: Simple, fast, direct access
- **Proposed (config classes)**: Supports N-species, batch experiments, validation

**Context**: Blocks batch experiments and multi-species

**Recommendation**: Refactor if extensibility is priority (see REFACTOR_PLAN.md #4)

---

## 7. Brain Architecture Constraints

**Question**: Should agents of the same species be required to have identical brain input dimensions?

**Current**: Yes (all prey have 32-dim input, all predators have 21-dim)

**Alternative**: Allow per-agent sensor configurations (some prey have better vision, etc.)

**Trade-off**:
- **Uniform**: Simpler, batching easier (GPU)
- **Variable**: More biological realism, harder to batch

**Recommendation**: Depends on whether individual variation is a goal

---

## 8. Performance vs Correctness

**Question**: What's the priority: GPU performance or matching CPU evolutionary dynamics exactly?

**Context**: GPU-CPU transfers for river/island checks are a bottleneck but fixing requires porting to GPU

**Options**:
A. Optimize (port river to GPU, 20-30% speedup, see REFACTOR_PLAN.md #7)
B. Accept performance cost for compatibility
C. Allow CPU/GPU to diverge in implementation

**Current**: A mix (some divergence exists already)

---

## 9. Species Roles

**Question**: Is the two-species (predator/prey) paradigm fundamental, or just the current implementation?

**Context**: Entire architecture assumes exactly 2 species with specific roles

**Future vision**:
- Support N species with flexible interactions?
- Keep 2-species but make extensible?
- Redesign for ecosystem complexity?

**Recommendation**: Clarify vision before starting refactor (impacts scope)

---

## 10. Batch Experiments Priority

**Question**: How important is running batch experiments with config variations?

**Current**: Not supported (global config)
**Effort to add**: Medium (1 week, see REFACTOR_PLAN.md #4)

**Use cases**:
- Parameter sweeps for research
- Hyperparameter tuning
- A/B testing evolutionary strategies

**Recommendation**: If scientific experiments are a goal, prioritize config refactor

---

## Next Steps

These questions should be answered before beginning major refactoring. The answers will inform:
1. Which refactors are critical vs nice-to-have
2. Whether to unify CPU/GPU or let them diverge
3. How much architectural change is acceptable
4. What the long-term vision for the platform is

**Suggested approach**:
1. Answer these questions
2. Update REFACTOR_PLAN.md priorities based on answers
3. Start with quick wins (utils extraction, type hints)
4. Plan major refactors incrementally
