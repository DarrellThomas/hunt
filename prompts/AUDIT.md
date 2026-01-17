# HUNT PLATFORM AUDIT

You are being asked to audit and prepare the HUNT ecosystem codebase for future extensibility.

**DO NOT implement new features yet.** This is an architecture review and refactoring pass.

## Context

HUNT is an evolutionary predator-prey simulation that has grown organically. It currently supports:
- 2 species (prey, predators)
- Neural network brains with neuroevolution
- GPU-accelerated simulation (10k+ agents)
- River/island environmental features
- Configurable parameters via config.py

We want to extend it to support:
- (N) Arbitrary number of species with different roles
- (F) New agent features (senses, abilities, traits)
- (E) New environmental features (weather, terrain, resources)
- (B) Batch experiment runner for scientific studies

But first, we need to make sure the foundation is solid.

## Your Task

### Phase 1: Audit (Do this first)

Review the codebase and document:

1. **ARCHITECTURE.md** - Create a clear map of:
   - File responsibilities
   - Class hierarchy
   - Data flow (how agents perceive → decide → act)
   - GPU vs CPU boundaries
   - Current coupling/dependencies between modules

2. **TECHNICAL_DEBT.md** - Identify:
   - Hardcoded assumptions (e.g., "exactly 2 species")
   - Copy-pasted code that should be abstracted
   - Magic numbers that should be config
   - Brittle patterns that will break when extended
   - Performance bottlenecks
   - Any bugs or edge cases you notice

3. **EXTENSION_POINTS.md** - Analyze:
   - What's easy to extend right now?
   - What would require significant refactoring?
   - Where are the natural seams for adding species?
   - Where are the natural seams for adding features?
   - What patterns should new code follow?

### Phase 2: Recommendations

After the audit, create:

4. **REFACTOR_PLAN.md** - Prioritized list of changes:
   - Critical (must fix before extending)
   - Important (will cause pain if not fixed)
   - Nice-to-have (cleaner but not blocking)
   
   For each item, estimate effort (small/medium/large) and risk.

5. **EXTENSION_ARCHITECTURE.md** - Propose patterns for:
   - Adding a new species (what files change? what's the interface?)
   - Adding a new agent trait (evolvable attribute)
   - Adding a new environment feature
   - Running batch experiments with config variations

### Phase 3: Critical Fixes Only

If you identify any **critical** issues that would make the codebase unstable or broken, fix them. But:
- Document what you changed and why in CHANGELOG.md
- Keep changes minimal and focused
- Do NOT add new features
- Do NOT refactor things that aren't critical

## Rules

1. Read ALL the code before writing anything
2. Git commit after each document you create
3. Ask clarifying questions in a QUESTIONS.md file if you're uncertain about design intent
4. Preserve all existing functionality
5. Budget: ~$30 for this audit (be thorough but efficient)

## Output

When complete, I should have:
- Clear understanding of the current architecture
- List of technical debt and risks
- Actionable refactoring plan
- Confidence that the platform can be extended safely

## Begin

Start by reading every .py and .md file. Understand before you prescribe.
