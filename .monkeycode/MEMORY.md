# User Instruction Memory

This file records user instructions, preferences, and teachings for reference in future interactions.

## Format

### User Instruction Entry
User instruction entries should follow this format:

[User Instruction Summary]
- Date: [YYYY-MM-DD]
- Context: [Mentioned scenario or time]
- Instructions:
  - [Content of user teaching or instruction, described line by line]

### Project Knowledge Entry
Entries discovered by the Agent during task execution should follow this format:

[Project Knowledge Summary]
- Date: [YYYY-MM-DD]
- Context: Discovered by Agent while performing [specific task description]
- Category: [Operations & Deployment|Build Methods|Testing Methods|Troubleshooting & Debugging|Workflow & Collaboration|Environment Configuration]
- Instructions:
  - [Specific knowledge points, described line by line]

## Deduplication Strategy
- Before adding a new entry, check for similar or identical instructions.
- If a duplicate is found, skip the new entry or merge it with the existing one.
- When merging, update the context or date information.
- This helps avoid redundant entries and keeps the memory file tidy.

## Entries

[Project Knowledge Summary]
- Date: 2026-10-27
- Context: Discovered by Agent while closing the second-round security review
- Category: Workflow & Collaboration
- Instructions:
  - Before pushing, all four checks must be green, with a baseline of 99 pytest +
    11 vitest and a clean ruff run:
      * `python3 -m ruff check backend/ training/` (from the repo root)
      * `python3 -m ruff format --check backend/ training/` (from the repo root)
      * `python3 -m pytest tests/ -q` (run from the `backend/` directory)
      * `npm test` (from the repo root; needs `npm install` first)
  - `ruff` only sees the repo layout when both `backend/` and `training/` are passed as
    explicit path arguments from the repo root; `training/` does not exist under `backend/`.
  - Security fixes must ship with a regression test that fails before the fix is applied
    (requestor requirement); add it to `backend/tests/test_api.py` unless the layer differs.
