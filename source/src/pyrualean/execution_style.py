"""Optional model guidance for reducing Python-cell round trips.

This is a prompt intervention, not a guard or an automatic code transformation.
"""

BATCH_ACTIONS_GUIDANCE = """
Execution efficiency: each return to the planner can take a long time.
Use the persistent Python namespace to avoid unnecessary model round trips.
- In one Python cell, include operations whose arguments and branches can be
  determined from current variables and public tool return values. Inspect failed
  motion/IK results before issuing dependent motion; stop on unexpected failures.
- When moving to an observation pose, include the required localization/state
  queries and robo.show calls in that same cell. When localizing an object, also
  request any camera image needed for the next decision in that cell.
- Define reusable helpers together with their first use, and reuse existing
  variables and helpers. Use bounded loops for repeated operations whose
  parameters and stopping conditions are already available.
- End the cell when the next action requires interpreting a new image or resolving
  an uncertain outcome. Do not guess image contents, skip outcome checks, or treat
  a successful API return as proof that a grasp or placement succeeded.
These are execution preferences; all task requirements and verification rules
still apply. Do not combine actions across a boundary that requires new evidence.
""".strip()


def execution_prompt(prompt: str, *, batch_actions: bool, arm: str) -> str:
    if type(batch_actions) is not bool:
        raise ValueError("batch_actions must be a boolean")
    if not batch_actions:
        return prompt
    if arm != "cells":
        raise ValueError("batch_actions requires the persistent Python cells arm")
    return prompt + "\n\n" + BATCH_ACTIONS_GUIDANCE
