"""Native mechanisms and an explicit evidence/history combination in PyRUA.

OpenVL recovery is explicitly opt-in; existing profiles retain their behavior. An explicit
baseline follows the same initialization path while adding no tools or prompt.
"""

from pathlib import Path

_STATE_HARNESSES = {
    "reflact": ("reflact_harness", "ReflActHarness"),
    "pos": ("pos_harness", "PoSHarness"),
    "pos_progress_lite": ("progress_lite", "ProgressLiteHarness"),
    "openvl_pos_reflact": ("composite_harness", "CompositeHarness"),
    "minimal_fusion": ("minimal_fusion", "MinimalFusionHarness"),
    "v17_ablation": ("ablation_v17", "build_harness"),
    "v35_mast3r": ("mast3r_adapter_v35", "build_harness"),
    "v36_mast3r_auto": ("mast3r_auto_v36", "build_harness"),
    "v18_feedback_guard": ("feedback_guard_v18", "build_harness"),
    "v19_prerequisites": ("prerequisites_v19", "build_harness"),
    "v22_v19_cached": ("v19_cached_v22", "build_harness"),
    "longharness": ("long_harness", "LongHarness"),
    "evoharness": ("evo_harness", "EvoHarness"),
    "reuseit": ("reuseit_harness", "ReUseItHarness"),
    "rlm": ("rlm_harness", "RLMHarness"),
}


def _state_harness_module(kind):
    from importlib import import_module
    return import_module("." + _STATE_HARNESSES[kind][0], __package__)


_CLI_OPENVL_CONFIGS = (
    ("openvl_visual_config", "openvl_visual", "Local Grounding/Zoom and visual memory inside PyRUA"),
    ("openvl_routed_config", "openvl_routed", "Local visual tools plus automatic text-perception routing inside PyRUA"),
    ("openvl_capabilities_config", "openvl", "Legacy native capability/evidence adapter (kind=openvl; keeps historical behavior)"),
)


def add_migration_arguments(parser):
    """Use the same explicit, mutually exclusive selectors in both CLIs."""
    group = parser.add_mutually_exclusive_group()
    for name, _kind, description in _CLI_OPENVL_CONFIGS:
        group.add_argument("--" + name.replace("_", "-"), type=Path, metavar="PATH",
                           help=description + "; requires --runtime openai")
    group.add_argument("--openvl-evidence", action="store_true",
                       help="Legacy evidence-only adapter without changing the agent loop")
    group.add_argument("--general-harness-config", type=Path, metavar="PATH",
                           help="General harness JSON; recap_full requires openai, lightweight variants also support codex")


def cli_migration(args):
    """Resolve only migration selection; never change model, budgets or API docs."""
    selected = [(name, kind, getattr(args, name, None)) for name, kind, _ in _CLI_OPENVL_CONFIGS
                if getattr(args, name, None) is not None]
    general = getattr(args, "general_harness_config", None)
    if general:
        import json
        if selected or getattr(args, "openvl_evidence", False):
            raise ValueError("Choose only one migration")
        config_path = Path(general).resolve()
        config = json.loads(config_path.read_text())
        if config.get("kind") not in {
            "baseline", "recap", "recap_full", "ace", "failure_memory", *_STATE_HARNESSES
        }:
            raise ValueError("not a general harness configuration")
        if config.get("kind") in {"openvl_pos_reflact", "minimal_fusion", "v17_ablation", "v35_mast3r", "v36_mast3r_auto", "v18_feedback_guard", "v19_prerequisites", "v22_v19_cached"} and config.get("openvl", {}).get("config"):
            config["openvl"]["config"] = str((config_path.parent / config["openvl"]["config"]).resolve())
        if "playbook" in config:
            config["playbook"] = str((config_path.parent / config["playbook"]).resolve())
        if "workflow" in config:
            config["workflow"] = str((config_path.parent / config["workflow"]).resolve())
        if "book" in config:
            config["book"] = str((config_path.parent / config["book"]).resolve())
        validate_migration_runtime(config, getattr(args, "runtime", None))
        return config
    if len(selected) > 1 or (selected and getattr(args, "openvl_evidence", False)):
        raise ValueError("Choose only one OpenVL config flag or --openvl-evidence")
    if not selected:
        return None
    name, kind, config = selected[0]
    if getattr(args, "runtime", None) != "openai":
        raise ValueError("--" + name.replace("_", "-") + " requires --runtime openai")
    return {"kind": kind, "config": str(Path(config).resolve())}


def validate_migration(config):
    if not isinstance(config, dict):
        raise ValueError("migration must be an object")
    if config.get("kind") in _STATE_HARNESSES:
        _state_harness_module(config["kind"]).validate_options(config)
        return
    allowed = {
        "baseline": {"kind"},
        "recap": {"kind"},
        "recap_full": {"kind", "window_rounds", "max_depth", "max_pending"},
        "ace": {"kind", "playbook", "sha256"},
        "failure_memory": {"kind"},
        "harnesspro": {"kind", "root", "policy"},
        "vista": {"kind", "root"},
        "openvl_evidence": {"kind", "validate_geometry", "max_failure_images"},
        "openvl": {"kind", "config", "local_zoom", "local_coding", "validate_geometry", "max_failure_images", "recovery", "feedback_mode", "completion_review", "expose_robot_tools", "directory_mode"},
        "openvl_visual": {"kind", "config"},
        "openvl_routed": {"kind", "config", "ambiguity_score_margin", "duplicate_mask_iou"},
        "openvl_vista": {"kind", "root", "validate_geometry", "max_failure_images"},
    }
    kind = config.get("kind")
    if kind not in allowed or set(config) - allowed[kind]:
        raise ValueError("choose a supported migration; implicit mixtures and recovery options are forbidden")
    if kind == "recap_full":
        from .recap_full import validate_options
        validate_options(config)
    if kind == "ace":
        if not config.get("playbook") or not config.get("sha256"):
            raise ValueError("ACE requires a frozen playbook and sha256")
        from .general_harness import load_playbook
        load_playbook(config)
    if kind in {"harnesspro", "vista", "openvl_vista"} and not config.get("root"):
        raise ValueError(f"{kind} needs root")
    if kind == "harnesspro" and not config.get("policy"):
        raise ValueError("harnesspro needs policy")
    if kind in {"openvl_visual", "openvl_routed"} and not config.get("config"):
        raise ValueError(f"{kind} needs a native grounding+zoom config")
    if kind == "openvl_routed":
        from .openvl_routed import validate_ambiguity_thresholds
        validate_ambiguity_thresholds(config.get("ambiguity_score_margin", .10), config.get("duplicate_mask_iou", .90))
    if kind == "openvl":
        if config.get("completion_review", "off") not in {"off", "observe", "gate"}:
            raise ValueError("completion_review must be off, observe, or gate")
        if config.get("completion_review", "off") != "off" and not config.get("config"):
            raise ValueError("completion_review requires OpenVL config")
        if not config.get("config"):
            raise ValueError("openvl needs a native capability config")
        if config.get("feedback_mode", "full") not in {"full", "ondemand", "auto"}:
            raise ValueError("feedback_mode must be full, ondemand, or auto")
        if "recovery" in config and type(config["recovery"]) is not bool:
            raise ValueError("recovery must be boolean")
        if config.get("directory_mode", "all") not in {"all", "evidence"}:
            raise ValueError("directory_mode must be all or evidence")
        if "expose_robot_tools" in config and type(config["expose_robot_tools"]) is not bool:
            raise ValueError("expose_robot_tools must be boolean")
        if "local_coding" in config and type(config["local_coding"]) is not bool:
            raise ValueError("local_coding must be boolean")
        if "local_zoom" in config and type(config["local_zoom"]) is not bool:
            raise ValueError("local_zoom must be boolean")


def validate_migration_runtime(config, runtime):
    validate_migration(config)
    if config["kind"] == "recap_full" and runtime != "openai":
        raise ValueError("recap_full requires runtime=openai to own recursive control and the sliding context window; use recap for Codex's lightweight mode")
    general = config["kind"] in {"baseline", "recap", "ace", "failure_memory"}
    if runtime != "openai" and not (general and runtime == "codex"):
        raise ValueError("this migration requires runtime=openai; general harness also supports codex")


def configure_migration(kit, config, *, task, initial_images, base_url=None, api_key=None,
                        model=None, runtime="openai"):
    validate_migration_runtime(config, runtime)
    kind = config["kind"]
    if (kit.evidence_adapter or kit.native_vista or kit.harnesspro_hook or kit.stage_verifier
            or kit.robo._action_feedback or kit.general_harness or kit.guardian_verifier):
        raise ValueError("independent migration requires a clean toolkit")
    agent_loop = "codex exec" if runtime == "codex" else "pyrualean.openai_runtime"
    record = {"kind": kind, "agent_loop": agent_loop,
              "execution_recovery": False, "stage_verification": False}
    if kind == "recap_full":
        from .recap_full import ReCAPFull
        kit.general_harness = ReCAPFull(kit, config, task)
        record["adaptation"] = kit.general_harness.provenance
    elif kind in {"openvl_pos_reflact", "minimal_fusion"}:
        from .composite_harness import CompositeHarness
        from .minimal_fusion import MinimalFusionHarness
        visual = config.get("openvl", {})
        if visual.get("enabled", True):
            from .openvl_routed import OpenVLRoutedAdapter
            kit.evidence_adapter = OpenVLRoutedAdapter(
                kit, config=visual["config"], initial_images=initial_images,
                ambiguity_score_margin=visual.get("ambiguity_score_margin", .10),
                duplicate_mask_iou=visual.get("duplicate_mask_iou", .90))
            record["native"] = kit.evidence_adapter.provenance
        cls = MinimalFusionHarness if kind == "minimal_fusion" else CompositeHarness
        kit.general_harness = cls(kit, config, task, initial_images=initial_images)
        record["adaptation"] = kit.general_harness.provenance
    elif kind in _STATE_HARNESSES:
        cls = getattr(_state_harness_module(kind), _STATE_HARNESSES[kind][1])
        kit.general_harness = cls(kit, config, task, initial_images=initial_images)
        if kind in {"v17_ablation", "v35_mast3r", "v36_mast3r_auto", "v18_feedback_guard", "v19_prerequisites", "v22_v19_cached"} and kit.evidence_adapter is not None:
            record["native"] = kit.evidence_adapter.provenance
        record["adaptation"] = kit.general_harness.provenance
    elif kind in {"recap", "ace", "failure_memory"}:
        from .general_harness import GeneralHarness
        kit.general_harness = GeneralHarness(kit, config)
        record["adaptation"] = kit.general_harness.provenance
    elif kind == "harnesspro":
        import json
        from .harnesspro_runtime import NativeHarnessProHook
        policy_path = Path(config["policy"])
        options = json.loads(policy_path.read_text()).get("pyrualean", {})
        if any(options.get(key) for key in ("action_feedback", "execution_recovery", "stage_verification")):
            raise ValueError("independent HarnessPro policy must disable extra robot recovery/feedback")
        kit.harnesspro_hook = NativeHarnessProHook(
            root=Path(config["root"]), policy_file=policy_path,
            workspace=kit.workspace, task_description=task, base_url=base_url, api_key=api_key)
        record["native"] = kit.harnesspro_hook.record()
        kit.robo._harnesspro_feedback = kit.harnesspro_hook
    elif kind in {"vista", "openvl_vista"}:
        from .vista_native import NativeVistaMemory
        kit.native_vista = NativeVistaMemory(
            source=Path(config["root"]), directory=kit.workspace.parent / "vista_native",
            task=task, initial_images=initial_images)
        record["native"] = kit.native_vista.provenance()
        if kind == "openvl_vista":
            kit.enable_openvl_evidence(
                validate_geometry=config.get("validate_geometry", True),
                max_failure_images=config.get("max_failure_images", 2),
                execution_recovery=False, stage_verification=False, temporal_feedback=True)
            record["native"] = {"vista": record["native"],
                                "openvl_evidence": kit.evidence_adapter.provenance}
    elif kind == "openvl_visual":
        from .openvl_visual import OpenVLVisualAdapter
        kit.evidence_adapter = OpenVLVisualAdapter(kit, config=config["config"], initial_images=initial_images)
        record["native"] = kit.evidence_adapter.provenance
    elif kind == "openvl_routed":
        from .openvl_routed import OpenVLRoutedAdapter
        kit.evidence_adapter = OpenVLRoutedAdapter(kit, config=config["config"], initial_images=initial_images,
            ambiguity_score_margin=config.get("ambiguity_score_margin", .10),
            duplicate_mask_iou=config.get("duplicate_mask_iou", .90))
        record["native"] = kit.evidence_adapter.provenance
    elif kind == "openvl":
        from .openvl_native import OpenVLNativeAdapter
        kit.evidence_adapter = OpenVLNativeAdapter(
            kit, config=config["config"], initial_images=initial_images, model=model,
            feedback_mode=config.get("feedback_mode", "full"),
            expose_robot_tools=config.get("expose_robot_tools", True),
            directory_mode=config.get("directory_mode", "all"),
            local_zoom=config.get("local_zoom", True),
            local_coding=config.get("local_coding", False), base_url=base_url, api_key=api_key,
            validate_geometry=config.get("validate_geometry", True),
            max_failure_images=config.get("max_failure_images", 2))
        mode = config.get("completion_review", "off")
        if mode != "off":
            if not base_url or not api_key or not model:
                raise ValueError("completion_review requires the planner model and gateway credentials")
            from .openvl_completion import CompletionReviewer
            kit.evidence_adapter.completion_reviewer = CompletionReviewer(
                kit.evidence_adapter, task, initial_images, base_url, api_key, model, mode)
        kit.evidence_adapter.provenance['completion_review'] = mode
        import json
        (kit.evidence_adapter.root / 'provenance.json').write_text(json.dumps(kit.evidence_adapter.provenance, indent=2) + '\n')
        record["native"] = kit.evidence_adapter.provenance
    elif kind == "openvl_evidence":
        kit.enable_openvl_evidence(
            validate_geometry=config.get("validate_geometry", True),
            max_failure_images=config.get("max_failure_images", 2),
            execution_recovery=False, stage_verification=False)
        record["native"] = kit.evidence_adapter.provenance
    if kind == "openvl" and config.get("recovery", False):
        from .openvl_recovery import install
        install(kit)
        record["execution_recovery"] = True
        record["recovery_protocol"] = "openvl-cross-turn-v1"
    return record


def migration_prompt(prompt, kit):
    if kit.general_harness:
        return prompt + kit.general_harness.prompt()
    if kit.native_vista:
        from .vista_native import adapt_robot_prompt
        prompt = adapt_robot_prompt(prompt, kit.arm) + kit.native_vista.prompt()
        if kit.evidence_adapter:
            prompt += """
# OpenVL evidence with VISTA history
OpenVL Robot_Perception_Tool/Robot_Evidence_Tool and Python evidence helpers are
also available. openvl_vista_link.binding_history maps each new binding to its
original perception event/view, source step and image hash. Its vista_event is
the separate end-of-tool event, which may be after motion. Use VISTA for historical image
comparison and notes, and OpenVL for binding lookup and validity checks.
An archived image or note never refreshes a binding's geometry. Evidence overlays
are labeled evidence_*; they are derived views, not a new calibrated camera.
Retrieve additional images when needed; both modules use the same PyRUA loop.
Use `with evidence_target(ref, intent='move'|'grasp'|'contact'|'release'):` in
Python (or robo.evidence_target) to explicitly associate actions with measured
target evidence. Here ref must be the ID STRING matches[0]['reference'] from a
nonempty evidence_find() result, never the entire matches[0] dictionary or None.
If an ID/type error occurs, correct the reference argument; bypassing the scope
does not repair the evidence association. The reference must be current at scope entry; motion inside
the scope does not refresh it. This scope declares intent, never task success,
and does not stop a failed motion. Use normal Python conditions on actual tool
returns when subsequent actions depend on them. Contact may be intentional.
Failure evidence includes the first anomaly's image and step alongside the
current image. Consult its linked VISTA event to compare history. Treat
unassigned targets and missing snapshots as unknown, rather than guessing.
Before claiming completion, inspect fresh visible outcomes; reaching a pose or
executing a command alone does not establish completion. Keep multi-action cells.
"""
        return prompt
    if kit.harnesspro_hook:
        context = (kit.harnesspro_hook.session_context.get("hookSpecificOutput") or {}).get("additionalContext")
        if context:
            return prompt + "\n# HarnessPro runtime\n" + str(context) + "\n"
    # OpenVL's prompt is adapted by the existing play_episode path exactly once.
    return prompt
