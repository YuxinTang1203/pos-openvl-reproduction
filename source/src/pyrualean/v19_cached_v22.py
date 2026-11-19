"""V19 B with action-boundary/cell-boundary calibration-only memoization.

Original V19 builds the harness, prompt, tools and feedback. This adapter adds
no model guidance, execution guard, recovery API, robot action, or sensing call.
Only consecutive identical camera_meta calls between action/cell/read boundaries
can reuse data. Other reads preserve their ordinary state-envelope side effects.
"""
from copy import deepcopy
import functools
import inspect
import time

from .prerequisites_v19 import build_harness as build_v19
from .prerequisites_v19 import validate_options as validate_v19

KIND = 'v22_v19_cached'


class CellCalibrationCache:
    """V20-derived calibration cache narrowed to consecutive same-key reads."""
    def __init__(self, robot):
        self.values = {}
        self.hits = 0
        self.active_key = None
        self.within_camera = 0
        self.cell_sequence = 0
        self.cell_stats = {'hits':0, 'misses':0, 'clears':0, 'hit_elapsed_s':0., 'miss_elapsed_s':0.}
        original = robot.camera_meta
        signature = inspect.signature(original)

        @functools.wraps(original)
        def cached(*args, **kwargs):
            started = time.perf_counter()
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key = tuple(bound.arguments.items())
            try:
                hash(key)
            except TypeError:
                # Preserve the original API's argument validation/error type.
                self.clear()
                return original(*args, **kwargs)
            if key != self.active_key:
                self.clear()
                self.active_key = key
            if key not in self.values:
                self.cell_stats['misses'] += 1
                self.within_camera += 1
                try:
                    self.values[key] = deepcopy(original(*args, **kwargs))
                finally:
                    self.within_camera -= 1
                    self.cell_stats['miss_elapsed_s'] += time.perf_counter()-started
            else:
                self.hits += 1
                self.cell_stats['hits'] += 1
                self.cell_stats['hit_elapsed_s'] += time.perf_counter()-started
            return deepcopy(self.values[key])
        robot.camera_meta = cached
        # artifact does not go through _readonly; show only queues an image.
        # Wrap both explicitly. OpenVL's internal reads also use these robot
        # boundaries. Reads internal to the one camera_meta miss remain intact.
        for name in ('_readonly', 'artifact', 'show'):
            if not hasattr(robot, name):
                continue
            method = getattr(robot, name)

            def wrap_reader(method):
                @functools.wraps(method)
                def reader(*args, **kwargs):
                    if not self.within_camera:
                        self.clear()
                    return method(*args, **kwargs)
                return reader
            setattr(robot, name, wrap_reader(method))
        if hasattr(robot, '_stateful'):
            stateful = robot._stateful
            @functools.wraps(stateful)
            def attempted_action(*args, **kwargs):
                # Also precedes _guard/stage/guardian refusals, which occur
                # before the ordinary action-feedback before() hook.
                self.clear()
                return stateful(*args, **kwargs)
            robot._stateful = attempted_action

    def clear(self):
        self.cell_stats['clears'] += 1
        self.values.clear()
        self.active_key = None

    def begin_cell(self):
        self.cell_sequence += 1
        self.cell_stats = {'hits':0, 'misses':0, 'clears':0, 'hit_elapsed_s':0., 'miss_elapsed_s':0.}
        self.clear()


def as_v19(config):
    if config.get('kind') != KIND:
        raise ValueError('expected v22_v19_cached')
    if config.get('guard_enabled', False) is not False:
        raise ValueError('v22_v19_cached is V19 B with guard disabled')
    if config.get('profile') != 'fusion':
        raise ValueError('v22_v19_cached requires the existing fusion profile')
    base = dict(config)
    base['kind'] = 'v19_prerequisites'
    base['guard_enabled'] = False
    return base


def validate_options(config):
    validate_v19(as_v19(config))


def build_harness(kit, config, task, initial_images=()):
    base = as_v19(config)
    validate_v19(base)
    harness = build_v19(kit, base, task, initial_images)
    cache = CellCalibrationCache(kit.robo)
    harness.calibration_cache = cache
    guard = harness.feedback_guard
    original_begin = guard.begin_turn
    original_before = guard.before

    def begin_turn():
        cache.begin_cell()
        return original_begin()

    def before(*, tool=None, arguments=None):
        # Runs before ALL atomic attempts, including subsequently rejected ones.
        cache.clear()
        return original_before(tool=tool, arguments=arguments)

    guard.begin_turn = begin_turn
    guard.before = before
    original_after = harness.after_tool
    def after_tool(name, result, records):
        try:
            return original_after(name, result, records)
        finally:
            if name in {'python', 'run_program'}:
                harness._log('v22_calibration_cache_cell', cell_sequence=cache.cell_sequence,
                    **dict(cache.cell_stats), cached_entries=len(cache.values),
                    interpretation='calibration read accounting, not recovery or task success')
    harness.after_tool = after_tool
    harness.provenance['v22_v19_cached'] = {
        'base': 'v19_B_conditional', 'guard_enabled': False,
        'cache': 'consecutive same camera/step camera_meta only; cell/action/other-read invalidation',
        'prompt_changed': False, 'tools_changed': False,
        'new_model_calls': 0, 'automatic_robot_actions': 0,
        'observation_or_target_cache': False,
    }
    harness._log('v22_v19_cached_configured', config=harness.provenance['v22_v19_cached'])
    return harness
