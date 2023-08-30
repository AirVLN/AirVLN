from enum import Enum
from typing import Dict, Union
import attr


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


class _DefaultAirsimActions(Enum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    GO_UP = 4
    GO_DOWN = 5
    MOVE_LEFT = 6
    MOVE_RIGHT = 7


@attr.s(auto_attribs=True, slots=True)
class AirsimActionsSingleton(metaclass=Singleton):
    r"""Implements an extendable Enum for the mapping of action names
    to their integer values.

    This means that new action names can be added, but old action names cannot
    be removed nor can their mapping be altered. This also ensures that all
    actions are always contigously mapped in :py:`[0, len(AirsimActions) - 1]`

    This accesible as the global singleton :ref:`AirsimActions`
    """

    _known_actions: Dict[str, int] = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        for action in _DefaultAirsimActions:
            self._known_actions[action.name] = action.value

    def __getattr__(self, name):
        return self._known_actions[name]

    def __getitem__(self, name):
        return self._known_actions[name]

    def __len__(self):
        return len(self._known_actions)

    def __iter__(self):
        return iter(self._known_actions)


class _DefaultAirsimActionSettings(Dict):
    FORWARD_STEP_SIZE = 5
    UP_DOWN_STEP_SIZE = 2
    LEFT_RIGHT_STEP_SIZE = 5
    TURN_ANGLE = 15
    TILT_ANGLE = 15


AirsimActions: AirsimActionsSingleton = AirsimActionsSingleton()
AirsimActionSettings: _DefaultAirsimActionSettings = _DefaultAirsimActionSettings()
