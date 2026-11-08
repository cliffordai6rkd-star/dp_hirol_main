import argparse
import os
import struct
import time
from dataclasses import dataclass, field

import yaml


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_SIZE = 8


@dataclass
class XboxState:
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    lt: float = 0.0
    r2: float = 0.0
    a: float = 0.0
    y: float = 0.0
    reset_modifier: float = 0.0
    buttons: dict[int, int] = field(default_factory=dict)


class LinuxXboxController:
    def __init__(self, config):
        self.device = config.get("xbox_device", "/dev/input/js0")
        self.deadzone = float(config.get("xbox_deadzone", 0.12))
        self.trigger_deadzone = float(config.get("xbox_trigger_deadzone", 0.02))
        self.axis_map = config.get(
            "xbox_axis_map",
            {
                "left_x": 0,
                "left_y": 1,
                "lt": 2,
                "right_x": 3,
                "right_y": 4,
                "r2": 5,
            },
        )
        self.axis_sign = config.get(
            "xbox_axis_sign",
            {
                "left_x": 1.0,
                "left_y": -1.0,
                "lt": 1.0,
                "right_x": 1.0,
                "right_y": -1.0,
                "r2": 1.0,
            },
        )
        self.button_map = config.get(
            "xbox_button_map",
            {
                "a": 0,
                "y": 3,
            },
        )
        self.lt_button = config.get("xbox_lt_button", None)
        self.r2_button = config.get("xbox_r2_button", None)
        self.reset_button = config.get("xbox_reset_button", None)

        if not os.path.exists(self.device):
            raise FileNotFoundError(
                f"Xbox joystick device not found: {self.device}. "
                "Check /dev/input/js* or pass the device into the container."
            )

        self.fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = {}
        self.buttons = {}
        self.poll()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _normalize_axis(self, value):
        value = max(-32767, min(32767, int(value)))
        return value / 32767.0

    def _apply_deadzone(self, value, deadzone):
        abs_value = abs(value)
        if abs_value < deadzone:
            return 0.0
        return (abs_value - deadzone) / (1.0 - deadzone) * (1.0 if value >= 0.0 else -1.0)

    def _read_events(self):
        while True:
            try:
                event = os.read(self.fd, JS_EVENT_SIZE)
            except BlockingIOError:
                break

            if len(event) != JS_EVENT_SIZE:
                break

            _timestamp_ms, value, event_type, number = struct.unpack("IhBB", event)
            event_type = event_type & ~JS_EVENT_INIT

            if event_type == JS_EVENT_AXIS:
                self.axes[number] = self._normalize_axis(value)
            elif event_type == JS_EVENT_BUTTON:
                self.buttons[number] = int(value)

    def _axis(self, name, default=0.0):
        axis_id = self.axis_map.get(name, None)
        if axis_id is None:
            return default
        sign = float(self.axis_sign.get(name, 1.0))
        return sign * float(self.axes.get(int(axis_id), default))

    def _stick_axis(self, name):
        return self._apply_deadzone(self._axis(name), self.deadzone)

    def _trigger(self, name):
        if name == "lt" and self.lt_button is not None:
            return float(self.buttons.get(int(self.lt_button), 0))
        if name == "r2" and self.r2_button is not None:
            return float(self.buttons.get(int(self.r2_button), 0))

        raw = self._axis(name, default=-1.0)
        value = 0.5 * (raw + 1.0)
        value = max(0.0, min(1.0, value))
        if value < self.trigger_deadzone:
            return 0.0
        return value

    def _button(self, name):
        button_id = self.button_map.get(name, None)
        if button_id is None:
            return 0.0
        return float(self.buttons.get(int(button_id), 0))

    def poll(self):
        self._read_events()
        return XboxState(
            left_x=self._stick_axis("left_x"),
            left_y=self._stick_axis("left_y"),
            right_x=self._stick_axis("right_x"),
            right_y=self._stick_axis("right_y"),
            lt=self._trigger("lt"),
            r2=self._trigger("r2"),
            a=self._button("a"),
            y=self._button("y"),
            reset_modifier=float(self.buttons.get(int(self.reset_button), 0))
            if self.reset_button is not None
            else 0.0,
            buttons=dict(self.buttons),
        )


def main():
    parser = argparse.ArgumentParser(description="Print normalized Xbox controller state.")
    parser.add_argument("--config", default="config/sim_cfg/replay_test.yaml")
    parser.add_argument("--raw", action="store_true", help="Print raw axis/button dictionaries too.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    controller = LinuxXboxController(config)
    try:
        while True:
            state = controller.poll()
            line = "lx={:+.2f} ly={:+.2f} rx={:+.2f} ry={:+.2f} lt={:.2f} r2={:.2f} a={:.0f} y={:.0f} reset={:.0f}".format(
                    state.left_x,
                    state.left_y,
                    state.right_x,
                    state.right_y,
                    state.lt,
                    state.r2,
                    state.a,
                    state.y,
                    state.reset_modifier,
            )
            if args.raw:
                axes = " ".join(f"{idx}:{value:+.2f}" for idx, value in sorted(controller.axes.items()))
                buttons = " ".join(f"{idx}:{value}" for idx, value in sorted(controller.buttons.items()))
                line = f"{line} | axes[{axes}] buttons[{buttons}]"
            print(line, end="\r", flush=True)
            time.sleep(0.05)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
