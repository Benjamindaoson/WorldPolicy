"""Read-only prerequisite check; does not download datasets or activate hardware."""

import importlib.util
import json
import platform


def main() -> None:
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("torch", "mujoco", "lerobot", "libero", "robosuite")}
    result = {"platform": platform.system(), "packages": packages,
              "ready_for_official_libero": platform.system() == "Linux" and all(packages.values())}
    print(json.dumps(result, indent=2))
    if not result["ready_for_official_libero"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
