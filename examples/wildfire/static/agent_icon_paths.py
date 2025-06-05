from dataclasses import dataclass


@dataclass(frozen=True)
class Icon:
    # Icon paths
    helicopter: str = "helicopter.svg"
    drone: str = "drone.svg"
    normal_repr: str = "agent_repr_prop.svg"
    blue_aircraft: str = "plane.svg"
    # Icon made by "https://www.flaticon.com/authors/pixel-perfect"
    # "Pixel perfect" from "https://www.flaticon.com/"
