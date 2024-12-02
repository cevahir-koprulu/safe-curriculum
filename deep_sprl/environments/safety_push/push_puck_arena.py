import pathlib
import os
import robosuite.models.arenas
from robosuite.models.arenas import TableArena


class PushPuckArena(TableArena):
    """
    Workspace that contains a tabletop with a slot made of blocks.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
    """

    def __init__(
        self,
        table_full_size=(0.45, 0.69, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0),
    ):
        env_path = pathlib.Path(__file__).parent.resolve()
        arenas_path = pathlib.Path(robosuite.models.arenas.__file__).parent.parent.resolve()
        if not os.path.exists(f"{env_path}/assets"):
            os.system(f"cp -r {arenas_path}/assets {env_path}/assets")
        os.system(f"cp {env_path}/push_puck_arena.xml {env_path}/assets/arenas/push_puck_arena.xml")
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            xml=f"{env_path}/assets/arenas/push_puck_arena.xml",
        )

        # Get references to walls body
        self.walls_body = self.worldbody.find("./body[@name='walls']")