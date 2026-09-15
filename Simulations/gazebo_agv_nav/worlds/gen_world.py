"""Generate a Gazebo SDF world from our D4-symmetric grid (map_generator.py),
so the same map layout/seed used in the lightweight prototype and later
generalization tests is reproduced exactly as box obstacles in Gazebo.
Each grid cell = 1 meter.
"""
import os, sys, argparse
sys.path.insert(0, "/workspace")
from envs.map_generator import symmetric_corridor, asymmetric_corridor

SDF_HEADER = """<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="agv_nav">
    <physics name="1ms" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>5</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"></plugin>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <direction>-0.5 0.1 -0.9</direction>
      <diffuse>0.9 0.9 0.85 1</diffuse>
    </light>
    <include>
      <uri>{model_uri}</uri>
      <name>burger</name>
      <pose>{spawn_x} {spawn_y} 0.05 0 0 0</pose>
    </include>
    <model name="goal_marker">
      <static>false</static>
      <pose>-5 -5 0.15 0 0 0</pose>
      <link name="link">
        <gravity>false</gravity>
        <visual name="visual">
          <geometry><cylinder><radius>0.25</radius><length>0.3</length></cylinder></geometry>
          <material>
            <ambient>0.1 0.9 0.2 1</ambient>
            <diffuse>0.15 1.0 0.25 1</diffuse>
            <emissive>0.05 0.5 0.05 1</emissive>
          </material>
        </visual>
      </link>
    </model>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.55 0.55 0.58 1</ambient>
            <diffuse>0.6 0.6 0.63 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

# Warehouse pallet-rack look: same 0.8x0.8x1.8 collision box (occupancy-grid /
# training physics unchanged) but visually a thin steel frame (4 posts + 2
# shelf planks) with a couple of cardboard-colored boxes sitting on the
# shelves, instead of one solid block. The gz-sim SERVER never renders
# anything (it runs headless with `-s`) -- only a GUI client you open
# yourself does -- so this adds zero training-time cost no matter how much
# visual detail is added.
POST_OFF = 0.34
RACK_LINK = """
      <link name="link">
        <collision name="collision"><geometry><box><size>0.8 0.8 1.8</size></box></geometry></collision>
        {posts}
        {planks}
        {boxes}
      </link>
"""
_POST = """
        <visual name="post_{n}">
          <pose>{px} {py} 0 0 0 0</pose>
          <geometry><box><size>0.06 0.06 1.8</size></box></geometry>
          <material><ambient>0.55 0.45 0.05 1</ambient><diffuse>0.65 0.52 0.06 1</diffuse></material>
        </visual>"""
_PLANK = """
        <visual name="plank_{n}">
          <pose>0 0 {pz} 0 0 0</pose>
          <geometry><box><size>0.8 0.8 0.04</size></box></geometry>
          <material><ambient>0.6 0.5 0.08 1</ambient><diffuse>0.72 0.6 0.1 1</diffuse></material>
        </visual>"""
_CBOX = """
        <visual name="cbox_{n}">
          <pose>{bx} {by} {bz} 0 0 {byaw}</pose>
          <geometry><box><size>{bs} {bs} {bs}</size></box></geometry>
          <material><ambient>0.72 0.56 0.38 1</ambient><diffuse>0.8 0.63 0.42 1</diffuse></material>
        </visual>"""

_posts_xml = "".join(
    _POST.format(n=k, px=px, py=py)
    for k, (px, py) in enumerate([(POST_OFF, POST_OFF), (-POST_OFF, POST_OFF),
                                   (POST_OFF, -POST_OFF), (-POST_OFF, -POST_OFF)])
)
_planks_xml = "".join(_PLANK.format(n=k, pz=pz) for k, pz in enumerate([-0.3, 0.3]))
_boxes_xml = "".join(
    _CBOX.format(n=k, bx=bx, by=by, bz=bz, bs=bs, byaw=byaw)
    for k, (bx, by, bz, bs, byaw) in enumerate([
        (0.05, -0.05, 0.47, 0.28, 0.3),
        (-0.15, 0.1, 0.47, 0.22, -0.4),
        (0.0, 0.0, -0.13, 0.3, 0.1),
    ])
)
BOX_TEMPLATE = ("""
    <model name="obs_{i}">
      <static>true</static>
      <pose>{x} {y} 0.9 0 0 0</pose>""" + RACK_LINK.format(
    posts=_posts_xml, planks=_planks_xml, boxes=_boxes_xml) + """
    </model>
""")

# Perimeter wall cells (same footprint as before) rendered as brick-red,
# floor-to-ceiling-ish walls instead of orange pallets, so the boundary reads
# as a building wall.
WALL_TEMPLATE = """
    <model name="wall_{i}">
      <static>true</static>
      <pose>{x} {y} 1.25 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>0.8 0.8 2.5</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>0.8 0.8 2.5</size></box></geometry>
          <material>
            <ambient>0.5 0.18 0.14 1</ambient>
            <diffuse>0.6 0.22 0.16 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

# Purely visual (no collision) floor lane stripe -- cheap way to get the
# "hazard/lane marking" warehouse-floor look from the reference images.
STRIPE_TEMPLATE = """
    <model name="stripe_{i}">
      <static>true</static>
      <pose>{x} {y} 0.01 0 0 {yaw}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{length} 0.15 0.01</size></box></geometry>
          <material>
            <ambient>0.9 0.75 0.05 1</ambient>
            <diffuse>0.95 0.8 0.1 1</diffuse>
            <emissive>0.15 0.12 0 1</emissive>
          </material>
        </visual>
      </link>
    </model>
"""

SDF_FOOTER = """
  </world>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_type", choices=["symmetric", "asymmetric"], default="symmetric")
    ap.add_argument("--grid_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/worlds/agv_nav.sdf")
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(args.seed)
    gen = symmetric_corridor if args.map_type == "symmetric" else asymmetric_corridor
    grid = gen(size=args.grid_size, rng=rng)

    spawn_x, spawn_y = args.grid_size / 2, args.grid_size / 2
    default_uri = "/opt/ros/jazzy/share/turtlebot3_gazebo/models/turtlebot3_burger"
    if not os.path.exists(default_uri):
        default_uri = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "models", "turtlebot3_burger")
    model_uri = os.environ.get("TB3_MODEL_URI", default_uri)
    sdf = SDF_HEADER.format(spawn_x=spawn_x, spawn_y=spawn_y, model_uri=model_uri)
    size = args.grid_size
    i = 0
    for y in range(size):
        for x in range(size):
            if not grid[y, x]:
                continue
            is_border = x == 0 or x == size - 1 or y == 0 or y == size - 1
            tmpl = WALL_TEMPLATE if is_border else BOX_TEMPLATE
            sdf += tmpl.format(i=i, x=x, y=y)
            i += 1

    # Two floor lane stripes crossing through the middle of the map -- cheap
    # visual detail, no effect on collision/occupancy.
    mid = size / 2
    sdf += STRIPE_TEMPLATE.format(i="h", x=mid, y=mid, yaw=0, length=size - 4)
    sdf += STRIPE_TEMPLATE.format(i="v", x=mid, y=mid, yaw=1.5708, length=size - 4)

    sdf += SDF_FOOTER

    with open(args.out, "w") as f:
        f.write(sdf)
    print(f"Wrote {args.out} with {i} obstacles ({args.map_type}, seed={args.seed})")


if __name__ == "__main__":
    main()
