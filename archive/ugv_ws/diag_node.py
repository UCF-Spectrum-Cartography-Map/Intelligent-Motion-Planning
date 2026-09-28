"""
diag_node.py — pinpoint the node-creation hang.

Auto-dumps the full stack of every thread after 20s if it's still stuck, so we
see the exact frame (rcl/DDS vs torch threads) without catching Ctrl-\\.

Run standalone in a TTY:  python3 -u diag_node.py
"""
import faulthandler
faulthandler.dump_traceback_later(20, exit=False)   # auto stack dump after 20s

print("A) creating ROS node BEFORE torch ...", flush=True)
import rclpy
from rclpy.node import Node
rclpy.init()
n = Node("diag_pre")
print("   -> node created OK before torch", flush=True)

print("B) importing torch ...", flush=True)
import torch
torch.set_num_threads(1)
print(f"   -> torch {torch.__version__} imported", flush=True)

print("C) creating a second ROS node AFTER torch ...", flush=True)
n2 = Node("diag_post")
print("   -> node created OK after torch", flush=True)

print("ALL OK — neither ordering hangs.", flush=True)
faulthandler.cancel_dump_traceback_later()
rclpy.shutdown()
