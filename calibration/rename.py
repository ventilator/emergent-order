#!/usr/bin/env python3

import sys
import shutil

for line in sys.stdin.readlines():
    tmp = line.split()
    label, ids = tmp[0], tmp[1:]
    ids = [x for x in ids if x not in ("", ",")]
    ids = [int(x) for x in ids]
    label = label.lower()
    for i, _id in enumerate(ids):
        shutil.copyfile(f"{label}_{_id:04d}.png", f"{label}_x{i}.png")

