# Lara Radovanovic
# 2025-09-21
# CAI5005 - Intro to AI
# Grid World Resource Gathering — A* Search (Tkinter)
# HUD + Speed Control + Legend + Show/Hide Path toggle

from tkinter import *
from tkinter import ttk
import time
import heapq
import threading

# CODE ADAPTED FROM https://www.youtube.com/watch?v=UgsVkRwh6mQ
# Last accessed: 2025-09-21


# --------------------------
# Problem constants
# --------------------------
ROWS, COLS = 5, 5
BASE = (0, 0)
CAPACITY = 2
REQUIRED = {"stone": 3, "iron": 2, "crystal": 1}

TERRAIN_COST = {
    "grassland": 1,
    "hills": 2,
    "swamp": 3,
    "mountains": 4,
}

# For drawing
TERRAIN_COLOR = {
    "grassland": "#dff5d8",  # light green, easiest terrain (cost = 1)
    "hills":     "#f3e5c6",  # tan/beige, slightly harder (cost = 2)
    "swamp":     "#d7e6ef",  # pale blue, more difficult (cost = 3)
    "mountains": "#e7d6e8",  # light purple, hardest terrain (cost = 4)
}

RESOURCE_COLOR = {
    "stone":   "#808080",   # gray (S)
    "iron":    "#a0522d",   # sienna (I)
    "crystal": "#5b9bd5",   # blue-ish (C)
}

RESOURCE_LETTER = {"stone": "S", "iron": "I", "crystal": "C"}

# Visuals
AGENT_COLOR = "#1b5e20"
PATH_COLOR = "#1565c0"
BASE_OUTLINE = "#2e7d32"

# 4-directional moves
DIRS = [(1,0), (-1,0), (0,1), (0,-1)]

# --------------------------
# Three test maps
# --------------------------
def make_maps():
    m1_terrain = [
        ["grassland","grassland","hills","grassland","mountains"],
        ["grassland","swamp","grassland","grassland","hills"],
        ["grassland","hills","grassland","swamp","grassland"],
        ["swamp","grassland","mountains","grassland","grassland"],
        ["grassland","grassland","hills","grassland","swamp"],
    ]
    m1_resources = [
        {"type":"stone","pos":(1,3)},
        {"type":"stone","pos":(3,0)},
        {"type":"stone","pos":(4,2)},
        {"type":"iron","pos":(2,1)},
        {"type":"iron","pos":(4,4)},
        {"type":"crystal","pos":(0,4)},
    ]

    m2_terrain = [
        ["grassland","hills","grassland","grassland","swamp"],
        ["grassland","grassland","hills","grassland","grassland"],
        ["mountains","grassland","grassland","hills","grassland"],
        ["grassland","swamp","grassland","grassland","mountains"],
        ["grassland","hills","grassland","swamp","grassland"],
    ]
    m2_resources = [
        {"type":"stone","pos":(1,2)},
        {"type":"stone","pos":(3,1)},
        {"type":"stone","pos":(4,3)},
        {"type":"iron","pos":(2,3)},
        {"type":"iron","pos":(3,4)},
        {"type":"crystal","pos":(0,3)},
    ]

    m3_terrain = [
        ["grassland","grassland","grassland","hills","grassland"],
        ["hills","swamp","grassland","grassland","grassland"],
        ["grassland","grassland","hills","grassland","mountains"],
        ["grassland","mountains","grassland","swamp","grassland"],
        ["swamp","grassland","grassland","hills","grassland"],
    ]
    m3_resources = [
        {"type":"stone","pos":(1,1)},
        {"type":"stone","pos":(2,2)},
        {"type":"stone","pos":(4,1)},
        {"type":"iron","pos":(2,4)},
        {"type":"iron","pos":(3,2)},
        {"type":"crystal","pos":(0,2)},
    ]

    return [
        {"name":"Map 1", "terrain":m1_terrain, "resources":m1_resources},
        {"name":"Map 2", "terrain":m2_terrain, "resources":m2_resources},
        {"name":"Map 3", "terrain":m3_terrain, "resources":m3_resources},
    ]

# --------------------------
# A* over full problem
# State = (r, c, invS, invI, invC, dS, dI, dC, mask)
# mask: bitmask of picked resource indices
# --------------------------
class Planner:
    def __init__(self, terrain, resources):
        self.terrain = terrain
        self.resources = resources
        self.H, self.W = ROWS, COLS
        self.total_items = dict(REQUIRED)
        self.min_step_cost = 1
        self.resource_by_pos = {r["pos"]:(i, r["type"]) for i, r in enumerate(resources)}

    def in_bounds(self, r, c): return 0 <= r < self.H and 0 <= c < self.W
    def terrain_cost(self, r, c): return TERRAIN_COST[self.terrain[r][c]]

    def goal_reached(self, dS, dI, dC):
        return (dS >= self.total_items["stone"] and
                dI >= self.total_items["iron"] and
                dC >= self.total_items["crystal"])

    def needed_remaining(self, dS, dI, dC):
        return {
            "stone": max(0, self.total_items["stone"] - dS),
            "iron":  max(0, self.total_items["iron"]  - dI),
            "crystal": max(0, self.total_items["crystal"] - dC),
        }

    def inv_count(self, invS, invI, invC): return invS + invI + invC

    def collect_if_possible(self, r, c, invS, invI, invC, dS, dI, dC, mask):
        # Deliver at base
        if (r, c) == BASE and (invS or invI or invC):
            need = self.needed_remaining(dS, dI, dC)
            takeS = min(invS, need["stone"])
            takeI = min(invI, need["iron"])
            takeC = min(invC, need["crystal"])
            dS += takeS; dI += takeI; dC += takeC
            invS = invI = invC = 0

        # Pick up if needed & capacity allows
        if (r, c) in self.resource_by_pos:
            idx, rtype = self.resource_by_pos[(r, c)]
            if ((mask >> idx) & 1) == 0:
                need = self.needed_remaining(dS, dI, dC)
                if need[rtype] > 0 and self.inv_count(invS, invI, invC) < CAPACITY:
                    if rtype == "stone":   invS += 1
                    elif rtype == "iron":  invI += 1
                    else:                  invC += 1
                    mask |= (1 << idx)

        return invS, invI, invC, dS, dI, dC, mask

    def nearest_needed_resource_manhattan(self, r, c, dS, dI, dC, mask):
        need = self.needed_remaining(dS, dI, dC)
        dists = []
        for idx, res in enumerate(self.resources):
            if ((mask >> idx) & 1):   # already collected
                continue
            if need[res["type"]] <= 0:
                continue
            rr, cc = res["pos"]
            dists.append(abs(rr - r) + abs(cc - c))
        if dists:
            return min(dists)
        return abs(BASE[0] - r) + abs(BASE[1] - c)

    def heuristic(self, state):
        r, c, invS, invI, invC, dS, dI, dC, mask = state
        if self.goal_reached(dS, dI, dC): return 0
        if self.inv_count(invS, invI, invC) >= CAPACITY:
            return (abs(BASE[0] - r) + abs(BASE[1] - c)) * self.min_step_cost
        return self.nearest_needed_resource_manhattan(r, c, dS, dI, dC, mask) * self.min_step_cost

    def neighbors(self, state):
        r, c, invS, invI, invC, dS, dI, dC, mask = state
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not self.in_bounds(nr, nc): continue
            step = self.terrain_cost(nr, nc)
            nS, nI, nC, ndS, ndI, ndC, nmask = self.collect_if_possible(
                nr, nc, invS, invI, invC, dS, dI, dC, mask
            )
            yield (nr, nc, nS, nI, nC, ndS, ndI, ndC, nmask), step

    def reconstruct_path(self, came_from, end_state):
        path, s = [], end_state
        while s in came_from:
            s, pos = came_from[s]  # (prev_state, position entered)
            path.append(pos)
        path.reverse()
        return path

    def astar(self, start_state):
        t0 = time.time()
        g = {start_state: 0}
        f = {start_state: self.heuristic(start_state)}
        came_from = {}
        pq = [(f[start_state], 0, start_state)]
        seen = set()
        expansions = 0

        while pq:
            _, _, s = heapq.heappop(pq)
            if s in seen: continue
            seen.add(s)

            r, c, invS, invI, invC, dS, dI, dC, mask = s
            if self.goal_reached(dS, dI, dC):
                path = self.reconstruct_path(came_from, s)
                if not path or path[0] != BASE:
                    path = [BASE] + path
                return {
                    "path": path,
                    "cost": g[s],
                    "expanded": expansions,
                    "runtime_sec": time.time() - t0,
                    "goal_state": s,
                }

            expansions += 1
            for nstate, step_cost in self.neighbors(s):
                new_g = g[s] + step_cost
                if nstate not in g or new_g < g[nstate]:
                    g[nstate] = new_g
                    came_from[nstate] = (s, (nstate[0], nstate[1]))
                    fn = new_g + self.heuristic(nstate)
                    f[nstate] = fn
                    heapq.heappush(pq, (fn, expansions, nstate))
        return None

# --------------------------
# Tkinter UI with HUD + speed + legend + path toggle
# --------------------------
class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("Grid World — A* Resource Gathering")
        self.state("zoomed")

        self.maps = make_maps()
        self.current_map_index = 0

        # Remember last planned path so we can toggle show/hide without re-planning
        self.last_path = None

        # ---------- Top bar ----------
        top = Frame(self)
        top.pack(side=TOP, fill=X)

        Label(top, text="Map:").pack(side=LEFT, padx=(10,5))
        self.map_var = StringVar(value=self.maps[0]["name"])
        self.map_menu = ttk.Combobox(
            top, textvariable=self.map_var,
            values=[m["name"] for m in self.maps],
            state="readonly", width=30
        )
        self.map_menu.pack(side=LEFT, padx=(0,10))
        self.map_menu.bind("<<ComboboxSelected>>", self.on_map_change)

        Label(top, text="Animation Speed:").pack(side=LEFT, padx=(10,5))
        self.speed_var = StringVar(value="Normal")
        self.speed_menu = ttk.Combobox(
            top, textvariable=self.speed_var,
            values=["Slow","Normal","Fast"], state="readonly", width=10
        )
        self.speed_menu.pack(side=LEFT, padx=(0,10))

        # New: Show/Hide Path toggle
        self.show_path_var = BooleanVar(value=True)
        self.path_chk = Checkbutton(top, text="Show Path", variable=self.show_path_var,
                                    command=self.on_toggle_path)
        self.path_chk.pack(side=LEFT, padx=(4,10))

        self.plan_btn = Button(top, text="Plan & Animate", command=self.plan_and_animate)
        self.plan_btn.pack(side=LEFT)

        self.stats_var = StringVar(value="")
        Label(top, textvariable=self.stats_var, font=("Consolas", 11)).pack(side=RIGHT, padx=10)

        # ---------- Main area: left canvas + right HUD ----------
        main = Frame(self)
        main.pack(fill=BOTH, expand=1)

        self.can = Canvas(main, bg="#ffffff")
        self.can.pack(side=LEFT, fill=BOTH, expand=1)
        self.can.bind("<Configure>", lambda e: self.redraw())

        # HUD panel
        hud = Frame(main, width=300)
        hud.pack(side=RIGHT, fill=Y)
        hud.pack_propagate(False)

        Label(hud, text="HUD", font=("Consolas", 14, "bold")).pack(pady=(12,8))

        # Inventory
        Label(hud, text="Inventory (max 2):", font=("Consolas", 12)).pack(anchor="w", padx=10)
        self.inv_var = StringVar(value="S:0  I:0  C:0")
        Label(hud, textvariable=self.inv_var, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,8))

        # Delivered
        Label(hud, text="Delivered:", font=("Consolas", 12)).pack(anchor="w", padx=10)
        self.deliv_var = StringVar(value="S:0/3  I:0/2  C:0/1")
        Label(hud, textvariable=self.deliv_var, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,8))

        # Needed
        Label(hud, text="Needed remaining:", font=("Consolas", 12)).pack(anchor="w", padx=10)
        self.need_var = StringVar(value="S:3  I:2  C:1")
        Label(hud, textvariable=self.need_var, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,8))

        # Current tile
        Label(hud, text="Current tile:", font=("Consolas", 12)).pack(anchor="w", padx=10, pady=(12,0))
        self.tile_var = StringVar(value="(0,0) grassland  cost=1")
        Label(hud, textvariable=self.tile_var, font=("Consolas", 12)).pack(anchor="w", padx=24, pady=(0,12))

        # --- Legend ---
        sep = ttk.Separator(hud, orient=HORIZONTAL)
        sep.pack(fill=X, padx=10, pady=6)

        Label(hud, text="Legend", font=("Consolas", 13, "bold")).pack(anchor="w", padx=10, pady=(6,4))

        self.legend_frame = Frame(hud)
        self.legend_frame.pack(anchor="w", padx=12, pady=(0,10))
        self.build_legend(self.legend_frame)

        # Drawing/animation state
        self.agent_pos = BASE
        self.agent_item = None
        self.path_preview_ids = []
        self.tile_size_cache = (1,1)

        # initial draw
        self.redraw()
        self.reset_hud()

    # --- Legend UI ---
    def legend_row(self, parent, swatch_color, text, is_line=False, is_dot=False, text_color="#000"):
        row = Frame(parent); row.pack(anchor="w", pady=2)
        if is_line:
            cv = Canvas(row, width=28, height=12, highlightthickness=0, bg="white"); cv.pack(side=LEFT)
            cv.create_line(2, 6, 26, 6, width=3, fill=swatch_color)
        elif is_dot:
            cv = Canvas(row, width=28, height=20, highlightthickness=0, bg="white"); cv.pack(side=LEFT)
            cv.create_oval(6, 6, 20, 20, fill=swatch_color, outline="#0b3d15", width=2)
        else:
            cv = Canvas(row, width=28, height=18, highlightthickness=0, bg="white"); cv.pack(side=LEFT)
            cv.create_rectangle(2, 2, 26, 16, fill=swatch_color, outline="#333333")
        Label(row, text=text, font=("Consolas", 11), fg=text_color).pack(side=LEFT, padx=6)

    def build_legend(self, parent):
        # Terrain colors
        Label(parent, text="Terrain:", font=("Consolas", 12, "underline")).pack(anchor="w", pady=(0,2))
        self.legend_row(parent, TERRAIN_COLOR["grassland"], f"grassland (cost {TERRAIN_COST['grassland']})")
        self.legend_row(parent, TERRAIN_COLOR["hills"],     f"hills (cost {TERRAIN_COST['hills']})")
        self.legend_row(parent, TERRAIN_COLOR["swamp"],     f"swamp (cost {TERRAIN_COST['swamp']})")
        self.legend_row(parent, TERRAIN_COLOR["mountains"], f"mountains (cost {TERRAIN_COST['mountains']})")

        # Resources
        Label(parent, text="Resources:", font=("Consolas", 12, "underline")).pack(anchor="w", pady=(8,2))
        self.legend_row(parent, RESOURCE_COLOR["stone"],   f"{RESOURCE_LETTER['stone']} = stone")
        self.legend_row(parent, RESOURCE_COLOR["iron"],    f"{RESOURCE_LETTER['iron']} = iron")
        self.legend_row(parent, RESOURCE_COLOR["crystal"], f"{RESOURCE_LETTER['crystal']} = crystal")

        # Base, Agent, Path
        Label(parent, text="Other:", font=("Consolas", 12, "underline")).pack(anchor="w", pady=(8,2))
        row = Frame(parent); row.pack(anchor="w", pady=2)
        cv = Canvas(row, width=28, height=18, highlightthickness=0, bg="white"); cv.pack(side=LEFT)
        cv.create_rectangle(2, 2, 26, 16, outline=BASE_OUTLINE, width=3)
        Label(row, text="B = base", font=("Consolas", 11)).pack(side=LEFT, padx=6)
        self.legend_row(parent, AGENT_COLOR, "agent", is_dot=True)
        self.legend_row(parent, PATH_COLOR, "planned path", is_line=True)

    # -------------
    # Map helpers
    # -------------
    def get_current_map(self):
        idx = self.current_map_index
        return self.maps[idx]["terrain"], self.maps[idx]["resources"]

    def on_map_change(self, _evt=None):
        name = self.map_var.get()
        for i, m in enumerate(self.maps):
            if m["name"] == name:
                self.current_map_index = i
                break
        self.agent_pos = BASE
        self.clear_preview()
        self.last_path = None
        self.reset_hud()
        self.redraw()

    # -------------
    # Drawing
    # -------------
    def grid_geom(self):
        W = self.can.winfo_width()
        H = self.can.winfo_height()
        tile_w = W / COLS
        tile_h = H / ROWS
        return W, H, tile_w, tile_h

    def redraw(self):
        self.can.delete("all")
        terrain, resources = self.get_current_map()
        W, H, tw, th = self.grid_geom()
        self.tile_size_cache = (tw, th)

        # draw terrain tiles
        for r in range(ROWS):
            for c in range(COLS):
                x0 = c*tw; y0 = r*th
                x1 = x0+tw; y1 = y0+th
                t = terrain[r][c]
                self.can.create_rectangle(x0, y0, x1, y1, fill=TERRAIN_COLOR[t], outline="#333333")

        # grid lines & labels
        for c in range(COLS+1):
            x = c*tw
            self.can.create_line(x, 0, x, H, fill="#888888")
        for r in range(ROWS+1):
            y = r*th
            self.can.create_line(0, y, W, y, fill="#888888")

        # axis labels 0..4
        for c in range(COLS):
            self.can.create_text(c*tw + tw/2, 12, text=str(c), font=("Consolas", 12))
        for r in range(ROWS):
            self.can.create_text(12, r*th + th/2, text=str(r), font=("Consolas", 12))

        # base
        bx0 = BASE[1]*tw; by0 = BASE[0]*th
        self.can.create_rectangle(bx0, by0, bx0+tw, by0+th, outline=BASE_OUTLINE, width=3)
        self.can.create_text(bx0+tw/2, by0+th/2, text="B", fill=BASE_OUTLINE, font=("Consolas", 18, "bold"))

        # resources (one per tile)
        seen = set()
        for res in resources:
            (rr, cc) = res["pos"]
            if (rr,cc) in seen:
                raise ValueError("Map invalid: more than one resource on a tile")
            seen.add((rr,cc))
            x0 = cc*tw; y0 = rr*th
            letter = RESOURCE_LETTER[res["type"]]
            self.can.create_text(x0+tw/2, y0+th/2, text=letter, fill=RESOURCE_COLOR[res["type"]],
                                 font=("Consolas", 18, "bold"))

        # agent
        self.draw_agent()

        # If we have a last planned path and the toggle is ON, re-draw it after a resize/redraw
        if self.last_path and self.show_path_var.get():
            self.draw_path_preview(self.last_path)

    def draw_agent(self):
        if self.agent_item is not None:
            self.can.delete(self.agent_item)
        tw, th = self.tile_size_cache
        r, c = self.agent_pos
        x = c*tw + tw/2
        y = r*th + th/2
        d = min(tw, th) * 0.45
        self.agent_item = self.can.create_oval(x-d/2, y-d/2, x+d/2, y+d/2,
                                               fill=AGENT_COLOR, outline="#0b3d15", width=2)

    def clear_preview(self):
        for item in self.path_preview_ids:
            self.can.delete(item)
        self.path_preview_ids.clear()

    def draw_path_preview(self, path):
        """Draws the blue planned path if the 'Show Path' toggle is ON."""
        # Always start by clearing any old preview
        self.clear_preview()
        # Respect toggle
        if not self.show_path_var.get():
            return
        if not path or len(path) < 2:
            return
        tw, th = self.tile_size_cache
        pts = []
        for (r,c) in path:
            pts.append(c*tw + tw/2)
            pts.append(r*th + th/2)
        self.path_preview_ids.append(self.can.create_line(*pts, fill=PATH_COLOR, width=3))

    # -------------
    # HUD helpers
    # -------------
    def reset_hud(self):
        self.inv_var.set("S:0  I:0  C:0")
        self.deliv_var.set(f"S:0/{REQUIRED['stone']}  I:0/{REQUIRED['iron']}  C:0/{REQUIRED['crystal']}")
        self.need_var.set(f"S:{REQUIRED['stone']}  I:{REQUIRED['iron']}  C:{REQUIRED['crystal']}")
        terrain, _ = self.get_current_map()
        t = terrain[BASE[0]][BASE[1]]
        self.tile_var.set(f"(0,0) {t}  cost={TERRAIN_COST[t]}")

    def update_hud(self, r, c, invS, invI, invC, dS, dI, dC):
        self.inv_var.set(f"S:{invS}  I:{invI}  C:{invC}")
        self.deliv_var.set(f"S:{dS}/{REQUIRED['stone']}  I:{dI}/{REQUIRED['iron']}  C:{dC}/{REQUIRED['crystal']}")
        needS = max(0, REQUIRED["stone"] - dS)
        needI = max(0, REQUIRED["iron"] - dI)
        needC = max(0, REQUIRED["crystal"] - dC)
        self.need_var.set(f"S:{needS}  I:{needI}  C:{needC}")
        terrain, _ = self.get_current_map()
        t = terrain[r][c]
        self.tile_var.set(f"({r},{c}) {t}  cost={TERRAIN_COST[t]}")

    def speed_delay(self):
        sp = self.speed_var.get()
        if sp == "Slow": return 0.6
        if sp == "Fast": return 0.08
        return 0.20  # Normal

    # New: handler for the Show Path toggle
    def on_toggle_path(self):
        if self.show_path_var.get():
            # toggled ON: if we have a path, draw it
            if self.last_path:
                self.draw_path_preview(self.last_path)
        else:
            # toggled OFF: clear whatever is drawn
            self.clear_preview()

    # -------------
    # Planning & animation
    # -------------
    def plan_and_animate(self):
        self.plan_btn.config(state=DISABLED)
        def work():
            try:
                terrain, resources = self.get_current_map()
                planner = Planner(terrain, resources)
                start = (BASE[0], BASE[1], 0,0,0, 0,0,0, 0)
                result = planner.astar(start)
                if not result:
                    self.stats_var.set("No solution found.")
                    self.last_path = None
                    self.clear_preview()
                    return

                path = result["path"]
                self.last_path = path  # remember it for toggling
                self.draw_path_preview(path)  # will respect toggle

                self.stats_var.set(
                    f"cost={result['cost']}  expanded={result['expanded']}  runtime={result['runtime_sec']:.3f}s"
                )

                self.animate_path_with_state(path, planner, start)
            finally:
                self.plan_btn.config(state=NORMAL)

        threading.Thread(target=work, daemon=True).start()

    def animate_path_with_state(self, path, planner, state_start):
        r, c, invS, invI, invC, dS, dI, dC, mask = state_start
        self.agent_pos = (r, c)
        self.draw_agent()
        self.update_hud(r, c, invS, invI, invC, dS, dI, dC)
        self.can.update()

        for pos in path[1:]:
            nr, nc = pos
            invS, invI, invC, dS, dI, dC, mask = planner.collect_if_possible(
                nr, nc, invS, invI, invC, dS, dI, dC, mask
            )
            self.agent_pos = (nr, nc)
            self.draw_agent()
            self.update_hud(nr, nc, invS, invI, invC, dS, dI, dC)
            self.can.update()
            time.sleep(self.speed_delay())

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    App().mainloop()
