"""Share the GPU with the user's desktop and games: detect a running game and throttle a job by a duty cycle.

The training machine is also the user's gaming PC. A job calls `Share.tick(dt)` after every unit of GPU work (one
training step, one prediction batch) that took `dt` seconds; the call sleeps so that the job uses only `duty` of the
wall time, a smaller `game_duty` while a game runs. A game is any GPU application that is not a known desktop program
(DESKTOP), or more than 5 GB of GPU memory held by other processes. `Share.must_yield()` says when the others hold more
than `yield_gb`: the job should then checkpoint and exit so the memory is free (train/guard_run.py restarts it).
`below_normal()` lowers the process priority. Only needs nvidia-smi; torch is optional.
"""
import ctypes, subprocess, time

DESKTOP = (':\\windows\\', 'chrome', 'msedge', 'firefox', 'claude', 'tradingview', 'steamwebhelper', 'steam.exe', 'blitz',
           'vgtray', 'riotclient', 'riot client', 'protonvpn', 'python', 'discord', 'code.exe', 'nvidia', 'yourphone',
           'phoneexperiencehost', 'spotify', 'whatsapp', 'telegram', 'obs64', 'epicgameslauncher', 'battle.net.exe',
           'slack', 'teams', 'notion', 'cursor', 'powertoys', 'widgets', 'onedrive', 'dwm.exe', 'textinputhost',
           'razer', 'windowsapps\\microsoft.', 'copilot', 'logitech', 'corsair', 'msi', 'asus', 'armoury', 'icue',
           'overwolf', 'geforce', 'radeon', 'wallpaper', 'lghub', 'steelseries', 'vanguard', 'anydesk', 'zoom')


def below_normal():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def _smi(query):
    try:
        return subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'], capture_output=True, text=True,
                              timeout=20).stdout
    except Exception:
        return None


def gpu_used_gb():
    out = _smi('--query-gpu=memory.used')
    try:
        return float(out.split()[0]) / 1024
    except Exception:
        return None


def gpu_other_gb(own_reserved_gb=None):
    """GPU memory held by everything except this process (GB); None when nvidia-smi fails."""
    used = gpu_used_gb()
    if used is None:
        return None
    if own_reserved_gb is None:
        try:
            import torch
            own_reserved_gb = torch.cuda.memory_reserved() / 2 ** 30
        except Exception:
            own_reserved_gb = 0.0
    return used - own_reserved_gb - 0.4                       # 0.4 GB: this process's CUDA context


def is_game(name):
    return not any(k in name.lower() for k in DESKTOP)


def gpu_games():
    """GPU applications that are not known desktop programs; [] when none or unknown."""
    out = _smi('--query-compute-apps=name')
    if out is None:
        return []
    return sorted({l.strip() for l in out.splitlines() if l.strip() and is_game(l.strip())})


class Share:
    def __init__(self, duty=0.8, game_duty=0.25, yield_gb=9.0, every=10.0, log=print):
        self.duty0, self.game_duty, self.yield_gb, self.every, self.log = duty, game_duty, yield_gb, every, log
        self.duty, self.games, self.other, self.slept, self._t = duty, None, None, 0.0, 0.0

    def poll(self, force=False):
        """Refresh the game state at most every `every` seconds."""
        now = time.time()
        if not force and now - self._t < self.every:
            return
        self._t = now
        g = gpu_games(); self.other = gpu_other_gb()
        if not g and self.other is not None and self.other > 5.0:      # an unlisted name, but something large is there
            g = [f'unknown application(s) holding {self.other:.1f} GB']
        if self.games is None or bool(g) != bool(self.games):
            self.duty = self.game_duty if g else self.duty0
            self.log('sharing: game running' if g else 'sharing: no game', g, f'-> duty {self.duty:.2f}')
        self.games = g

    def must_yield(self):
        return self.yield_gb > 0 and self.other is not None and self.other > self.yield_gb

    def tick(self, dt):
        """Call after a unit of GPU work that took dt seconds (synchronise CUDA first)."""
        self.poll()
        if self.duty < 1.0:
            z = dt * (1.0 / self.duty - 1.0); self.slept += z; time.sleep(z)
