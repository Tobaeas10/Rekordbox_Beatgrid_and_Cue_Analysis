"""Adaptive number of parallel workers for predict_beatgrids.py, predict_cues.py and predict_beatgrids_and_cues.py.

Idea: split the machine's load (CPU and RAM, percent of the whole machine) with psutil into three parts:
  * OURS: this program and its worker processes. Proportional to the number of busy workers: the controller learns what one busy worker costs.
  * REKORDBOX: rekordbox.exe and its child processes. Its analysis runs while we work and stops suddenly, so it only gets a FAST channel (last 5 s).
  * REST: everything else (browser, ...). Exogenous and noisy: slow exponential average (time constant 20 s at the start, growing to 60 s) plus 10 s blocks of the last 40 s.
Workers that fit = (limit - rest - rekordbox) / cost of one worker, separately for CPU (limit 85 %) and RAM (limit 95 %); the smaller number wins. We do not hold workers
back to protect Rekordbox: the CPU is filled up to the limit and shared.
The number of workers changes by at most a third of the physical cores per decision (6 cores: 2, 32 cores: 10; at least 1), only when BOTH the slow and the 40 s view say so (a short spike of another program is gone from the 30 s view,
a short dip is not yet in the slow one), with a dead band (more only if at least a whole extra worker fits, fewer only if more than half a worker is too many) and a cooldown (10 s up,
20 s down). Emergency: RAM at 97 % or more for 10 s takes one worker away at once. When the last song stops waiting for Rekordbox's analysis (update(..., waiting=0)),
the cooldown is cleared and the decision is made at once, so the ramp-up after Rekordbox is done is quick.
If no Rekordbox process is found everything else works the same (Rekordbox then simply counts as REST).

Usage in a pipeline:
    controller = adaptive_workers.WorkerController(forced_workers=arguments.force_number_of_jobs)   # as early as possible
    controller.update(number_of_running_jobs, waiting=number_of_songs_still_waiting_for_rekordbox)  # call in the main loop, cheap (waiting is optional)
    if number_of_running_jobs < controller.allowed_workers(): submit the next job
    controller.stop()                              # at the end
"""
import math
import os
import threading
import time
from collections import deque

CPU_LIMIT_PERCENT = 85.0
RAM_LIMIT_PERCENT = 95.0
RAM_EMERGENCY_PERCENT = 97.0
SAMPLE_INTERVAL_SECONDS = 1.0
DECISION_INTERVAL_SECONDS = 5.0
FAST_WINDOW_SECONDS = 40.0
FAST_BLOCK_SECONDS = 10.0
REKORDBOX_WINDOW_SECONDS = 5.0
EMERGENCY_WINDOW_SECONDS = 10.0
REKORDBOX_SCAN_SECONDS = 5.0
SLOW_TIME_CONSTANT_MINIMUM, SLOW_TIME_CONSTANT_MAXIMUM, SLOW_TIME_CONSTANT_RUN_FRACTION = 20.0, 60.0, 0.15
INCREASE_MARGIN = 1.0           # one more worker only if a whole extra worker fits
DECREASE_MARGIN = 0.5           # one less only if the current number is more than half a worker too many
COOLDOWN_UP_SECONDS, COOLDOWN_DOWN_SECONDS, COOLDOWN_EARLY_SECONDS, EARLY_PHASE_SECONDS = 10.0, 20.0, 5.0, 60.0
MINIMUM_COST_CPU, MINIMUM_COST_RAM = 2.0, 0.2    # percentage points of the whole machine per worker; avoids absurd targets from noise
MINIMUM_BUSY_SAMPLES = 5


def add_argument(argument_parser):
    argument_parser.add_argument("--force-number-of-jobs", type=int, default=None, metavar="N",
                                 help="use exactly N parallel workers; default: adjust the number automatically to the free CPU and RAM (start with 1)")


class WorkerController:
    def __init__(self, forced_workers=None):
        self.forced_workers = max(1, forced_workers) if forced_workers else None
        self.maximum_workers = os.cpu_count() or 1
        self.maximum_step = 1                        # set from the number of physical cores when psutil is available
        self.workers = self.forced_workers or 1
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.running_jobs = 0
        self.started = time.time()
        self.last_sample = None
        self.last_change = self.started
        self.last_decision = self.started
        self.fast_samples = deque()                  # (time, total cpu, total ram, rest cpu, rest ram, rekordbox cpu, rekordbox ram) of the last FAST_WINDOW_SECONDS
        self.others_cpu = self.others_ram = None     # slow averages of REST (percentage points of the whole machine)
        self.rekordbox_processes, self.rekordbox_scan_time, self.last_rekordbox_times = [], 0.0, {}
        self.waiting = None
        self.ours_cpu = self.busy_workers = None     # slow averages, only over samples where at least one job ran
        self.ram_per_process = None                  # slow average of the memory of one worker process (% of RAM)
        self.main_ram = 0.0
        self.total_cpu = self.total_ram = 0.0        # latest, for the description
        self.busy_samples = 0
        self.last_cpu_times = {}
        self.thread = None
        self.psutil = None
        if self.forced_workers:
            return
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            print("note: psutil is not installed (pip install psutil), so the number of workers cannot adapt; using 2 workers. Or give --force-number-of-jobs N.", flush=True)
            self.workers = 2
            return
        self.processor_count = psutil.cpu_count() or 1
        self.maximum_step = max(1, (psutil.cpu_count(logical=False) or self.processor_count // 2 or 1) // 3)   # a third of the physical cores per decision, at least 1
        self.memory_total = float(psutil.virtual_memory().total)
        psutil.cpu_percent(None)                     # first call only sets the reference point
        self.thread = threading.Thread(target=self.sample_loop, daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------- measuring
    def sample_loop(self):
        while not self.stop_event.wait(SAMPLE_INTERVAL_SECONDS):
            try:
                self.take_sample()
            except Exception:
                pass                                 # a process that vanished while it was read, a transient OS error: just skip this sample

    def take_sample(self):
        psutil = self.psutil
        now = time.time()
        total_cpu, total_ram = psutil.cpu_percent(None), psutil.virtual_memory().percent
        me = psutil.Process()
        children = me.children(recursive=True)
        cpu_seconds, children_rss, current_times = 0.0, 0, {}
        for process in [me] + children:
            try:
                times = process.cpu_times()
                used = times.user + times.system
                rss = process.memory_info().rss
            except psutil.Error:
                continue
            current_times[process.pid] = used
            cpu_seconds += used - self.last_cpu_times.get(process.pid, used if process.pid == me.pid else 0.0)   # a new worker started at 0 cpu seconds
            if process.pid != me.pid:
                children_rss += rss
        self.last_cpu_times = current_times
        main_rss = me.memory_info().rss
        rekordbox_seconds, rekordbox_rss = self.read_rekordbox(now, {process.pid for process in [me] + children})
        elapsed = now - self.last_sample if self.last_sample else None
        self.last_sample = now
        if not elapsed or elapsed <= 0:
            return
        ours_cpu = max(0.0, 100.0 * cpu_seconds / elapsed / self.processor_count)
        rekordbox_cpu = max(0.0, 100.0 * rekordbox_seconds / elapsed / self.processor_count)
        self.record(now, elapsed, total_cpu, total_ram, ours_cpu, 100.0 * children_rss / self.memory_total, 100.0 * main_rss / self.memory_total, len(children),
                    rekordbox_cpu, 100.0 * rekordbox_rss / self.memory_total)

    def read_rekordbox(self, now, own_pids):
        """(cpu seconds used since the last call, rss bytes) of Rekordbox's processes (every process whose name contains 'rekordbox', plus their children)."""
        psutil = self.psutil
        if now - self.rekordbox_scan_time >= REKORDBOX_SCAN_SECONDS:
            self.rekordbox_scan_time = now
            found = {}
            for process in psutil.process_iter(["name"]):
                try:
                    if "rekordbox" in (process.info["name"] or "").lower() and process.pid not in own_pids:
                        found[process.pid] = process
                        for child in process.children(recursive=True):
                            found.setdefault(child.pid, child)
                except psutil.Error:
                    continue
            self.rekordbox_processes = list(found.values())
        seconds, rss, current_times = 0.0, 0, {}
        for process in self.rekordbox_processes:
            try:
                times = process.cpu_times()
                used = times.user + times.system
                rss += process.memory_info().rss
            except psutil.Error:
                continue
            current_times[process.pid] = used
            seconds += used - self.last_rekordbox_times.get(process.pid, used)     # a process seen for the first time counts from now on
        self.last_rekordbox_times = current_times
        return seconds, rss

    def record(self, now, elapsed, total_cpu, total_ram, ours_cpu, children_ram, main_ram, child_count, rekordbox_cpu=0.0, rekordbox_ram=0.0):
        """One measurement. All values are percent of the whole machine. (A separate method so the logic can be tested without psutil.)"""
        with self.lock:
            others_cpu = max(0.0, total_cpu - ours_cpu - rekordbox_cpu)
            others_ram = max(0.0, total_ram - children_ram - main_ram - rekordbox_ram)
            self.total_cpu, self.total_ram, self.main_ram = total_cpu, total_ram, main_ram
            self.fast_samples.append((now, total_cpu, total_ram, others_cpu, others_ram, rekordbox_cpu, rekordbox_ram))
            while self.fast_samples and now - self.fast_samples[0][0] > FAST_WINDOW_SECONDS:
                self.fast_samples.popleft()
            alpha = 1.0 - math.exp(-elapsed / self.slow_time_constant(now))
            self.others_cpu = self.average(self.others_cpu, others_cpu, alpha)
            self.others_ram = self.average(self.others_ram, others_ram, alpha)
            if self.running_jobs > 0:
                self.busy_samples += 1
                self.ours_cpu = self.average(self.ours_cpu, ours_cpu, alpha)
                self.busy_workers = self.average(self.busy_workers, float(self.running_jobs), alpha)
            if child_count > 0:
                self.ram_per_process = self.average(self.ram_per_process, children_ram / child_count, alpha)

    def slow_time_constant(self, now):
        return min(SLOW_TIME_CONSTANT_MAXIMUM, max(SLOW_TIME_CONSTANT_MINIMUM, SLOW_TIME_CONSTANT_RUN_FRACTION * (now - self.started)))

    @staticmethod
    def average(old_value, new_value, alpha):
        return new_value if old_value is None else (1.0 - alpha) * old_value + alpha * new_value

    # ---------------------------------------------------------------- deciding
    def allowed_workers(self):
        """Number of jobs that may run at the same time right now."""
        return self.workers

    def update(self, running_jobs, waiting=None):
        """Tell the controller how many jobs run now (and optionally how many songs still wait for Rekordbox's analysis); every 5 seconds this re-decides the allowed number of workers."""
        if self.forced_workers or self.thread is None:
            return
        with self.lock:
            self.running_jobs = running_jobs
            now = time.time()
            if waiting is not None:
                if self.waiting and waiting == 0:                    # Rekordbox's analysis of the batch is over: its load is about to vanish, so decide at once
                    self.last_change = now - COOLDOWN_UP_SECONDS
                    self.last_decision = now - DECISION_INTERVAL_SECONDS
                self.waiting = waiting
            if now - self.last_decision < DECISION_INTERVAL_SECONDS:
                return
            self.last_decision = now
            self.decide(now)

    def workers_that_fit(self, others_cpu, others_ram):     # others = rest + rekordbox
        """(fit by cpu, fit by ram) as fractional numbers of workers, from the learned cost of one worker."""
        cost_cpu = max(MINIMUM_COST_CPU, self.ours_cpu / max(self.busy_workers, 1.0))
        cost_ram = max(MINIMUM_COST_RAM, self.ram_per_process or MINIMUM_COST_RAM)
        return (CPU_LIMIT_PERCENT - others_cpu) / cost_cpu, (RAM_LIMIT_PERCENT - others_ram - self.main_ram) / cost_ram

    def decide(self, now):
        if self.fast_samples and now - self.fast_samples[-1][0] < 5 * SAMPLE_INTERVAL_SECONDS:
            emergency_samples = [sample[2] for sample in self.fast_samples if now - sample[0] <= EMERGENCY_WINDOW_SECONDS]
            if emergency_samples and sum(emergency_samples) / len(emergency_samples) >= RAM_EMERGENCY_PERCENT and self.workers > 1 and now - self.last_change >= EMERGENCY_WINDOW_SECONDS:
                self.workers -= 1
                self.last_change = now
                return
        if self.others_cpu is None or self.ours_cpu is None or self.busy_samples < MINIMUM_BUSY_SAMPLES or not self.fast_samples:
            return                                   # not enough measurements yet (no job has run long enough to know what a worker costs)
        rekordbox_samples = [sample for sample in self.fast_samples if now - sample[0] <= REKORDBOX_WINDOW_SECONDS] or [self.fast_samples[-1]]
        rekordbox_cpu = sum(sample[5] for sample in rekordbox_samples) / len(rekordbox_samples)
        rekordbox_ram = sum(sample[6] for sample in rekordbox_samples) / len(rekordbox_samples)
        blocks = {}
        for sample in self.fast_samples:
            blocks.setdefault(int((now - sample[0]) // FAST_BLOCK_SECONDS), []).append(sample)
        blocks = [block for block in blocks.values() if len(block) >= 3] or [list(self.fast_samples)]
        block_rest = [(sum(sample[3] for sample in block) / len(block), sum(sample[4] for sample in block) / len(block)) for block in blocks]   # (cpu, ram) of each 10 s block
        slow_fit = min(self.workers_that_fit(self.others_cpu + rekordbox_cpu, self.others_ram + rekordbox_ram))
        # a load of another program only counts when it lasted the whole 40 s window: to remove a worker even the lightest 10 s block must be too heavy, to add one even the heaviest must leave room
        fit_lightest_block = max(min(self.workers_that_fit(cpu + rekordbox_cpu, ram + rekordbox_ram)) for cpu, ram in block_rest)
        fit_heaviest_block = min(min(self.workers_that_fit(cpu + rekordbox_cpu, ram + rekordbox_ram)) for cpu, ram in block_rest)
        early = now - self.started < EARLY_PHASE_SECONDS
        since_change = now - self.last_change
        if min(slow_fit, fit_heaviest_block) >= self.workers + INCREASE_MARGIN and self.running_jobs >= self.workers and self.workers < self.maximum_workers \
                and since_change >= (COOLDOWN_EARLY_SECONDS if early else COOLDOWN_UP_SECONDS):
            fits = min(slow_fit, fit_heaviest_block)
            self.workers = min(self.maximum_workers, self.workers + min(self.maximum_step, max(1, int(fits - self.workers))))   # both averages agree that this many more fit (and the current ones are all busy: otherwise it would change nothing)
            self.last_change = now
        elif max(slow_fit, fit_lightest_block) < self.workers - DECREASE_MARGIN and self.workers > 1 and since_change >= (COOLDOWN_EARLY_SECONDS if early else COOLDOWN_DOWN_SECONDS):
            fits = max(slow_fit, fit_lightest_block)
            self.workers = max(1, self.workers - min(self.maximum_step, max(1, int(self.workers - round(fits)))))   # both averages agree that this many too many
            self.last_change = now

    def live_workers(self):
        """Current number of workers (a function, so the progress table can show the live value for its time estimate)."""
        return max(1, self.workers)

    def description(self):
        """Short text for headers/footers, e.g. '3 worker(s) (auto)'."""
        if self.forced_workers:
            return f"{self.workers} worker(s) (fixed by --force-number-of-jobs)"
        return f"{self.workers} worker(s) (auto)"

    def stop(self):
        self.stop_event.set()
