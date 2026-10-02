"""Recover completed stage telemetry from the native log, including interruptions."""
from datetime import datetime
import json
from pathlib import Path
import re
import time


def save_progress(run, started, initial_scaling):
    path = Path(run)/'sampler.log'
    text = path.read_text(errors='replace') if path.exists() else ''
    stages, stamp, previous_time = [], None, None
    peak_memory = None
    for line in text.splitlines():
        match = re.search(r'[\w.-]+: time: (.+)', line)
        if match:
            stamp = datetime.fromisoformat(match[1].strip())
        match = re.search(r'iteration: (\d+), beta: ([\deE.+-]+), scaling: ([\deE.+-]+)', line)
        if match:
            stages.append(dict(iteration=int(match[1]), beta=float(match[2]), next_scaling=float(match[3]),
                               actual_scaling=stages[-1]['next_scaling'] if stages else initial_scaling,
                               seconds=(stamp-previous_time).total_seconds() if stamp and previous_time else None))
            previous_time = stamp
        match = re.search(r'stats\(accepted/invalid/rejected\): \((\d+), (\d+), (\d+)\)', line)
        if match and stages:
            accepted, invalid, rejected = map(int, match.groups())
            total = accepted+invalid+rejected
            stages[-1].update(accepted=accepted, invalid=invalid, rejected=rejected,
                              attempted=total, acceptance=accepted/total if total else None)
        match = re.search(r'unique samples (\d+) out of (\d+)', line)
        if match and stages:
            stages[-1]['resampling_for_next_stage'] = dict(unique=int(match[1]), population=int(match[2]))
        match = re.search(r'slipkit: peak_memory_bytes = (\d+)', line)
        if match:
            peak_memory = int(match[1])
        match = re.search(r'(\d+)\s+maximum resident set size', line)
        if match:
            peak_memory = int(match[1])  # macOS /usr/bin/time -l reports bytes
        match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', line)
        if match:
            peak_memory = int(match[1])*1024
    report = dict(stages=stages, elapsed_seconds=time.monotonic()-started, peak_memory_bytes=peak_memory,
                  last_beta=stages[-1]['beta'] if stages else None)
    target = Path(run)/'progress.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(target)
    return report
