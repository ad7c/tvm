# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Restart only Conv2D CS+H from empty history and leave other tuning paused."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import psutil

import complete_operator800_study as recovery
from operator800_cohort import collect_cohort
from report_single_seed_study import digest
from run_operator800_study import environment, verify_sources
from watch_operator800_cohort import ensure_running
from watch_operator800_study import read_json, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'experiments/results/paper_v1/operator800_20261011_cohort'
FIGURES = ROOT / 'papers/figures/operator800_20261011_cohort'
TASK = 'CONVG1C01'


def preflight():
    manifest = read_json(COHORT / 'cohort.json')
    protocol = read_json(Path(manifest['protocol_path']))
    verify_sources(protocol)
    report = collect_cohort(COHORT / 'cohort.json')
    if TASK in manifest['accepted']:
        raise RuntimeError('Conv2D CS+H is already accepted; refusing a duplicate run')
    active = [str(root) for root in recovery.task_roots(manifest) if recovery.guard_is_alive(root)]
    controllers = []
    scripts = {'complete_operator800_study.py', 'finish_conv_d_only.py',
               Path(__file__).name, 'guard_experiment_resources.py',
               'run_operator800_study.py', 'e1_thread_filter.py'}
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid != os.getpid() and any(Path(arg).name in scripts
                for arg in (process.info['cmdline'] or [])):
            controllers.append(process.pid)
    if active or controllers:
        raise RuntimeError(f'An experiment or controller is already running: {active} {controllers}')
    spec = next(spec for spec in protocol['order'] if spec['id'] == TASK)
    smoke = next(spec for spec in protocol['order']
                 if spec['smoke'] and spec['workload'] == 'conv2d_nchw3x3')
    if smoke['id'] not in manifest['accepted']:
        raise RuntimeError('Registered Conv2D smoke has not passed')
    return manifest, protocol, spec, report


def stop_publisher():
    path = COHORT / 'live_status/watcher.json'
    record = read_json(path)
    process = recovery.verified_process(record['pid'], record['created'])
    if process is not None:
        if not any(Path(arg).name == 'watch_operator800_cohort.py' for arg in process.cmdline()):
            raise RuntimeError('Status publisher identity changed')
        process.terminate()
        process.wait(timeout=15)
    record.update(status='stopped_at_requested_boundary', ended_utc=recovery.utc_now())
    write_json_atomic(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest, protocol, spec, report = preflight()
    if args.check:
        print(json.dumps({'sources_verified': True, 'only_task': TASK, 'trials': spec['trials'],
            'retained_formal_groups': len(report['runs']), 'restart_from_empty_history': True,
            'writes': False, 'launches': False}, ensure_ascii=False))
        return
    lock = recovery.acquire_lock(COHORT)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    control = COHORT / 'resumes' / ('conv_c_only_' + stamp)
    control.mkdir(parents=True, exist_ok=False)
    shutil.copy2(COHORT / 'cohort.json', control / 'cohort.before.json')
    paper_hashes = {str(path): digest(path) for path in
        (ROOT / 'papers/main.tex', ROOT / 'papers/sections/single_seed_experiments.tex')}
    request = {'utc': recovery.utc_now(), 'task': TASK,
               'instruction': 'Restart Conv2D CS+H from empty history, keep all other groups paused.',
               'old_pause_record': manifest.get('pause_record'), 'paper_sha256': paper_hashes,
               'protocol_sha256': manifest['protocol_sha256'], 'controller': lock}
    write_json_atomic(control / 'request.json', request)
    pause = read_json(Path(manifest['pause_record']))
    if pause['actual_task'] != TASK:
        raise RuntimeError('Recorded paused task differs from the requested restart')
    paused_root = str(Path(pause['active_root']).resolve())
    existing = [f for f in manifest['failures']
                if f['root'] == paused_root and f['id'] == TASK and f['phase'] == 'tuning']
    if not existing:
        existing = [{'root': paused_root, 'phase': 'tuning', 'id': TASK,
                     'utc': pause['ended_utc'], 'reason': 'User-requested pause; partial attempt retained.',
                     'short_budget': False, 'classification': 'user_requested_pause'}]
        manifest['failures'].extend(existing)
    for failure in existing:
        failure.update(resolved_by='user_pause_isolated_with_explicit_restart',
                       resolution_evidence=str(control / 'request.json'))
    config = SimpleNamespace(input=Path(manifest['protocol_path']).parent, output=COHORT,
                             figure_output=FIGURES, interval=900, poll=5, max_attempts=1)
    ctl = recovery.Controller(config, manifest, protocol)
    manifest['controller'] = {**lock, 'source_sha256': digest(Path(__file__)),
                              'mode': 'restart_single_arm', 'only_task': TASK, 'interval_s': 900}
    success = False
    failure = None
    raw = None
    try:
        ctl.save(status='restarting_single_arm', active_root=None, paused_task=None,
                 requires_explicit_continue=False, stop_after=TASK, resume_request=str(control / 'request.json'))
        ensure_running(COHORT)
        raw = ctl.new_root(TASK, 'tuning')
        request['new_raw_root'] = str(raw)
        write_json_atomic(control / 'request.json', request)
        print(f'RESTART ONLY {TASK}: empty history in {raw}', flush=True)
        command = [sys.executable, '-u', '-X', 'utf8', str(ROOT / 'experiments/run_operator800_study.py'),
                   '--output', str(raw), '--seed', str(protocol['seed']), '--only', TASK]
        ctl.execute(raw, command, 'tuning', TASK, environment())
        if TASK not in ctl.manifest['accepted'] or Path(ctl.manifest['accepted'][TASK]['root']) != raw:
            raise RuntimeError('New full-budget Conv2D CS+H arm did not pass acceptance')
        result = collect_cohort(COHORT / 'cohort.json')
        row = next(row for row in result['runs'] if row['id'] == TASK)
        paper_unchanged = {path: digest(Path(path)) == expected
                           for path, expected in paper_hashes.items()}
        write_json_atomic(control / 'DONE.json', {'utc': recovery.utc_now(), 'task': TASK,
            'submitted': row['budget']['actual_submitted'], 'xgb_samples': row['xgb_data_size'],
            'best_us': row['best_us'], 'raw_root': str(raw), 'other_tasks_started': False,
            'report': str(FIGURES / 'arms' / TASK / 'RESULTS.md'), 'independent_retiming': 'pending',
            'paper_edits_by_this_controller': False, 'paper_files_unchanged': paper_unchanged})
        success = True
    except BaseException as error:
        failure = repr(error)
        write_json_atomic(control / 'ERROR.json', {'utc': recovery.utc_now(), 'error': failure,
                                                   'other_tasks_started': False})
        raise
    finally:
        # No scheduling loop: only this explicit arm can run.
        ctl.active_root = None
        ctl.save(status='paused_by_user' if success else 'needs_attention', active_root=None,
                 stop_after=TASK, requires_explicit_continue=True, closeout_error=failure)
        ctl.monitor(force=True)
        stop_publisher()
        if success:
            content = (f'# 卷积 CS+H 重跑完成，后续队列保持暂停\n\n更新时间 UTC：{recovery.utc_now()}\n\n'
                       f'- {TASK}：800/800，模型累计 {row["xgb_data_size"]} 个样本，搜索最佳 {row["best_us"]:.6f} μs。\n'
                       '- 已更新该组收敛图、训练图及卷积三组对比；中断的 298 次记录单独保留。\n'
                       '- 后续算子等待明确的继续指令；论文正文未改动，独立重测仍待完成。\n')
        else:
            content = (f'# 卷积 CS+H 需要检查，后续队列未启动\n\n{failure}\n\n'
                       '- 失败尝试原样保留；本入口不会自动重试。\n')
        (COHORT / 'monitor15min/STATUS.md').write_text(content, encoding='utf-8')
        lock_path = COHORT / 'controller.lock.json'
        if lock_path.exists() and read_json(lock_path) == lock:
            lock_path.unlink()
    print('DONE: Conv2D CS+H accepted; remaining queue paused', flush=True)


if __name__ == '__main__':
    main()
