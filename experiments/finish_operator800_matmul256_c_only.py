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
"""Finish the existing MatMul256 CS+H attempt, record results, then pause."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import psutil

import complete_operator800_study as recovery
from operator800_cohort import collect_cohort, write_report
from report_single_seed_study import digest
from run_operator800_study import verify_sources
from watch_operator800_cohort import live_check, render
from watch_operator800_study import read_json, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'experiments/results/paper_v1/operator800_20261011_cohort'
TASK = 'M256G1C01'


def verify_process(record, script):
    process = recovery.verified_process(record['pid'], record['created'])
    if process is None or not any(Path(arg).name == script for arg in process.cmdline()):
        raise RuntimeError(f'Expected process identity is absent or changed: {script}')
    return process


def preflight():
    manifest = read_json(COHORT / 'cohort.json')
    protocol = read_json(Path(manifest['protocol_path']))
    verify_sources(protocol)
    previous = read_json(COHORT / 'controller.lock.json')
    controller = verify_process(previous, 'complete_operator800_study.py')
    raw = Path(manifest['active_root']).resolve()
    if not raw.is_relative_to((ROOT / 'experiments/results').resolve()):
        raise RuntimeError('Active experiment is outside the intended workspace')
    progress = read_json(raw / 'progress.json')
    guard = read_json(raw / 'resource_guard/status.json')
    if progress['current'] != TASK or guard['status'] != 'running':
        raise RuntimeError('Requested tuning arm is no longer the active guarded task')
    if TASK in manifest['accepted']:
        raise RuntimeError('Requested arm was already accepted; inspect the queue before handover')
    owned = [psutil.Process(pid) for pid in
             (guard['guard_pid'], guard['child_pid'], progress['pid'])]
    if owned[1].create_time() != guard['child_created']:
        raise RuntimeError('Guarded driver identity changed')
    scripts = ('guard_experiment_resources.py', 'run_operator800_study.py', 'e1_thread_filter.py')
    for process, script in zip(owned, scripts):
        if not any(Path(arg).name == script for arg in process.cmdline()):
            raise RuntimeError(f'Unexpected active process: {process.pid}')
    return manifest, protocol, previous, controller, raw, owned


def stop_publisher():
    path = COHORT / 'live_status/watcher.json'
    record = read_json(path)
    process = recovery.verified_process(record['pid'], record['created'])
    if process is not None:
        if not any(Path(arg).name == 'watch_operator800_cohort.py' for arg in process.cmdline()):
            raise RuntimeError('Status publisher identity changed')
        process.terminate()
        process.wait(timeout=15)
    record.update(status='paused_by_user', ended_utc=recovery.utc_now())
    write_json_atomic(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest, protocol, previous, controller, raw, owned = preflight()
    identities = {str(process.pid): process.create_time() for process in owned}
    if args.check:
        print(json.dumps({'task': TASK, 'raw': str(raw), 'sources_verified': True,
            'preserved_processes': identities, 'old_controller': previous,
            'new_gpu_tasks': False, 'writes': False}, ensure_ascii=False))
        return
    folder = COHORT / 'pauses' / ('after_matmul256_c_' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(COHORT / 'cohort.json', folder / 'cohort.before.json')
    shutil.copy2(COHORT / 'monitor15min/AUDIT_STATUS.md', folder / 'AUDIT_STATUS.before.md')
    request = {'utc': recovery.utc_now(), 'instruction':
        'Finish MatMul256 CS+H, record the data, then pause until explicit continue.',
        'actual_task': TASK, 'active_root': str(raw), 'old_controller': previous,
        'preserved_identities_before': identities,
        'paper_sha256': {str(path): digest(path) for path in
            (ROOT / 'papers/main.tex', ROOT / 'papers/sections/single_seed_experiments.tex')},
        'protocol_sha256': manifest['protocol_sha256']}
    write_json_atomic(folder / 'request.json', request)
    # Stop only the scheduler; its existing guarded experiment remains running.
    controller.terminate()
    controller.wait(timeout=15)
    for process in owned:
        if not (process.is_running() and process.num_threads() > 0
                and process.create_time() == identities[str(process.pid)]):
            raise RuntimeError('An active tuning identity changed during handover')
    lock_path = COHORT / 'controller.lock.json'
    if lock_path.exists():
        if read_json(lock_path) != previous:
            raise RuntimeError('Controller lock changed during handover')
        lock_path.rename(folder / 'controller.previous.lock.json')
    manifest = read_json(COHORT / 'cohort.json')
    lock = recovery.acquire_lock(COHORT)
    config = SimpleNamespace(input=Path(manifest['protocol_path']).parent, output=COHORT,
        figure_output=Path(manifest['figure_output']), interval=900, poll=5, max_attempts=1)
    ctl = recovery.Controller(config, manifest, protocol)
    manifest['controller'] = {**lock, 'source_sha256': digest(Path(__file__)),
        'mode': 'finish_existing_arm_then_pause', 'only_task': TASK, 'interval_s': 900}
    request.update(new_controller=lock,
        preserved_identities_after={str(p.pid): p.create_time() for p in owned})
    write_json_atomic(folder / 'request.json', request)
    success = False
    failure = None
    try:
        ctl.save(stop_after=TASK, requires_explicit_continue=True,
                 stop_after_record=str(folder / 'request.json'))
        print(f'ADOPT ONLY {TASK}: {raw}; no new GPU tasks permitted', flush=True)
        # This controller has no scheduling loop and no experiment-launch call.
        ctl.wait_existing(raw)
        reference = ctl.manifest['accepted'].get(TASK)
        if reference is None or Path(reference['root']).resolve() != raw:
            raise RuntimeError('The existing full-budget arm did not pass acceptance')
        ctl.save(status='recording_before_pause', active_root=None)
        report_input = folder / 'report_input.json'
        shutil.copy2(COHORT / 'cohort.json', report_input)
        report = write_report(report_input, config.figure_output)
        row = next(row for row in report['runs'] if row['id'] == TASK)
        if row['budget']['actual_submitted'] != 800 or row['xgb_data_size'] != 800:
            raise RuntimeError('Unexpected final submitted or model sample count')
        done = {**request, 'utc': recovery.utc_now(), 'status': 'paused_by_user',
            'submitted': row['budget']['actual_submitted'], 'xgb_samples': row['xgb_data_size'],
            'best_us': row['best_us'], 'tune_s': row['tune_s'], 'phases_s': row['phases_s'],
            'accepted_reference': reference, 'other_tasks_started': False,
            'arm_report': str(config.figure_output / 'arms' / TASK / 'RESULTS.md'),
            'aggregate_report': str(config.figure_output / 'RESULTS.md'),
            'independent_retiming': 'pending', 'paper_files_unchanged':
                {path: digest(Path(path)) == expected
                 for path, expected in request['paper_sha256'].items()}}
        write_json_atomic(folder / 'DONE.json', done)
        success = True
        print(json.dumps(done, ensure_ascii=False), flush=True)
    except BaseException as error:
        failure = repr(error)
        write_json_atomic(folder / 'ERROR.json', {'utc': recovery.utc_now(),
            'error': failure, 'other_tasks_started': False, 'raw_root': str(raw)})
        raise
    finally:
        ctl.active_root = None
        ctl.save(status='paused_by_user' if success else 'needs_attention', active_root=None,
            stop_after=TASK, requires_explicit_continue=True, closeout_error=failure,
            stop_after_record=str(folder / ('DONE.json' if success else 'ERROR.json')))
        ctl.monitor(force=True)
        stop_publisher()
        check = live_check(COHORT / 'cohort.json', {})
        # The finishing process is still alive only to publish this final state.
        # It has no further work and exits immediately after releasing its lock.
        check['controller_alive'] = False
        # The finishing process is still alive only to publish this final state.
        # It has no further work and exits immediately after releasing its lock.
        check['controller_alive'] = False
        header = ('# MatMul256 CS+H 已完成，后续队列已暂停\n\n'
            '- 已保存原始数据、审核记录、收敛图、模型训练图和结果报告。\n'
            '- MatMul256 D、baseline 及其他 GPU 实验等待明确的继续指令。\n'
            '- 独立重测仍待完成；论文正文未改动。\n\n' if success else
            '# 本组需要检查，后续队列已停止\n\n' + str(failure) + '\n\n')
        content = header + render(check, 30).replace(
            '未运行，请检查控制器', '已按用户要求暂停，等待继续指令')
        for name in ('AUDIT_STATUS.md', 'STATUS.md'):
            (COHORT / 'monitor15min' / name).write_text(content, encoding='utf-8')
        write_json_atomic(COHORT / 'live_status/latest.json', check)
        if lock_path.exists() and read_json(lock_path) == lock:
            lock_path.rename(folder / 'controller.stopped.lock.json')
    print('DONE: original MatMul256 CS+H completed; all GPU work paused', flush=True)


if __name__ == '__main__':
    main()
