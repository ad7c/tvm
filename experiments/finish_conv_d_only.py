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
"""Adopt only the running Conv2D D arm, publish its results, then stop the queue."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import psutil

import complete_operator800_study as recovery
from operator800_cohort import collect_cohort
from report_single_seed_study import digest
from run_operator800_study import environment, verify_sources
from watch_operator800_study import read_json, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'experiments/results/paper_v1/operator800_20261011_cohort'
RAW = ROOT / 'experiments/results/o8_6ff1d8_006'
FIGURES = ROOT / 'papers/figures/operator800_20261011_cohort'
TASK = 'CONVG1D01'
XE = Path('C:/Users/徐奎/AppData/Local/Programs/MiKTeX/miktex/bin/x64/xelatex.exe')
CONTROL = COHORT / 'finish_conv_d_only'
PAPER = ROOT / 'papers/sections/single_seed_experiments.tex'
PROTECTED = [ROOT / 'papers/main.tex', ROOT / 'papers/sections/appendix_measurements.tex',
             ROOT / 'papers/sections/appendix_artifacts.tex', ROOT / 'papers/实验评估.md']


def preflight():
    manifest = read_json(COHORT / 'cohort.json')
    protocol = read_json(Path(manifest['protocol_path']))
    verify_sources(protocol)
    if Path(manifest['active_root']).resolve() != RAW.resolve():
        raise RuntimeError('Active root changed; refusing to take over a different experiment')
    progress = read_json(RAW / 'progress.json')
    guard = read_json(RAW / 'resource_guard/status.json')
    if progress['current'] != TASK or guard['argv'][-2:] != ['--only', TASK]:
        raise RuntimeError('Active driver is not restricted to Conv2D D')
    old = read_json(COHORT / 'controller.lock.json')
    process = recovery.verified_process(old['pid'], old['created'])
    if process is None or not any(Path(arg).name == 'complete_operator800_study.py'
                                  for arg in process.cmdline()):
        raise RuntimeError('Original controller identity does not match')
    if process.pid in (guard['guard_pid'], guard['child_pid'], progress['pid']):
        raise RuntimeError('Controller identity overlaps active experiment')
    if not recovery.guard_wrapper_alive(RAW):
        raise RuntimeError('Resource guard is not active')
    return manifest, protocol, old, process


def update_paper(row, expected):
    for path, checksum in expected.items():
        if digest(Path(path)) != checksum:
            raise RuntimeError('Paper changed since user request; refusing automatic overwrite: ' + path)
    original = PAPER.read_text(encoding='utf-8')
    phrase = '其余三个算子的长预算结果仍待补齐。'
    if original.count(phrase) != 1 or 'conv-d-completed' in original:
        raise RuntimeError('Experiment-section insertion point changed')
    hit = row['own_best']
    block = r'''
% conv-d-completed: audited single-arm result, no cross-method comparison.
\paragraph{Conv2D 的完整 D 运行}
Conv2D 的 CS+H+Draft-GA 已完成 800 次提交，有限且通过正确性检查的测量为 FINITE 次，
模型累计样本为 SAMPLES。原搜索最佳延迟为 $BEST\,\mu$s，首次出现在第 FIRST 次提交、
FIRSTTIME min；完整调优耗时为 TOTALTIME min。学习模型评分调用累计 MODEL s，
已记录的 Draft 评分为 DRAFT s。图~\ref{fig:conv-d-completed} 展示本组的完整搜索轨迹。
这些数值仅描述 D 配置的搜索结果；Conv2D 的 B/C 对照、固定赢家独立重测与 PyTorch
参考尚未完成，因此不据此计算卷积的相对加速比。
\begin{figure*}[tb]
\centering
\includegraphics[width=\textwidth]{figures/operator800_20261011_cohort/arms/CONVG1D01/convergence.pdf}
\caption{Conv2D 的 CS+H+Draft-GA 单组实测收敛，800 次真实提交、单 seed。
曲线按完整批次反馈更新；标记为首次观察到本组最终最佳候选的位置，独立重测尚待完成。}
\label{fig:conv-d-completed}
\end{figure*}
'''
    values = {'FINITE': str(row['outcomes']['finite']), 'SAMPLES': str(row['xgb_data_size']),
              'BEST': f"{row['best_us']:.6f}", 'FIRSTTIME': f"{hit['tune_s']/60:.2f}",
              'FIRST': str(hit['submitted']), 'TOTALTIME': f"{row['tune_s']/60:.2f}",
              'MODEL': f"{row['xgb_prediction_s']:.3f}", 'DRAFT': f"{row['draft_scoring_s']:.3f}"}
    for key, value in values.items():
        block = block.replace(key, value)
    marker = '\\FloatBarrier\n\\subsection{CS+H 对无效探索的削减}'
    if original.count(marker) != 1:
        raise RuntimeError('Section boundary changed')
    updated = original.replace(phrase, 'Conv2D 的 D 配置也已完成，B/C 对照仍待补齐；MatMul256 与 Reduction 的长预算结果待补。')
    updated = updated.replace(marker, block + '\n' + marker)
    shutil.copy2(PAPER, CONTROL / 'experiment_section.before.tex')
    PAPER.write_text(updated, encoding='utf-8')
    for attempt in range(3):
        with (CONTROL / f'xelatex-{attempt+1}.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(XE), '-interaction=nonstopmode', '-halt-on-error',
                            '-file-line-error', 'main.tex'], cwd=ROOT / 'papers',
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        log_text = (ROOT / 'papers/main.log').read_text(encoding='utf-8', errors='replace')
        if not any(s in log_text for s in ('There were undefined references', 'Label(s) may have changed',
                                         'Rerun to get outlines right')):
            break
    else:
        raise RuntimeError('PDF references did not stabilize')
    if 'Overfull \\hbox' in log_text or 'Missing character:' in log_text:
        raise RuntimeError('PDF needs layout review; see compilation log')
    for path in PROTECTED:
        if digest(path) != expected[str(path)]:
            raise RuntimeError('Protected paper content changed')
    render = ROOT / 'tmp/pdfs/experiment_rewrite_20261011/render_review.py'
    env = environment()
    env['PYTHONPATH'] = str(ROOT / 'tmp/pdf-tools') + os.pathsep + env['PYTHONPATH']
    subprocess.run([sys.executable, '-X', 'utf8', str(render)], cwd=ROOT, env=env,
                   check=True, timeout=180)


def stop_publisher():
    path = COHORT / 'live_status/watcher.json'
    record = read_json(path)
    process = recovery.verified_process(record['pid'], record['created'])
    if process is not None:
        if not any(Path(arg).name == 'watch_operator800_cohort.py' for arg in process.cmdline()):
            raise RuntimeError('Status publisher identity changed')
        process.terminate()
        process.wait(timeout=15)
    record.update(status='stopped_at_user_requested_boundary', ended_utc=recovery.utc_now())
    write_json_atomic(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest, protocol, old, process = preflight()
    if args.check:
        print(json.dumps({'verified': True, 'only_task': TASK, 'old_controller': old,
                          'guard_alive': True, 'writes': False}, ensure_ascii=False))
        return
    CONTROL.mkdir(exist_ok=True)
    expected = {str(path): digest(path) for path in PROTECTED + [PAPER]}
    write_json_atomic(CONTROL / 'request.json', {'utc': recovery.utc_now(), 'only_task': TASK,
        'instruction': 'Finish current Conv2D D, update data and section 4 only, then stop all further tuning.',
        'old_controller': old, 'protected_hashes': expected, 'script_sha256': digest(Path(__file__))})
    shutil.copy2(COHORT / 'cohort.json', CONTROL / 'cohort.before.json')
    process.terminate()  # Only the scheduler; never terminate its child/guard tree.
    process.wait(timeout=15)
    if not recovery.guard_wrapper_alive(RAW):
        raise RuntimeError('Guard unexpectedly exited during controller handoff')
    lock = recovery.acquire_lock(COHORT)
    manifest = read_json(COHORT / 'cohort.json')
    config = SimpleNamespace(input=Path(manifest['protocol_path']).parent, output=COHORT,
                             figure_output=FIGURES, interval=900, poll=5, max_attempts=2)
    ctl = recovery.Controller(config, manifest, protocol)
    ctl.manifest['controller'] = {**lock, 'source_sha256': digest(Path(__file__)),
                                  'mode': 'finish_current_only', 'only_task': TASK, 'interval_s': 900}
    ctl.save(status='finishing_current_only', active_root=str(RAW), stop_after=TASK)
    print('ADOPTED existing Conv2D D; no new tuning will be launched', flush=True)
    success = False
    failure = None
    try:
        ctl.wait_existing(RAW)  # Existing audit, plots and 15-minute checks; no scheduling loop.
        if TASK not in ctl.manifest['accepted']:
            raise RuntimeError('Conv2D D did not produce an accepted full-budget result')
        verify_sources(protocol)
        report = collect_cohort(COHORT / 'cohort.json')
        row = next(r for r in report['runs'] if r['id'] == TASK)
        if row['budget']['actual_submitted'] != 800:
            raise RuntimeError('Expected 800 real submissions, including failed candidates')
        if not (FIGURES / 'arms' / TASK / 'convergence.pdf').is_file():
            raise RuntimeError('Convergence plot missing')
        write_json_atomic(CONTROL / 'audited_result.json', row)
        update_paper(row, expected)
        success = True
        write_json_atomic(CONTROL / 'DONE.json', {'utc': recovery.utc_now(), 'task': TASK,
            'submitted': 800, 'next_tasks_started': False, 'best_us': row['best_us'],
            'paper_section_sha256': digest(PAPER), 'pdf_sha256': digest(ROOT / 'papers/main.pdf'),
            'protected_hashes': {str(path): digest(path) for path in PROTECTED},
            'result_report': str(FIGURES / 'arms' / TASK / 'RESULTS.md'),
            'independent_retiming': 'pending', 'pdf_visual_review': 'rendered; pending visual inspection'})
    except BaseException as error:
        failure = repr(error)
        write_json_atomic(CONTROL / 'ERROR.json', {'utc': recovery.utc_now(), 'error': failure,
            'no_new_tuning_started': True})
        raise
    finally:
        ctl.active_root = None
        ctl.save(status='paused_by_user' if success else 'needs_attention', active_root=str(RAW),
                 stop_after=TASK, closeout_error=failure)
        ctl.monitor(force=True)
        stop_publisher()
        title = '卷积 D 完成，已按要求结束本次运行' if success else '收尾需要检查，未启动后续任务'
        content = '# ' + title + '\n\n更新 UTC：' + recovery.utc_now() + '\n\n'
        content += '- 后续卷积 B/C、规约和 MatMul256 均未由本收尾流程启动。\n'
        if success:
            content += (f"- 卷积 D：800/800，累计模型样本 {row['xgb_data_size']}，搜索最佳 {row['best_us']:.6f} μs。\n"
                        '- 已更新该组报告、收敛/训练曲线与论文第 4 节，其他正文保持原样。\n'
                        '- 全套为 4/12 个完整调优组；独立重测仍待完成，不标记整套实验完成。\n')
        else:
            content += '- 错误：' + str(failure) + '\n'
        content += '\n[详细收尾记录](../finish_conv_d_only/)\n'
        (COHORT / 'monitor15min/STATUS.md').write_text(content, encoding='utf-8')
        lock_path = COHORT / 'controller.lock.json'
        if lock_path.exists() and read_json(lock_path) == lock:
            lock_path.unlink()
    print('DONE: data updated, queue stopped', flush=True)


if __name__ == '__main__':
    main()
