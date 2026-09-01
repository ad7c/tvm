"""E0 误杀检验：检查 vk!=1 约束是否会误杀合法候选。

从 baseline 的 tuning record 里提取所有"合法候选"（已通过 postproc 并进 database），
看它们的 k 循环 SamplePerfectTile 决策（3 因子）中间值是否为 1。
若任一合法候选 vk==1，则 vk!=1 硬约束误杀，必须放弃。
"""
import glob
import json
import os

os.environ["TVM_LIBRARY_PATH"] = "F:/tvm/build-prune/lib"
os.environ["USERPROFILE"] = "C:\\tvm_home"
os.environ["HOME"] = "C:\\tvm_home"
os.environ["TEMP"] = "F:\\tvm_tmp"
os.environ["TMP"] = "F:\\tvm_tmp"
os.environ["PYTHONPATH"] = "F:/tvm/python"


def load_records(path):
    # tuning_record.json 是逐行 JSON（NDJSON），每行一条 record
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def k_loop_tiles(records):
    """返回每个 record 的 (k_loop_tile_list, middle_factor)。"""
    out = []
    for r in records:
        # r = [workload_id, [trace, run_secs, target, args_info]]
        # trace = [insts, decisions]
        insts = r[1][0][0]
        decisions = r[1][0][1]
        run_secs = r[1][1][0] if isinstance(r[1][1], list) and r[1][1] else r[1][1]
        inst_by_idx = {i: inst for i, inst in enumerate(insts)}
        k_tile = None
        for d in decisions:
            idx, val = d[0], d[1]
            inst = inst_by_idx[idx]
            # SamplePerfectTile 的 attrs[0] 是 n（切分份数）
            if inst[0] == "SamplePerfectTile" and inst[2][0] == 3:
                k_tile = val  # [bk0, vk, bk1]，中间即 vk
        if k_tile is None:
            out.append((None, None, run_secs))
        else:
            out.append((k_tile, k_tile[1], run_secs))
    return out


def summarize(name, records):
    tiles = k_loop_tiles(records)
    n = len(tiles)
    n_vk1 = sum(1 for t in tiles if t[1] == 1)
    print(f"=== {name} ===")
    print(f"  合法候选数: {n}")
    print(f"  vk==1 (会被 vk!=1 误杀): {n_vk1}")
    print(f"  vk!=1 (安全): {n - n_vk1}")
    if n:
        print(f"  误杀率: {n_vk1}/{n} = {n_vk1 / n * 100:.1f}%")
    print(f"  k-loop tile 分布:")
    for i, (kt, mid, rs) in enumerate(tiles):
        flag = "  <-- 误杀" if mid == 1 else ""
        print(f"    record {i}: k_tile={kt}  vk={mid}  run_secs={rs}{flag}")
    print()
    return tiles


if __name__ == "__main__":
    targets = []
    for p in [
        "F:/tvm_tmp/verify_mode0/database_tuning_record.json",
    ]:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            targets.append(("verify_mode0 (128^3 baseline)", p))

    # 顺带扫旧的 baseline 数据（256^3，样本更多）
    for name, pat in [
        ("prune_cmp_0 (256^3)", "F:/tvm/experiments/data/prune_cmp_0/database_tuning_record.json"),
        ("prune_cmp_1 (256^3)", "F:/tvm/experiments/data/prune_cmp_1/database_tuning_record.json"),
        ("prune_cmp_2 (256^3)", "F:/tvm/experiments/data/prune_cmp_2/database_tuning_record.json"),
        ("prune_run_0 (128^3, 剪枝后)", "F:/tvm/experiments/data/prune_run_0/database_tuning_record.json"),
    ]:
        if os.path.exists(pat) and os.path.getsize(pat) > 0:
            targets.append((name, pat))

    all_legal = 0
    all_vk1 = 0
    for name, path in targets:
        try:
            recs = load_records(path)
            tiles = summarize(name, recs)
            if tiles:
                all_legal += len(tiles)
                all_vk1 += sum(1 for t in tiles if t[1] == 1)
        except Exception as e:
            print(f"=== {name} === 解析失败: {e}\n")

    if all_legal:
        print(f"=== 汇总 ===")
        print(f"  合法候选总数: {all_legal}")
        print(f"  vk==1 总数: {all_vk1}")
        print(f"  总误杀率: {all_vk1}/{all_legal} = {all_vk1 / all_legal * 100:.1f}%")
        print()
        if all_vk1 > 0:
            print("  >> 结论：vk!=1 会误杀合法候选，硬约束不成立，必须放弃（与方案 v1.3 决策一致，现获实证）")
        else:
            print("  >> 结论：当前样本内 vk!=1 零误杀")
