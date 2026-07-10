"""
RAG 评估脚本：逐题运行 + LLM 打分

使用方式：
  python evaluate.py                # 跑全部 20 题
  python evaluate.py --ids 1,2,3    # 只跑指定题号
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent  # week10_homework/
EVAL_DIR   = Path(__file__).parent
RESULT_DIR = EVAL_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


def load_questions(ids: list[int] = None) -> list[dict]:
    with open(EVAL_DIR / "questions.json", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def score_answer(question: str, answer: str, qtype: str, pipeline) -> dict:
    """用 LLM 对比问题+答案，打 correctness 分（0-1）"""
    if qtype == "should_refuse":
        # 拒绝类评分：答案含拒绝关键词=1.0（正确拒绝），否则=0.0（不该回答的瞎答了）
        refused = any(kw in answer for kw in ["无法", "不能", "超出", "不包含", "未提供"])
        return {"score": 1.0 if refused else 0.0, "reason": "正确拒绝" if refused else "应该拒绝但做了回答"}

    judge_prompt = f"""你是一个评测专家。请根据以下问答对，判断答案是否正确。

【问题】{question}

【答案】{answer}

评分标准（得分含义）：
- 1.0：答案准确，数字精确、单位正确、结论完整，与年报数据一致
- 0.5：方向对（如公司\年份\指标正确），但数字有偏差、漏了明细或描述不完整
- 0.0：完全错误，答错公司/年份/指标；或本应能从年报检索到却说"无法回答"

请只输出一个 JSON：{{"score": <0-1之间的浮点数>, "reason": "<一句话理由>"}}"""

    try:
        resp = pipeline.client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是一个评测专家，只输出JSON。"},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
        )
        result = json.loads(resp.choices[0].message.content.strip().strip("```json").strip("```").strip())
        return result
    except Exception as e:
        logger.warning(f"  打分失败: {e}")
        return {"score": -1, "reason": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG 问答评估")
    parser.add_argument("--ids", type=str, default=None, help="指定题号，逗号分隔")
    args = parser.parse_args()

    q_ids = [int(x.strip()) for x in args.ids.split(",")] if args.ids else None
    questions = load_questions(q_ids)
    logger.info(f"加载 {len(questions)} 道测试题")

    # 导入 pipeline
    sys.path.insert(0, str(BASE_DIR))
    from rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(use_bm25=True, use_rerank=False)
    logger.info("Pipeline 初始化完成\n")

    results = []
    type_scores = {}

    for q in questions:
        qid   = q["id"]
        qtype = q["type"]
        logger.info(f"[{qid:02d}/{len(questions)}] [{qtype}] {q['question'][:50]}...")

        result = pipeline.query(q["question"], verbose=False)
        answer = result["answer"]
        score_info = score_answer(q["question"], answer, qtype, pipeline)

        logger.info(f"  得分={score_info['score']}  理由={score_info['reason']}")

        results.append({
            "id":          qid,
            "type":        qtype,
            "question":    q["question"],
            "answer":      answer,
            "score":       score_info["score"],
            "reason":      score_info["reason"],
            "num_citations": len(result["citations"]),
            "citations":   [c["source"] for c in result["citations"]],
        })

        # 统计
        if qtype not in type_scores:
            type_scores[qtype] = []
        type_scores[qtype].append(score_info["score"])

    # ── 输出汇总 ──
    valid = [r for r in results if r["score"] >= 0]
    avg = sum(r["score"] for r in valid) / len(valid) if valid else 0

    print(f"\n{'='*60}")
    print(f"评估完成  |  总题数={len(results)}  有效={len(valid)}  平均分={avg:.2f}")
    print(f"{'='*60}")
    print(f"\n{'题型':<25} {'题数':>4} {'平均分':>8}")
    print(f"{'-'*40}")
    for t, scores in type_scores.items():
        valid_s = [s for s in scores if s >= 0]
        print(f"{t:<25} {len(scores):>4} {sum(valid_s)/len(valid_s):>8.2f}" if valid_s else f"{t:<25} {len(scores):>4} {'N/A':>8}")
    print(f"{'-'*40}")

    # ── 逐题详情 ──
    print(f"\n{'题号':<5} {'题型':<22} {'得分':>5}")
    print(f"{'-'*35}")
    for r in results:
        s = f"{r['score']:.2f}" if r['score'] >= 0 else "N/A"
        print(f"{r['id']:<5} {r['type']:<22} {s:>5}")

    # ── 保存 ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"timestamp": ts, "avg_score": avg, "results": results}
    json_path = RESULT_DIR / f"eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存 → {json_path}")

    # CSV
    csv_path = RESULT_DIR / f"eval_{ts}.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("id,type,score,reason,num_citations\n")
        for r in results:
            f.write(f'{r["id"]},{r["type"]},{r["score"]},{r["reason"]},{r["num_citations"]}\n')
    logger.info(f"CSV 已保存 → {csv_path}")


if __name__ == "__main__":
    main()
