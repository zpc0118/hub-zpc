"""调试脚本：逐题测试 RAG 问答效果"""
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(use_bm25=True, use_rerank=False)

questions = [
    ("格力电器2023年的营业收入是多少？", "simple_fact"),
    ("恒瑞医药2023年归属于上市公司股东的净利润是多少？", "simple_fact"),
    ("比亚迪2023年的研发费用是多少亿元？占营业收入的比例是多少？", "precise_number"),
    ("长江电力2022年的净利润是多少？", "simple_fact"),
    ("顺丰控股2024年的营业收入是多少？", "simple_fact"),
    ("比较比亚迪和格力电器2023年的毛利率，哪家更高？", "cross_doc_compare"),
    ("格力电器2022年到2024年营业收入的变化趋势如何？", "time_trend"),
    ("比亚迪未来三年的股价走势如何？", "should_refuse"),
]

for q, qtype in questions:
    print(f"\n{'='*60}")
    print(f"[{qtype}] {q}")
    print(f"{'='*60}")
    result = pipeline.query(q, verbose=True)
    print(result["answer"])
    if result["citations"]:
        print("\n── 来源 ──")
        for c in result["citations"]:
            print(f"  {c['source']}")
