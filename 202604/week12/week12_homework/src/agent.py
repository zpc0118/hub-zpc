"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  # 单次问答
  python agent.py --mode manual   --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc       --question "五粮液近一年股价涨跌幅？"

  # 多轮对话
  python agent.py --mode manual --interactive
  python agent.py --mode fc --interactive

环境变量：
  DASHSCOPE_API_KEY  必填
  AGENT_MODEL        默认 qwen-max，可换 deepseek-v3 等
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"


def run_one_shot(mode: str, question: str, max_steps: int):
    """单次问答（兼容原有行为）"""
    if mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print
    run_and_print(question, max_steps)


def run_interactive(mode: str, max_steps: int):
    """多轮对话模式"""
    if mode == "manual":
        from react_manual import run, SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT
        mode_label = "手写 Prompt 解析"
    else:
        from react_function_calling import run, FC_SYSTEM_PROMPT
        system_prompt = FC_SYSTEM_PROMPT
        mode_label = "Function Calling"

    messages = [{"role": "system", "content": system_prompt}]
    print(f"\n多轮对话模式 [{mode_label}]")
    print("输入 /clear 清空历史  /exit 退出\n")

    round_num = 0
    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not question:
            continue
        if question == "/exit":
            break
        if question == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            round_num = 0
            print("历史已清空\n")
            continue

        round_num += 1
        print()  # 空行分隔
        for step_data in run(question, max_steps=max_steps, messages=messages):
            stype = step_data["type"]
            if stype == "action":
                print(f"  [Step {step_data['step']}] {step_data['action']}"
                      f"({step_data.get('action_input', {})})"
                      f" -> {str(step_data.get('observation', ''))[:100]}..."
                      )

        # run() 返回值中最后一个 step_data 即 final/error/max_steps
        if step_data["type"] == "final":
            print(f"\n  -> {step_data['answer']}")
        elif step_data["type"] in ("error", "max_steps"):
            print(f"\n  !! {step_data.get('answer', step_data.get('observation', ''))}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--interactive", action="store_true",
                        help="多轮对话模式")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.mode, args.max_steps)
    else:
        run_one_shot(args.mode, args.question, args.max_steps)
