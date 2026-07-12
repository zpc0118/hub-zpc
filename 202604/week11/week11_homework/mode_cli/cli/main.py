"""
main.py — fincli：A股年报检索 + 天气查询 统一命令行入口

把 src/ 后端能力封装成一条"看起来像 git/ls 那样"的真实命令，而不是
`python xxx.py ...`。通过 pyproject.toml 的 [project.scripts] 注册为
console_script，`pip install -e .` 后即可全局调用：

  fincli list-companies
  fincli search --query "营收和净利润" --stock-code 300750 --year 2023 --top-k 3
  fincli weather --city 宁德

不想安装也可直接跑：
  python mode_cli/cli/main.py search --query "营收" --stock-code 300750 --year 2023
  python -m mode_cli.cli.main weather --city 宁德

教学点：
  1. CLI 作为"工具实现层"，本质就是一个能跑的脚本——跟协议无关
  2. 用 pyproject + console_script 把脚本变成 PATH 上的真实命令，是 Python CLI 工具的标准发布方式
  3. 一个 fincli 含多个子命令（search/list-companies/weather），对应 git 的子命令设计

依赖：
  pip install faiss-cpu numpy openai httpx
  环境变量：DASHSCOPE_API_KEY（Embedding）
"""

import argparse
import sys
from pathlib import Path

# 让本脚本能 import 项目根的 src/（无论从哪个工作目录 / 是否安装）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import geocode, get_weather_by_coords  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        prog="fincli",
        description="fincli — A股年报检索 + 天气查询 命令行工具",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fincli search ...
    p_search = sub.add_parser("search", help="检索年报段落")
    p_search.add_argument("--query", required=True,
                          help="检索问题（不要含公司名/年份，用简短财务术语，如 '营收和净利润'）")
    p_search.add_argument("--stock-code", default=None, help="按公司过滤，如 300750")
    p_search.add_argument("--year", default=None, help="按年份过滤：2021/2022/2023")
    p_search.add_argument("--top-k", type=int, default=5, help="返回段落数，默认5")

    # fincli list-companies
    sub.add_parser("list-companies", help="列出知识库收录的公司")

    # fincli geocode ...
    p_geocode = sub.add_parser("geocode", help="城市名→经纬度")
    p_geocode.add_argument("--city", required=True, help="城市中文名，如 宁德")

    # fincli weather-by-coords ...
    p_wbc = sub.add_parser("weather-by-coords", help="经纬度→天气")
    p_wbc.add_argument("--lat", type=float, required=True, help="纬度")
    p_wbc.add_argument("--lon", type=float, required=True, help="经度")
    p_wbc.add_argument("--location-name", default="", help="可选，地名用于天气报告标题")

    args = parser.parse_args()

    if args.cmd == "search":
        print(search_annual_report(args.query, args.stock_code, args.year, args.top_k))
    elif args.cmd == "list-companies":
        print(list_companies())
    elif args.cmd == "geocode":
        print(geocode(args.city))
    elif args.cmd == "weather-by-coords":
        print(get_weather_by_coords(args.lat, args.lon, args.location_name))


if __name__ == "__main__":
    main()
