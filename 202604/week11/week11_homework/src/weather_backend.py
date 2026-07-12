"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. "纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 提供 geocode（城市名→经纬度）+ get_weather_by_coords（经纬度→天气）
     两个原子函数，支持 LLM ReAct 多轮链式调用
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_backend import geocode, get_weather_by_coords
  loc = geocode("宁德")       # 第一步：城市 → 坐标
  print(get_weather_by_coords(26.66, 119.55, location_name="中国 福建省 宁德市"))  # 第二步：坐标 → 天气

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


# ── 内部结构化 geocoding ──────────────────────────────────────────────────

def _geocode(city: str) -> dict:
    """
    城市名 → 经纬度的结构化查询（内部函数，返回 dict 供链式调用）。

    中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
    而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
    策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
    且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。

    Returns:
        {"lat": float, "lon": float, "name": str, "country": str,
         "admin1": str, "display_name": str}
        或 {"error": str}
    """
    with httpx.Client(timeout=10.0) as client:
        def _query(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _query(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _query(city + "市")
            if retry:
                results = retry

        if not results:
            return {"error": f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"}

        # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
        # 其次取有人口数据的，避免落到同名小村庄
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return {
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "name": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
            "display_name": f"{loc.get('country', '')} {loc.get('admin1', '')} {loc.get('name', city)}".strip(),
        }


# ── 对外接口 ──────────────────────────────────────────────────────────────

def geocode(city: str) -> str:
    """
    将城市名转换为经纬度坐标。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        坐标文本，包含经纬度和完整地名，例如：
        中国 福建省 宁德市
        纬度=26.66, 经度=119.55
    """
    loc = _geocode(city)
    if "error" in loc:
        return loc["error"]
    return f"{loc['display_name']}\n纬度={loc['lat']}, 经度={loc['lon']}"


def get_weather_by_coords(lat: float, lon: float, location_name: str = "") -> str:
    """
    根据经纬度坐标查询当前天气及未来3天预报。

    Args:
        lat:           纬度
        lon:           经度
        location_name: 可选，用于天气报告标题的地名，如 "中国 福建省 宁德市"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        # 格式化输出
        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        header = location_name if location_name else f"坐标 ({lat:.2f}°N, {lon:.2f}°E)"

        lines = [
            f"【{header}】天气报告",
            f"坐标：{lat:.2f}°N, {lon:.2f}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )

        return "\n".join(lines)


# ── 自检 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="weather_backend 自检")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("geocode")
    p1.add_argument("--city", required=True)

    p2 = sub.add_parser("weather")
    p2.add_argument("--lat", type=float, required=True)
    p2.add_argument("--lon", type=float, required=True)
    p2.add_argument("--location-name", default="")

    args = parser.parse_args()

    if args.cmd == "geocode":
        print(geocode(args.city))
    else:
        print(get_weather_by_coords(args.lat, args.lon, args.location_name))
