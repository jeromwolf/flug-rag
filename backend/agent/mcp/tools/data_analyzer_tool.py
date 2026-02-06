"""Data analysis tool for basic statistics and chart data generation."""

import csv
import io
import math
from collections import Counter
from typing import Any

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)


class DataAnalyzerTool(BaseTool):
    """Analyze data: compute statistics and generate chart-ready data."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="data_analyzer",
            description="데이터를 분석합니다. 기본 통계(평균, 중앙값, 표준편차 등) 계산 및 차트 데이터 생성을 지원합니다.",
            category="analytics",
            parameters=[
                ToolParameter(
                    name="data",
                    type=ToolParamType.OBJECT,
                    description="분석할 데이터 (리스트 또는 딕셔너리)",
                ),
                ToolParameter(
                    name="analysis_type",
                    type=ToolParamType.STRING,
                    description="분석 유형",
                    required=False,
                    default="statistics",
                    enum=["statistics", "chart_data", "csv_summary"],
                ),
                ToolParameter(
                    name="chart_type",
                    type=ToolParamType.STRING,
                    description="차트 유형 (chart_data 분석 시)",
                    required=False,
                    default="bar",
                    enum=["bar", "line", "pie", "scatter"],
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        data = kwargs.get("data")
        analysis_type = kwargs.get("analysis_type", "statistics")
        chart_type = kwargs.get("chart_type", "bar")

        if data is None:
            return ToolResult(success=False, error="data parameter is required")

        try:
            if analysis_type == "statistics":
                result = self.compute_statistics(data)
            elif analysis_type == "chart_data":
                result = self.generate_chart_data(data, chart_type)
            elif analysis_type == "csv_summary":
                result = self._analyze_csv_data(data)
            else:
                return ToolResult(success=False, error=f"Unknown analysis_type: {analysis_type}")

            return ToolResult(
                success=True,
                data=result,
                metadata={"analysis_type": analysis_type},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Analysis error: {e}")

    def compute_statistics(self, data: Any) -> dict:
        """Compute basic statistics for numeric data."""
        values = self._extract_numeric_values(data)

        if not values:
            return {"error": "No numeric values found", "count": 0}

        n = len(values)
        sorted_values = sorted(values)
        total = sum(values)
        mean = total / n

        # Median
        if n % 2 == 0:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            median = sorted_values[n // 2]

        # Mode
        counter = Counter(values)
        max_count = max(counter.values())
        modes = [v for v, c in counter.items() if c == max_count]
        mode = modes[0] if len(modes) == 1 else modes

        # Standard deviation
        variance = sum((x - mean) ** 2 for x in values) / n
        std = math.sqrt(variance)

        # Percentiles
        def percentile(data_sorted, p):
            k = (len(data_sorted) - 1) * (p / 100)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data_sorted[int(k)]
            return data_sorted[f] * (c - k) + data_sorted[c] * (k - f)

        return {
            "count": n,
            "sum": round(total, 4),
            "mean": round(mean, 4),
            "median": round(median, 4),
            "mode": mode,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "range": round(max(values) - min(values), 4),
            "std": round(std, 4),
            "variance": round(variance, 4),
            "percentiles": {
                "25": round(percentile(sorted_values, 25), 4),
                "50": round(percentile(sorted_values, 50), 4),
                "75": round(percentile(sorted_values, 75), 4),
                "90": round(percentile(sorted_values, 90), 4),
            },
        }

    def generate_chart_data(self, data: Any, chart_type: str = "bar") -> dict:
        """Generate chart-ready data for frontend visualization."""
        if chart_type == "pie":
            return self._pie_chart_data(data)
        elif chart_type == "scatter":
            return self._scatter_chart_data(data)
        else:
            return self._bar_line_chart_data(data, chart_type)

    def _bar_line_chart_data(self, data: Any, chart_type: str) -> dict:
        """Generate bar/line chart data."""
        if isinstance(data, dict):
            labels = list(data.keys())
            values = [self._to_numeric(v) for v in data.values()]
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                # List of dicts: use first string key as label, first numeric as value
                labels = []
                values = []
                for item in data:
                    label_key = next((k for k, v in item.items() if isinstance(v, str)), None)
                    value_key = next((k for k, v in item.items() if isinstance(v, (int, float))), None)
                    if label_key and value_key:
                        labels.append(str(item[label_key]))
                        values.append(float(item[value_key]))
                    elif value_key:
                        labels.append(str(len(labels)))
                        values.append(float(item[value_key]))
            else:
                labels = [str(i) for i in range(len(data))]
                values = [self._to_numeric(v) for v in data]
        else:
            return {"error": "Unsupported data format for chart"}

        return {
            "chart_type": chart_type,
            "labels": labels,
            "datasets": [{"data": values}],
        }

    def _pie_chart_data(self, data: Any) -> dict:
        """Generate pie chart data."""
        if isinstance(data, dict):
            labels = list(data.keys())
            values = [self._to_numeric(v) for v in data.values()]
        elif isinstance(data, list):
            counter = Counter(data)
            labels = [str(k) for k in counter.keys()]
            values = list(counter.values())
        else:
            return {"error": "Unsupported data format for pie chart"}

        total = sum(values) if values else 1
        percentages = [round(v / total * 100, 1) for v in values]

        return {
            "chart_type": "pie",
            "labels": labels,
            "datasets": [{"data": values, "percentages": percentages}],
        }

    def _scatter_chart_data(self, data: Any) -> dict:
        """Generate scatter plot data."""
        points = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    points.append({"x": self._to_numeric(item[0]), "y": self._to_numeric(item[1])})
                elif isinstance(item, dict) and "x" in item and "y" in item:
                    points.append({"x": self._to_numeric(item["x"]), "y": self._to_numeric(item["y"])})

        return {
            "chart_type": "scatter",
            "datasets": [{"data": points}],
        }

    def _analyze_csv_data(self, data: Any) -> dict:
        """Analyze CSV string data."""
        if isinstance(data, str):
            reader = csv.DictReader(io.StringIO(data))
            rows = list(reader)
        elif isinstance(data, dict) and "csv_content" in data:
            reader = csv.DictReader(io.StringIO(data["csv_content"]))
            rows = list(reader)
        elif isinstance(data, list):
            rows = data
        else:
            return {"error": "Provide CSV string or list of dicts"}

        if not rows:
            return {"row_count": 0, "columns": []}

        columns = list(rows[0].keys()) if isinstance(rows[0], dict) else []
        summary: dict[str, Any] = {
            "row_count": len(rows),
            "columns": columns,
            "column_stats": {},
        }

        for col in columns:
            col_values = [row.get(col) for row in rows if row.get(col)]
            numeric_values = []
            for v in col_values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    pass

            if numeric_values and len(numeric_values) > len(col_values) * 0.5:
                summary["column_stats"][col] = self.compute_statistics(numeric_values)
            else:
                counter = Counter(col_values)
                summary["column_stats"][col] = {
                    "type": "categorical",
                    "unique_count": len(counter),
                    "top_values": dict(counter.most_common(5)),
                    "null_count": len(rows) - len(col_values),
                }

        return summary

    async def analyze_csv(self, file_path: str) -> dict:
        """Analyze a CSV file from disk."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        result = await self.execute(data=content, analysis_type="csv_summary")
        if result.success:
            return result.data
        raise RuntimeError(result.error)

    def _extract_numeric_values(self, data: Any) -> list[float]:
        """Extract numeric values from various data structures."""
        values = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    values.append(float(item))
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            values.append(float(v))
                elif isinstance(item, str):
                    try:
                        values.append(float(item))
                    except ValueError:
                        pass
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))
                elif isinstance(v, list):
                    values.extend(self._extract_numeric_values(v))
        return values

    @staticmethod
    def _to_numeric(value: Any) -> float:
        """Convert a value to numeric, defaulting to 0."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
