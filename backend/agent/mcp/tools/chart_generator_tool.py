"""Chart generator tool -- produces ECharts-compatible JSON option specs."""

from typing import Any

from agent.mcp.tools.base import (
    BaseTool, ToolDefinition, ToolParameter, ToolParamType, ToolResult,
)

# Professional Korean government report color palette (navy, blue, gray tones)
_DEFAULT_COLORS = [
    "#1B3A5C",  # navy
    "#2E6DA4",  # blue
    "#5B8DB8",  # steel blue
    "#8AB4D6",  # light blue
    "#A8C6D8",  # powder blue
    "#4A4A4A",  # dark gray
    "#7A8B99",  # slate gray
    "#B0BEC5",  # light gray
]

_DARK_COLORS = [
    "#5B9BD5",  # soft blue
    "#7EC8E3",  # cyan
    "#70AD47",  # green
    "#FFC000",  # amber
    "#ED7D31",  # orange
    "#A5A5A5",  # gray
    "#4472C4",  # royal blue
    "#9B59B6",  # purple
]


class ChartGeneratorTool(BaseTool):
    """데이터 시각화 도구 -- ECharts JSON 사양 생성"""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="chart_generator",
            description="데이터를 ECharts 차트 사양으로 변환합니다.",
            category="visualization",
            help_text=(
                "데이터를 ECharts 호환 JSON 옵션 사양으로 변환합니다.\n"
                "지원 차트: bar, line, pie, scatter, radar\n"
                "data 형식: {labels: [...], datasets: [{name: '...', values: [...]}]}\n"
                "프론트엔드에서 ECharts로 직접 렌더링할 수 있는 option 객체를 반환합니다."
            ),
            parameters=[
                ToolParameter(
                    name="chart_type",
                    type=ToolParamType.STRING,
                    description="차트 유형",
                    enum=["bar", "line", "pie", "scatter", "radar"],
                ),
                ToolParameter(
                    name="title",
                    type=ToolParamType.STRING,
                    description="차트 제목",
                ),
                ToolParameter(
                    name="data",
                    type=ToolParamType.OBJECT,
                    description="차트 데이터 ({labels: [...], datasets: [{name, values: [...]}]})",
                ),
                ToolParameter(
                    name="theme",
                    type=ToolParamType.STRING,
                    description="테마 (default 또는 dark)",
                    required=False,
                    default="default",
                    enum=["default", "dark"],
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        chart_type = kwargs.get("chart_type", "")
        title = kwargs.get("title", "")
        data = kwargs.get("data")
        theme = kwargs.get("theme", "default")

        if not chart_type:
            return ToolResult(success=False, error="chart_type parameter is required")
        if not title:
            return ToolResult(success=False, error="title parameter is required")
        if not data or not isinstance(data, dict):
            return ToolResult(success=False, error="data parameter is required and must be an object")

        labels = data.get("labels", [])
        datasets = data.get("datasets", [])

        if not datasets:
            return ToolResult(success=False, error="data.datasets is required and must not be empty")

        colors = _DARK_COLORS if theme == "dark" else _DEFAULT_COLORS

        try:
            if chart_type == "pie":
                option = self._build_pie(title, labels, datasets, colors, theme)
            elif chart_type == "radar":
                option = self._build_radar(title, labels, datasets, colors, theme)
            elif chart_type == "scatter":
                option = self._build_scatter(title, datasets, colors, theme)
            else:
                option = self._build_cartesian(chart_type, title, labels, datasets, colors, theme)

            return ToolResult(
                success=True,
                data={"echarts_option": option, "chart_type": chart_type},
                metadata={"title": title, "theme": theme},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Chart generation error: {e}")

    def _base_option(self, title: str, theme: str) -> dict[str, Any]:
        """Common option fields."""
        text_color = "#E0E0E0" if theme == "dark" else "#333333"
        bg_color = "#1E1E2F" if theme == "dark" else "#FFFFFF"
        return {
            "title": {
                "text": title,
                "left": "center",
                "textStyle": {"color": text_color, "fontSize": 16},
            },
            "backgroundColor": bg_color,
            "tooltip": {"trigger": "item"},
        }

    def _build_cartesian(
        self,
        chart_type: str,
        title: str,
        labels: list,
        datasets: list[dict],
        colors: list[str],
        theme: str,
    ) -> dict:
        """Build bar or line chart option."""
        option = self._base_option(title, theme)
        option["tooltip"]["trigger"] = "axis"

        axis_color = "#AAAAAA" if theme == "dark" else "#666666"

        option["xAxis"] = {
            "type": "category",
            "data": labels,
            "axisLabel": {"color": axis_color},
        }
        option["yAxis"] = {
            "type": "value",
            "axisLabel": {"color": axis_color},
        }

        series = []
        legend_data = []
        for i, ds in enumerate(datasets):
            name = ds.get("name", f"Series {i + 1}")
            values = ds.get("values", [])
            legend_data.append(name)
            series.append({
                "name": name,
                "type": chart_type,
                "data": values,
                "itemStyle": {"color": colors[i % len(colors)]},
            })

        option["legend"] = {
            "data": legend_data,
            "bottom": 0,
            "textStyle": {"color": axis_color},
        }
        option["series"] = series

        return option

    def _build_pie(
        self,
        title: str,
        labels: list,
        datasets: list[dict],
        colors: list[str],
        theme: str,
    ) -> dict:
        """Build pie chart option."""
        option = self._base_option(title, theme)
        option["tooltip"]["formatter"] = "{b}: {c} ({d}%)"

        # Use first dataset
        ds = datasets[0]
        values = ds.get("values", [])

        pie_data = []
        for i, (label, val) in enumerate(zip(labels, values)):
            pie_data.append({
                "name": str(label),
                "value": val,
                "itemStyle": {"color": colors[i % len(colors)]},
            })

        axis_color = "#AAAAAA" if theme == "dark" else "#666666"

        option["legend"] = {
            "orient": "vertical",
            "left": "left",
            "textStyle": {"color": axis_color},
        }
        option["series"] = [{
            "name": ds.get("name", title),
            "type": "pie",
            "radius": "55%",
            "center": ["50%", "55%"],
            "data": pie_data,
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                },
            },
            "label": {"color": axis_color},
        }]

        return option

    def _build_radar(
        self,
        title: str,
        labels: list,
        datasets: list[dict],
        colors: list[str],
        theme: str,
    ) -> dict:
        """Build radar chart option."""
        option = self._base_option(title, theme)

        # Determine max values per indicator
        all_values = []
        for ds in datasets:
            vals = ds.get("values", [])
            all_values.append(vals)

        # Compute max per axis
        max_vals = []
        for i in range(len(labels)):
            col_max = 0
            for vals in all_values:
                if i < len(vals):
                    col_max = max(col_max, vals[i])
            max_vals.append(col_max * 1.2 if col_max > 0 else 100)

        indicators = [
            {"name": str(label), "max": max_vals[i]}
            for i, label in enumerate(labels)
        ]

        axis_color = "#AAAAAA" if theme == "dark" else "#666666"

        option["radar"] = {
            "indicator": indicators,
            "name": {"textStyle": {"color": axis_color}},
        }

        series_data = []
        legend_data = []
        for i, ds in enumerate(datasets):
            name = ds.get("name", f"Series {i + 1}")
            values = ds.get("values", [])
            legend_data.append(name)
            series_data.append({
                "name": name,
                "value": values,
                "lineStyle": {"color": colors[i % len(colors)]},
                "itemStyle": {"color": colors[i % len(colors)]},
                "areaStyle": {"color": colors[i % len(colors)], "opacity": 0.15},
            })

        option["legend"] = {
            "data": legend_data,
            "bottom": 0,
            "textStyle": {"color": axis_color},
        }
        option["series"] = [{
            "type": "radar",
            "data": series_data,
        }]

        return option

    def _build_scatter(
        self,
        title: str,
        datasets: list[dict],
        colors: list[str],
        theme: str,
    ) -> dict:
        """Build scatter chart option."""
        option = self._base_option(title, theme)
        option["tooltip"]["trigger"] = "item"

        axis_color = "#AAAAAA" if theme == "dark" else "#666666"

        option["xAxis"] = {
            "type": "value",
            "axisLabel": {"color": axis_color},
        }
        option["yAxis"] = {
            "type": "value",
            "axisLabel": {"color": axis_color},
        }

        series = []
        legend_data = []
        for i, ds in enumerate(datasets):
            name = ds.get("name", f"Series {i + 1}")
            values = ds.get("values", [])
            legend_data.append(name)
            # values should be [[x,y], [x,y], ...] or [{x,y}, {x,y}, ...]
            scatter_data = []
            for point in values:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    scatter_data.append([point[0], point[1]])
                elif isinstance(point, dict):
                    scatter_data.append([point.get("x", 0), point.get("y", 0)])
                else:
                    scatter_data.append([0, point if isinstance(point, (int, float)) else 0])

            series.append({
                "name": name,
                "type": "scatter",
                "data": scatter_data,
                "itemStyle": {"color": colors[i % len(colors)]},
            })

        option["legend"] = {
            "data": legend_data,
            "bottom": 0,
            "textStyle": {"color": axis_color},
        }
        option["series"] = series

        return option
