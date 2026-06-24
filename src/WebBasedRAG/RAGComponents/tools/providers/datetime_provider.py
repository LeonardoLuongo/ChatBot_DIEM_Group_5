from typing import List
from datetime import datetime
from langchain_core.tools import StructuredTool
from WebBasedRAG.RAGComponents.tools.core.interfaces import ToolProvider

class DateTimeToolProvider(ToolProvider):
    """
    Fornisce consapevolezza temporale in modo isolato.
    """
    def get_tools(self) -> List[StructuredTool]:
        def current_datetime_action() -> str:
            now = datetime.now()
            days_in_english = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_of_week = days_in_english[now.weekday()]
            
            return (
                f"CURRENT TEMPORAL CONTEXT:\n"
                f"- Day: {day_of_week}\n"
                f"- Date (YYYY-MM-DD): {now.strftime('%Y-%m-%d')}\n"
                f"- Time: (HH:MM:SS){now.strftime('%H:%M:%S')}\n"
                f"- Current Year: {now.year}\n"
            )

        # Non passiamo un args_schema perché la funzione non prende argomenti in input
        datetime_tool = StructuredTool.from_function(
            func=current_datetime_action,
            name="CurrentDateTime",
            description=(
                "Use this tool to find out the current date, day of the week, and exact time. "
                "Call this tool FIRST if the user's prompt contains words like 'today', 'now', 'tomorrow', "
                "'this year', or any relative time references, so you have the correct temporal context."
                "Knowing the current date is essential also to search for up-to-date information."
            )
        )

        return [datetime_tool]
