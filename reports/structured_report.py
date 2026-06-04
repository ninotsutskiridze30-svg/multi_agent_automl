from dataclasses import dataclass

@dataclass
class StructuredReport:
    agent_name: str
    summary: str
    recommendations: str

    def to_text(self):
        return f"""
Agent: {self.agent_name}

Summary:
{self.summary}

Recommendations:
{self.recommendations}
"""