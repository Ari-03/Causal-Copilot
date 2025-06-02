from crewai import Agent, Task, Crew, Process
from typing import Dict, List
import json
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class AlgorithmType(Enum):
    PC = "PC"
    GES = "GES"
    FCI = "FCI"
    NOTEARS = "NOTEARS"
    DIRECT_LINGAM = "DirectLiNGAM"
    ICA_LINGAM = "ICALiNGAM"
    CDNOD = "CDNOD"


@dataclass
class AlgorithmCharacteristics:
    name: str
    description: str
    assumptions: List[str]
    advantages: List[str]
    limitations: List[str]
    suitable_cases: List[str]


class CrewFilter:
    def __init__(self, args):
        self.args = args
        self.algorithm_characteristics = self._load_algorithm_characteristics()

    def _load_algorithm_characteristics(self) -> Dict[str, AlgorithmCharacteristics]:
        """Load characteristics of each algorithm"""
        return {
            AlgorithmType.PC.value: AlgorithmCharacteristics(
                name="PC",
                description="Constraint-based algorithm using conditional independence tests",
                assumptions=["No hidden confounders", "Faithfulness"],
                advantages=[
                    "Good balance between generality and efficiency",
                    "Works well with large datasets",
                ],
                limitations=[
                    "May miss some causal relationships",
                    "Sensitive to independence test choice",
                ],
                suitable_cases=[
                    "Large datasets",
                    "When all relevant variables are observed",
                ],
            ),
            AlgorithmType.GES.value: AlgorithmCharacteristics(
                name="GES",
                description="Score-based algorithm using greedy equivalence search",
                assumptions=["No hidden confounders", "Faithfulness"],
                advantages=[
                    "Efficient for larger datasets",
                    "Provides good general approach",
                ],
                limitations=[
                    "May get stuck in local optima",
                    "Requires score function choice",
                ],
                suitable_cases=[
                    "Large datasets",
                    "When score function is well-defined",
                ],
            ),
            AlgorithmType.FCI.value: AlgorithmCharacteristics(
                name="FCI",
                description="Constraint-based algorithm that handles hidden confounders",
                assumptions=["Faithfulness"],
                advantages=["Can handle hidden confounders", "More general than PC"],
                limitations=[
                    "Computationally intensive",
                    "May produce ambiguous results",
                ],
                suitable_cases=[
                    "When hidden confounders are suspected",
                    "Small to medium datasets",
                ],
            ),
            AlgorithmType.NOTEARS.value: AlgorithmCharacteristics(
                name="NOTEARS",
                description="Continuous optimization approach for causal discovery",
                assumptions=["Linear relationships", "No hidden confounders"],
                advantages=[
                    "Efficient for high-dimensional data",
                    "Continuous optimization",
                ],
                limitations=[
                    "Assumes linearity",
                    "May not capture complex relationships",
                ],
                suitable_cases=[
                    "High-dimensional data",
                    "When linearity assumption holds",
                ],
            ),
            AlgorithmType.DIRECT_LINGAM.value: AlgorithmCharacteristics(
                name="DirectLiNGAM",
                description="Linear non-Gaussian acyclic model",
                assumptions=["Linear relationships", "Non-Gaussian noise"],
                advantages=["Efficient", "Can identify full DAG"],
                limitations=["Assumes linearity", "Requires non-Gaussian noise"],
                suitable_cases=["When noise is non-Gaussian", "Linear relationships"],
            ),
            AlgorithmType.ICA_LINGAM.value: AlgorithmCharacteristics(
                name="ICALiNGAM",
                description="Independent component analysis based LiNGAM",
                assumptions=["Linear relationships", "Non-Gaussian noise"],
                advantages=["Can identify full DAG", "More robust than DirectLiNGAM"],
                limitations=["Computationally intensive", "Assumes linearity"],
                suitable_cases=[
                    "When computational resources allow",
                    "Non-Gaussian noise",
                ],
            ),
            AlgorithmType.CDNOD.value: AlgorithmCharacteristics(
                name="CDNOD",
                description="Causal discovery for nonstationary/heterogeneous data",
                assumptions=["Nonstationary data", "Heterogeneous relationships"],
                advantages=[
                    "Handles nonstationarity",
                    "Can detect changing relationships",
                ],
                limitations=[
                    "May be overkill for stationary data",
                    "Computationally intensive",
                ],
                suitable_cases=["Nonstationary data", "Heterogeneous relationships"],
            ),
        }

    def create_agents(self):
        """Create the CrewAI agents for algorithm selection"""
        data_analyzer = Agent(
            role="Data Analysis Expert",
            goal="Analyze data characteristics and requirements",
            backstory="""You are an expert in data analysis with deep knowledge of 
            statistical properties and data characteristics. Your role is to analyze 
            the dataset and identify key features that will influence algorithm selection.""",
            verbose=True,
        )

        algorithm_expert = Agent(
            role="Causal Discovery Algorithm Expert",
            goal="Select the most appropriate causal discovery algorithm",
            backstory="""You are an expert in causal discovery algorithms with deep 
            knowledge of their assumptions, advantages, and limitations. Your role is 
            to select the most appropriate algorithm based on data characteristics.""",
            verbose=True,
        )

        return data_analyzer, algorithm_expert

    def create_tasks(self, data_analyzer, algorithm_expert, data, statistics_desc):
        """Create the tasks for the agents"""
        analyze_data_task = Task(
            description=f"""Analyze the following dataset characteristics and requirements:
            Data Statistics: {statistics_desc}
            Columns: {", ".join(data.columns)}
            
            Identify key features that will influence algorithm selection, including:
            1. Data dimensionality and sample size
            2. Expected graph density
            3. Data completeness
            4. Variable types
            5. Relationship types
            6. Error characteristics
            7. Computational constraints
            
            Provide your analysis in JSON format.""",
            agent=data_analyzer,
        )

        select_algorithm_task = Task(
            description="""Based on the data analysis, select the most appropriate 
            causal discovery algorithm. Consider:
            1. Algorithm assumptions and data characteristics match
            2. Computational requirements and constraints
            3. Expected performance and accuracy
            
            Provide your selection and justification in JSON format.""",
            agent=algorithm_expert,
        )

        return analyze_data_task, select_algorithm_task

    def forward(
        self,
        global_state,
        query="What is the best causal discovery algorithm for this dataset?",
    ):
        """Select the appropriate causal discovery algorithm using CrewAI"""
        # Create agents
        data_analyzer, algorithm_expert = self.create_agents()

        # Create tasks
        analyze_data_task, select_algorithm_task = self.create_tasks(
            data_analyzer,
            algorithm_expert,
            global_state.user_data.processed_data,
            global_state.statistics.description,
        )

        # Create and run the crew
        crew = Crew(
            agents=[data_analyzer, algorithm_expert],
            tasks=[analyze_data_task, select_algorithm_task],
            process=Process.sequential,
        )

        result = crew.kickoff()

        # Parse the result and update global state
        try:
            algo_candidates = json.loads(result)
            global_state.algorithm.algorithm_candidates = algo_candidates
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON response from CrewAI")
            return global_state

        return global_state
