from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from preprocess.dataset import knowledge_info
from preprocess.stat_info_functions import (
    stat_info_collection,
    convert_stat_info_to_text,
)
from agentic.tools import DataAnalysisTool, AlgorithmSelectionTool, CausalDiscoveryTool, VisualizationTool, ReportGenerationTool
from algorithm.crew_filter import CrewFilter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from algorithm.hyperparameter_selector import HyperparameterSelector
from postprocess.judge import Judge
from postprocess.visualization import Visualization, convert_to_edges
from preprocess.eda_generation import EDA
from report.report_generation import Report_generation
from global_setting.Initialize_state import global_state_initialization, load_data
import json
import os
import logging
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class CausalDiscoveryState:
    """State object to track the progress of causal discovery"""

    data: pd.DataFrame
    statistics: Dict
    knowledge_docs: str
    selected_algorithm: str
    results: Dict
    output_dirs: Dict[str, str]
    global_state: Any

    def update_from_global_state(self):
        """Update state from global state"""
        self.data = self.global_state.user_data.processed_data
        self.statistics = self.global_state.statistics.__dict__
        self.knowledge_docs = self.global_state.user_data.knowledge_docs
        self.selected_algorithm = (
            self.global_state.algorithm.selected_algorithm
            if hasattr(self.global_state.algorithm, "selected_algorithm")
            else ""
        )
        self.results = {
            "graph": self.global_state.results.converted_graph
            if hasattr(self.global_state.results, "converted_graph")
            else None,
            "metrics": self.global_state.results.metrics
            if hasattr(self.global_state.results, "metrics")
            else None,
        }

class CausalDiscoveryAgent:
    def __init__(self, args):
        self.args = args
        self.state = None
        self.initialize_state()
        self.initialize_tools()
        self.initialize_agents()
        self.initialize_tasks()

    def initialize_state(self):
        """Initialize the global state and load data"""
        try:
            logger.info("Initializing state")
            global_state = global_state_initialization(self.args)
            global_state = load_data(global_state, self.args)

            if self.args.data_mode == "real":
                logger.info(f"Loading real-world data from {self.args.data_file}")
                global_state.user_data.raw_data = self.load_real_world_data(
                    self.args.data_file
                )

            global_state.user_data.processed_data = self.process_user_query(
                self.args.initial_query, global_state.user_data.raw_data
            )

            # Initialize selected features
            global_state.user_data.selected_features = (
                global_state.user_data.processed_data.columns.tolist()
            )
            global_state.user_data.visual_selected_features = (
                global_state.user_data.selected_features
            )

            # Initialize logging structure
            if not hasattr(global_state.logging, "select_conversation"):
                global_state.logging.select_conversation = [
                    {"response": self.args.initial_query}
                ]
            if not hasattr(global_state.logging, "graph_conversion"):
                global_state.logging.graph_conversion = {}

            # Collect statistics and knowledge
            logger.info("Collecting statistics and knowledge")
            global_state = stat_info_collection(global_state)
            global_state = knowledge_info(self.args, global_state)
            global_state.statistics.description = convert_stat_info_to_text(
                global_state.statistics
            )

            self.state = CausalDiscoveryState(
                data=global_state.user_data.processed_data,
                statistics=global_state.statistics.__dict__,
                knowledge_docs=global_state.user_data.knowledge_docs,
                selected_algorithm="",
                results={},
                output_dirs={
                    "report": self.args.output_report_dir,
                    "graph": self.args.output_graph_dir,
                },
                global_state=global_state,
            )
            logger.info("State initialization completed")
        except Exception as e:
            logger.error(f"Error initializing state: {str(e)}")
            raise

    def initialize_tools(self):
        """Initialize all tools for the agent"""
        try:
            logger.info("Initializing tools")
            self.eda = EDA(self.state.global_state)

            # Initialize EDA results
            if not hasattr(self.state.global_state.results, "eda"):
                self.state.global_state.results.eda = {}

            self.filter = CrewFilter(self.args)
            self.reranker = Reranker(self.args)
            self.hp_selector = HyperparameterSelector(self.args)
            self.programmer = Programming(self.args)
            self.judge = Judge(self.state.global_state, self.args)
            self.visualizer = Visualization(self.state.global_state)
            self.report_gen = Report_generation(self.state.global_state, self.args)

            self.tools = [
                DataAnalysisTool(self.eda),
                AlgorithmSelectionTool(self.filter, self.reranker, self.hp_selector),
                CausalDiscoveryTool(self.programmer),
                VisualizationTool(self.visualizer),
                ReportGenerationTool(self.report_gen),
            ]
            logger.info("Tools initialization completed")
        except Exception as e:
            logger.error(f"Error initializing tools: {str(e)}")
            raise

    def initialize_agents(self):
        """Initialize the CrewAI agents"""
        try:
            logger.info("Initializing agents")

            self.data_analyzer = Agent(
                role="Data Analysis Expert",
                goal="Analyze data characteristics and perform EDA",
                backstory="""You are an expert in data analysis with deep knowledge of 
                statistical properties and data characteristics. Your role is to analyze 
                the dataset and identify key features that will influence causal discovery.""",
                tools=[self.tools[0]],  # DataAnalysisTool
                verbose=True,
            )

            self.algorithm_selector = Agent(
                role="Causal Discovery Algorithm Expert",
                goal="Select and configure the most appropriate causal discovery algorithm",
                backstory="""You are an expert in causal discovery algorithms with deep 
                knowledge of their assumptions, advantages, and limitations. Your role is 
                to select and configure the most appropriate algorithm based on data characteristics.""",
                tools=[self.tools[1]],  # AlgorithmSelectionTool
                verbose=True,
            )

            self.causal_analyst = Agent(
                role="Causal Analysis Expert",
                goal="Perform causal discovery and interpret results",
                backstory="""You are an expert in causal analysis with deep knowledge of 
                causal inference and interpretation. Your role is to perform causal discovery 
                and provide meaningful interpretations of the results.""",
                tools=[
                    self.tools[2],
                    self.tools[3],
                ],  # CausalDiscoveryTool and VisualizationTool
                verbose=True,
            )

            self.report_generator = Agent(
                role="Report Generation Expert",
                goal="Generate comprehensive reports of the causal discovery process",
                backstory="""You are an expert in technical report generation with deep 
                knowledge of causal discovery and data analysis. Your role is to create 
                clear and comprehensive reports of the causal discovery process and results.""",
                tools=[self.tools[4]],  # ReportGenerationTool
                verbose=True,
            )

            logger.info("Agents initialization completed")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise

    def initialize_tasks(self):
        """Initialize the tasks for the agents"""
        try:
            logger.info("Initializing tasks")

            self.analyze_data_task = Task(
                description=f"""Analyze the following dataset:
                Data Shape: {self.state.data.shape}
                Columns: {", ".join(self.state.data.columns)}
                Statistics: {json.dumps(self.state.statistics, indent=2)}
                Knowledge: {self.state.knowledge_docs}
                
                Perform exploratory data analysis and identify:
                1. Data quality issues
                2. Statistical properties
                3. Potential causal relationships
                4. Domain-specific considerations""",
                agent=self.data_analyzer,
                expected_output="A comprehensive analysis of the dataset including data quality, statistical properties, and potential causal relationships.",
            )

            self.select_algorithm_task = Task(
                description="""Based on the data analysis, select and configure the most 
                appropriate causal discovery algorithm. Consider:
                1. Algorithm assumptions and data characteristics match
                2. Computational requirements
                3. Expected performance""",
                agent=self.algorithm_selector,
                expected_output="The selected causal discovery algorithm with its configuration and justification for the selection.",
            )

            self.perform_analysis_task = Task(
                description="""Using the selected algorithm, perform causal discovery and 
                interpret the results. Consider:
                1. Causal relationships identified
                2. Confidence in the relationships
                3. Potential confounding factors
                4. Limitations of the analysis""",
                agent=self.causal_analyst,
                expected_output="A detailed analysis of the causal relationships found, including confidence levels and potential limitations.",
            )

            self.generate_report_task = Task(
                description="""Generate a comprehensive report of the causal discovery process, 
                including:
                1. Data analysis summary
                2. Algorithm selection rationale
                3. Causal discovery results
                4. Interpretation and implications
                5. Limitations and future work""",
                agent=self.report_generator,
                expected_output="A comprehensive PDF report documenting the entire causal discovery process and its findings.",
            )

            logger.info("Tasks initialization completed")
        except Exception as e:
            logger.error(f"Error initializing tasks: {str(e)}")
            raise

    def run(self, query: str) -> str:
        """Run the causal discovery agent"""
        try:
            logger.info(f"Running agent with query: {query}")

            # Create and run the crew
            crew = Crew(
                agents=[
                    self.data_analyzer,
                    self.algorithm_selector,
                    self.causal_analyst,
                    self.report_generator,
                ],
                tasks=[
                    self.analyze_data_task,
                    self.select_algorithm_task,
                    self.perform_analysis_task,
                    self.generate_report_task,
                ],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            self.state.update_from_global_state()
            logger.info("Agent run completed")
            return result
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return f"Error: {str(e)}"

    def load_real_world_data(self, file_path):
        """Load real-world data from file"""
        try:
            logger.info(f"Loading data from {file_path}")
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            elif file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    return pd.DataFrame(json.load(f))
            else:
                raise ValueError(f"Unsupported file format for {file_path}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def process_user_query(self, query, data):
        """Process user query and filter data accordingly"""
        try:
            logger.info(f"Processing query: {query}")
            query_dict = {}
            if ";" in query or ":" in query:
                for part in query.split(";"):
                    key, value = part.strip().split(":")
                    query_dict[key.strip()] = value.strip()

            if "filter" in query_dict and query_dict["filter"] == "continuous":
                data = data.select_dtypes(include=["float64", "int64"])

            return data
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise


def main(args):
    """Main function to run the causal discovery agent"""
    try:
        logger.info("Starting causal discovery agent")
        agent = CausalDiscoveryAgent(args)
        return agent
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Causal Discovery Agent")
    parser.add_argument(
        "--data-file", type=str, required=True, help="Path to input data file"
    )
    parser.add_argument(
        "--output-report-dir",
        type=str,
        default="output/reports",
        help="Directory for output reports",
    )
    parser.add_argument(
        "--output-graph-dir",
        type=str,
        default="output/graphs",
        help="Directory for output graphs",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="real",
        choices=["real", "simulated"],
        help="Data mode",
    )
    parser.add_argument(
        "--simulation-mode",
        type=str,
        default="offline",
        choices=["online", "offline"],
        help="Simulation mode",
    )
    parser.add_argument(
        "--initial-query",
        type=str,
        default="Do causal discovery on this dataset",
        help="Initial query",
    )
    parser.add_argument(
        "--demo-mode",
        type=bool,
        default=False,
        help="Run in demo mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode",
    )
    parser.add_argument(
        "--parallel",
        type=bool,
        default=False,
        help="Enable parallel computing",
    )

    args = parser.parse_args()
    agent = main(args)

    # Example interaction
    response = agent.run("Analyze this dataset and find causal relationships")
    print(response)
