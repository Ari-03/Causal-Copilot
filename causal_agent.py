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
from crewai_tools import FileReadTool, FileWriterTool, EXASearchTool, SerperDevTool, ScrapeWebsiteTool
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

            self.data_analysis_tool=DataAnalysisTool(self.eda)
            self.algorithm_selection_tool=AlgorithmSelectionTool(self.filter, self.reranker, self.hp_selector)
            self.causal_discovery_tool=CausalDiscoveryTool(self.programmer)
            self.visualization_tool=VisualizationTool(self.visualizer)
            self.report_generation_tool=ReportGenerationTool(self.report_gen)
            self.file_read_tool=FileReadTool()
            self.file_writer_tool=FileWriterTool()
            self.EXA_search_tool=EXASearchTool(num_results=10, type='auto')
            self.serper_dev_tool=SerperDevTool()
            self.scrape_website_tool=ScrapeWebsiteTool()

            logger.info("Tools initialization completed")
        except Exception as e:
            logger.error(f"Error initializing tools: {str(e)}")
            raise

    def initialize_agents(self):
        """Initialize the CrewAI agents"""
        try:
            logger.info("Initializing agents")
            
            self.scientist_agent = Agent(
                role="Planner and Evaluator of the causal analysis process",
                goal="""Design a sequence of subgoals to answer a scientific causal question provided by the user.
                    Iteratively assign subgoals to the assistant agent, review returned results, critique their quality, 
                    adjust the analysis plan as needed, and determine when the causal query has been fully answered.""",
                backstory="""You are a domain-aware scientific thinker and planner with extensive experience in causal inference workflows.
                    You specialize in breaking down complex scientific questions into structured analytical tasks, critically evaluating outputs, 
                    and steering the research process with rigor and precision. You maintain high scientific standards, adapt plans dynamically, 
                    and ensure all steps stay aligned with the original causal question. You know how to judge when enough evidence has been gathered
                    to make a defensible causal claim.""",
                tools=[self.file_read_tool, self.file_writer_tool],  # FileReadTool, FileWriterTool
                verbose=True,
            )

            self.assistant_agent = Agent(
                role="Orchestrator of the causal analysis process",
                goal="""Coordinate the execution of each subgoal by delegating specialized tasks to domain-specific agents, 
                    monitor their progress, collect results, and relay the findings to the Scientist Agent for evaluation.
                    Incorporate feedback from the Scientist Agent to refine execution strategies and iterate on subgoal completion.
                    Once all subgoals are complete and approved, compile a final deliverable report summarizing the full causal analysis.""",
                backstory="""You are a process-oriented orchestrator and supervisor who oversees the entire execution of a causal analysis workflow.
                    You specialize in translating high-level plans into targeted actions by delegating work to specialized agents like search, coding, and discovery agents.
                    You track task progress using shared documents, update subgoal states, and adapt your orchestration strategy based on feedback from the Scientist Agent.
                    You are the glue that keeps all moving parts coordinated, ensuring that subgoals are executed efficiently and aligned with scientific expectations.""",
                tools=[self.file_read_tool, self.file_writer_tool],  # FileReadTool, FileWriterTool
                verbose=True,
                allow_delegation=True,
            )

            self.search_agent = Agent(
                role="Causal Inference Search Agent",
                goal="""Uncover cutting-edge developments in causal inference""",
                backstory="""You're a seasoned researcher with a knack for uncovering the latest
                    developments in causal inference. Known for your ability to find the most relevant
                    information and present it in a clear and concise manner.""",
                tools=[self.serper_dev_tool, self.scrape_website_tool],  # SerperDevTool, ScrapeWebsiteTool
                verbose=True,
                allow_delegation=True,
            )

            self.verification_agent = Agent(
                role="Causal Inference Verification Agent",
                goal="""Verify the correctness of causal inference tasks and results""",
                backstory="""You are a meticulous verifier. You check the validity of data, DAGs, SCMs, and results, 
                    performing a thorough literature review and ensuring that all results are logically sound 
                    and consistent with causal principles.""",
                tools=[self.EXA_search_tool],  # EXASearchTool
                verbose=True,
                allow_delegation=True,
            )

            self.data_analyzer = Agent(
                role="Data Analysis Expert",
                goal="Analyze data characteristics and perform EDA",
                backstory="""You are an expert in data analysis with deep knowledge of 
                statistical properties and data characteristics. Your role is to analyze 
                the dataset and identify key features that will influence causal discovery.""",
                tools=[self.data_analysis_tool],  # DataAnalysisTool
                verbose=True,
            )

            self.algorithm_selector = Agent(
                role="Causal Discovery Algorithm Expert",
                goal="Select and configure the most appropriate causal discovery algorithm",
                backstory="""You are an expert in causal discovery algorithms with deep 
                knowledge of their assumptions, advantages, and limitations. Your role is 
                to select and configure the most appropriate algorithm based on data characteristics.""",
                tools=[self.algorithm_selection_tool],  # AlgorithmSelectionTool
                verbose=True,
            )

            self.causal_analyst = Agent(
                role="Causal Analysis Expert",
                goal="Perform causal discovery and interpret results",
                backstory="""You are an expert in causal analysis with deep knowledge of 
                causal inference and interpretation. Your role is to perform causal discovery 
                and provide meaningful interpretations of the results.""",
                tools=[self.causal_discovery_tool, self.visualization_tool],  # CausalDiscoveryTool and VisualizationTool
                verbose=True,
            )

            self.report_generator = Agent(
                role="Report Generation Expert",
                goal="Generate comprehensive reports of the causal discovery process",
                backstory="""You are an expert in technical report generation with deep 
                knowledge of causal discovery and data analysis. Your role is to create 
                clear and comprehensive reports of the causal discovery process and results.""",
                tools=[self.report_generation_tool],  # ReportGenerationTool
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
            
            self.create_plan_task = Task(
                description="""You are given a causal question or a dataset with a user query. Begin by identifying the treatment(s), outcome(s),
                    and any relevant confounders or domain assumptions. Use this information to formulate a causal analysis plan consisting
                    of structured high-level subgoals, each targeting a phase in the pipeline such as:
                    - Background literature search and constraint identification
                    - Data preprocessing and diagnostics
                    - Causal graph discovery (e.g., PC, NOTEARS, LiNGAM)
                    - Validation using cross-evidence or simulation
                    - Causal inference (e.g., effect estimation)
                    - Result interpretation and reporting

                    For each subgoal, specify the objective and expected output only. Do not assign agents — this will be handled in later steps.
                    Write the resulting subgoal list to 'SubGoals.json'.

                    User query: {causal_query}
                    Dataset: {dataset_name}""",
                agent=self.scientist_agent,
                expected_output="""A file named 'SubGoals.json' containing a list of 4-6 high-level subgoals tailored to the causal question. Each subgoal must include:

                    - description: a clear, concise goal (e.g., \"Run causal discovery on the cleaned dataset\")
                    - expected_output: the expected deliverable for the subgoal (e.g., \"a DAG with estimated edge weights and adjacency matrix\")
                    - status: always initialized as 'pending'
                    - score: null
                    - feedback: null
                    - combined_results: null

                    Example structure:
                    [
                    {
                        "id": 1,
                        "description": "Perform literature review on potential confounders between variable X and Y",
                        "expected_output": "A list of at least 3 relevant papers and a summary of domain constraints",
                        "status": "pending",
                        "score": null,
                        "feedback": null,
                        "combined_results": null,
                    },
                    ...
                    ]""",
            )

            self.start_subgoal_task = Task(
                description="""Read 'SubGoals.json' and identify the first subgoal where status != 'completed' and (score == null or score < {threshold}). Utilize feedback from the Scientist Agent if available.
                    Decompose this subgoal into 2-4 smaller, very concrete subsubgoals that can be executed independently, each meant for and assigned to a specialized agent (for now, assume all are for dummy_specialized_agent).

                    For example:
                    - If the subgoal involves causal discovery, assign:
                    • Data diagnostics to CodingAgent
                    • Run NOTEARS algorithm to DiscoveryAgent
                    • Literature cross-check to VerificationAgent
                    - If the subgoal involves validation, assign:
                    • Literature search to SearchAgent
                    • Counterfactual simulation to VerificationAgent

                    For now, assume all are for dummy_specialized_agent

                    Save the resulting subsubgoals to a file named 'SubSubGoals.json'. Each subsubgoal must include:
                    - id: unique subsubgoal ID
                    - parent_id: corresponding subgoal ID
                    - description: a detailed, concrete task definition
                    - assigned_agent: name of the agent responsible (for now, assume all are for dummy_specialized_agent)
                    - status: initialized as 'pending'
                    - tool_hints: optional suggestions (e.g., algorithm or dataset to use)
                    - output: null

                    Also, update the corresponding subgoal in 'SubGoals.json' to status='in_progress'.

                    Threshold: {threshold}""",
                agent=self.assistant_agent,
                expected_output="""'SubSubGoals.json' containing 2-4 well-defined subsubgoals, each with:
                    - id
                    - parent_id
                    - description
                    - assigned_agent
                    - status='pending'
                    - Optional: tool_hints
                    - output: null

                    'SubGoals.json' is updated with the selected subgoal marked as status='in_progress'.""",
            )

            self.collect_subgoal_results_task = Task(
                description="""Read 'SubSubGoals.json' and identify all subsubgoals with status='completed'. Group them by their parent_id.

                    For each group:
                    1. Synthesize a comprehensive summary of the outputs based on subgoal type:
                        - *Discovery*: include DAG or CPDAG structure, algorithm used, edge confidence, and any assumptions or limitations
                        - *Validation*: include literature support, consistency checks, and simulation results
                        - *Inference*: include effect estimates, standard errors, confidence intervals, and assumptions

                    2. Write the synthesized summary into the parent subgoal (in 'SubGoals.json') under the field combined_result.

                    3. Mark the parent subgoal's status as 'for_review'.

                    Format the summary according to the requested style:
                    - 'structured': use numbered bullet points and section headers
                    - 'narrative': use paragraph-style explanation with contextual transitions

                    Summary style: {summary_style}""",
                agent=self.assistant_agent,
                expected_output="""'SubGoals.json' is updated with:
                    - combined_result: a synthesized, human-readable summary for each updated subgoal
                    - status='for_review' for each parent subgoal""",
            )

            self.evaluate_subgoal_task = Task(
                description="""Read 'SubGoals.json' and identify all subgoals with status='for_review'. For each, review the combined_result field and evaluate its quality using the following criteria:

                    Evaluation by subgoal type:
                    - *Discovery*: assess graph clarity, validity of methods, and confidence in inferred relationships
                    - *Validation*: examine consistency across literature, simulations, and assumptions
                    - *Inference*: verify interpretability of causal effect estimates and robustness of assumptions

                    Use the following weighted scoring rubric to assign a score between 0.0 and 1.0 for each subgoal:

                    Scoring Rubric (rate each on 0.0-1.0 scale):

                    1. Clarity and completeness of output (weight: 0.25)  
                    - Are key components (results, assumptions, methodology) clearly presented?

                    2. Methodological soundness (weight: 0.30)  
                    - Are methods appropriate and properly applied?

                    3. Evidence strength or support (weight: 0.25)  
                    - Is the result backed by sufficient data, literature, or simulation?

                    4. Interpretability and usefulness (weight: 0.20)  
                    - Is the result understandable and actionable for the main causal query?

                    Compute the final score as a weighted average. Interpret scores as follows:
                    - 0.85 - 1.0: Excellent. No revision needed.
                    - 0.70 - 0.84: Good. Acceptable but could be improved.
                    - 0.50 - 0.69: Marginal. Needs refinement.
                    - < 0.50: Inadequate. Subgoal must be redone.

                    For each subgoal:
                    1. Assign a score between 0.0 and 1.0.
                    2. Provide concise but specific feedback highlighting strengths or deficiencies.
                    3. Compare the score to the threshold:
                        - If score >= {threshold}, mark the subgoal as 'completed'
                        - If score < {threshold}, mark it 'pending' and suggest refinements

                    Re-evaluate the entire plan by looking at the subgoals. If deemed necessary, make changes to the plan. Remove or add subgoals if necessary. Maintain the json fields of the unchanged subgoals as they are, such as status, combined results, etc.

                    Save all updates back to 'SubGoals.json'.

                    Important note: for now, provide a score of 1.0 no matter what output is returned by the dummy_specialized_agent.

                    Evaluation threshold: {threshold}""",
                agent=self.scientist_agent,
                expected_output="""'SubGoals.json' is updated with:
                    - score: float between 0.0 and 1.0
                    - feedback: evaluator's written response
                    - status: set to 'completed' or 'pending' based on whether the score meets the threshold
                    for each subgoal with status='for_review'.""",
            )

            self.verification_task = Task(
                description="""The results of our causal inference task are given here:
                    {causal_results}
                    Break down the results into bullet points and verify, or refute, each one with sources.
                    Ensure that all conclusions are logically sound and consistent with information given in the literature.
                    If there is evidence contrary to the conclusions, please provide a detailed explanation, with sources""",
                agent=self.verification_agent,
                expected_output="""A verification report detailing the validity of data, DAGs, SCMs, and results, along with links to any 
                    sources or references used. Formatted as markdown with bullet points.""",
            )

            self.search_task = Task(
                description="""Conduct a deep and comprehensive literature review on the topic: **{topic}**.
                    Your goal is to identify and compile **all relevant and significant publications** related to this topic
                    — including foundational works, recent breakthroughs, state-of-the-art techniques, and ongoing debates.
                    Focus on academic research papers, conference proceedings, preprints, technical reports, and any
                    high-quality secondary sources. Use only trusted sources and scholarly databases (e.g., Google Scholar, arXiv, Semantic Scholar).
                    The agent should attempt to retrieve more than 10 papers by either paginating through search results or issuing multiple semantically similar queries.
                    The review should reflect the literature available as of {date}.""",
                agent=self.search_agent,
                expected_output="""Return a detailed list of **all relevant research papers and sources** for the topic **{topic}**.
                    For each entry, include the following:
                    - **Title** of the paper or publication
                    - **Author(s)**
                    - **Year** of publication
                    - **Venue** (e.g., NeurIPS, Nature, arXiv, JMLR, etc.)
                    - A **2–3 sentence summary** of the main contribution
                    - A **direct link to the Google Scholar search result** or profile for that paper
                    - A **direct PDF or publisher URL** if available (e.g., arXiv, ACM, Springer, etc.)
                    - [Optional] **DOI** or citation metadata (if accessible)

                    The output should be structured clearly in bullet-point or tabular format, and include **as many papers as are genuinely relevant**, not just a fixed number.
                    Emphasize papers that are **widely cited, novel, or influential**, but include niche papers if they add conceptual or empirical value.""",
            )

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
