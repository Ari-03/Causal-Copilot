from crewai.tools import BaseTool
import logging
import numpy as np
from pydantic import BaseModel, Field
from typing import Any

from algorithm.crew_filter import CrewFilter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from algorithm.hyperparameter_selector import HyperparameterSelector
from postprocess.visualization import Visualization
from preprocess.eda_generation import EDA
from report.report_generation import Report_generation

logger = logging.getLogger(__name__)


class DataAnalysisToolSchema(BaseModel):
    """Schema for DataAnalysisTool inputs"""

    query: str = Field(
        default="Perform exploratory data analysis",
        description="The analysis query or description",
    )


class DataAnalysisTool(BaseTool):
    name: str = "perform_data_analysis"
    description: str = """Perform exploratory data analysis on the dataset.
    This tool will:
    1. Analyze data quality and characteristics
    2. Generate statistical summaries
    3. Create visualizations
    4. Identify potential causal relationships
    """
    eda: Any = None
    args_schema: type[BaseModel] = DataAnalysisToolSchema

    def __init__(self, eda: EDA):
        super().__init__()
        self.eda = eda

    def _run(self, query: str = "Perform exploratory data analysis") -> str:
        """Run the EDA tool

        Args:
            query (str): The analysis query or description. Defaults to "Perform exploratory data analysis".

        Returns:
            str: Result of the EDA analysis
        """
        try:
            logger.info(f"Starting EDA analysis with query: {query}")
            self.eda.generate_eda()
            logger.info("EDA completed successfully")
            return "EDA completed successfully. Check the output directory for visualizations and analysis."
        except Exception as e:
            logger.error(f"Error in EDA: {str(e)}")
            return f"Error performing EDA: {str(e)}"


class AlgorithmSelectionToolSchema(BaseModel):
    """Schema for AlgorithmSelectionTool inputs"""

    query: str = Field(
        default="Select and configure the most appropriate causal discovery algorithm",
        description="The query or description for algorithm selection",
    )


class AlgorithmSelectionTool(BaseTool):
    name: str = "select_algorithm"
    description: str = """Select and configure the most appropriate causal discovery algorithm.
    This tool will:
    1. Filter suitable algorithms based on data characteristics
    2. Rerank algorithms by suitability
    3. Select optimal hyperparameters
    4. Configure the chosen algorithm
    """
    filter: Any = None
    reranker: Any = None
    hp_selector: Any = None
    args_schema: type[BaseModel] = AlgorithmSelectionToolSchema

    def __init__(
        self,
        filter: CrewFilter,
        reranker: Reranker,
        hp_selector: HyperparameterSelector,
    ):
        super().__init__()
        self.filter = filter
        self.reranker = reranker
        self.hp_selector = hp_selector

    def _run(
        self,
        query: str = "Select and configure the most appropriate causal discovery algorithm",
    ) -> str:
        """Run the algorithm selection pipeline

        Args:
            query (str): The query or description for algorithm selection.

        Returns:
            str: Result of the algorithm selection process
        """
        try:
            logger.info(f"Starting algorithm selection with query: {query}")

            # Filter algorithms
            logger.info("Filtering algorithms")
            global_state = self.filter.forward(self.filter.global_state)

            # Rerank algorithms
            logger.info("Reranking algorithms")
            global_state = self.reranker.forward(global_state)

            # Select hyperparameters
            logger.info("Selecting hyperparameters")
            global_state = self.hp_selector.forward(global_state)

            logger.info(
                f"Algorithm selection completed: {global_state.algorithm.selected_algorithm}"
            )
            return f"Algorithm selection completed. Selected algorithm: {global_state.algorithm.selected_algorithm}"
        except Exception as e:
            logger.error(f"Error in algorithm selection: {str(e)}")
            return f"Error selecting algorithm: {str(e)}"


class CausalDiscoveryToolSchema(BaseModel):
    """Schema for CausalDiscoveryTool inputs"""

    query: str = Field(
        default="Run causal discovery algorithm and generate results",
        description="The query or description for causal discovery",
    )


class CausalDiscoveryTool(BaseTool):
    name: str = "run_causal_discovery"
    description: str = """Run the causal discovery algorithm and generate results.
    This tool will:
    1. Execute the selected algorithm
    2. Generate causal graphs
    3. Compute causal relationships
    4. Store results for visualization and reporting
    """
    programmer: Any = None
    args_schema: type[BaseModel] = CausalDiscoveryToolSchema

    def __init__(self, programmer: Programming):
        super().__init__()
        self.programmer = programmer

    def _run(
        self, query: str = "Run causal discovery algorithm and generate results"
    ) -> str:
        """Run the causal discovery algorithm

        Args:
            query (str): The query or description for causal discovery.

        Returns:
            str: Result of the causal discovery process
        """
        try:
            logger.info(f"Starting causal discovery with query: {query}")
            global_state = self.programmer.forward(self.programmer.global_state)
            logger.info("Causal discovery completed")
            return (
                "Causal discovery completed. Results are available in the global state."
            )
        except Exception as e:
            logger.error(f"Error in causal discovery: {str(e)}")
            return f"Error running causal discovery: {str(e)}"


class VisualizationToolSchema(BaseModel):
    """Schema for VisualizationTool inputs"""

    query: str = Field(
        default="Create visualizations of the causal discovery results",
        description="The query or description for visualization",
    )


class VisualizationTool(BaseTool):
    name: str = "visualize_results"
    description: str = """Create visualizations of the causal discovery results.
    This tool will:
    1. Generate causal graphs
    2. Create summary visualizations
    3. Compare with ground truth (if available)
    4. Save visualizations to output directory
    """
    visualizer: Any = None
    args_schema: type[BaseModel] = VisualizationToolSchema

    def __init__(self, visualizer: Visualization):
        super().__init__()
        self.visualizer = visualizer

    def _run(
        self, query: str = "Create visualizations of the causal discovery results"
    ) -> str:
        """Create visualizations

        Args:
            query (str): The query or description for visualization.

        Returns:
            str: Result of the visualization process
        """
        try:
            logger.info(f"Starting visualization with query: {query}")
            global_state = self.visualizer.global_state

            if (
                global_state.statistics.time_series
                and global_state.results.lagged_graph is not None
            ):
                logger.info("Generating time series visualizations")
                converted_graph = global_state.results.lagged_graph
                pos_est = self.visualizer.get_pos(converted_graph[0])
                for i in range(converted_graph.shape[0]):
                    _ = self.visualizer.plot_pdag(
                        converted_graph[i],
                        f"{global_state.algorithm.selected_algorithm}_initial_graph_{i}.svg",
                        pos=pos_est,
                    )
                summary_graph = np.any(converted_graph, axis=0).astype(int)
                _ = self.visualizer.plot_pdag(
                    summary_graph,
                    f"{global_state.algorithm.selected_algorithm}_initial_graph_summary.svg",
                    pos=pos_est,
                )
            else:
                logger.info("Generating standard visualizations")
                pos_est = self.visualizer.get_pos(global_state.results.converted_graph)
                if global_state.user_data.ground_truth is not None:
                    _ = self.visualizer.plot_pdag(
                        global_state.user_data.ground_truth,
                        "true_graph.pdf",
                        pos=pos_est,
                    )
                _ = self.visualizer.plot_pdag(
                    global_state.results.converted_graph,
                    f"{global_state.algorithm.selected_algorithm}_initial_graph.pdf",
                    pos=pos_est,
                )
            logger.info("Visualization completed")
            return "Visualization completed. Check the output directory for graphs."
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            return f"Error creating visualizations: {str(e)}"


class ReportGenerationToolSchema(BaseModel):
    """Schema for ReportGenerationTool inputs"""

    query: str = Field(
        default="Generate comprehensive report of the causal discovery process",
        description="The query or description for report generation",
    )


class ReportGenerationTool(BaseTool):
    name: str = "generate_report"
    description: str = """Generate a comprehensive report of the causal discovery process.
    This tool will:
    1. Summarize data analysis
    2. Document algorithm selection
    3. Present causal discovery results
    4. Include visualizations
    5. Generate PDF report
    """
    report_gen: Any = None
    args_schema: type[BaseModel] = ReportGenerationToolSchema

    def __init__(self, report_gen: Report_generation):
        super().__init__()
        self.report_gen = report_gen

    def _run(
        self,
        query: str = "Generate comprehensive report of the causal discovery process",
    ) -> str:
        """Generate the report

        Args:
            query (str): The query or description for report generation.

        Returns:
            str: Result of the report generation process
        """
        try:
            logger.info(f"Starting report generation with query: {query}")
            report = self.report_gen.generation()
            self.report_gen.save_report(report)
            logger.info("Report generation completed")
            return "Report generation completed. Check the output directory for the PDF report."
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            return f"Error generating report: {str(e)}"
