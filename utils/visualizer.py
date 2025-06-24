import pandas as pd
import plotly.express as px
import streamlit as st
from typing import Dict, Any, Optional
from utils.logger import Logger

class Visualizer:
    def __init__(self, logger):
        self.logger = logger
    
    def create_histogram(self, df: pd.DataFrame, column: str, title: str) -> Any:
        """Create a histogram for a given column."""
        try:
            fig = px.histogram(df, x=column, nbins=20, title=title)
            self.logger.log_info(f"Created histogram for column: {column}")
            return fig
        except Exception as e:
            self.logger.log_error(f"Histogram creation failed: {str(e)}")
            return None
    
    def create_bar(self, x: list, y: list, title: str) -> Any:
        """Create a bar chart."""
        try:
            fig = px.bar(x=x, y=y, title=title)
            self.logger.log_info(f"Created bar chart: {title}")
            return fig
        except Exception as e:
            self.logger.log_error(f"Bar chart creation failed: {str(e)}")
            return None
    
    def create_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Any:
        """Create a scatter plot."""
        try:
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
            self.logger.log_info(f"Created scatter plot for {x_col} vs {y_col}")
            return fig
        except Exception as e:
            self.logger.log_error(f"Scatter plot creation failed: {str(e)}")
            return None
    
    def display_visualization(self, fig: Any, st_container: Any = st) -> None:
        """Display visualization in Streamlit."""
        try:
            if fig:
                st_container.plotly_chart(fig)
                self.logger.log_info("Displayed visualization in Streamlit")
            else:
                self.logger.log_warning("No visualization to display")
        except Exception as e:
            self.logger.log_error(f"Visualization display failed: {str(e)}")