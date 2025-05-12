"""
html_reporter.py - Creates visual HTML reports for KBO candidates

This module generates visually appealing HTML reports for KBO detection results,
providing interactive visualizations and formatted data for human review.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import base64
import jinja2
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from astropy.time import Time

# Set up logging
logger = logging.getLogger(__name__)

class HTMLReporter:
    """
    Creates visual HTML reports for KBO candidates.
    
    This class provides methods to generate HTML reports with interactive
    visualizations for KBO detection results, matches, and orbit analysis.
    """
    
    def __init__(self, 
                template_dir: Optional[str] = None, 
                output_dir: Optional[str] = None,
                include_plots: bool = True):
        """
        Initialize the HTML reporter.
        
        Args:
            template_dir: Directory containing Jinja2 templates.
            output_dir: Directory to save HTML reports to.
            include_plots: Whether to include plots in reports.
        """
        self.output_dir = output_dir
        self.include_plots = include_plots
        
        # Set up template directory
        if template_dir is None:
            # Use default templates directory within the module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.template_dir = os.path.join(module_dir, "templates")
        else:
            self.template_dir = template_dir
        
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_candidate_report(self, 
                               candidate: Dict[str, Any], 
                               matches: Dict[str, Any],
                               catalog_results: Dict[str, Any] = None,
                               detection_images: List[Dict[str, Any]] = None) -> str:
        """
        Generate an HTML report for a KBO candidate.
        
        Args:
            candidate: Dictionary with candidate data.
            matches: Dictionary with match results.
            catalog_results: Optional catalog query results.
            detection_images: Optional list of detection image data.
            
        Returns:
            HTML string or path to saved HTML file.
        """
        # Load the candidate template
        template = self.env.get_template("candidate_report.html")
        
        # Create context for template
        context = {
            "title": f"KBO Candidate Report: {candidate.get('id', 'Unknown')}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "candidate": candidate,
            "matches": matches,
            "has_catalog_results": catalog_results is not None,
            "catalog_results": catalog_results,
            "has_detection_images": detection_images is not None,
            "detection_images": detection_images,
            "include_plots": self.include_plots
        }
        
        # Generate plots if requested
        if self.include_plots:
            context["plots"] = self._generate_candidate_plots(candidate, matches)
        
        # Render the template
        html = template.render(**context)
        
        # Save to file if output directory specified
        if self.output_dir:
            # Create a unique filename
            candidate_id = candidate.get("id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kbo_candidate_{candidate_id}_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(html)
            
            logger.info(f"Saved candidate HTML report to {filepath}")
            return filepath
        
        # Otherwise return the HTML string
        return html
    
    def generate_detection_session_report(self, 
                                       session_info: Dict[str, Any], 
                                       candidates: List[Dict[str, Any]]) -> str:
        """
        Generate an HTML report for an entire KBO detection session.
        
        Args:
            session_info: Dictionary with detection session information.
            candidates: List of dictionaries with candidate data.
            
        Returns:
            HTML string or path to saved HTML file.
        """
        # Load the session template
        template = self.env.get_template("session_report.html")
        
        # Count candidates by classification
        classifications = {
            "new": sum(1 for c in candidates if c.get("classification") == "possible_new"),
            "high_conf": sum(1 for c in candidates if c.get("classification") == "known_high_confidence"),
            "medium_conf": sum(1 for c in candidates if c.get("classification") == "known_medium_confidence"),
            "low_conf": sum(1 for c in candidates if c.get("classification") == "known_low_confidence"),
            "uncertain": sum(1 for c in candidates if c.get("classification") == "uncertain"),
            "total": len(candidates)
        }
        
        # Create context for template
        context = {
            "title": f"KBO Detection Session Report: {session_info.get('id', 'Unknown')}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session": session_info,
            "candidates": candidates,
            "classifications": classifications,
            "include_plots": self.include_plots
        }
        
        # Generate plots if requested
        if self.include_plots:
            context["plots"] = self._generate_session_plots(session_info, candidates)
        
        # Render the template
        html = template.render(**context)
        
        # Save to file if output directory specified
        if self.output_dir:
            # Create a unique filename
            session_id = session_info.get("id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_session_{session_id}_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(html)
            
            logger.info(f"Saved session HTML report to {filepath}")
            return filepath
        
        # Otherwise return the HTML string
        return html
    
    def _generate_candidate_plots(self, 
                               candidate: Dict[str, Any], 
                               matches: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate plots for a candidate report.
        
        Args:
            candidate: Dictionary with candidate data.
            matches: Dictionary with match results.
            
        Returns:
            Dictionary mapping plot names to base64-encoded PNG data.
        """
        plots = {}
        
        # 1. Generate position plot if we have match data
        if matches and "matches" in matches and matches["matches"]:
            # Create a scatter plot of the candidate and matches
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # Plot matches
            match_distances = []
            for i, match in enumerate(matches["matches"]):
                if "ra" in match and "dec" in match and "separation" in match:
                    # Calculate offset from candidate
                    delta_ra = (match["ra"] - candidate["ra"]) * 3600  # Convert to arcsec
                    delta_dec = (match["dec"] - candidate["dec"]) * 3600
                    
                    # Plot as scatter point
                    ax.scatter(delta_ra, delta_dec, s=100//(i+1), alpha=0.7, 
                              label=f"{match.get('catalog', 'Unknown')}: {match.get('id', 'Unknown')}")
                    
                    match_distances.append(match["separation"])
            
            # Plot candidate at origin
            ax.scatter(0, 0, s=100, color='red', marker='*', label='Candidate')
            
            # Add details
            ax.set_xlabel("ΔRA (arcsec)")
            ax.set_ylabel("ΔDec (arcsec)")
            ax.set_title("Candidate Position vs. Catalog Matches")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add circle showing typical match radius (10 arcsec)
            circle = plt.Circle((0, 0), 10, fill=False, linestyle='--', color='gray')
            ax.add_patch(circle)
            
            # Ensure axes are equal scale
            ax.set_aspect('equal')
            
            # Convert to base64 PNG
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            plots["position"] = base64.b64encode(buf.read()).decode("utf-8")
        
        # 2. Generate motion vector plot if available
        if "motion_rate" in candidate and "motion_angle" in candidate:
            fig = Figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='polar')
            
            # Convert motion angle to radians
            angle_rad = np.radians(candidate["motion_angle"])
            rate = candidate["motion_rate"]
            
            # Plot motion vector as arrow
            ax.arrow(angle_rad, 0, 0, rate, alpha=0.8, width=0.1,
                    head_width=0.3, head_length=0.3, fc='blue', ec='blue')
            
            # Plot any matching object motion vectors
            if matches and "matches" in matches:
                for i, match in enumerate(matches["matches"][:3]):
                    if "motion_rate" in match and "motion_angle" in match:
                        match_angle_rad = np.radians(match["motion_angle"])
                        match_rate = match["motion_rate"]
                        
                        ax.arrow(match_angle_rad, 0, 0, match_rate, alpha=0.5, width=0.05,
                               head_width=0.15, head_length=0.15, fc='green', ec='green')
            
            # Set plot limits and labels
            ax.set_title("Motion Vector Polar Plot")
            ax.set_rmax(max(rate * 1.5, 5))  # Set radial limit
            ax.set_rticks([1, 2, 3, 4, 5])  # Set radial ticks
            ax.grid(True)
            
            # Add cardinal directions
            ax.text(np.radians(0), ax.get_rmax()*1.05, "E", ha='center', va='center')
            ax.text(np.radians(90), ax.get_rmax()*1.05, "N", ha='center', va='center')
            ax.text(np.radians(180), ax.get_rmax()*1.05, "W", ha='center', va='center')
            ax.text(np.radians(270), ax.get_rmax()*1.05, "S", ha='center', va='center')
            
            # Convert to base64 PNG
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            plots["motion"] = base64.b64encode(buf.read()).decode("utf-8")
        
        return plots
    
    def _generate_session_plots(self, 
                             session_info: Dict[str, Any], 
                             candidates: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate plots for a detection session report.
        
        Args:
            session_info: Dictionary with session information.
            candidates: List of candidate dictionaries.
            
        Returns:
            Dictionary mapping plot names to base64-encoded PNG data.
        """
        plots = {}
        
        # 1. Generate classification pie chart
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Count candidates by classification
        class_counts = {
            "New Objects": sum(1 for c in candidates if c.get("classification") == "possible_new"),
            "High Confidence Matches": sum(1 for c in candidates if c.get("classification") == "known_high_confidence"),
            "Medium Confidence Matches": sum(1 for c in candidates if c.get("classification") == "known_medium_confidence"),
            "Low Confidence Matches": sum(1 for c in candidates if c.get("classification") == "known_low_confidence"),
            "Uncertain": sum(1 for c in candidates if c.get("classification") == "uncertain")
        }
        
        # Filter out zero counts
        class_counts = {k: v for k, v in class_counts.items() if v > 0}
        
        # Create pie chart
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors = ['green', 'blue', 'purple', 'orange', 'gray']
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=90)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect('equal')
        ax.set_title("Candidates by Classification")
        
        # Make text legible
        for text in texts + autotexts:
            text.set_fontsize(9)
        
        # Convert to base64 PNG
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        plots["classification"] = base64.b64encode(buf.read()).decode("utf-8")
        
        # 2. Generate sky plot of candidate positions
        if candidates and all("ra" in c and "dec" in c for c in candidates):
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Get RA/Dec for each candidate
            ra_values = [c["ra"] for c in candidates]
            dec_values = [c["dec"] for c in candidates]
            
            # Color by classification
            colors = []
            for c in candidates:
                classification = c.get("classification", "uncertain")
                if classification == "possible_new":
                    colors.append("green")
                elif classification == "known_high_confidence":
                    colors.append("blue")
                elif classification == "known_medium_confidence":
                    colors.append("purple")
                elif classification == "known_low_confidence":
                    colors.append("orange")
                else:
                    colors.append("gray")
            
            # Plot candidates
            scatter = ax.scatter(ra_values, dec_values, c=colors, alpha=0.7, s=50)
            
            # Add field boundaries if available
            if "field_ra_min" in session_info and "field_ra_max" in session_info and \
               "field_dec_min" in session_info and "field_dec_max" in session_info:
                
                ra_min = session_info["field_ra_min"]
                ra_max = session_info["field_ra_max"]
                dec_min = session_info["field_dec_min"]
                dec_max = session_info["field_dec_max"]
                
                # Draw field boundary rectangle
                rect = plt.Rectangle((ra_min, dec_min), ra_max-ra_min, dec_max-dec_min,
                                   fill=False, edgecolor='red', linestyle='--')
                ax.add_patch(rect)
            
            # Add labels
            ax.set_xlabel("RA (degrees)")
            ax.set_ylabel("Dec (degrees)")
            ax.set_title("Sky Distribution of Candidate Objects")
            ax.grid(True, alpha=0.3)
            
            # Invert RA axis (increasing right to left)
            ax.invert_xaxis()
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=8, label='Possible New'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=8, label='High Confidence Match'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                      markersize=8, label='Medium Confidence Match'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=8, label='Low Confidence Match'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=8, label='Uncertain')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Convert to base64 PNG
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            plots["sky_distribution"] = base64.b64encode(buf.read()).decode("utf-8")
        
        return plots