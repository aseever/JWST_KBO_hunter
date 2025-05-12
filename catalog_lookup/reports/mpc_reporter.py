"""
mpc_reporter.py - Formats KBO detections for MPC submission

This module creates reports formatted according to Minor Planet Center (MPC)
submission guidelines, facilitating the reporting of potential new KBO discoveries
and generating MPC-compliant observation files.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord

# Set up logging
logger = logging.getLogger(__name__)

class MPCReporter:
    """
    Formats KBO detections for submission to the Minor Planet Center (MPC).
    
    This class provides methods to generate MPC-compliant observation files
    for reporting KBO detections and potential discoveries.
    """
    
    def __init__(self, output_dir: Optional[str] = None, observer_code: str = "XXX"):
        """
        Initialize the MPC reporter.
        
        Args:
            output_dir: Directory to save MPC format files to.
            observer_code: MPC observatory code (default: 'XXX' for testing).
        """
        self.output_dir = output_dir
        self.observer_code = observer_code
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    def generate_observation_report(self, 
                                  detections: List[Dict[str, Any]],
                                  provisional_designation: Optional[str] = None,
                                  observer_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an MPC-compliant observation report.
        
        Args:
            detections: List of detection dictionaries.
            provisional_designation: Optional provisional designation for the object.
            observer_info: Optional dictionary with observer information.
            
        Returns:
            MPC-formatted observation report or path to saved file.
        """
        # Generate provisional designation if not provided
        if provisional_designation is None:
            # Format: YYYY XXNN where YYYY is the year, and NN is a counter
            year = datetime.now().year
            provisional_designation = f"{year} AA"  # Default - should be incremented in practice
        
        # Validate and set observer information
        observer_code = self.observer_code
        if observer_info and "mpc_code" in observer_info:
            observer_code = observer_info["mpc_code"]
        
        # Generate the report header
        header = f"COD {observer_code}\n"
        
        if observer_info:
            if "name" in observer_info:
                header += f"OBS {observer_info['name']}\n"
            if "telescope" in observer_info:
                header += f"TEL {observer_info['telescope']}\n"
            if "contact" in observer_info:
                header += f"CON {observer_info['contact']}\n"
        
        header += f"MEA {os.getlogin()}\n"  # Measurer - use current username as default
        header += "NET ICRF\n"  # Reference frame (Gaia-based)
        
        # Begin observation section
        lines = [header]
        
        # Process each detection
        for detection in detections:
            # Format the position in MPC format
            mpc_line = self._format_mpc_line(detection, provisional_designation, observer_code)
            if mpc_line:
                lines.append(mpc_line)
        
        # Add end of file marker
        lines.append("END")
        
        # Join all lines with newlines
        report = "\n".join(lines)
        
        # Save to file if output directory specified
        if self.output_dir:
            # Create a unique filename
            clean_desig = provisional_designation.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mpc_report_{clean_desig}_{timestamp}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(report)
            
            logger.info(f"Saved MPC observation report to {filepath}")
            return filepath
        
        # Otherwise return the report string
        return report
    
    def generate_discovery_report(self, 
                               discovery_info: Dict[str, Any],
                               detections: List[Dict[str, Any]]) -> str:
        """
        Generate an MPC-compliant discovery report.
        
        Args:
            discovery_info: Dictionary with discovery information.
            detections: List of detection dictionaries.
            
        Returns:
            MPC-formatted discovery report or path to saved file.
        """
        # MPC discovery format includes both observations and discovery details
        
        # First, create the observation report
        provisional_designation = discovery_info.get("provisional_designation")
        observer_info = discovery_info.get("observer_info", {})
        
        obs_report = self.generate_observation_report(
            detections, provisional_designation, observer_info)
        
        # Then add discovery-specific information
        discovery_lines = []
        
        # Include discovery header
        discovery_lines.append("--- Discovery Details ---")
        discovery_lines.append(f"Discovery Date: {discovery_info.get('discovery_date', 'Unknown')}")
        
        # Include discovery circumstances
        if "description" in discovery_info:
            discovery_lines.append(f"Circumstances: {discovery_info['description']}")
            
        # Include orbital elements if available
        if "orbital_elements" in discovery_info:
            elements = discovery_info["orbital_elements"]
            discovery_lines.append("Preliminary Orbital Elements:")
            discovery_lines.append(f"  a = {elements.get('a', 'Unknown')} AU")
            discovery_lines.append(f"  e = {elements.get('e', 'Unknown')}")
            discovery_lines.append(f"  i = {elements.get('i', 'Unknown')} deg")
            if "Omega" in elements:
                discovery_lines.append(f"  Ω = {elements['Omega']} deg")
            if "omega" in elements:
                discovery_lines.append(f"  ω = {elements['omega']} deg")
        
        # Include observer notes
        if "notes" in discovery_info:
            discovery_lines.append(f"Notes: {discovery_info['notes']}")
            
        # Combine observation report with discovery details
        if isinstance(obs_report, str) and not os.path.exists(obs_report):
            # If it's a report string
            full_report = obs_report + "\n\n" + "\n".join(discovery_lines)
            
            # Save to file if output directory specified
            if self.output_dir:
                # Create a unique filename
                clean_desig = provisional_designation.replace(" ", "_") if provisional_designation else "new"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mpc_discovery_{clean_desig}_{timestamp}.txt"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, "w") as f:
                    f.write(full_report)
                
                logger.info(f"Saved MPC discovery report to {filepath}")
                return filepath
            
            # Otherwise return the report string
            return full_report
        else:
            # If it's a file path, append to the file
            with open(obs_report, "a") as f:
                f.write("\n\n" + "\n".join(discovery_lines))
            
            logger.info(f"Appended discovery details to {obs_report}")
            return obs_report
    
    def _format_mpc_line(self, 
                       detection: Dict[str, Any], 
                       designation: str, 
                       observer_code: str) -> Optional[str]:
        """
        Format a single detection in MPC 80-column format.
        
        Args:
            detection: Detection dictionary.
            designation: Provisional designation.
            observer_code: MPC observatory code.
            
        Returns:
            MPC-formatted observation line, or None if formatting fails.
        """
        try:
            # Check required fields
            if not all(key in detection for key in ["ra", "dec", "epoch"]):
                logger.warning("Detection missing required fields (ra, dec, epoch)")
                return None
            
            # Convert epoch to Time object if needed
            if isinstance(detection["epoch"], str):
                epoch = Time(detection["epoch"])
            elif isinstance(detection["epoch"], Time):
                epoch = detection["epoch"]
            else:
                logger.warning("Invalid epoch format")
                return None
            
            # Format the date in MPC format: YYYY MM DD.ddddd
            date_str = epoch.strftime("%Y %m %d.%f")[:-3]  # Get first 5 decimal places
            
            # Format the designation (11 characters, left-justified)
            if len(designation) > 11:
                designation = designation[:11]
            desig_formatted = f"{designation:<11}"
            
            # Convert RA from degrees to hours/min/sec
            ra_deg = detection["ra"]
            ra_coord = SkyCoord(ra=ra_deg*u.deg, dec=0*u.deg, frame='icrs')
            ra_hms = ra_coord.ra.hms
            ra_h = int(ra_hms.h)
            ra_m = int(ra_hms.m)
            ra_s = ra_hms.s
            
            # Format RA as HH MM SS.ss (2 decimal places for seconds)
            ra_formatted = f"{ra_h:02d} {ra_m:02d} {ra_s:05.2f}"
            
            # Convert Dec from degrees to deg/min/sec
            dec_deg = detection["dec"]
            dec_coord = SkyCoord(ra=0*u.deg, dec=dec_deg*u.deg, frame='icrs')
            dec_dms = dec_coord.dec.signed_dms
            dec_sign = '+' if dec_dms.sign >= 0 else '-'
            dec_d = abs(int(dec_dms.d))
            dec_m = int(dec_dms.m)
            dec_s = dec_dms.s
            
            # Format Dec as sDD MM SS.s (1 decimal place for seconds)
            dec_formatted = f"{dec_sign}{dec_d:02d} {dec_m:02d} {dec_s:04.1f}"
            
            # Magnitude formatting (empty if not available)
            mag_formatted = "       "
            if "mag" in detection:
                filter_code = detection.get("filter_code", "V")  # Default to V
                mag = detection["mag"]
                mag_formatted = f"{mag:4.1f} {filter_code:3s}"
            
            # Construct the MPC format line
            # Columns 1-12: designation
            # Columns 13-14: discovery asterisk (empty for now)
            # Columns 16-32: date
            # Columns 33-44: RA
            # Columns 45-56: Dec
            # Columns 57-65: magnitude and band
            # Columns 66-71: observatory code (right-justified)
            mpc_line = (
                f"{desig_formatted}  {date_str}  {ra_formatted}  {dec_formatted}  "
                f"{mag_formatted}     {observer_code:>3s}"
            )
            
            return mpc_line
            
        except Exception as e:
            logger.error(f"Error formatting MPC line: {e}")
            return None