"""
test_orbit_tools.py - Test script for the orbit tools

Usage:
    python test_orbit_tools.py
"""

import sys
import json
import logging
import os
import numpy as np
from datetime import datetime
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
from catalog_lookup.core.orbit_tools import OrbitTools, OrbitalElements, ObservedPosition, KBODynamicalClass

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    print("Initializing Orbit Tools...")
    
    # Create orbit tools
    orbit_tools = OrbitTools(verbose=True)
    
    # Example 1: Calculate position from elements
    print("\n=== Example 1: Calculate Position from Elements ===")
    
    # Define orbital elements for a sample KBO (similar to Pluto)
    pluto_elements = OrbitalElements(
        a=39.5,           # Semi-major axis (AU)
        e=0.25,           # Eccentricity
        i=17.1,           # Inclination (degrees)
        Omega=110.3,      # Longitude of ascending node (degrees)
        omega=113.8,      # Argument of perihelion (degrees)
        M=14.9,           # Mean anomaly (degrees)
        epoch=Time('2023-01-01T00:00:00', scale='tdb')  # Epoch
    )
    
    # Calculate position for a specific date
    epoch = Time('2023-07-01T00:00:00', scale='tdb')
    ra, dec = orbit_tools.calculate_position(pluto_elements, epoch)
    
    print(f"Calculated position for epoch {epoch.iso}:")
    print(f"  RA: {ra:.6f} degrees")
    print(f"  Dec: {dec:.6f} degrees")
    
    # Example 2: Convert between elements and state vector
    print("\n=== Example 2: Elements to State Vector and Back ===")
    
    # Convert elements to state vector
    pos, vel = orbit_tools.elements_to_state_vector(pluto_elements, epoch)
    
    print("State vector at the same epoch:")
    print(f"  Position (AU): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"  Velocity (AU/day): [{vel[0]:.5f}, {vel[1]:.5f}, {vel[2]:.5f}]")
    
    # Convert back to elements
    reconstructed_elements = orbit_tools.state_vector_to_elements(pos, vel, epoch)
    
    print("\nReconstructed elements:")
    print(f"  a: {reconstructed_elements.a:.3f} AU")
    print(f"  e: {reconstructed_elements.e:.3f}")
    print(f"  i: {reconstructed_elements.i:.3f} degrees")
    print(f"  Ω: {reconstructed_elements.Omega:.3f} degrees")
    print(f"  ω: {reconstructed_elements.omega:.3f} degrees")
    print(f"  M: {reconstructed_elements.M:.3f} degrees")
    
    # Example 3: Classify orbits
    print("\n=== Example 3: Classify Orbits ===")
    
    # Define different types of orbits
    orbit_examples = {
        "Classical KBO": OrbitalElements(
            a=44.0, e=0.05, i=2.0, Omega=100.0, omega=50.0, M=30.0, 
            epoch=Time('2023-01-01')
        ),
        "Plutino (3:2 resonant)": OrbitalElements(
            a=39.4, e=0.22, i=15.0, Omega=90.0, omega=120.0, M=45.0, 
            epoch=Time('2023-01-01')
        ),
        "Scattered disk": OrbitalElements(
            a=50.0, e=0.5, i=20.0, Omega=150.0, omega=70.0, M=10.0, 
            epoch=Time('2023-01-01')
        ),
        "Detached object": OrbitalElements(
            a=80.0, e=0.4, i=35.0, Omega=200.0, omega=300.0, M=5.0, 
            epoch=Time('2023-01-01')
        ),
        "Centaur": OrbitalElements(
            a=18.0, e=0.3, i=10.0, Omega=40.0, omega=60.0, M=90.0, 
            epoch=Time('2023-01-01')
        )
    }
    
    for name, elements in orbit_examples.items():
        kbo_class = orbit_tools.classify_orbit(elements)
        print(f"{name}: {kbo_class.name}")
        
        # Print orbital characteristics
        print(f"  a: {elements.a:.1f} AU, e: {elements.e:.2f}, i: {elements.i:.1f}°")
        print(f"  Perihelion: {elements.q:.1f} AU, Aphelion: {elements.Q:.1f} AU")
        print(f"  Period: {elements.P:.1f} years")
    
    # Example 4: Fit orbit to observations
    print("\n=== Example 4: Fit Orbit to Observations ===")
    
    # Create synthetic observations for a known orbit
    true_elements = OrbitalElements(
        a=42.5, e=0.12, i=8.5, Omega=130.0, omega=75.0, M=25.0,
        epoch=Time('2023-01-01T00:00:00', scale='tdb')
    )
    
    # Generate synthetic observations
    observations = []
    for days in [0, 10, 20, 30, 40]:
        epoch = Time('2023-01-01T00:00:00') + days*u.day
        ra, dec = orbit_tools.calculate_position(true_elements, epoch)
        
        # Add small random noise (0.1 arcsec)
        ra += np.random.normal(0, 0.1/3600)
        dec += np.random.normal(0, 0.1/3600)
        
        observations.append(ObservedPosition(
            ra=ra,
            dec=dec,
            epoch=epoch,
            ra_err=0.1,
            dec_err=0.1
        ))
    
    print(f"Generated {len(observations)} synthetic observations")
    
    # Create initial guess (intentionally incorrect)
    initial_guess = OrbitalElements(
        a=45.0, e=0.15, i=10.0, Omega=120.0, omega=80.0, M=20.0,
        epoch=Time('2023-01-01T00:00:00', scale='tdb')
    )
    
    print("Initial guess:")
    print(f"  a: {initial_guess.a:.2f} AU")
    print(f"  e: {initial_guess.e:.2f}")
    print(f"  i: {initial_guess.i:.2f} degrees")
    
    # Fit orbit
    fitted_elements, rms_error = orbit_tools.fit_orbit(observations, initial_guess)
    
    print("\nFitted elements:")
    print(f"  a: {fitted_elements.a:.2f} AU")
    print(f"  e: {fitted_elements.e:.2f}")
    print(f"  i: {fitted_elements.i:.2f} degrees")
    print(f"  Ω: {fitted_elements.Omega:.2f} degrees")
    print(f"  ω: {fitted_elements.omega:.2f} degrees")
    print(f"  M: {fitted_elements.M:.2f} degrees")
    print(f"  RMS error: {rms_error:.3f} arcseconds")
    
    # Example 5: Calculate orbit similarity
    print("\n=== Example 5: Calculate Orbit Similarity ===")
    
    # Define two similar orbits
    orbit1 = OrbitalElements(
        a=42.0, e=0.10, i=5.0, Omega=100.0, omega=200.0, M=0.0,
        epoch=Time('2023-01-01T00:00:00', scale='tdb')
    )
    
    orbit2 = OrbitalElements(
        a=42.5, e=0.12, i=5.5, Omega=105.0, omega=195.0, M=10.0,
        epoch=Time('2023-01-01T00:00:00', scale='tdb')
    )
    
    similarity = orbit_tools.calculate_orbit_similarity(orbit1, orbit2)
    print(f"Similarity between similar orbits: {similarity:.3f}")
    
    # Define two very different orbits
    orbit3 = OrbitalElements(
        a=42.0, e=0.10, i=5.0, Omega=100.0, omega=200.0, M=0.0,
        epoch=Time('2023-01-01T00:00:00', scale='tdb')
    )
    
    orbit4 = OrbitalElements(
        a=25.0, e=0.50, i=25.0, Omega=200.0, omega=100.0, M=180.0,
        epoch=Time('2023-01-01T00:00:00', scale='tdb')
    )
    
    similarity = orbit_tools.calculate_orbit_similarity(orbit3, orbit4)
    print(f"Similarity between different orbits: {similarity:.3f}")
    
    # Example 6: Calculate encounter parameters
    print("\n=== Example 6: Calculate Encounter Parameters ===")
    
    # Define an orbit that crosses Neptune's orbit
    crossing_orbit = OrbitalElements(
        a=35.0, e=0.2, i=10.0, Omega=150.0, omega=120.0, M=30.0,
        epoch=Time('2023-01-01T00:00:00', scale='tdb')
    )
    
    neptune_params = orbit_tools.calculate_encounter_parameters(crossing_orbit, 'neptune')
    
    print("Neptune encounter parameters:")
    print(f"  Orbits cross: {neptune_params['orbits_cross']}")
    print(f"  Minimum orbit intersection distance: {neptune_params['moid']:.3f} AU")
    print(f"  Resonance: {neptune_params['best_resonance']}")
    print(f"  Perihelion: {neptune_params['perihelion']:.2f} AU")
    print(f"  Aphelion: {neptune_params['aphelion']:.2f} AU")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())