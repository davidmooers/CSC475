# Vibration-Based Pickleball Impact Detection and Localization
This project will investigate whether surface vibrations measured using piezoelectric sensors can be used to detect 
pickleball impacts, distinguish them from other vibration sources, and estimate where an impact occurred relative to a court line.

The work is divided into three phases:
1) Vibration Detection and Source Classification
2) Multi-Sensor Impact Localization
3) Simple Court Line Demo

The project will begin with controlled experiments on a small test surface before being tested on a real pickleball court.

### Research Questions
1) Can a piezoelectric sensor reliably detect surface vibrations caused by a pickleball bounce?
2) Is vibration signal data precise enough to distinguish pickleball bounces from footsteps and/or noise in the data?
3) Can multiple sensors be used to approximate an impact location, and with what precision and accuracy?
4) How well does the combined detection, classification, and localization system perform on a real pickleball court?

## System Design
**TODO:** Circuit diagram

**TODO:** Data flow diagram

## Phase 1 - Vibration Detection and Source Classification
### Goal
Detect individual vibration events and classify each event as either:
- Pickleball Bounce
- Non-pickleball Bounce

Non-pickleball events include footsteps, paddle strikes, environmental vibrations, etc.

### Setup
- Use one piezoelectric disc and an Arduino circuit to amplify and sample the sensor data
- Stream the data from the Arduino to my laptop running a Python program
- Python program processes and records the raw ADC samples
- Attach the piezo sensor to a quiet, court-like surface and fix it in place with electrical tape
- Mark a few reference distance points away from the sensor: ~10cm, ~30cm, ~50cm, etc.
- Record and label simulated events (ex. drop balls from various positions, footsteps, etc.)
- Use recorded data to classify and predict new input

### Evaluation
Phase 1 will measure:
- Detection success at different distances
- Signal strength relative to background noise
- Classification accuracy between pickleball and non-pickleball events

## Phase 2 - Multi-Sensor Impact Localization
### Goal
Use multiple piezoelectric sensors to estimate where an impact occured.

### Setup
- Place at least 3 sensors around a small area
- Produce impacts at known locations
- Record the signal from each sensor
- Compare differences in signal strength and timing between sensors
- Use the measurements to estimate the impact position

### Evaluation
Compare the estimated impact position with the known position, and measure the average localization error.

## Phase 3 - Simple Court Line Demo
### Goal
Combine detection, classification, and localization on a small section of a real pickleball court line.

### Setup
- Place multiple sensors around a short section of a court line
- Drop pickleballs at known locations inside, outside, and near the line
- Detect whether the event was a pickleball bounce
- Estimate the impact location
- Use the estimated position to determine an IN or OUT decision

### Evaluation
Measure how often the system correctly identifies:
- Pickleball impacts
- Impact locations
- IN/OUT calls

These results will then determine whether this kind of system is feasible on a larger, full court scale.
