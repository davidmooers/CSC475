# Vibration-Based Pickleball Impact Detection and Localization
This project will attempt to use piezoelectric sensors to detect and analyze surface vibrations caused
by pickleball impacts to estimate whether a ball landed inside or outside a court line.

The work is divided into three phases:
1) Vibration Detection and Source Classification
2) Multi-Sensor Impact Localization
3) Simple Court Line Demo

## Research Questions
1) Can a piezoelectric sensor reliably detect surface vibrations caused by a pickleball bounce?
2) Is vibration signal data precise enough to distinguish pickleball bounces from footsteps and/or noise in the data?
3) Can multiple sensors be used to approximate an impact location, and with what precision and accuracy?
4) How well does the combined system work on a real pickleball court?

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
- Use one piezoelectric disc attached consistently to a test surface
- Use an Arduino circuit to capture and amplify the sensor data
- Stream the amplified data from the Arduino to my laptop running a Python program
- Python program processes and records the raw ADC samples
- Attach the piezo sensor to a quiet, court-like surface and fix it in place with electrical tape
- Mark a few reference distance points away from the sensor: ~10cm, ~30cm, ~50cm, etc.
- Record and label simulated events (ex. drop balls from various positions, footsteps, etc.)
- Use recorded data to classify and predict new input

