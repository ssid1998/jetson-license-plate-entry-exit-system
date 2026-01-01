# Project Use Case: Automated Vehicle Dwell Time Management at Airport Terminals

## The Problem: Departure Terminal Congestion

At many airports in India, departure terminals face significant traffic congestion. This issue is frequently caused by private vehicles overstaying the allotted drop-off time, leading to traffic jams and major inconveniences for passengers rushing to catch their flights.

---

## The Current Manual System

To manage this situation, airport authorities have implemented a manual system that relies on officials stationed at entry and exit booths.

- **At the Entry Gate**  
  An official manually generates a paper slip with a timestamp and hands it to the driver upon entering the departure area.

- **At the Exit Gate**  
  Another official collects the slip, manually calculates the time the vehicle has spent in the terminal, and checks whether it has exceeded the **10-minute limit**.  
  - If the limit is exceeded, a fee is collected before allowing the vehicle to exit.  
  - If the limit is not exceeded, the vehicle is allowed to leave without any charge.

This manual process is inefficient, prone to human error, and can create additional queues, ultimately defeating its primary purpose of ensuring smooth traffic flow.

---

## The Proposed Automated Solution

Our project replaces this outdated manual process with a modern, efficient system using **Automatic Number Plate Recognition (ANPR)** technology powered by a **Jetson Nano**. The result is a seamless, fully automated workflow for managing vehicle entry, exit, and fee collection.

---

## Automated System Workflow

### At the Entry Gate

A camera, display screen, and automated barrier are installed.

1. As a vehicle approaches, the ANPR camera captures and reads its license plate.
2. The system records the unique vehicle number along with the precise entry timestamp in a central database.
3. The barrier opens automatically, allowing the vehicle to proceed.

---

### At the Exit Gate

A similar setup is installed, consisting of a camera, display, automated barrier, and an integrated payment terminal (supporting card, cash, and digital payments).

1. The ANPR camera detects the vehicle’s license plate at the exit.
2. The system queries the database to retrieve the corresponding entry timestamp and calculates the total dwell time.
3. The system follows one of the two scenarios:
   - **Within the 10-Minute Limit**  
     If the dwell time is within the allowed limit, a confirmation message is displayed and the exit barrier opens immediately.
   - **Exceeded Time Limit**  
     If the vehicle has overstayed, the display shows the applicable fee and prompts the driver to complete payment using the terminal. Once payment is confirmed, the barrier opens.
4. In both cases, the vehicle’s exit timestamp is recorded in the database, enabling further analysis of airport traffic patterns.

---