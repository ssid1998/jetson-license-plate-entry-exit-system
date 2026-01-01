# Application Use Case

In multiple airports back in India, there is a problem of people overcrowding the departure terminal while they come to drop their close ones. This often creates a traffic jam at the departure terminal, causing inconvenience to other travellers coming to the airport to catch a flight.

For this, two booths have been installed by the airport authority at the entry and exit gates of the departure terminal, where the following tasks are performed to ensure the vehicles entering the departure limit their stay to less than **10 minutes**.

---

## Existing Manual System

### At Entry Gate
- An official at the booth generates a slip which contains a timestamp of when the car is entering the departure terminal.
- After the concerned vehicle finishes dropping the passenger at the departure gate, it proceeds towards the exit gate.

### At Exit Gate
- A different official collects the slip with the timestamp.
- The official manually checks whether the current time exceeds **10 minutes** from the entry timestamp.
  - If **yes**, the vehicle owner is asked to pay a fixed fee.
  - If **no**, the vehicle is allowed to exit without any fee.

---

## Proposed Automated System (Using License Plate Detection)

We, with our project of detecting car licence plates using **Jetson Nano**, aim to execute this entire system by replacing the two officials with a set of cameras and displays installed at the entry and exit points.

---

## System After Automating the Process

### At Entry Gate
- A camera and display are installed along with a barrier.
- The camera detects the vehicle number plate.
- The unique vehicle number is recorded along with the **entry timestamp** in a database.

**Example Record:**

| Vehicle Number | Entry Timestamp | Exit Timestamp |
|---------------|-----------------|----------------|
| EMD AA 432    | 16:58           | —              |

---

### At Exit Gate
- A separate set of cameras, display, and a card/cash payment machine are installed at the exit booth along with a barrier.
- When a vehicle reaches the exit gate:
  - The vehicle number is detected.
  - The database is checked for the corresponding entry timestamp.
- If the current time is **within 10 minutes** of the entry timestamp:
  - The vehicle is allowed to exit without any fee.
- If the current time **exceeds 10 minutes**:
  - The driver is prompted to pay the fee using the cash/card machine.
  - After payment, the vehicle is allowed to exit.
- In both cases, the **exit timestamp** is recorded.

**Example Record After Exit:**

| Vehicle Number | Entry Timestamp | Exit Timestamp |
|---------------|-----------------|----------------|
| EMD AA 432    | 16:58           | 17:06          |

---