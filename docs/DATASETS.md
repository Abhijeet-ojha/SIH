# Candidate Datasets

Ranked by what they fix, not by size. The current training set (IO-VNBD) has the phone
rigidly fixed in the vehicle frame, so it contains **no examples of vibration without
motion** and **no examples of arbitrary phone orientation**. Those two gaps are the cause
of the shake bug and of the negative held-out R². Pick datasets that close them.

## Tier 1 — closes the shake gap directly

### SHL (University of Sussex–Huawei Locomotion)

The single highest-value addition. 750 h of labelled data across 8 transport modes
(Car 88 h, Bus 107 h, Train 115 h, Subway 89 h, Walk 127 h, Run 21 h, Bike 79 h,
Still 127 h), recorded on **four Huawei Mate 9 phones simultaneously** at four body
positions — hand, torso, hip pocket, backpack. Full IMU plus magnetometer, barometer,
GPS, gravity and linear-acceleration channels.

Why it matters here: the *hand* position while *Still* or *Walk* is exactly the negative
class the model has never seen — vibration and handling energy with zero vehicle motion.
The four simultaneous positions give free orientation augmentation on identical ground
truth. Train the Phase 3 motion gate on this and the shake bug is addressed with data
rather than with a threshold.

- <https://www.sussex.ac.uk/strc/research/wearable/locomotion-transportation>
- <https://ieee-dataport.org/documents/sussex-huawei-locomotion-and-transportation-dataset>
- Full set is ~950 GB. Take the Car / Still / Walk subsets only.

### Your own Android logger

Once Phase 5 registers the extra sensors, an afternoon of collection gives you what no
public set has: **Indian roads, Indian traffic, arbitrary phone placement**. Record
deliberate adversarial cases — phone in hand while stationary at a light, phone in pocket,
phone on the seat, passenger handing it around, a genuine tunnel or basement ramp. Fifty
minutes of this is worth more for the demo than another 100 GB of UK motorway.

## Tier 2 — vehicle IMU with real ground truth

### comma2k19

33 h of California highway driving, 2019 one-minute segments, collected on comma EON
devices whose sensor suite is deliberately phone-grade: 9-axis IMU, phone GPS, plus **raw
GNSS** from both Qualcomm and u-blox receivers, and **CAN bus vehicle speed, wheel speeds
and steering angle**. CAN speed is far better ground truth than GPS speed — it does not
drop out in the tunnels you are trying to survive.

- <https://github.com/commaai/comma2k19>
- <https://huggingface.co/datasets/commaai/comma2k19> · <https://arxiv.org/abs/1812.05752>
- ~100 GB in 10 GB chunks; a few chunks is plenty.

### UAH-DriveSet

Smartphone-collected, six drivers, six vehicles, three behaviours (normal / aggressive /
drowsy) on motorway and secondary roads, 500 minutes. Accelerometer at 10 Hz, GPS at 1 Hz
— the same rates you already handle, so the loader work is near-zero. Its value is
**driver-style and vehicle diversity**: six different cars is the direct test of the
generalisation failure your LODRO CV already exposes.

- <http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/>
- Free for non-commercial academic use.

## Tier 3 — filter validation only

**KITTI** (OXTS RT3003 IMU/GNSS, ~10 cm ground truth, 39.2 km) and **Oxford RobotCar**
(NovAtel SPAN, 50 Hz, 1010 km) have ground truth good enough to validate the EKF itself
in isolation. Do not train the speed model on them — tactical-grade IMU noise is nothing
like a phone's, and a model trained there will not transfer.

**OxIOD** and **RoNIN** are pedestrian, phone-in-arbitrary-placement, with motion-capture
ground truth. Not vehicle data, but RoNIN's body-heading network is the reference
implementation of the Phase 2 problem — how to get a motion direction out of a phone whose
orientation you do not control. Read the method, borrow the idea, do not train on the data.

## Note on Indian data

There is no public Indian smartphone-IMU driving dataset with usable ground truth. Do not
spend days looking; the honest position for the writeup is "we validated on public UK/US
data and collected our own Indian pilot data with the logger app," which is both true and
more defensible than a vague claim of Indian coverage. IDD (India Driving Dataset) is
vision-only and will not help.

## Suggested split

| Purpose | Data |
|---|---|
| Speed regressor training | IO-VNBD (fixed) + UAH-DriveSet + comma2k19 CAN speed |
| Motion gate (shake rejection) | SHL Car/Still/Walk across all four body positions |
| Orientation robustness | SHL four-position pairs + synthetic rotation augmentation |
| Held-out generalisation | one full driver and one full dataset never trained on |
| Demo / Indian relevance | your own logger recordings |

The held-out row matters most. The current LODRO result is the only honest number in the
repo about generalisation, and it says the model does not generalise. Keep that test, keep
reporting it, and let the new data move it.
