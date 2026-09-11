# NavPulse — demo script (round 2)

Install `outputs/apk/navpulse.apk`. Same app ID and signing key as before, higher version
code, so it installs as an update.

Build it yourself with `.\tool\build.ps1 all` (analyze + test + apk). That script exists
because OneDrive holds file handles on `build\` and Flutter cannot clean in place — it
mirrors the source to a build copy outside OneDrive and runs there. Building the repo
directory directly fails with *"Flutter failed to delete a directory at
build\unit_test_assets"*.

---

## The 90-second demo

The app opens on **INDOOR WALK**. The mode selector is the first thing on screen; both
options are always visible and say what they mean.

### Beat 1 — it works with the radios off (30 s)

1. Airplane mode on. Location can be off entirely.
2. Open NavPulse. Status reads **INDOOR WALK / READY**.
3. Hold the phone flat, top edge forward, facing the way you will walk.
4. Press the green **START** pill. Hold still two seconds.
5. Walk 8–10 normal steps. Steps, distance and the trail update from the accelerometer.
6. Turn your body **and phone together** 90°, walk 8–10 more. The trail turns.

Point at the legend: the trail is drawn from sensors alone. No fix, no network, no
permission was requested.

### Beat 2 — the tilt test (20 s) ← *lead with this if a judge is sceptical*

Hand the judge the phone while it is running and ask them to **slant or twist it**.

The heading does not move, and a banner appears: **HEADING HELD — phone moved, not the
vehicle**, with a live trust percentage.

What to say: *"A gyroscope cannot tell whether the phone turned or the car turned — it sees
one rotation either way. We separate them physically. A car turning on a road rotates about
the vertical and essentially nothing else; a hand rotates the phone about a horizontal axis.
We measured it: a real turn shows 0.00 rad/s of off-vertical rate, a hand slant shows 0.44.
So we believe yaw in proportion to how still the mounting is."*

The numbers behind it, all in `test/tilt_heading_test.dart`:

| Case | Before | After |
| :--- | ---: | ---: |
| Realistic hand slant | 7.9° of false turn | < 3° |
| Re-seating the phone in a cradle | 31.6° | < 5° |
| Genuine 90°/6 s turn | tracked exactly | still tracked |

### Beat 3 — cut the satellites (40 s, vehicle mode)

1. Stop. Switch the mode selector to **VEHICLE**. Start. Wait for **GNSS LOCKED**.
2. Drive. The trail is blue while satellites are correcting the filter.
3. Press **CUT GPS SIGNAL**. The banner turns amber, the trail continues in amber, and the
   uncertainty figure starts growing on its own.
4. Press **RESTORE GPS**. The gap between where the system thought it was and the first
   real fix is the honest error, live, in front of them.

---

## If a judge pushes on accuracy

Do not oversell. The measured result on real IO-VNBD drives is in the repo:

- **51% median blackout drift** (exit error ÷ distance travelled with GPS off) against a
  10% target, on the 6 drives where the phone genuinely tracked the vehicle.
- **0 of 6 meet the target.** Fused beats the naive baseline on 5 of 6 (126.3% → 51.3%).
- The speed model beats a constant on every held-out driver, but R² is still negative on
  two of three.

The stronger card is the data work, which is defensible and ours:

> *"The published IO-VNBD files are labelled synchronised and are not — the phone and
> vehicle streams are offset by up to 39 seconds, different per drive. And the gyroscope
> axis labels are wrong: the channel labelled 'Yaw' has 0.000 correlation with the vehicle
> turning, while the one labelled 'Pitch' has 0.97. We found both by cross-checking one
> sensor against another, corrected them, and our heading error went from unusable to
> 1–7° over 90 seconds."*

That is a result about the dataset everyone in this problem statement is using.

---

## Failure modes, and what to do

| Symptom | Cause | Action |
| :--- | :--- | :--- |
| Step count stays 0 | step threshold too high for a gentle gait | Stop → Navigate → lower threshold to 0.4 |
| Status stuck on SEARCHING | vehicle mode indoors | switch to INDOOR WALK |
| CUT GPS pill missing | you are in indoor mode | it is hidden by design — there is no GPS to cut |
| Distance obviously wrong | step length not yours | Stop → Navigate → measure a known distance, divide by steps |
| Heading frozen | the mount guard is active | that is the feature — hold the phone still and it returns in 0.6 s |

Do not shake the phone to show it "working" — repeated oscillation can register as steps.
Keep the app foregrounded; sessions live in memory and are lost when the process dies.

---

## What this is, said accurately

Indoor mode is pedestrian dead reckoning: step detection, relative heading, and a
user-calibrated step length. It estimates a **relative path**, not an absolute indoor
position and not a floor plan. The drift figure shown is a heuristic allowance, not a
measured accuracy.

Vehicle mode is the trained pipeline: frame alignment → motion gate → on-device
gradient-boosted speed model (43 KB, trained on real IO-VNBD with CAN-bus speed labels) →
6-state Joseph-form EKF. The vehicle accuracy numbers above do not validate the pedestrian
mode, and vice versa.

---

## Build environment

Flutter 3.24.5, Java 17, Android SDK 34 under `C:/Users/souri/navpulse-build-tools`.
Note `Color.withValues()` does **not** exist in 3.24.5 (3.27+ only) — use `withOpacity`.

```powershell
cd flutter_app
.\tool\build.ps1 all        # analyze, test, apk -> outputs/apk/navpulse.apk
.\tool\build.ps1 test       # tests only
```
