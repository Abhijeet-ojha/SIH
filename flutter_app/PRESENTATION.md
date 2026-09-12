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

### Beat 4 — the part that is worth money (30 s)

This is the women's-safety case, and it is what dead reckoning is actually *for*: knowing
where somebody is when the network does not.

1. Tap **Safety** in the sheet, add a phone number, put a name in.
2. Press the red **SOS** button on the map.
3. The exact SMS is shown before anything is sent. Read it out:

   ```
   SOS for Asha at 21:14
   12.971604,77.594612
   within 31m (approximate)
   last GPS 2min ago, 150m by sensors since
   https://maps.google.com/?q=12.971604,77.594612
   ```

4. Press SEND. It goes by **SMS**, so it needs no data connection — the one channel that
   still works in a basement, a tunnel, or on one bar.

Three things to point at, because each is a decision a competitor will not have made:

- **It sends a circle, not a dot.** GNSS accuracy at the last fix and the drift since,
  combined in quadrature. A responder told "within 31 m" searches correctly; one told a
  bare coordinate searches the wrong building with total confidence.
- **It states how stale the fix is,** rounded *up*. A 2-minute-old position during a walk
  is a different instruction from a live one.
- **With no fix it refuses to invent one.** The message says "position unknown" and gives
  the distance walked instead. Turn location off and press SOS to show this — the coordinate
  never appears. Anything else would send help somewhere wrong.

If there is no signal the message is **queued**, not dropped, and goes out automatically
when signal returns.

### Beat 5 — offline navigation (20 s, only if tiles were fetched)

Long-press anywhere on the map to drop a destination. The card shows distance and which way
it lies relative to the way you are facing, and it keeps working with the radios off.

Say what it is: *"straight-line bearing and distance, not turn-by-turn. There is no road
graph on the device, so a route down streets would be a drawing, not a route."*

The basemap is bundled into the APK, not fetched at run time:

```powershell
python tool/fetch_tiles.py --lat <demo lat> --lon <demo lon> --radius-km 1.2 --name "Venue"
.	ooluild.ps1 apk
```

With no tiles fetched the map falls back to a metre grid and everything else still works —
so this beat is optional, and skipping it breaks nothing.

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

| SOS says NO FIX | no GPS fix yet this session | that is correct behaviour — it will not invent a coordinate |
| SOS opens the messaging app instead of sending | SEND_SMS not granted | press send there; grant the permission for one-tap next time |
| Map shows a grid, not streets | no tiles bundled for this area | run `tool/fetch_tiles.py` and rebuild, or skip Beat 5 |

Shaking the phone is now a *feature demo*, not a hazard: hand it to a judge and let them
shake it. Steps stay at zero, because a step has to point along gravity, keep the phone
steady, and hold a rhythm for ~2.5 s before any distance is credited
(`test/shake_rejection_test.dart`). Keep the app foregrounded; sessions live in memory and
are lost when the process dies.

---

## What this is, said accurately

Indoor mode is pedestrian dead reckoning: step detection, relative heading, and a
user-calibrated step length. It estimates a **relative path**, not an absolute indoor
position and not a floor plan. The drift figure shown is a heuristic allowance, not a
measured accuracy.

The safety layer turns that relative path into a shareable latitude and longitude by
anchoring it to the last real GNSS fix. The accuracy circle it quotes is honest about its
inputs: real GNSS accuracy at the anchor, combined with a drift allowance of 20% of the
distance walked since. **That 20% is an allowance, not a measurement** — published
step-length methods land at 5–15%, and it is set deliberately above them because an
over-large circle costs a searcher time while an over-small one sends them to the wrong
place. Do not quote it as a measured accuracy.

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
