# NavPulse indoor walking prototype (2.1.0)

## Install and demonstrate on the OnePlus Nord CE 3

Use the NEW `outputs/apk/navpulse-indoor-walk.apk`, not the previous vehicle APK.
It has the same app ID/signing key and a higher version code, so install it as an update.

1. Turn on airplane mode. Turn Wi-Fi and Bluetooth off as well if they remain enabled. Phone Location can be off.
2. Open NavPulse Localizer. The status must say INDOOR WALK / READY, not SEARCHING.
3. Hold the phone screen-up at waist height, with its top edge pointing forward. Face the direction you will first walk.
4. Tap green Play. Hold still for two seconds while the step detector warms up.
5. Walk 8-10 normal steps forward. Step count, estimated distance and the amber trail update from actual accelerometer readings.
6. Turn your body AND phone together through 90 degrees, then walk another 8-10 steps. The trail should turn.
7. Stop walking. Speed drops to zero in about 1.4 seconds. Position only advances on detected steps.
8. Tap Stop, then Sessions to inspect the recorded session. Play again resets the relative origin and heading.

The map's START point is your physical starting point, represented as local (0,0). No GNSS fix, network or location permission is requested in indoor mode. No prerecorded animation or vehicle ML estimate drives this path.

## If the step count stays at zero

Open Pipeline and check the accelerometer rate and step signal. Around 50 Hz is requested, though the actual phone rate can differ. A zero rate or an error indicates the sensor stream is unavailable. If readings are live but gentle steps are missed, Stop, open Navigate, lower the step threshold from 0.6 to 0.4 and restart. Avoid deliberately shaking the phone: repeated hand oscillations can cause false steps. Keep the phone facing the walking direction, not turning it independently to show someone the screen.

Adjust step length while stopped: the default is 0.65 m. Measure a known walking distance and divide by the detected step count to choose a better length for the presenter.

## What to tell the jury

This is pedestrian dead reckoning: step detection + relative heading + calibrated step length. Airplane mode does not supply location; local motion sensors supply the measurements. It estimates a relative path, not an absolute indoor latitude/longitude or a floor plan. The displayed drift allowance is a heuristic, not measured accuracy. Indoor magnetic disturbances are avoided when Android's game rotation sensor is available; a rotation-vector or gyro fallback is used otherwise. Carry orientation changes, missed steps and heading drift affect the estimate.

The original vehicle pipeline is still available by turning off Indoor walking while stopped. Its data/model accuracy measurements do not validate the new pedestrian mode. Sessions are kept in memory and are lost when the app process closes. Keep the app foregrounded for this demo.

## Build and recovery

Flutter 3.24.5, Java 17, Android SDK 34 are installed under `C:/Users/souri/navpulse-build-tools`.
Build copy outside OneDrive: `C:/Users/souri/navpulse-build-tools/indoor-build`.
Canonical source remains in this repository's `flutter_app`; sync changes to the build copy before rebuilding.

```powershell
$env:JAVA_HOME = 'C:/Users/souri/navpulse-build-tools/jdk-17.0.20.1+1'
$env:ANDROID_HOME = 'C:/Users/souri/navpulse-build-tools/android-sdk'
Set-Location C:/Users/souri/navpulse-build-tools/indoor-build
& C:/Users/souri/navpulse-build-tools/flutter/bin/flutter.bat test
& C:/Users/souri/navpulse-build-tools/flutter/bin/flutter.bat build apk --release
```

The APK is signed with the local debug key for sideloaded demonstration. Phone sensor performance must still be checked on the physical device.
