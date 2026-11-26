# annotations/

Corrected labels and the decisions behind them. The original set in `test_data/traffic_v5_mini/`
is never modified.

| File | What it is |
|---|---|
| `final/<stem>.txt` | Final YOLO labels for all 40 images. An empty file means "checked, no target objects". |
| `source_groups.csv` | `image, group, pin_split, note`: images that must stay in one split (same video, camera or incident). `pin_split=train` keeps COCO train2017 images out of val/test. |
| `class_fixes.csv` | Class corrections confirmed by looking at the image, applied before the CVAT import. |
| `changes.csv` | Every change from the original labels to `final/`, with the reason. |

## Class rules

Class id = line in `test_data/traffic_v5_mini/classes.txt`, 0-based. When an object fits several
classes, the first matching rule wins. The rules follow how the original labels already use the
classes (checked on crops of every special class), so corrections stay consistent with them.

1. **Emergency vehicle with service livery** → its service class, whatever the base vehicle:
   - `police`: police markings (ДПС livery, light bar). A police officer on foot is `pedestrian`.
   - `ambulance`: ambulance markings (red stripe/cross, «Скорая»), usually a van.
   - `fire_truck`: fire service vehicle (red, ladders/equipment).
   - No visible livery → classify by body type below.
2. **`lorry`**: articulated truck, tractor unit with a semi-trailer, labeled as **one** box.
   The semi-trailer never gets a separate `trailer` box.
3. **`truck`**: rigid truck on one chassis: dump truck, flatbed, box truck, light flatbed (Gazelle-type).
4. **`bus`**: vehicle built for passengers beyond a car: full-size bus, route minibus, passenger van
   (Sprinter, Vito-type).
5. **`trailer`**: a trailer on its own axles (behind a car or a rigid truck, or standing alone), labeled
   **separately** from the towing vehicle.
6. **`car`**: passenger car, SUV, pickup.
7. **`motorcycle`**, **`bicycle`**: the vehicle; its rider is `pedestrian`.
8. **`pedestrian`**: any person on foot or riding (officers, workers, cyclists' riders).

Box rules:
- Cover the visible part only; never extend a box past the image border for a cut-off object.
- Partly hidden object: box the visible part if the class is still recognisable.
- Too small or too blurred to tell the class: leave it out and note the image in `changes.csv`.
- Motion blur: box the vehicle's silhouette if it is recognisable, not the trail behind it.
- Dense crowd: every person who can be told apart from the others gets a box. People merged into one
  blob are not separate target objects and are not labeled (YOLO has no "ignore" regions, so this
  residual is a known limitation; it applies to `Снимок экрана от 2023-06-05 16-05-35`).
