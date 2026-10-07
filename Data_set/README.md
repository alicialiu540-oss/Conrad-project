# Raw data uploads

Upload raw movement data for the Conrad rehabilitation project to this folder.

Accepted data may include:

- Original movement videos (`.mp4`, `.mov`)
- MediaPipe or YOLO landmark data (`.csv`)
- Joint-angle data (`.csv`)
- Non-identifying exercise metadata (`.json` or `.csv`)

Create a separate subfolder for each exercise or collection. Use clear file names
that do not contain a participant's name or other personal information.

Example:

```text
Data_set/
└── quadruped_alternating_arms_legs/
    ├── participant_001.mp4
    ├── participant_001_joint_angles.csv
    └── metadata.csv
```

## Privacy

Only upload data that you are authorized to share. Obtain appropriate consent
before uploading human-participant recordings. Remove names and other direct
identifiers. Do not upload medical records, contact information, or other
sensitive personal information to a public repository.

GitHub has practical file-size limits. Large video collections should be stored
in an approved research-data service, with a non-sensitive manifest or link kept
here instead.
