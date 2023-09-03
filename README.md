# CinC2023

[![docker-ci-and-test](https://github.com/wenh06/cinc2023/actions/workflows/docker-test.yml/badge.svg?branch=docker-test)](https://github.com/wenh06/cinc2023/actions/workflows/docker-test.yml)
[![format-check](https://github.com/wenh06/cinc2023/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/cinc2023/actions/workflows/check-formatting.yml)

Predicting Neurological Recovery from Coma After Cardiac Arrest: The George B. Moody PhysioNet Challenge 2023

<!-- toc -->

- [The Conference](#the-conference)
- [External Resources Used](#external-resources-used)
  - [SQI (Signal Quality Index) Calculation](#sqi--signal-quality-index--calculation)

<!-- tocstop -->

## The Conference

[Conference Website](https://cinc2023.org/)

[Official Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vTa94VmPIbywGJEBYjNkzJiGZuPLaajzPIZpoxsi12_X5DF66ccUFB6Qi3U41UEpVu2q1rzTF7nlSpY/pubhtml?gid=0&widget=true&headers=false)

:point_right: [Back to TOC](#cinc2023)

## External Resources Used

### SQI (Signal Quality Index) Calculation

[Source Code](utils/sqi.py) integrated from [bdsp-core/icare-dl](https://github.com/bdsp-core/icare-dl/blob/main/Artifact_pipeline.zip).

As stated in the `Artfiact Screening (Signal Quality)` subsection of the `Data Description` section of the
[I-CARE database version 1.0 hosted at PhysioNet](https://physionet.org/content/i-care/1.0/), the SQI is calculated as follows:
<blockquote>
...This artifact score is based on how many 10-second epochs within a 5-minute EEG window are contaminated by artifacts. Each 10-second epoch was scored for the presence of the following artifacts including: 1) flat signal, 2) extreme high or low values, 3) muscle artifact, 4) non-physiological spectra, and 5) implausibly fast rising or decreasing signal amplitude...
</blockquote>

Precomputed SQI (5min window, 1min step) for all EEGs: [Google Drive](https://drive.google.com/u/0/uc?id=1yPeLkL7WmHzXfSi5XK7hzWTfcAvrL8_q) | [Alternative](https://deep-psp.tech/Data/CinC2023-SQI.zip)

:point_right: [Back to TOC](#cinc2023)
