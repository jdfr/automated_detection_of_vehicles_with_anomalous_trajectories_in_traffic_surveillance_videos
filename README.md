# Anomalous Trajectory Detection

This is the source code used to get the results in the paper "Automated detection of vehicles with anomalous trajectories in traffic surveillance videos":

```
@article{fernandez2023automated,
  title={Automated detection of vehicles with anomalous trajectories in traffic surveillance videos},
  author={Fern{\'a}ndez-Rodr{\'\i}guez, Jose D and Garc{\'\i}a-Gonz{\'a}lez, Jorge and Ben{\'\i}tez-Rochel, Rafaela and Molina-Cabello, Miguel A and Ramos-Jim{\'e}nez, Gonzalo and L{\'o}pez-Rubio, Ezequiel},
  journal={Integrated Computer-Aided Engineering},
  number={Preprint},
  pages={293--309},
  year={2023},
  publisher={IOS Press}
}
```

The object detector used as a base was yolov5 from https://github.com/ultralytics/yolov5 at commit 9ef94940aa5e9618e7e804f0758f9a6cebfc63a9, but the code was somewhat modified.

We compared our results with SORT (in folder `sort/`, taken from https://github.com/abewley/sort ) and BYTE (in folder `byte/`, taken from https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker ).

The code used to detect anomalous trajectories is in `anom_traj_detector.py`. There is a large multiline string at the end of `anom_traj_detector.py` with the command line arguments used to test the anomalous trajectory detection system, both with our method and wwith other methods. The code used to crunch the resulting data is in `summarize.py` and `trajectories.py`.

The original, unmodified code from yolov5 is still licensed under its original license, the GPLv2. The code from SORT and BYTE is still licensed under the respective licenses of each project. All original code and the modifications made to yolov5 to use it as a substrate for our anomalous trajectory detector are now licensed under the AGPLv3 (as part of a drive to harmonize licensing terms over all my repositories for journal papers).

Most of the used datasets are from previous papers:

* The dataset from "Multiple Object Tracking in Urban Mixed Traffic" ( https://www.jpjodoin.com/urbantracker/ ) should be put in `/media/jpjodoin_urbantracker/`

* The dataset from "CDnet 2014: An Expanded Change Detection Benchmark Dataset" ( http://changedetection.net/ ) should be put in `/media/changedetection`

* The dataset from "Vehicle Tracking by Simultaneous Detection and Viewpoint Estimation" ( https://gram.web.uah.es/data/datasets/rtm/index.html ) should be put in `/media/GRAM_RTM_UAH`

* The dataset from "The Ko-PER Intersection Laserscanner and Video Dataset" ( https://www.uni-ulm.de/in/mrm/forschung/datensaetze/ ) should be put in `/media/uni_ulm`

* The dataset from "AI City Challenge. Track 4" ( https://www.aicitychallenge.org/2021-data-and-evaluation/ ) should be put in `/media/aic21-track4-train-data`

* The dataset from "Rain Removal in Traffic Surveillance: Does it Matter?" ( https://www.kaggle.com/datasets/aalborguniversity/aau-rainsnow ) should be put in `/media/aalborguniversity`

* The previously unpublished synthetic videos from "Videovigilancia de trayectorias anomalas de vehiculos en videos de trafico" are in folder `originales/` and should be put in `/media/originales`


