import numpy as np
import os
import glob
import warnings
import matplotlib.pyplot as plt

groundTruth = [
  {'prefix': 'original_video1', 'frames': 313, 'segments': ((36,70), (160,250))},
  {'prefix': 'original_video2', 'frames': 1145, 'segments': ((933,1030),)},
  {'prefix': 'original_video3', 'frames': 649, 'segments': ()},
  {'prefix': 'original_video4', 'frames': 949, 'segments': ((740,881),)},
  {'prefix': 'uni_ulm_Sequence2_1', 'frames': 947, 'segments': ((500,925),)},
  {'prefix': 'uni_ulm_Sequence2_4', 'frames': 947, 'segments': ((500,925),)},
  {'prefix': 'citinews1.', 'frames': 389, 'segments': ((1,386),)},
  {'prefix': 'citinews1_stabilized', 'frames': 389, 'segments': ((1,389),)},
  {'prefix': 'citinews2.', 'frames': 245, 'segments': ((1,245),)},
  {'prefix': 'citinews2_stabilized', 'frames': 245, 'segments': ((1,245),)},
  {'prefix': 'citinews3.', 'frames': 279, 'segments': ((1,279),)},
  {'prefix': 'citinews3_stabilized', 'frames': 279, 'segments': ((1,279),)},
  {'prefix': 'gram_rtm_uah_Urban1', 'frames': 23435, 'segments': ()},#((1500,1880), (2410,2960), (3680,3810), (4470,4970), (5670,6040), (9490,9515), (9720,10190), (11060,11230), (13010,13300), (13615,13670), (13820,14340), (16610,16740), (17090,17450), (19345,19470), (19970,20590), (20770,20880), (21260,21610))},
  {'prefix': 'uni_ulm_Sequence1a_1', 'frames': 2419, 'segments': ()},
  {'prefix': 'uni_ulm_Sequence1a_4', 'frames': 2419, 'segments': ()},
  {'prefix': 'uni_ulm_Sequence3_1', 'frames': 1020, 'segments': ()},
  {'prefix': 'uni_ulm_Sequence3_4', 'frames': 1020, 'segments': ()},
  {'prefix': 'changedetection_highway', 'frames': 1700, 'segments': ()},
  {'prefix': 'changedetection_streetLight', 'frames': 3200, 'segments': ()},
  {'prefix': 'changedetection_traffic', 'frames': 1570, 'segments': ()},
  {'prefix': 'gram_rtm_uah_M30.', 'frames': 7520, 'segments': ()},
  {'prefix': 'gram_rtm_uah_M30HD', 'frames': 9390, 'segments': ()},
  {'prefix': 'jodoin_urbantracker_rene', 'frames': 8501, 'segments': ()},
  {'prefix': 'jodoin_urbantracker_rouen', 'frames': 628, 'segments': ()},
  {'prefix': 'jodoin_urbantracker_sherbrooke', 'frames': 4000, 'segments': ()},
  {'prefix': 'jodoin_urbantracker_stmarc', 'frames': 2000, 'segments': ()},
]

def inAnySegment(values, segments):
  result = np.zeros(values.shape, dtype=np.bool_)
  for segment in segments:
    result = np.logical_or(result, np.logical_and(values>=segment[0], values<=segment[1]))
  return result

def computeSummaries(directory, groundTruth=groundTruth):
  shape  = (len(groundTruth),)
  TPs    = np.zeros(shape, dtype=np.float64)
  FPs    = np.zeros(shape, dtype=np.float64)
  FNs    = np.zeros(shape, dtype=np.float64)
  TNs    = np.zeros(shape, dtype=np.float64)
  GT_Ps  = np.zeros(shape, dtype=np.float64)
  frames = np.zeros(shape, dtype=np.float64)
  precs  = np.zeros(shape, dtype=np.float64)
  recs   = np.zeros(shape, dtype=np.float64)
  names  = []
  for i,g in enumerate(groundTruth):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    prefix     = g['prefix']
    nframes    = g['frames']
    segments   = g['segments']
    pattern    = os.path.join(directory, prefix)+'*.txt'
    files      = glob.glob(pattern)
    if len(files)!=1:
      raise Exception(f'There should be just one file for {pattern}. Files: {files}')
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', 'loadtxt: Empty input file')
      anomFrames = np.loadtxt(files[0], comments=':', dtype=np.int32)
    #print(f'MIRA: {anomFrames.shape} for {files[0]}')
    if anomFrames.shape==():
      #import code; code.interact(local=locals())
      anomFrames = anomFrames.reshape((-1,))
    repeated   = np.diff(anomFrames, prepend=1)
    anomFrames = anomFrames[repeated!=0]
    allFrames  = np.arange(1, nframes+1, dtype=np.int32)
    GT_P       = inAnySegment(allFrames, segments)
    GT_N       = np.logical_not(GT_P)
    detected   = np.zeros(allFrames.shape, dtype=np.bool_)
    detected[anomFrames] = True
    not_detected         = np.logical_not(detected)
    TP = np.logical_and(GT_P, detected)
    FP = np.logical_and(GT_N, detected)
    FN = np.logical_and(GT_P, not_detected)
    TN = np.logical_and(GT_N, not_detected)
    TPs[i]    = TP.sum()
    FPs[i]    = FP.sum()
    FNs[i]    = FN.sum()
    TNs[i]    = TN.sum()
    GT_Ps[i]  = GT_P.sum()
    if TPs[i]==0:
      if FPs[i]==0 and FNs[i]==0:
        precs[i]  = 1
        recs[i]   = 1
      else:
        precs[i]  = 0
        recs[i]   = 0
    else:
      precs[i]  = TPs[i]/(TPs[i]+FPs[i])
      recs[i]   = TPs[i]/(TPs[i]+FNs[i])
    frames[i] = nframes
    names.append(prefix)
  return {'TP': TPs, 'FP': FPs, 'TN': TNs, 'FN': FNs, 'GT_P': GT_Ps, 'frames': frames, 'precision': precs, 'recall': recs, 'name': names}

def writeSummarySingleExperiment(path, summary):
  name,frames,TP,FP,TN,FN,GT_P,precision,recall = summary['name'], summary['frames'], summary['TP'], summary['FP'], summary['TN'], summary['FN'], summary['GT_P'], summary['precision'], summary['recall']
  #import code; code.interact(local=locals())
  with open(path, 'w') as f:
    f.write('name,frames,GT_P,TP,FP,TN,FN,precision,recall\n')
    for i in range(len(summary['TP'])):
      f.write(f'{name[i]},{frames[i]},{GT_P[i]},{TP[i]},{FP[i]},{TN[i]},{FN[i]},{precision[i]},{recall[i]}\n')
    f.write(f',,,,,,,{np.mean(precision)},{np.mean(recall)}\n')
    f.write(f',,,,,,,avg. precision,avg.recall\n')

def writeSummariesSingleExperiment(path, summaries):
  with open(path, 'w') as f:
    for directory, spec, summary in summaries:
      name,frames,TP,FP,TN,FN,GT_P,precision,recall = summary['name'], summary['frames'], summary['TP'], summary['FP'], summary['TN'], summary['FN'], summary['GT_P'], summary['precision'], summary['recall']
      f.write(f'{directory}\n')
      f.write('name,frames,GT_P,TP,FP,TN,FN,precision,recall\n')
      for i in range(len(summary['TP'])):
        f.write(f'{name[i]},{frames[i]},{GT_P[i]},{TP[i]},{FP[i]},{TN[i]},{FN[i]},{precision[i]},{recall[i]}\n')
      f.write(f',,,,,,,{np.mean(precision)},{np.mean(recall)}\n')
      f.write(f',,,,,,,avg. precision,avg.recall\n\n')


def getMatches(summaries, spec):
  matches = []
  for tup in summaries:
    thisspec = tup[1]
    if all(thisspec[name]==value for name, value in spec.items()):
      matches.append(tup)
  return matches

def getSingleMatch(summaries, spec):
  matches = getMatches(summaries, spec)
  if len(matches)!=1:
    raise Exception(f'There should be only one match for this spec: {spec}, but there are {len(matches)}')
  return matches[0]

def writeSummaryAllExperiments(path, summaries, specs, header, rowTemplate):
  null = ''
  with open(path, 'w') as f:
    f.write(header)
    f.write('\n')
    for spec in specs:
      _, _, match = getSingleMatch(summaries, spec)
      strf = f'f"{rowTemplate}"'
      f.write(eval(strf))
      f.write('\n')

def getCurves(summaries, specsToShow, specName, measureNames, funs):
  xs  = []
  yss = [[] for _ in range(len(measureNames))]
  for s in specsToShow:
    _, spec, match = getSingleMatch(summaries, s)
    xs.append(spec[specName])
    for i,(mn,fun) in enumerate(zip(measureNames, funs)):
      yss[i].append(fun(match[mn]))
      print(f'Adding for {specName} value {xs[-1]}, for {mn} value {yss[i][-1]}')
  return xs, yss

def showGraph(summaries, specsToShow, specName, measureNames, funs):
  xs, yss = getCurves(summaries, specsToShow, specName, measureNames, funs)
  everything = []
  for ys in yss:
    everything.append(xs)
    everything.append(ys)
  fig, axs = plt.subplots(1, 2)
  axs[0].plot(*everything)
  axs[0].legend(measureNames)
  axs[1].plot(yss[1], yss[0])
  axs[1].set_xlabel(measureNames[1])
  axs[1].set_ylabel(measureNames[0])
  plt.show()

def writeAllSummaries(prefixpath, directories, specs, header, rowTemplate):
  summaries = []
  for d, spec in directories:
    s = computeSummaries(os.path.join(prefixpath, d))
    summaries.append((d,spec,s))
    #writeSummarySingleExperiment(os.path.join(d, 'results.csv'), s)
  writeSummariesSingleExperiment(os.path.join(prefixpath, 'allresults.csv'), summaries)
  writeSummaryAllExperiments(os.path.join(prefixpath, 'summary.csv'), summaries, specs, header, rowTemplate)
  return summaries

def getDataByBackbone(backbone):
  directories = [
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.1_allvehicles_use_distance_limit_small_False', {'BC': 0, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.1, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.2_allvehicles_use_distance_limit_small_False', {'BC': 0, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.2, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.3_allvehicles_use_distance_limit_small_False', {'BC': 0, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.3, 'JSV': 0}),
    #(f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.4_allvehicles_use_distance_limit_small_False', {'BC': 0, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.4, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.5_allvehicles_use_distance_limit_small_False', {'BC': 0, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.5, 'JSV': 0}),

    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.1_allvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.1, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.2_allvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.2, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.3_allvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.3, 'JSV': 0}),
    #(f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.4_allvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.4, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.5_allvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.5, 'JSV': 0}),

    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.1_smallvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.1, 'JSV': 1}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.2_smallvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.2, 'JSV': 1}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.3_smallvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.3, 'JSV': 1}),
    #(f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.4_smallvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.4, 'JSV': 1}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.5_smallvehicles_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.5, 'JSV': 1}),

    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.1_allvehicles_use_distance_limit_small_False', {'BC': 1, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.1, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.2_allvehicles_use_distance_limit_small_False', {'BC': 1, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.2, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.3_allvehicles_use_distance_limit_small_False', {'BC': 1, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.3, 'JSV': 0}),
    #(f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.4_allvehicles_use_distance_limit_small_False', {'BC': 1, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.4, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.5_allvehicles_use_distance_limit_small_False', {'BC': 1, 'DLS': 0,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.5, 'JSV': 0}),

    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.1_allvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.1, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.2_allvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.2, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.3_allvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.3, 'JSV': 0}),
    #(f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.4_allvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.4, 'JSV': 0}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.5_allvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.5, 'JSV': 0}),

    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.1_smallvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.1, 'JSV': 1}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.2_smallvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.2, 'JSV': 1}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.3_smallvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.3, 'JSV': 1}),
    #(f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.4_smallvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.4, 'JSV': 1}),
    (f'quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_{backbone}_thresh_0.5_smallvehicles_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': backbone, 'BBT': 0.5, 'JSV': 1}),
  ]
  thrs = (0.1,0.2,0.3,0.5)
  specs = (
    [{'BC': 0, 'DLS': 0, 'FT': 1, 'MT': 'cos', 'BB': backbone, 'JSV': 0, 'BBT': x} for x in thrs]+
    [{'BC': 0, 'DLS': 1, 'FT': 1, 'MT': 'cos', 'BB': backbone, 'JSV': 0, 'BBT': x} for x in thrs]+
    [{'BC': 0, 'DLS': 1, 'FT': 1, 'MT': 'cos', 'BB': backbone, 'JSV': 1, 'BBT': x} for x in thrs]+
    [{'BC': 1, 'DLS': 0, 'FT': 1, 'MT': 'cos', 'BB': backbone, 'JSV': 0, 'BBT': x} for x in thrs]+
    [{'BC': 1, 'DLS': 1, 'FT': 1, 'MT': 'cos', 'BB': backbone, 'JSV': 0, 'BBT': x} for x in thrs]+
    [{'BC': 1, 'DLS': 1, 'FT': 1, 'MT': 'cos', 'BB': backbone, 'JSV': 1, 'BBT': x} for x in thrs]+
    [])
  return directories, specs

def getOurData():
  prefixpath = '/media/Data/experiments/to_measure/'
  dirs_resnet18, specs_resnet18 = getDataByBackbone('resnet18')
  #dirs_resnet101, specs_resnet101 = getDataByBackbone('resnet101')
  #dirs_vgg11_bn, specs_vgg11_bn = getDataByBackbone('vgg11_bn')
  #dirs_vgg19_bn, specs_vgg19_bn = getDataByBackbone('vgg19_bn')
  directories = [
  #BC==BORDER CORRECTION
  #DLS==USE DISTANCE LIMIT SMALL
  #FT==USE FEATURES
  #MT: METRIC FOR FEATURES
  #BB: BACKBONE TO EXTRACT FEATURES
  #BBT: threshold for features
  #JSV: CONSIDER FEATURES JUST FOR SMALL VEHICLES
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0__no_features_use_distance_limit_small_False', {'BC': 0, 'DLS': 0,'FT': 0, 'MT': None, 'BB': None, 'BBT': None, 'JSV': None}),
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0__no_features_use_distance_limit_small_True', {'BC': 0, 'DLS': 1,'FT': 0, 'MT': None, 'BB': None, 'BBT': None, 'JSV': None}),
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0__no_features_use_distance_limit_small_False', {'BC': 1, 'DLS': 0,'FT': 0, 'MT': None, 'BB': None, 'BBT': None, 'JSV': None}),
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0__no_features_use_distance_limit_small_True', {'BC': 1, 'DLS': 1,'FT': 0, 'MT': None, 'BB': None, 'BBT': None, 'JSV': None}),
  ] + dirs_resnet18 #+ dirs_resnet101 + dirs_vgg11_bn + dirs_vgg19_bn
  specs = (
    [{'BC': 0, 'DLS': 0,'FT': 0}, {'BC': 0, 'DLS': 1,'FT': 0}, {'BC': 1, 'DLS': 0,'FT': 0}, {'BC': 1, 'DLS': 1,'FT': 0}]
    #[{'BC': 0, 'FT': 0}, {'BC': 1, 'FT': 0}]
    )+specs_resnet18#+specs_resnet101+specs_vgg11_bn+specs_vgg19_bn
  header = 'use border correction,use distance limit small,use features,features are just for small vehicles,backbone,metric,threshold,average precision,average recall,nonzero precision,nonzero recall,total FP,total FN'
  rowTemplate = "{spec.get('BC',null)},{spec.get('DLS',null)},{spec.get('FT',null)},{spec.get('JSV',null)},{spec.get('BB',null)},{spec.get('MT',null)},{spec.get('BBT',null)},{match['precision'].mean()},{match['recall'].mean()},{(match['precision']!=0).sum()},{(match['recall']!=0).sum()},{match['FP'].sum()},{match['FN'].sum()}"
  #header = 'use border correction,use features,features are just for small vehicles,backbone,metric,threshold,average precision,average recall,nonzero precision,nonzero recall'
  #rowTemplate = "{spec.get('BC',null)},{spec.get('FT',null)},{spec.get('JSV',null)},{spec.get('BB',null)},{spec.get('MT',null)},{spec.get('BBT',null)},{match['precision'].mean()},{match['recall'].mean()},{(match['precision']!=0).sum()},{(match['recall']!=0).sum()}"
  return {'prefixpath': prefixpath, 'directories': directories, 'specs': specs, 'header': header, 'rowTemplate': rowTemplate}

def getOtherData():
  prefixpath = '/media/Data/experiments/others/'
  directories = [
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_byte', {'BC': 0, 'AL': 'BYTE', 'BF': 1}),
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_sort', {'BC': 0, 'AL': 'SORT', 'BF': 1}),
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF1_THR0.25', {'BC': 1, 'AL': 'BYTE', 'BF': 1}),
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF1_THR0.25', {'BC': 1, 'AL': 'SORT', 'BF': 1}),
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF5_THR0.25', {'BC': 1, 'AL': 'BYTE', 'BF': 5}),
    ('quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF5_THR_0.25', {'BC': 1, 'AL': 'SORT', 'BF': 5}),
  ]
  specs = [
    {'BC': 0, 'AL': 'BYTE', 'BF': 1},
    {'BC': 1, 'AL': 'BYTE', 'BF': 1},
    {'BC': 1, 'AL': 'BYTE', 'BF': 5},
    {'BC': 0, 'AL': 'SORT', 'BF': 1},
    {'BC': 1, 'AL': 'SORT', 'BF': 1},
    {'BC': 1, 'AL': 'SORT', 'BF': 5},
  ]
  header='use border correction, algorithm, buffer size,average precision,average recall,nonzero precision,nonzero recall,total FP,total FN'
  rowTemplate = "{spec.get('BC',null)},{spec.get('AL',null)},{spec.get('BF',null)},{match['precision'].mean()},{match['recall'].mean()},{(match['precision']!=0).sum()},{(match['recall']!=0).sum()},{match['FP'].sum()},{match['FN'].sum()}"
  return {'prefixpath': prefixpath, 'directories': directories, 'specs': specs, 'header': header, 'rowTemplate': rowTemplate}
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_byte
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_sort
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF1_THR0.25
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF1_THR0.25_mindim0.015
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF1_THR0.25_mindim0.02
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF1_THR0.25_mindim0.025
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF1_THR0.5
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF5_THR0.25
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF5_THR0.25_mindim0.015
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF5_THR0.25_mindim0.02
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF5_THR0.25_mindim0.025
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_byte_BUF5_THR0.5
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0.015
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0.02
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0.025
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.5
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF1_THR0.25
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF1_THR0.25_mindim0.015
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF1_THR0.25_mindim0.02
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF1_THR0.25_mindim0.025
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF1_THR0.5
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF5_THR_0.25
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF5_THR0.25_mindim0.015
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF5_THR0.25_mindim0.02
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF5_THR0.25_mindim0.025
#quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_sort_BUF5_THR0.5


if __name__=='__main__':
  summaries = writeAllSummaries(**getOurData())
  thrs = (0.1,0.2,0.3,0.5)
  specsToShow = [{'BC': 0, 'DLS': 1,'FT': 1, 'MT': 'cos', 'BB': 'resnet18', 'BBT': x, 'JSV': 0} for x in thrs]
  showGraph(summaries, specsToShow, 'BBT', ('precision', 'recall'), [np.mean]*2)
  
  writeAllSummaries(**getOtherData())


