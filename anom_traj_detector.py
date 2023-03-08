import sys
import os

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from detect import load_weights_for_streamlined, run_streamlined
from utils.torch_utils import select_device

from byte.byte_tracker import BYTETracker
from sort.sort import Sort
import explore_layers as sim_util
import sys
import argparse
import time

TRACKER_CLASSIC = 0
TRACKER_BYTE    = 1
TRACKER_SORT    = 2

#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

baseprefix = '/media/Data/'

WRITE_TO_COMMON_FILE = False
COMMONFILE = f'{baseprefix}tracerecords.txt'
#ffmpeg -ss 00:00:21.3 -to 00:00:34.3 -i "Driver caught going wrong-way on highway-sAQkcweENiY.mp4" -c libx264 citinews1.mp4
#ffmpeg -ss 00:01:43.8 -to 00:01:52.0 -i "Driver caught going wrong-way on highway-sAQkcweENiY.mp4" -c libx264 citinews2.mp4

def cosine_similarity_by_row(x, y):
  return (x*y).sum(axis=1) / ( np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) )

def create_costs_matrix(A : np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Function to get a matrix C from two vectors of positions (A and B) so C_i_j is the cost (distance) between i-th element from A and j-th element from B.
  inputs:
      A : np.ndarray -> Numpy array with shape (n,2).
      B : np.ndarray -> Numpy array with shape (m,2).

  outputs:
      C : np.ndarray -> Numpy array with shape (n,m).
  """

  assert len(A.shape) == 2
  assert len(B.shape) == 2
  assert A.shape[1] == 2
  assert B.shape[1] == 2

  n = A.shape[0]
  m = B.shape[0]
  #C = -1*np.ones(shape=(n, m))

  #for i in range(n):
  #  for j in range(m):
  #    C[i,j] = np.linalg.norm(A[i]-B[j])
  C = cdist(A, B)

  return C


class Trace:
  """
  Class to represent a complete trace.
  """

  def __init__(self, sz:np.ndarray, t:int, trace_id:int, initial_position:np.ndarray, boxFeats:np.ndarray):
    """
    inputs:
        trace_id : int -> trace identification number.
        initial_position : np.ndarray -> Numpy array with shape (2).
    """
    self.id = trace_id                          # Trace id.
    self.positions = [(t, initial_position)]    # Last known position.
    self.skipped_frames = 0                     # Number of frames already skipped.
    self.sz = sz
    self.boxFeats = boxFeats

  def skipped(self, t):
    self.positions.append((t, None))
    self.skipped_frames += 1

  def get_last_position(self):
    return self.positions[-1][-1]

  def get_last_three_positions(self):
    return self.positions[-3][-1], self.positions[-2][-1], self.positions[-1][-1]

  def get_last_not_None_position(self):
    for p in reversed(self.positions):
      if not p[-1] is None:
        #import code; code.interact(local=locals())
        return p[-1]

  def get_last_step(self):
    """ return tuple containing last position, last step, and id info, if at all possible """
    ps = self.positions
    if len(ps)>1 and ps[-1][-1] is not None and ps[-2][-1] is not None:
      return (ps[-1][-1], ps[-1][-1]-ps[-2][-1], (self.id, ps[-1][0]))
    else:
      return None

  def add_position(self, sz, t, position, boxFeats):
    assert not position is None
    self.sz = sz
    self.boxFeats = boxFeats
    self.positions.append((t, position))
    self.skipped_frames = 0

  def get_skipped_frames(self):
    return self.skipped_frames

  def get_id(self):
    return self.id

  def get_positions(self):
    return self.positions

  def get_not_None_positions(self):
    l = []
    for p in self.positions:
      if not p is None:
        l.append(p)
    return l

class ClassicTracker:
    """
    Class to rerepresent a Tracker.
    """
    def __init__(self, scale_diff_limit:float, scale_step_big:float, size_distance_limit_big:float, size_distance_limit_small:float, maximum_distance_to_assign:int, maximum_frames_to_skip_before_set_trace_as_inactive:int, value_to_use_as_inf = 50000, min_dim=-1, feat_sim_args=None, use_distance_limit_small=True):
        """
        inputs:
            maximum_distance_to_assign : int -> The reference we will use as maximum distance in order to avoid assignments between positions too far.
            maximum_frames_to_skip_before_set_trace_as_inactive : int -> The amount of frames we will allow to be skkiped by a trace before setting it as inactive.
            value_to_use_as_inf : int -> The value to use instead of infinite as "very large value" in order to avoid numerical problems.
        """
        self.scale_diff_limit = scale_diff_limit
        self.scale_step_big = scale_step_big
        self.size_distance_limit_big = size_distance_limit_big
        self.size_distance_limit_small = size_distance_limit_small
        self.active_traces = dict()                                         # Active traces.
        self.inactive_traces = dict()                                       # Old traces. self.active_traces and self.inactive_traces should be disjoint sets.
        self.next_trace_id = 0                                          # Next trace id.
        self.maximum_distance_to_assign = maximum_distance_to_assign    # Beyond this distance, any association will be discarded.
        self.maximum_frames_to_skip_before_set_trace_as_inactive = maximum_frames_to_skip_before_set_trace_as_inactive # Maximum skipped frames number before set a trace as inactive.
        self.min_dim = min_dim # boxes with any dimension below this limit won't be tracked. This is expressed as a percentage of the largest side of the image
        self.feat_sim_args = feat_sim_args
        if self.feat_sim_args.usefs:
          if   self.feat_sim_args.metric=='cos':
            self.feat_metric = 0
          elif self.feat_sim_args.metric=='L2':
            self.feat_metric = 1
          else:
            raise Exception('If measuring feature vector similarities, use valid metric!!!!')

        self.value_to_use_as_inf = value_to_use_as_inf
        self.use_distance_limit_small = use_distance_limit_small

    def new_trace(self, sz, t, position, boxFeats):
        self.active_traces[self.next_trace_id] = Trace(sz, t, self.next_trace_id, position, boxFeats)
        self.next_trace_id += 1

        return self.next_trace_id-1

    def active_traces_last_positions(self):
        last_positions = []
        trace_indexes = []
        szs = []
        boxFeats = []
        for trace_index, trace in self.active_traces.items():
            last_positions.append(trace.get_last_not_None_position())
            trace_indexes.append(trace_index)
            szs.append(trace.sz)
            if self.feat_sim_args.usefs:
              boxFeats.append(trace.boxFeats)
        return np.array(last_positions), np.array(trace_indexes), np.array(szs), np.array(boxFeats)

    def get_active_traces_steps(self):
      """ returns Nx3x2 array: first dimension represents instances, second dimension selects between positions, steps and (id,t), last dimension are coordinates """
      steps = []
      for trace in self.active_traces.values():
        step = trace.get_last_step()
        if step is not None:
          steps.append(step)
      if len(steps)==0:
        return np.zeros((0,3,2), dtype=np.float64)
      else:
        return np.array(steps)

    def assign_incomming_positions(self, t:int, new_boxesFeats:np.ndarray, new_positions:np.ndarray, new_sizes:np.ndarray, img_long_side):#, other:np.ndarray):
        """
        Method to insert new positions in order to be associated to active traces. All position without valid association will start its own new trace.
        intpus:
            new_positions : np.ndarray -> A numpy array with shape (n,2).

        outputs:
            associated_ids : np.ndarray -> Each trace id associated to the incomming positions.
        """
        associated_ids = new_positions.shape[0]*[None]

        # If there are no active traces.
        if len(self.active_traces) == 0:
            for pos_index in range(new_positions.shape[0]):
                new_trace_id = self.new_trace(new_sizes[pos_index], t, new_positions[pos_index], new_boxesFeats[pos_index] if self.feat_sim_args.usefs else None)
                associated_ids[pos_index] = new_trace_id

        # If there are no new positions.
        elif new_positions.shape[0] == 0:
            # We will increase skipped frames for each trace.
            for trace in self.active_traces.values():
                trace.skipped(t)

            # We will move each active trace with too much skipped frames to inactive traces.
            for trace_index in list(self.active_traces.keys()):
                if self.active_traces[trace_index].get_skipped_frames() > self.maximum_frames_to_skip_before_set_trace_as_inactive:
                    trace = self.active_traces.pop(trace_index)
                    self.inactive_traces[trace_index] = trace

        elif len(self.active_traces) > 0 and  new_positions.shape[0] > 0:
            traces_last_positions, trace_indexes, last_positions_szs, boxFeats_last_positions = self.active_traces_last_positions()

            p=0
            while True:
              p=p+1
              distances = create_costs_matrix(traces_last_positions, new_positions) # We get the assignment cost between incomming positions and active traces last known positions.
              costs_matrix = np.power(distances, 2)

              assert costs_matrix.shape[0] == traces_last_positions.shape[0]
              assert costs_matrix.shape[1] == new_positions.shape[0]
              sn1           = new_sizes         [:,0].reshape((1,-1))
              sn2           = new_sizes         [:,1].reshape((1,-1))
              sl1           = last_positions_szs[:,0].reshape((-1,1))
              sl2           = last_positions_szs[:,1].reshape((-1,1))
              #mean_new_szs  = np.mean(new_sizes,          axis=1).reshape((1,-1))
              #mean_last_szs = np.mean(last_positions_szs, axis=1).reshape((-1,1))
              min_new_szs   = np.min (new_sizes,          axis=1).reshape((1,-1))
              new_szs_big   = (min_new_szs/img_long_side)>=self.scale_step_big
              new_szs_small = np.logical_not(new_szs_big)
              #min_last_szs  = np.min (last_positions_szs, axis=1).reshape((-1,1))
              #rat_last_szs  = min_last_szs/img_long_side

              # We set any cost value greater than self.maximum_distance_to_assign as self.value_to_use_as_inf.
              costs_matrix[distances > self.maximum_distance_to_assign] = self.value_to_use_as_inf
              # forbid assignment when the sizes differ too greatly
              #scale_diffs = np.abs(mean_new_szs-mean_last_szs)/np.maximum(mean_new_szs, mean_last_szs)
              scale_diffs = np.maximum( (sn1-sl1)/np.maximum(sn1, sl1), np.abs(sn2-sl2)/np.maximum(sn2, sl2) )
              costs_matrix[scale_diffs>self.scale_diff_limit] = self.value_to_use_as_inf
              # forbid assignment when distance is higher than scaled size
              if self.use_distance_limit_small:
                costs_matrix[np.logical_and(distances>(min_new_szs*self.size_distance_limit_big),   new_szs_big  )] = self.value_to_use_as_inf
                costs_matrix[np.logical_and(distances>(min_new_szs*self.size_distance_limit_small), new_szs_small)] = self.value_to_use_as_inf
              else:
                costs_matrix[distances>(min_new_szs*self.size_distance_limit_big)] = self.value_to_use_as_inf
              # forbid assignment when too dissimilar
              if self.feat_sim_args.usefs:
                if   self.feat_metric==0:
                  sim_matrix = cdist(boxFeats_last_positions, new_boxesFeats, 'cos')
                elif self.feat_metric==1:
                  sim_matrix = cdist(boxFeats_last_positions, new_boxesFeats, 'euclidean')
                #xx=sim_matrix.copy()
                #xx.sort(axis=1)
                #import code; code.interact(local=locals())
                #print(f'MIRA: sim_matrix->{sim_matrix.shape}, costs_matrix->{costs_matrix.shape}')
                #import code; code.interact(local=locals())
                #idxs1, idxs2 = np.nonzero(np.logical_and(sim_matrix>self.feat_sim_args.threshold, costs_matrix<self.value_to_use_as_inf))
                if WRITE_TO_COMMON_FILE and len(idxs1)>0:
                  with open(COMMONFILE, 'a') as f:
                    f.write(f'  AT t={t}, THERE ARE PAIRS DISALLOWED BECAUSE OF SIMILARITY: {list(zip(idxs1, idxs2))}\n')
                    f.write(f'    sim_matrix[dxs1, idxs2] = {sim_matrix[idxs1, idxs2]}\n')
                if self.feat_sim_args.just_small_detections:
                  costs_matrix[np.logical_and(sim_matrix>self.feat_sim_args.threshold, distances>(min_new_szs*self.size_distance_limit_small))] = self.value_to_use_as_inf
                else:
                  costs_matrix[sim_matrix>self.feat_sim_args.threshold] = self.value_to_use_as_inf

              #costs_matrix[distances> min_new_szs*self.size_distance_limit] = self.value_to_use_as_inf
              #costs_matrix[distances>min_last_szs*self.size_distance_limit] = self.value_to_use_as_inf

              # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
              trace_indices, pos_indices = linear_sum_assignment(costs_matrix)
              if np.any(costs_matrix[trace_indices, pos_indices]>=self.value_to_use_as_inf):
                do_delete = False
                dsts = distances[trace_indices, pos_indices]
                # is there any assignment whose distance is too large?
                mxi         = np.argmax(dsts)
                do_delete   = dsts[mxi]>self.maximum_distance_to_assign
                # is there any assignment whose distance is too large relative to its last size?
                #if not do_delete:
                #  scaled    = dsts/(min_last_szs[trace_indices,0]*self.size_distance_limit)
                #  mxi       = np.argmax(scaled)
                #  do_delete = scaled[mxi]>1
                # is there any assignment whose distance is too large relative to its new size?
                if not do_delete:
                  #scaled    = dsts/(min_new_szs[0,pos_indices]*self.size_distance_limit)
                  scaled    = dsts/min_new_szs[0,pos_indices]
                  if self.use_distance_limit_small:
                    scaled[new_szs_big[0,pos_indices]]   /= self.size_distance_limit_big
                    scaled[new_szs_small[0,pos_indices]] /= self.size_distance_limit_small
                  else:
                    scaled   /= self.size_distance_limit_big
                  mxi       = np.argmax(scaled)
                  do_delete = scaled[mxi]>1
                # is there any assignment whose sizes differ too greatly?
                if not do_delete:
                  scaled    = scale_diffs[trace_indices, pos_indices]
                  mxi       = np.argmax(scaled)
                  do_delete = scaled[mxi]>self.scale_diff_limit
                # is there any pair whose feature vectors are too dissimilar?
                if not do_delete and self.feat_sim_args.usefs:
                  #dists = cosine_similarity_by_row(old_bfeats, new_bfeats)
                  dists = sim_matrix[trace_indices, pos_indices]
                  mxi       = np.argmax(dists)
                  do_delete = dists[mxi]>self.feat_sim_args.threshold
                  if WRITE_TO_COMMON_FILE:
                    with open(COMMONFILE, 'a') as f:
                      f.write(f'  AT t={t}, GOT COS SIM BETWEEN {trace_indices} and {pos_indices}\n')
                      f.write(f'    dists = {dists}\n')
                      f.write(f'    do_delete = {do_delete}\n')
                  #import code; code.interact(local=locals())
                if False and not do_delete:
                  #make sure to kill disallowed pairs!
                  costs = costs_matrix[trace_indices, pos_indices]
                  mxi   = np.nonzero(costs==self.value_to_use_as_inf)
                  do_delete = mxi.size>0
                if do_delete:
                  if self.feat_sim_args.usefs:
                    boxFeats_last_positions = np.delete(boxFeats_last_positions, trace_indices[mxi], axis=0)
                  traces_last_positions = np.delete(traces_last_positions, trace_indices[mxi], axis=0)
                  trace_indexes         = np.delete(trace_indexes,         trace_indices[mxi], axis=0)
                  last_positions_szs    = np.delete(last_positions_szs,    trace_indices[mxi], axis=0)
                  continue
                else:
                  print('SOMETHING REALLY FUNKY IS GOING ON!!!!')
              #if t==5:
              #  import code; code.interact(local=locals())

              assigned_positions = []
              assigned_traces = []

              # Now the i-th index in trace_indices and i-th in pos_indices should be the optimal assignment.
              for cost_matrix_trace_index, pos_index in zip(trace_indices, pos_indices):
                  cost = distances[cost_matrix_trace_index, pos_index]
                  # If the assignment has lesser cost than self.maximum_distance_to_assign, we assignt the position to the trace.
                  if cost < self.maximum_distance_to_assign:
                      self.active_traces[trace_indexes[cost_matrix_trace_index]].add_position(new_sizes[pos_index], t, new_positions[pos_index], new_boxesFeats[pos_index] if self.feat_sim_args.usefs else None)
                      assigned_positions.append(pos_index)
                      assigned_traces.append(trace_indexes[cost_matrix_trace_index])
                      associated_ids[pos_index] = trace_indexes[cost_matrix_trace_index] #self.active_traces[trace_indexes[cost_matrix_trace_index]].get_id()

              # We will increase skipped frames for each non assigned trace.
              for trace_index in self.active_traces.keys():
                  if not trace_index in assigned_traces:
                      self.active_traces[trace_index].skipped(t)

              # We will move each active trace with too much skipped frames to inactive traces.
              for trace_index in list(self.active_traces.keys()):
                  if self.active_traces[trace_index].get_skipped_frames() > self.maximum_frames_to_skip_before_set_trace_as_inactive:
                      trace = self.active_traces.pop(trace_index)
                      self.inactive_traces[trace_index] = trace

              # We will generate new traces from non assigned positions.
              for pos_index in range(new_positions.shape[0]):
                  if not pos_index in assigned_positions:
                      new_trace_id = self.new_trace(new_sizes[pos_index], t, new_positions[pos_index], new_boxesFeats[pos_index] if self.feat_sim_args.usefs else None)
                      associated_ids[pos_index] = new_trace_id

              break

        return associated_ids

class TrackerBYTE:
    """
    Class to rerepresent a Tracker.
    """
    def __init__(self, scale_diff_limit:float, scale_step_big:float, size_distance_limit_big:float, size_distance_limit_small:float, maximum_distance_to_assign:int, maximum_frames_to_skip_before_set_trace_as_inactive:int, value_to_use_as_inf = 50000, min_dim=-1, byte_args=None):
        """
        inputs:
            maximum_distance_to_assign : int -> The reference we will use as maximum distance in order to avoid assignments between positions too far.
            maximum_frames_to_skip_before_set_trace_as_inactive : int -> The amount of frames we will allow to be skkiped by a trace before setting it as inactive.
            value_to_use_as_inf : int -> The value to use instead of infinite as "very large value" in order to avoid numerical problems.
        """
        self.scale_diff_limit = scale_diff_limit
        self.scale_step_big = scale_step_big
        self.size_distance_limit_big = size_distance_limit_big
        self.size_distance_limit_small = size_distance_limit_small
        self.active_traces = dict()                                         # Active traces.
        self.inactive_traces = dict()                                       # Old traces. self.active_traces and self.inactive_traces should be disjoint sets.
        self.maximum_distance_to_assign = maximum_distance_to_assign    # Beyond this distance, any association will be discarded.
        self.maximum_frames_to_skip_before_set_trace_as_inactive = maximum_frames_to_skip_before_set_trace_as_inactive # Maximum skipped frames number before set a trace as inactive.
        self.min_dim = min_dim # boxes with any dimension below this limit won't be tracked. This is expressed as a percentage of the largest side of the image

        self.value_to_use_as_inf = value_to_use_as_inf
        self.byte_args = byte_args
        self.byte = BYTETracker(self.byte_args, frame_rate=30)

    def active_traces_last_positions(self):
        last_positions = []
        trace_indexes = []
        szs = []
        for trace_index, trace in self.active_traces.items():
            last_positions.append(trace.get_last_not_None_position())
            trace_indexes.append(trace_index)
            szs.append(trace.sz)

        return np.array(last_positions), np.array(trace_indexes), np.array(szs)

    def get_active_traces_steps(self):
      """ returns Nx3x2 array: first dimension represents instances, second dimension selects between positions, steps and (id,t), last dimension are coordinates """
      steps = []
      for trace in self.active_traces.values():
        step = trace.get_last_step()
        if step is not None:
          steps.append(step)
      if len(steps)==0:
        return np.zeros((0,3,2), dtype=np.float64)
      else:
        return np.array(steps)

    def assign_incomming_positions(self, t:int, scores:np.ndarray, xyxys:np.ndarray, new_positions:np.ndarray, new_sizes:np.ndarray, img_long_side):#, other:np.ndarray):
        """
        Method to insert new positions in order to be associated to active traces. All position without valid association will start its own new trace.
        intpus:
            new_positions : np.ndarray -> A numpy array with shape (n,2).

        outputs:
            associated_ids : np.ndarray -> Each trace id associated to the incomming positions.
        """
        online_targets = self.byte.update(scores, xyxys)
        associated_ids = new_positions.shape[0]*[None]
        assigned_traces = []
        #update online tracks
        for track in online_targets:
          track_id = track.track_id
          assigned_traces.append(track_id)
          if track.current_index is not None:
            associated_ids[track.current_index] = track_id
            c  = new_positions[track.current_index]
            sz = new_sizes[track.current_index]
            if track_id in self.active_traces:
              self.active_traces[track_id].add_position(sz, t, c, None)
            else:
              self.active_traces[track_id] = Trace(sz, t, track_id, c, None)
          else:
            #cwh = track.cwh
            #c = cwh[2:4]
            #sz = cwh[:2]
            if track_id in self.active_traces:
              self.active_traces[track_id].skipped(t)
        #update lost tracks
        for track in self.byte.lost_stracks:
          track_id = track.track_id
          if track_id in self.active_traces:
            #cwh = track.cwh
            assigned_traces.append(track_id)
            #self.active_traces[track_id].add_position(cwh[2:4], t, cwh[:2], None)
            self.active_traces[track_id].skipped(t)
        # We will move traces neither online nor lost to inactive traces.
        for trace_index in list(self.active_traces.keys()):
          if not trace_index in assigned_traces:
            trace = self.active_traces.pop(trace_index)
            self.inactive_traces[trace_index] = trace
        return associated_ids


class TrackerSORT:
    """
    Class to rerepresent a Tracker.
    """
    def __init__(self, scale_diff_limit:float, scale_step_big:float, size_distance_limit_big:float, size_distance_limit_small:float, maximum_distance_to_assign:int, maximum_frames_to_skip_before_set_trace_as_inactive:int, value_to_use_as_inf = 50000, min_dim=-1, sort_args=None):
        """
        inputs:
            maximum_distance_to_assign : int -> The reference we will use as maximum distance in order to avoid assignments between positions too far.
            maximum_frames_to_skip_before_set_trace_as_inactive : int -> The amount of frames we will allow to be skkiped by a trace before setting it as inactive.
            value_to_use_as_inf : int -> The value to use instead of infinite as "very large value" in order to avoid numerical problems.
        """
        self.scale_diff_limit = scale_diff_limit
        self.scale_step_big = scale_step_big
        self.size_distance_limit_big = size_distance_limit_big
        self.size_distance_limit_small = size_distance_limit_small
        self.active_traces = dict()                                         # Active traces.
        self.inactive_traces = dict()                                       # Old traces. self.active_traces and self.inactive_traces should be disjoint sets.
        self.maximum_distance_to_assign = maximum_distance_to_assign    # Beyond this distance, any association will be discarded.
        self.maximum_frames_to_skip_before_set_trace_as_inactive = maximum_frames_to_skip_before_set_trace_as_inactive # Maximum skipped frames number before set a trace as inactive.
        self.min_dim = min_dim # boxes with any dimension below this limit won't be tracked. This is expressed as a percentage of the largest side of the image

        self.value_to_use_as_inf = value_to_use_as_inf
        self.sort_args = sort_args
        self.sort = Sort(max_age=self.sort_args.max_age, min_hits=self.sort_args.min_hits, iou_threshold=self.sort_args.iou_threshold)

    def active_traces_last_positions(self):
        last_positions = []
        trace_indexes = []
        szs = []
        for trace_index, trace in self.active_traces.items():
            last_positions.append(trace.get_last_not_None_position())
            trace_indexes.append(trace_index)
            szs.append(trace.sz)

        return np.array(last_positions), np.array(trace_indexes), np.array(szs)

    def get_active_traces_steps(self):
      """ returns Nx3x2 array: first dimension represents instances, second dimension selects between positions, steps and (id,t), last dimension are coordinates """
      steps = []
      for trace in self.active_traces.values():
        step = trace.get_last_step()
        if step is not None:
          steps.append(step)
      if len(steps)==0:
        return np.zeros((0,3,2), dtype=np.float64)
      else:
        return np.array(steps)

    def assign_incomming_positions(self, t:int, xyxys:np.ndarray, new_positions:np.ndarray, new_sizes:np.ndarray, img_long_side):#, other:np.ndarray):
        """
        Method to insert new positions in order to be associated to active traces. All position without valid association will start its own new trace.
        intpus:
            new_positions : np.ndarray -> A numpy array with shape (n,2).

        outputs:
            associated_ids : np.ndarray -> Each trace id associated to the incomming positions.
        """
        active_ids, deleted = self.sort.update(xyxys)
        allids = [track.id for track in self.sort.trackers]
        associated_ids = new_positions.shape[0]*[None]
        for track in self.sort.trackers:
          track_id = track.id
          if track.current_index is not None:
            associated_ids[track.current_index] = track_id
            c  = new_positions[track.current_index]
            sz = new_sizes[track.current_index]
            if track_id in self.active_traces:
              self.active_traces[track_id].add_position(sz, t, c, None)
            else:
              self.active_traces[track_id] = Trace(sz, t, track_id, c, None)
          else:
            #cwh = track.cwh
            #c = cwh[2:4]
            #sz = cwh[:2]
            if track_id in self.active_traces:
              self.active_traces[track_id].skipped(t)
        for trk in deleted:
          trace = self.active_traces.pop(trk.id)
          self.inactive_traces[trk.id] = trace
        return associated_ids


MEASURE_VECTOR_ABS_DIFFERENCE = 0
MEASURE_ANGLE_DIFFERENCE      = 1
MEASURE_SPEED_ABS_DIFFERENCE  = 2
MEASURE_SPEED_REL_DIFFERENCE  = 3
MEASURE_POLAR_REL_DIFFERENCE  = 4
MEASURE_NAMES = ['vector', 'angle', 'speedabs', 'speedrel', 'polar']

def compute_raw_anomalies(all_positioned_steps:np.ndarray, last_positioned_steps:np.ndarray, num_neighs:int, useMeans, measureMethods):
  allpositions     =  all_positioned_steps[:,0,:]
  lastpositions    = last_positioned_steps[:,0,:]
  allsteps         =  all_positioned_steps[:,1,:]
  laststeps        = last_positioned_steps[:,1,:]
  allids           =  all_positioned_steps[:,2,0]
  lastids          = last_positioned_steps[:,2,0]
  #alltimes         =  all_positioned_steps[:,2,1]
  #lasttimes        = last_positioned_steps[:,2,1]
  anomalies        = np.zeros((laststeps.shape[0], 1+len(measureMethods)))
  neighs           = []
  consensus        = [np.mean if useMean else np.median for useMean in useMeans]
  for k in range(len(lastids)):
    pos            = lastpositions[k,:].reshape((1,-1))
    step           = laststeps[k,:].reshape((1,-1))
    sel            = allids!=lastids[k]
    posdsts        = np.sqrt(np.power(allpositions[sel,:]-pos, 2).sum(axis=1));
    nn             = min(num_neighs, len(posdsts))
    idxs           = np.argsort(posdsts)[:nn]
    neighs.append(allpositions[sel,:2][idxs,:])
    steps          = allsteps[sel,:][idxs,:]
    if steps.size>0:
      for i, measureMethod in enumerate(measureMethods):
        if   measureMethod==MEASURE_VECTOR_ABS_DIFFERENCE:
          measures     = np.sqrt(np.power(steps-step, 2).sum(axis=1))
        elif measureMethod==MEASURE_ANGLE_DIFFERENCE:
          unit_step    = step /np.sqrt(np.power(step,  2).sum())
          if np.any(np.isnan(unit_step)):
            measures   = np.logical_not(np.all(steps==0, axis=1)).astype(np.float64)
          else:
            unit_steps = steps/np.sqrt(np.power(steps, 2).sum(axis=1)).reshape((-1,1))
            dot        = unit_steps*unit_step
            dot[np.isnan(unit_steps)] = -1
            measures   = np.arccos(np.clip(dot.sum(axis=1), -1.0, 1.0))/np.pi
        elif measureMethod==MEASURE_SPEED_ABS_DIFFERENCE:
          speed        = np.sqrt(np.power(step,  2).sum())
          speeds       = np.sqrt(np.power(steps, 2).sum(axis=1))
          measures     = np.abs(speeds-speed)
        elif measureMethod==MEASURE_SPEED_REL_DIFFERENCE:
          speed        = np.sqrt(np.power(step,  2).sum())
          speeds       = np.sqrt(np.power(steps, 2).sum(axis=1))
          measures     = speed/speeds
          measures[np.isnan(measures)] = 1 if speed==0 else 0
          m            = measures>1
          measures[m]  = 1/measures[m]
          measures     = 1-measures
        elif measureMethod==MEASURE_POLAR_REL_DIFFERENCE:
          speed        = np.sqrt(np.power(step,  2).sum())
          speeds       = np.sqrt(np.power(steps, 2).sum(axis=1))
          spdsrel      = speed/speeds
          spdsrel[np.isnan(spdsrel)] = 1 if speed==0 else 0
          s            = spdsrel>1
          spdsrel[s]   = 1/spdsrel[s]
          spdsrel      = 1-spdsrel
          unit_step    = step /speed
          if np.any(np.isnan(unit_step)):
            angles     = np.logical_not(np.all(steps==0, axis=1)).astype(np.float64)
          else:
            unit_steps = steps/speeds.reshape((-1,1))
            dot        = unit_steps*unit_step
            dot[np.isnan(unit_steps)] = -1
            angles     = np.arccos(np.clip(dot.sum(axis=1), -1.0, 1.0))/np.pi
          measures     = (spdsrel+angles)/2
        assert measures.shape[0]==steps.shape[0]
        assert (len(measures.shape)==1) or (measures.shape[1]==1)
        anomalies[k,i] = consensus[i](measures)
    else:
      anomalies[k,:] = 0
    #if lastids[k] in [25]:
    #  print(f'   MIRA PARA {lastids[k]}: <<{anomalies[k,0]}, {measures}, {step}, {steps}>>')
    anomalies[k,-1] = lastids[k]
    #if lastids[k]==28:
    #  print(f'For {lastids[k]}, step is {step}, anomaly is {anomalies[k,0]}')
  return anomalies, neighs

def cull_anomalies_dict(anomalies_dict, all_anomalies_dict, current_t, diff_t):
  for idx in list(anomalies_dict.keys()):
    v = anomalies_dict[idx]
    if current_t-v[-1][0]>=diff_t:
      all_anomalies_dict[idx] = v
      del anomalies_dict[idx]

def vehicle_is_not_on_image_edge(idx, tracker, imgshapexy, toleranceEdge):
  tr  = tracker.active_traces[idx]
  sz  = tr.sz
  pos = tr.get_last_position()
  mn  = pos-sz
  mx  = pos+sz
  vs  = np.floor(sz*toleranceEdge)
  return np.all(mn>=0) and np.all(mx<imgshapexy) and np.all(mn>=vs) and np.all((imgshapexy-1-mx)>=vs)

def compute_actual_anomalies(ftext, num_anomaly_measures, filter_size, quantiles, frames_for_thresholding, num_persistent, too_big_factor, anomalies_dict, allfiltered, last_anomalies, t, tracker, imgshapexy, toleranceEdge, do_border_correction):
  #anomalies_dict has tracking ids as keys and a list of [t, raw, filtered, thresholded, thresholded_all, thresholded_all_persistent] as values.
  #allfiltered has a new element for each frame
  # first, add raw values
  new_all_filtered = []
  for k in range(last_anomalies.shape[0]):
    idx = last_anomalies[k,-1]
    if idx not in anomalies_dict:
      anomalies_dict[idx] = []
    current_vehicle = anomalies_dict[idx]
    current_vehicle.append([t, last_anomalies[k,:-1], None, None, None, None])
    # also add filtered values if possible
    if len(current_vehicle)>=filter_size and (not do_border_correction or vehicle_is_not_on_image_edge(idx, tracker, imgshapexy, toleranceEdge)):
      raws = tuple(vs[1] for vs in current_vehicle[-filter_size:])
      filtered = np.median(np.vstack(raws), axis=0)
      current_vehicle[-1][2] = filtered
      new_all_filtered.append(filtered)
  #get all relevant filtered values to compute the quantiles
  if len(new_all_filtered)>0:
    allfiltered.append(np.vstack(new_all_filtered))
  else:
    allfiltered.append(np.zeros((0, num_anomaly_measures)))
  filtered_touse = np.vstack(allfiltered[-frames_for_thresholding:])
  assert(filtered_touse.shape[1]==len(quantiles))
  thresholds=None
  if filtered_touse.shape[0]>0:
    # compute a threshold for each anomaly measure
    if False and filtered.size==2:
      thresholds = np.zeros(filtered.shape)
      thresholds[0] = np.quantile(filtered_touse[:,0], quantiles[0])
      thresholded_0 = filtered_touse[:,0]>=thresholds[0]
      thresholds[1] = np.quantile(filtered_touse[thresholded_0,1], quantiles[1])
    else:
      thresholds = np.fromiter((np.quantile(filtered_touse[:,k], quantiles[k]) for k in range(filtered_touse.shape[1])), dtype=np.float64)
    thresholds_str = 'tresholds: '+', '.join(f'{thresholds[k]:0.4}' for k in range(len(thresholds)) )
    # threshold each anomaly measure
    for k in range(last_anomalies.shape[0]):
      idx = last_anomalies[k,-1]
      current_vehicle = anomalies_dict[idx]
      filtered = current_vehicle[-1][2]
      if filtered is not None:
        thresholded = filtered>=thresholds
        thresholded_all = np.all(thresholded)
        current_vehicle[-1][3:5] = thresholded, thresholded_all
        thresholded_alls = False
        if thresholded_all:
          if too_big_factor is not None:
            thresholded_alls = thresholded_alls or (filtered/thresholds)>=too_big_factor
          if num_persistent is not None:
            thresholded_alls = thresholded_alls or (sum(vs[4] is not None and vs[4] for vs in current_vehicle[-num_persistent:])==num_persistent)
          current_vehicle[-1][5] = thresholded_alls
        if thresholded_alls and ftext is not None:
          txt = f'{t}: {idx}\n'
          ftext.write(txt)
  else:
    thresholds_str = 'tresholds: None'
  return thresholds, thresholds_str

def initialize_conf(obj, kwargs, slots, defaults):
  if len(slots)!=len(defaults):
    raise Exception('There MUST be an equal amount of slots and default values!!!')
  for s, d in zip(slots, defaults):
    setattr(obj, s, kwargs.get(s, d))

def print_conf(obj, slots, indent, incr):
  noindent = indent is None
  next     = None if noindent else indent+incr
  name     = type(obj).__name__
  values   = (getattr(obj, s) for s in slots)
  strings  = ((v.pprint(next, incr) if hasattr(v, 'pprint') else repr(v)) for v in values)
  pairs    = (f'{s}={ss}' for s, ss in zip(slots, strings))
  sep      = ', ' if noindent else f',\n{" "*next}'
  prefix   = ''   if noindent else  f'\n{" "*next}'
  posfix   = ''   if noindent else  f'\n{" "*indent}'
  args     = sep.join(pairs)
  return f'{name}({prefix}{args}{posfix})'

class ConfInference:
  __slots__ = 'device', 'weights', 'imgsz', 'allowedClasses', 'add_snowflakes', 'add_rain'
  def __init__(self, **kwargs):
    initialize_conf(self, kwargs, self.__slots__, (0, 'yolov5x6.pt', (1280, 1280), ('car', 'motorcycle', 'bus', 'truck'), False, False))
  def pprint(self, indent=None, incr=2):
    return print_conf(self, self.__slots__, indent, incr)

class ConfIO:
  __slots__ = 'prefixinput', 'videoinput', 'frameskip', 'framespec', 'show', 'prefix', 'saveconf', 'confoutput', 'saveText', 'textfile', 'logfile', 'saveDetections', 'detectfile', 'savedata', 'dataoutput', 'savevideo', 'videooutput', 'outputfps', 'return_results', 'graphname', 'graphsize'
  def __init__(self, **kwargs):
    initialize_conf(self, kwargs, self.__slots__, ('', '', 0, None, False, '', False, '', False, '', '', False, '', False, '', False, '', None, False, '', (19.2,9.86)))
  def pprint(self, indent=None, incr=2):
    return print_conf(self, self.__slots__, indent, incr)

class ConfTracker:
  __slots__ = 'tracker_type', 'byte_args', 'sort_args', 'feat_sim_args', 'mindim', 'scale_diff_limit', 'scale_step_big', 'size_distance_limit_big', 'size_distance_limit_small', 'maximum_distance_to_assign', 'maximum_frames_to_skip_before_set_trace_as_inactive', 'value_to_use_as_inf', 'use_distance_limit_small',
  def __init__(self, **kwargs):
    initialize_conf(self, kwargs, self.__slots__, (0, None, None, None, -1, 0.75, 0.025, 1.25, 0.75, 150, 0, 500000000, True))
  def pprint(self, indent=None, incr=2):
    return print_conf(self, self.__slots__, indent, incr)

class ConfFeatureSimilarity:
  __slots__ = 'usefs', 'backbone', 'last_layer', 'input_size', 'metric', 'threshold', 'weightname', 'just_small_detections'
  def __init__(self, **kwargs):
    initialize_conf(self, kwargs, self.__slots__, (False, 'vgg11_bn', 'features', (224, 224), 'cos', 0.5, 'vgg11_bn-6002323d.pth', True))
  def pprint(self, indent=None, incr=2):
    return print_conf(self, self.__slots__, indent, incr)

class ByteArgs:
  __slots__ = 'mot20', 'track_thresh', 'track_buffer', 'match_thresh'
  def __init__(self, **kwargs):
    initialize_conf(self, kwargs, self.__slots__, (True, 0.25, 1, 0.9))
  def pprint(self, indent=None, incr=2):
    return print_conf(self, self.__slots__, indent, incr)

class SortArgs:
  __slots__ = 'max_age', 'min_hits', 'iou_threshold'
  def __init__(self, **kwargs):
    initialize_conf(self, kwargs, self.__slots__, (1, 1, 0.1))
  def pprint(self, indent=None, incr=2):
    return print_conf(self, self.__slots__, indent, incr)

class ConfAnomaly:
  __slots__ = 'num_neighs', 'num_prev_frames', 'useMeans', 'measureMethods', 'quantiles', 'worryingPersistence', 'worryingScale', 'toleranceEdge', 'do_border_correction'
  def __init__(self, **kwargs):
    initialize_conf(self, kwargs, self.__slots__, (5, 1000000, (True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0,), 1, 1, 0.05, True))
  def pprint(self, indent=None, incr=2):
    return print_conf(self, self.__slots__, indent, incr)

class Conf:
  __slots__ = 'inf', 'io', 'ctr', 'anom'
  def __init__(self, **kwargs):
    initialize_conf(self, kwargs, self.__slots__, (ConfInference(), ConfIO(), ConfTracker(), ConfAnomaly()))
  def pprint(self, indent=None, incr=2):
    return print_conf(self, self.__slots__, indent, incr)

def refreshIO(io, videoinputname, num_prev_frames, quantiles, videoname, framespec, frameskip):
  io.videoinput  = videoinputname
  io.framespec   = framespec
  io.frameskip   = frameskip
  io.detectfile  = f'{videoname}.detections'
  io.confoutput  = f'conf.txt'
  qstr           = '_'.join(f'{q:0.3}' for q in quantiles)
  io.logfile     = f'{videoname}.PF{num_prev_frames}.QUANTILES_{qstr}.log.txt'
  io.textfile    = f'{videoname}.PF{num_prev_frames}.QUANTILES_{qstr}.summary.txt'
  io.dataoutput  = f'{videoname}.PF{num_prev_frames}.QUANTILES_{qstr}.anomalies.npz'
  io.videooutput = f'{videoname}.PF{num_prev_frames}.QUANTILES_{qstr}.result.mp4'
  io.graphname   = f'{videoname}.PF{num_prev_frames}.QUANTILES_{qstr}.%spng'

def makeconf(device, videoinputname, videoname, prefixinput, prefix, show, num_neighs, num_prev_frames, worryingPersistence, worryingScale, useMeans, measureMethods, quantiles, graphsize):
  conf = Conf(
    inf=ConfInference(device=device),
    io=ConfIO(show=show,
              prefixinput=prefixinput,
              prefix=prefix,
              saveconf=True,
              savedata=True,
              saveText=True,
              saveDetections=True,
              savevideo=not show,
              outputfps=5,
              graphsize=graphsize),
    #ctr=ConfTracker(),
    anom=ConfAnomaly(
              num_neighs=num_neighs,
              num_prev_frames=num_prev_frames,
              useMeans=useMeans,
              measureMethods=measureMethods,
              quantiles=quantiles,
              worryingPersistence=worryingPersistence,
              worryingScale=worryingScale))
  refreshIO(conf.io, videoinputname, num_prev_frames, quantiles, videoname, None, 0)
  return conf

class DrawFrame:

  def __init__(self, conf):
    self.mycolors_tuples = ( (255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,0,255), (255,255,0) )
    self.mycolors = tuple(np.array(x).reshape((1,1,3)) for x in self.mycolors_tuples )
    self.whitet   = (255,255,255)
    self.white    = np.array(self.whitet)
    self.redt     = (0,0,255)
    self.red      = np.array(self.redt)
    self.oranget  = (0,165,255)
    self.orange   = np.array(self.oranget)
    self.forange  = self.orange*0.5
    self.alpha    = 0.5
    self.methodId = 0
    if   conf.anom.measureMethods[self.methodId]==MEASURE_VECTOR_ABS_DIFFERENCE:
      self.anom_rectangle_scale = 0.5
    if   conf.anom.measureMethods[self.methodId]==MEASURE_SPEED_ABS_DIFFERENCE:
      self.anom_rectangle_scale = 0.5
    elif conf.anom.measureMethods[self.methodId] in [MEASURE_ANGLE_DIFFERENCE, MEASURE_SPEED_REL_DIFFERENCE, MEASURE_POLAR_REL_DIFFERENCE]:
      self.anom_rectangle_scale = 10

  def draw(self, frame, frame_number, result, tracker, last_anomalies, lastpositions, neighs, anomalies_dict, thresholds_str):
    newframe = frame.copy()
    #print(f'MAKING FRAME {frame_number}, last_anomalies.shape=={last_anomalies.shape}, centers.shape={centers.shape}, all_positioned_steps.shape=={all_positioned_steps.shape}, positioned_steps.shape=={positioned_steps.shape}, last_anomalies=={last_anomalies}')
    # draw rectangles that ARE PAST THE ANOMALOUS THRESHOLD
    for k in range(last_anomalies.shape[0]):
      idx = last_anomalies[k,-1]
      current_vehicle = anomalies_dict[idx][-1]
      color = None
      if current_vehicle[3] is not None and np.any(current_vehicle[3]):
        color = self.forange
      if current_vehicle[4] is not None and current_vehicle[4]:
        #print(f'vehicle {idx}={current_vehicle[4]}: orange')
        color = self.orange
      if current_vehicle[5] is not None and current_vehicle[5]:
        #print(f'vehicle {idx}={current_vehicle[5]}: red')
        color = self.red
      if color is not None:
        sz = np.array(tracker.active_traces[idx].sz)/2
        xy1 = (lastpositions[k,:]-sz).astype(int)
        xy2 = (lastpositions[k,:]+sz).astype(int)
        subimg = frame[xy1[1]:xy2[1], xy1[0]:xy2[0]]
        newframe[xy1[1]:xy2[1], xy1[0]:xy2[0]] = subimg*self.alpha + (1-self.alpha)*color
    # draw anomaly rectangles
    for k in []:#range(last_anomalies.shape[0]):
      ano = last_anomalies[k,0]
      idx = last_anomalies[k,-1]
      col = self.white #self.mycolors[int(idx) % len(self.mycolors)]
      rad = int(ano*self.anom_rectangle_scale)
      xy1 = (lastpositions[k,:]-rad).astype(int)
      xy2 = (lastpositions[k,:]+rad).astype(int)
      subimg = frame[xy1[1]:xy2[1], xy1[0]:xy2[0]]
      newframe[xy1[1]:xy2[1], xy1[0]:xy2[0]] = subimg*self.alpha + (1-self.alpha)*col
    #draw updated traces
    for k in []:#range(last_anomalies.shape[0]):
      idx = last_anomalies[k,-1]
      tr = tracker.active_traces[idx]
      ps = tr.positions
      if np.any(ps[-1][-1]!=lastpositions[k,:]):
        print(f'MIRA: last ps=={ps[-1][-1]}, laspositions=={lastpositions[k,:]}')
      col = self.whitet #self.mycolors_tuples[int(idx % len(self.mycolors_tuples))]
      pprev = [None]
      pi = None
      for p in reversed(ps):
        if p[-1] is not None:
          pip = pi
          pi = p[-1].astype(int)
          cv2.circle(newframe, (pi[0], pi[1]), 3, col, -1)
          if pprev[-1] is not None:
            cv2.line(newframe, (pi[0], pi[1]), (pip[0], pip[1]), col, 1)
        pprev = p
    #draw yolo predictions
    for k in range(result.shape[0]):
      cv2.rectangle(newframe, (int(result[k,0]),int(result[k,1])), (int(result[k,2]),int(result[k,3])), (0, 165, 255), 3)
    #draw connections to neighbours
    for k in range(last_anomalies.shape[0]):
      ngs = neighs[k]
      for n in ngs:
        cv2.line(newframe, (int(lastpositions[k,0]), int(lastpositions[k,1])), (int(n[0]), int(n[1])), (255,165,0), 1)
    #draw inactive traces
    for idx in []:#tracker.inactive_traces.keys():
      tr = tracker.inactive_traces[idx]
      ps = tr.positions
      col = self.mycolors_tuples[int(idx % len(self.mycolors_tuples))]
      pprev = None
      pi = None
      for p in reversed(ps):
        if p[-1] is not None:
          pip = pi
          pi = p[-1].astype(int)
          cv2.circle(newframe, (pi[0], pi[1]), 3, col, -1)
          if pprev[-1] is not None:
            cv2.line(newframe, (pi[0], pi[1]), (pip[0], pip[1]), col, 1)
        pprev = p
    # draw active paths that have NOT been updated in this frame
    for idx in []:#tracker.active_traces.keys():
      tr = tracker.active_traces[idx]
      ps = tr.positions
      if ps[-1][-1] is not None:
        continue
      #import code; code.interact(local=locals())
      col = self.mycolors_tuples[int(idx % len(self.mycolors_tuples))]
      pprev = None
      pi = None
      for p in reversed(ps):
        if p[-1] is not None:
          pip = pi
          pi = p[-1].astype(int)
          cv2.circle(newframe, (pi[0], pi[1]), 3, (0,0,0), -1)
          if pprev[-1] is not None:
            cv2.line(newframe, (pi[0], pi[1]), (pip[0], pip[1]), (0,0,0), 1)
        pprev = p
    # draw text
    for k, tr in tracker.active_traces.items():
      ps = tr.get_last_not_None_position().astype(int)
      #last are thickness and lineType
      cv2.putText(newframe, f'{k}', (ps[0], ps[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, 2)
      cv2.putText(newframe, f'{k}', (ps[0], ps[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, 2)
    # draw frame number
    cv2.putText(newframe, f'{frame_number}. {thresholds_str}', (0, newframe.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, 2)
    cv2.putText(newframe, f'{frame_number}. {thresholds_str}', (0, newframe.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, 2)
    return newframe

def showIMG(imgs):
  for i, img in enumerate(imgs):
    cv2.namedWindow(f'n{i}', cv2.WINDOW_NORMAL)
    cv2.imshow(f'n{i}', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def anomaly_detection(conf):
  inf  = conf.inf
  io   = conf.io
  ctr  = conf.ctr
  anom = conf.anom
  if len(io.prefix)>0:
    os.system(f'mkdir -p {conf.io.prefix}')

  augment = inf.add_snowflakes or inf.add_rain
  if augment:
    import imgaug
    aug_s = imgaug.augmenters.Snowflakes(flake_size=(0.5, 0.8), speed=(0.001, 0.03), density=(0.005, 0.01))
    aug_r = imgaug.augmenters.Rain(drop_size=(0.10, 0.20), speed=(0.04, 0.12))

  confidenceBYTE = False
  useXYXYs = True
  mindim = ctr.mindim
  feat_sim = False
  if ctr.tracker_type==TRACKER_CLASSIC:
    if ctr.feat_sim_args.usefs:
      feat_sim = True
      device = select_device(inf.device)
      featExtr = sim_util.getFeatureExtractor(ctr.feat_sim_args.backbone, device, input_size=ctr.feat_sim_args.input_size, output_layer=ctr.feat_sim_args.last_layer, weights=ctr.feat_sim_args.weightname)
    tracker = ClassicTracker(ctr.scale_diff_limit, ctr.scale_step_big, ctr.size_distance_limit_big, ctr.size_distance_limit_small, ctr.maximum_distance_to_assign, ctr.maximum_frames_to_skip_before_set_trace_as_inactive, ctr.value_to_use_as_inf, feat_sim_args=ctr.feat_sim_args, use_distance_limit_small=ctr.use_distance_limit_small)
    useXYXYs = False
  elif ctr.tracker_type==TRACKER_BYTE:
    tracker = TrackerBYTE(ctr.scale_diff_limit, ctr.scale_step_big, ctr.size_distance_limit_big, ctr.size_distance_limit_small, ctr.maximum_distance_to_assign, ctr.maximum_frames_to_skip_before_set_trace_as_inactive, ctr.value_to_use_as_inf, byte_args=ctr.byte_args)
    confidenceBYTE = True
  elif ctr.tracker_type==TRACKER_SORT:
    tracker = TrackerSORT(ctr.scale_diff_limit, ctr.scale_step_big, ctr.size_distance_limit_big, ctr.size_distance_limit_small, ctr.maximum_distance_to_assign, ctr.maximum_frames_to_skip_before_set_trace_as_inactive, ctr.value_to_use_as_inf, sort_args=ctr.sort_args)
  #conf_threshold = 0.25
  #conf_threshold = 0.5
  conf_threshold = ctr.byte_args.track_thresh
  df = io.prefixinput+io.detectfile+'.npz'
  #df = io.prefixinput+io.detectfile+f'{conf_threshold}'+'.npz'
  #df = io.prefixinput+io.detectfile+'_byte'+'.npz'
  already_results = False
  if os.path.isfile(df) and not augment:
    d = np.load(df, allow_pickle=True)
    results = d['results']
    already_results = True
  else:
    model, stride, names, device = load_weights_for_streamlined(weights=inf.weights, device=inf.device)
    allowed = [names.index(x) for x in inf.allowedClasses]
    if io.saveDetections:
      results = []

  cap = cv2.VideoCapture(io.prefixinput+io.videoinput)
  if (cap.isOpened()== False):
    #Si no se puede encontrar el vídeo, muestra error.
    print(f"Error al abrir el vídeo: {io.prefixinput}{io.videoinput}')")
    exit()

  #Obtención de los fps y los frames totales del vídeo
  fps = cap.get(cv2.CAP_PROP_FPS)
  if io.outputfps is not None:
    fps = io.outputfps
  frames_totales = cap.get(cv2.CAP_PROP_FRAME_COUNT)

  ok, frame = cap.read()
  frame_number              = 0
  positioned_steps_by_frame = []
  anomalies_by_frame        = []
  allfiltered               = []
  thresholds_by_frame       = []
  anomalies_dict            = dict()
  median_filter_size        = 3
  #all_anomalies_dict        = dict()
  #time_to_remove_tracking   = 3

  if io.saveconf:
    with open(io.prefix+io.confoutput, 'w') as f:
      f.write(conf.pprint(indent=0, incr=2))

  if io.saveText:
    ftext = open(io.prefix+io.textfile, 'w')
  else:
    ftext = None
  if io.show or io.savevideo:
    drawer = DrawFrame(conf)
  if io.savevideo:
    outv = None
  if io.show:
    cv2.namedWindow('n', cv2.WINDOW_NORMAL)

  if WRITE_TO_COMMON_FILE:
    with open(COMMONFILE, 'a') as f:
      f.write(f'STARTING WITH {io.prefixinput+io.videoinput}\n')

  use_frameskip = io.frameskip > 0
  use_framespec = io.framespec is not None
  not_use_max_frame_number = not use_framespec
  if use_framespec:
    to_skip, max_frame_number = io.framespec
    not_use_max_frame_number = max_frame_number is None
    if to_skip is not None:
      for _ in range(to_skip):
        ok, frame = cap.read()

  times = []
  while (cap.isOpened() and (not_use_max_frame_number or (frame_number<max_frame_number))):# and frame_number<80):

    t0 = time.time()
    #Obtención del siguiente frame
    ok, frame = cap.read()
    if use_frameskip:
      for _ in range(io.frameskip):
        ok, frame = cap.read()
    t1 = time.time()

    #Si el frame se ha podido obtener
    if not ok:
      break

    if inf.add_snowflakes:
      frame = aug_s(images=[frame])[0]
    elif inf.add_rain:
      frame = aug_r(images=[frame])[0]

    if already_results:
      result = results[frame_number]
      if not confidenceBYTE:
        thresh = result[:,4]>=conf_threshold
        result = result[thresh,:]
        del thresh
    else:
      if confidenceBYTE:
        result = run_streamlined(model=model, conf_thres=0.1, stride=stride, imgsz=inf.imgsz, device=device, sourceimgs=[frame], classes=allowed, agnostic_nms=True)
      else:
        result = run_streamlined(model=model, conf_thres=conf_threshold, stride=stride, imgsz=inf.imgsz, device=device, sourceimgs=[frame], classes=allowed, agnostic_nms=True)
      result = np.squeeze(np.array(result, dtype=np.float64), axis=0)
      if result.size==0:
        result = np.empty((0,6), dtype=np.float64)
      if io.saveDetections and not augment:
        results.append(result)
    centers = (result[:,[0,1]]+result[:,[2,3]])/2
    szs = result[:,[2,3]]-result[:,[0,1]]
    boxesFeats = None
    t2 = time.time()
    if feat_sim:
      rs = result[:,0:4].astype(np.int32)
      if rs.shape[0]>0:
        imgboxes = [frame[rs[k,1]:rs[k,3], rs[k,0]:rs[k,2], :] for k in range(result.shape[0])]
        #for i, imgb in enumerate(imgboxes):
        #  cv2.imwrite(f'{baseprefix}boxes/box_T{frame_number:03d}_BB{i:03d}.png', imgb)
        #showIMG([frame, imgboxes[0], imgboxes[1], imgboxes[2], imgboxes[3], imgboxes[4], imgboxes[5]])
        #import code; code.interact(local=locals())
        #boxesFeats = np.zeros((len(imgboxes), 7*7*512))
        boxesFeats = featExtr(imgboxes, device).cpu().detach().numpy()
      else:
        boxesFeats = np.zeros((0,0), dtype=np.float32)
    t3 = time.time()
    imgshapexy = np.array((frame.shape[1], frame.shape[0]), dtype=np.float64)
    if mindim>0:
      maxd    = imgshapexy.max()
      mind    = szs.min(axis=1)
      dims    = mind/maxd
      tokeep  = dims>=mindim
      result  = result[tokeep,:]
      centers = centers[tokeep,:]
      szs     = szs[tokeep,:]
      del maxd, mind, dims, tokeep
    if useXYXYs:
      xyxys = result[:,:4]
      if ctr.tracker_type==TRACKER_BYTE:
        confs = result[:,4]
        tracker.assign_incomming_positions(frame_number, confs,      xyxys, centers, szs, np.max(imgshapexy))#, other)
      else:
        tracker.assign_incomming_positions(frame_number, xyxys, centers, szs, np.max(imgshapexy))#, other)
    else:
      #other = result[:,[4,5]]
      tracker.assign_incomming_positions(frame_number, boxesFeats, centers, szs, np.max(imgshapexy))#, other)
    positioned_steps = tracker.get_active_traces_steps()
    positioned_steps_by_frame.append(positioned_steps)
    t4 = time.time()

    nf = min(frame_number+1, anom.num_prev_frames)
    all_positioned_steps = np.concatenate(positioned_steps_by_frame[-nf:], axis=0)
    last_anomalies, neighs = compute_raw_anomalies(all_positioned_steps, positioned_steps, anom.num_neighs, anom.useMeans, anom.measureMethods)
    anomalies_by_frame.append(last_anomalies)
    t5 = time.time()
    #cull_anomalies_dict(anomalies_dict, all_anomalies_dict, frame_number, time_to_remove_tracking)
    thresholds, thresholds_str = compute_actual_anomalies(
             ftext, len(anom.measureMethods), median_filter_size,
             anom.quantiles, nf, anom.worryingPersistence, anom.worryingScale,
             anomalies_dict, allfiltered, last_anomalies, frame_number,
             tracker, imgshapexy, anom.toleranceEdge, anom.do_border_correction)
    thresholds_by_frame.append(thresholds)
    lastpositions = positioned_steps[:,0,:]
    t6 = time.time()

    if frame_number==0:
      height, width, _ = frame.shape
    if io.show or io.savevideo:
      newframe = drawer.draw(frame, frame_number, result, tracker, last_anomalies, lastpositions, neighs, anomalies_dict, thresholds_str)
      if io.savevideo:
        if outv is None:
          height, width, _ = frame.shape
          outv = cv2.VideoWriter(io.prefix+io.videooutput,cv2.VideoWriter_fourcc(*'H264'), fps, (width,height))
        outv.write(newframe)
      if io.show:
        cv2.imshow('n', newframe)
        cv2.waitKey(0)

    frame_number += 1
    print(f'{frame_number}/{frames_totales}')
    t7 = time.time()
    times.append((t1-t0, t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6))

  if io.saveText:
    ftext.close()
  with open(io.prefix+io.logfile, 'w') as f:
    f.write('frame acquisition,vehicle detection,feature acquisition,tracking,compute Ai(t),anomaly detection,processed frame drawing\n')
    for ts in times:
      f.write(f'{ts[0]},{ts[1]},{ts[2]},{ts[3]},{ts[4]},{ts[5]},{ts[6]}\n')
    f.write('means\n')
    times = np.array(times, dtype=np.float64)
    ms = times.mean(axis=0)
    f.write(f'{ms[0]},{ms[1]},{ms[2]},{ms[3]},{ms[4]},{ms[5]},{ms[6]}\n')
    f.write('totals\n')
    ss = times.sum(axis=0)
    f.write(f'{ss[0]},{ss[1]},{ss[2]},{ss[3]},{ss[4]},{ss[5]},{ss[6]}\n')
    f.write('total\n')
    f.write(f'{ss.sum()}\n')
  if not already_results and io.saveDetections and not augment:
    results = np.array(results, dtype=object)
    np.savez(df, results=results)
  if io.savevideo:
    outv.release()
  if io.show:
    cv2.destroyAllWindows()
  if io.savedata:
    np.savez(io.prefix+io.dataoutput,
             height=height,
             width=width,
             positioned_steps_by_frame=np.array(positioned_steps_by_frame, dtype=object),
             anomalies_by_frame=np.array(anomalies_by_frame, dtype=object),
             anomalies_dict=np.array(anomalies_dict, dtype=object),
             #anomalies_dict=all_anomalies_dict,
             thresholds_by_frame=np.array(thresholds_by_frame, dtype=object))
  if io.return_results:
    return ((height, width), positioned_steps_by_frame, anomalies_by_frame)
  else:
    return None

def medfiltered(rad, segments):
  newsegments = []
  for j in range(1, len(segments)-1):
    t = segments[j][0]
    md = np.median([r[1] for r in segments[j-rad:j+rad+1] ])
    newsegments.append([t, md])
  return newsegments

def remove_empty(segments, cols):
  return zip(*((s, c) for s, c in zip(segments, cols) if len(s)>0))

def show_anomaly_values_original(interactive, savesiz, figname, anomalies_by_frame, idxs_red, idxs_green):
  import matplotlib
  if interactive:
    matplotlib.use('TkAgg')
  import matplotlib.pyplot as plt
  nt = len(anomalies_by_frame)
  anomaly_tracks = dict()
  mn =  np.inf
  mx = -np.inf
  for k in range(nt):
    anomalies = anomalies_by_frame[k]
    for j in range(anomalies.shape[0]):
      idx = anomalies[j,-1]
      v   = anomalies[j,0]
      p   = (k, v)
      mn  = min(mn, v)
      mx  = max(mx, v)
      if idx in anomaly_tracks:
        anomaly_tracks[idx][0].append(p)
      else:
        if idx in idxs_red:
          color = 'r'
        elif idx in idxs_green:
          color = 'g'
        else:
          color = 'b'
        anomaly_tracks[idx] = ([p], color)
  anomaly_segments, anomaly_colors = zip(*[v for v in anomaly_tracks.values() if len(v[0])>1])
  anomaly_segments_mf1 = [medfiltered(1, seg) for seg in anomaly_segments]
  anomaly_segments_mf1, anomaly_colors_mf1 = remove_empty(anomaly_segments_mf1, anomaly_colors)
  to_present = ( (anomaly_segments, anomaly_colors, 'unfiltered.'), (anomaly_segments_mf1, anomaly_colors_mf1, 'filt1.') )
  #print(f'Number of tracks: {len(anomaly_segments)}')
  #import code; code.interact(local=locals())
  figs = []
  for segs, cols, name in to_present:
    f = plt.figure()
    a = plt.gca()
    a.set_xlim(0, nt)
    a.set_ylim(mn-abs(mn*0.05), mx+abs(mx*0.05))
    anos = matplotlib.collections.LineCollection(segs, colors=cols)
    a.add_collection(anos)
    if interactive:
      figs.append(f)
    else:
      plt.draw()
      f.set_size_inches(*savesiz)
      f.savefig(figname % name)
  if interactive:
    plt.show()

def checkTrackLengths(anomalies_dict):
  lens1 = [any(len(v)>2 and v[-1] for v in vs) for (idx, vs) in anomalies_dict.items() if len(vs)>10]
  lens2 = [any(len(v)>2 and v[-1] for v in vs) for (idx, vs) in anomalies_dict.items()]
  lens3 = [idx for (idx, vs) in anomalies_dict.items() if any(len(v)>2 and v[-1] for v in vs)]

  return len(lens1), sum(lens1), len(lens2), sum(lens2), lens3

def show_anomaly_values_quantiles(interactive, savesiz, figname, anomalies_dict, measure_methods, use_means, thresholds_by_frame, threshold, legend_loc):
  import matplotlib
  if interactive:
    matplotlib.use('TkAgg')
  import matplotlib.pyplot as plt
  anomaly_points_by_color   = []
  anomaly_segments_by_color = []
  anomaly_colors            = ('b', '#FFA500', 'r')#((0,0,255), (255,165,0), (255,0,0))
  anomaly_widths            = ((3, 6), (3, 6), (3, 9))#(1.5, 1.5, 3)
  thresholds_segments       = []
  thresholds_segments2      = []
  for measure_method in measure_methods:
    anomaly_segments_by_color.append(([], [], []))
    anomaly_points_by_color.append(([], [], []))
    thresholds_segments.append([])
    thresholds_segments2.append([])
  mn  = [ np.inf]*len(measure_methods)
  mx  = [-np.inf]*len(measure_methods)
  mxt =  -np.inf
  #build threshold lines (one set for each plot)
  prev_thresholds = None
  for t, thresholds in enumerate(thresholds_by_frame):
    for k in range(len(measure_methods)):
      if thresholds is not None:
        v = thresholds[k]
        p1 = (t, v)
        if prev_thresholds is None:
          thresholds_segments[k].append([p1])
        else:
          thresholds_segments[k][-1].append(p1)
        if threshold is not None:
          p2 = (t, v*threshold)
          if prev_thresholds is None:
            thresholds_segments2[k].append([p2])
          else:
            thresholds_segments2[k][-1].append(p2)
    prev_thresholds = thresholds
  #build anomaly lines for all plots. This is more complex because we have different lines with different colors, and we must connect them whenever a color change happens
  for idx, vals in anomalies_dict.items():
    prev_i      = [None, None, None]
    i           = [None, None, None]
    prev_p      = [None, None, None]
    for t, anomval, filtered, thresholded, thresholded_all, thresholded_alls in vals:
      mxt       = max(mxt, t)
      if filtered is None:
        i       = [None, None, None]
        prev_i  = [None, None, None]
        prev_p  = [None, None, None]
      else:
        for k in range(len(measure_methods)):
          v     = filtered[k]
          mn[k] = min(mn[k], v)
          mx[k] = max(mx[k], v)
          p     = (t, v)
          if not thresholded[k]:
            i[k]=0
          elif not thresholded_alls:
            i[k]=1
          else:
            i[k]=2
          anomaly_points_by_color[k][i[k]].append(p)
          if i[k]!=prev_i[k]:
            #anomaly_segments_by_color[k][i[k]].append([p])
            if prev_i[k] is None:
              anomaly_segments_by_color[k][i[k]].append([p])
            else:
              anomaly_segments_by_color[k][i[k]].append([prev_p[k], p])
            prev_i[k] = i[k]
          else:
            anomaly_segments_by_color[k][i[k]][-1].append(p)
          prev_p[k] = p
  point_names = ['non-anomalous $A_{i}(t)$', 'potentially anomalous $A_{i}(t)$', 'anomalous $A_{i}(t)$']
  fontsize = 18 #'large'
  to_present = [(thresholds_segments[k], thresholds_segments2[k], anomaly_segments_by_color[k], anomaly_points_by_color[k], mn[k], mx[k], f'{MEASURE_NAMES[measure_method]}_{"mean" if use_mean else "median"}.') for k, (measure_method, use_mean) in enumerate(zip(measure_methods, use_means))]
  #print(f'Number of tracks: {len(anomaly_segments)}')
  if np.isinf(mxt):
    mxt = 1
  figs = []
  put_legend = legend_loc is not None
  for thresh_segs, thresh_segs2, segs_by_color, pts_by_color, mni, mxi, name in to_present:
    if np.isinf(mni):
      mni = 0
      mxi = 1
    f = plt.figure()
    a = plt.gca()
    a.set_xlim(0, mxt)
    a.set_ylim(mni-abs(mni*0.05), mxi+abs(mxi*0.05))
    #import code; code.interact(local=locals())
    pointset=[]
    for segs, col, pts, (width, msize) in zip(segs_by_color, anomaly_colors, pts_by_color, anomaly_widths):
      #anos = matplotlib.collections.LineCollection(segs, colors=col, linewidths=width)
      anos = matplotlib.collections.LineCollection(segs, colors='gray', linewidths=width)
      a.add_collection(anos)
      ptss = np.array(pts)
      if ptss.size>0:
        pointset.append(plt.plot(ptss[:,0], ptss[:,1], '.', color=col, markersize=msize)[0])
    threshs = matplotlib.collections.LineCollection(thresh_segs, colors='g', linewidths=width)
    a.add_collection(threshs)
    if threshold is not None:
      threshs2 = matplotlib.collections.LineCollection(thresh_segs2, colors='lime', linewidths=width)
      a.add_collection(threshs2)
    if put_legend:
      handles = pointset+[threshs]
      labels  = point_names+['$P_{95}$ percentile']
      if thresholds is not None:
        handles.append(threshs2)
        labels .append('4 times $P_{95}$ percentile')
      plt.legend(handles, labels, fontsize=fontsize, loc=legend_loc)
    plt.xlabel('$t$', fontsize=fontsize)
    plt.ylabel('$A_{i}(t)$', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tick_params(axis='both', which='minor', labelsize=fontsize)
    if interactive:
      figs.append(f)
    else:
      plt.draw()
      f.set_size_inches(*savesiz)
      f.savefig(figname % name)
  if interactive:
    print(figname % name)
    plt.show()
  else:
    plt.close('all')
    #for f in figs:
    #  plt.close(f)

def main_run(mainconf, action, args):
  graph_siz   = (19.2,9.86)
  show        = False
  #for measure_method in [[MEASURE_VECTOR_ABS_DIFFERENCE], [MEASURE_ANGLE_DIFFERENCE], [MEASURE_SPEED_ABS_DIFFERENCE], [MEASURE_SPEED_REL_DIFFERENCE], [MEASURE_POLAR_REL_DIFFERENCE]]:
  anomaly_confs = (
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.99,), 60, 3),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.98,), 60, 3),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.95,), 60, 3),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.90,), 60, 3),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.85,), 60, 3),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.99,), 60, 4),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.98,), 60, 4),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.95,), 60, 4),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.90,), 60, 4),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.85,), 60, 4),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.99,), 60, 5),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.98,), 60, 5),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.95,), 60, 5),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.90,), 60, 5),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.85,), 60, 5),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.99,), 60, 6),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.98,), 60, 6),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.95,), 60, 6),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.90,), 60, 6),
                    ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.85,), 60, 6),
                  )
  anomaly_confs = ( ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.95,), 60, 4), )#((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.95,), 20), )
  #anomaly_confs = ( ((True,), (MEASURE_VECTOR_ABS_DIFFERENCE,), (0.95,), 20), ((True,True), (MEASURE_SPEED_ABS_DIFFERENCE,MEASURE_ANGLE_DIFFERENCE), (0.95,0.95), 20) )
  #anomaly_confs = ( ((True,True), (MEASURE_SPEED_ABS_DIFFERENCE,MEASURE_ANGLE_DIFFERENCE), (0.95,0.95), 10), )
  num_neighs = 5
  byte_args = ByteArgs(mot20=True, track_thresh=0.25, track_buffer=5, match_thresh=0.9)
  sort_args = SortArgs(max_age=5, min_hits=1, iou_threshold=0.1)

  feat_sim_args = ConfFeatureSimilarity(usefs=args.feat_sim_use, metric=args.feat_sim_metric,
                 threshold=args.feat_sim_threshold,
                 just_small_detections=args.feat_sim_just_small_detections,
                 **modelZoo[args.feat_sim_backbone])
  if feat_sim_args.usefs:
    if feat_sim_args.just_small_detections:
      feat_sim_args_affected_vehicles = '_smallvehicles'
    else:
      feat_sim_args_affected_vehicles = '_allvehicles'
    feat_sim_args_str = f'{feat_sim_args.backbone}_thresh_{feat_sim_args.threshold}{feat_sim_args_affected_vehicles}'
  else:
    feat_sim_args_str = '_no_features'
  if args.do_border_correction:
    border_correction_str = '_with_border_correction'
  else:
    border_correction_str = '_no_border_correction'
  use_means = (args.use_mean,)
  measure_methods = ([i for i, x in enumerate(MEASURE_NAMES) if x == args.measure_method][0],)
  quantiles = (args.quantile,)
  worryingPersistence = args.worryingPersistence
  worryingScale = args.worryingScale
  #for num_neighs in [5]:
  #for num_neighs in [50]:
  #for num_neighs in [5, 10, 50, 100]:
  #for num_neighs in [5, 50]:
  if True:
  #for tracker_type, track_thresh, buf, mindim, trkstr in [
       #(TRACKER_SORT,    0.25, 1,    0.025, '_sort'),
       #(TRACKER_BYTE,    0.25, 1,    0.025, '_byte'),
       #(TRACKER_SORT,    0.25, 5,    0.025, '_sort'),
       #(TRACKER_BYTE,    0.25, 5,    0.025, '_byte'),
      # (TRACKER_CLASSIC, 0.25, None, 0.025, '_classic'),
       #(TRACKER_SORT,    0.25, 1,    0.02, '_sort'),
       #(TRACKER_BYTE,    0.25, 1,    0.02, '_byte'),
       #(TRACKER_SORT,    0.25, 5,    0.02, '_sort'),
       #(TRACKER_BYTE,    0.25, 5,    0.02, '_byte'),
       #(TRACKER_CLASSIC, 0.25, None, 0.02, '_classic'),
       #(TRACKER_SORT,    0.25, 1,    0.015, '_sort'),
       #(TRACKER_BYTE,    0.25, 1,    0.015, '_byte'),
       #(TRACKER_SORT,    0.25, 5,    0.015, '_sort'),
       #(TRACKER_BYTE,    0.25, 5,    0.015, '_byte'),
      # (TRACKER_CLASSIC, 0.25, None, 0.015, '_classic'),

       #(TRACKER_CLASSIC, 0.25, None, 0,     '_classic'),
       #]:
    if   args.tracker_type=='classic':
      tracker_type = TRACKER_CLASSIC
    elif args.tracker_type=='byte':
      tracker_type = TRACKER_BYTE
    elif args.tracker_type=='sort':
      tracker_type = TRACKER_SORT
    else:
      raise Exception(f'Tracker type not understood: {args.tracker_type}')
    trkstr='_'+args.tracker_type
    track_thresh = 0.25
    buf = 1
    mindim = 0
    byte_args.track_thresh = track_thresh
    byte_args.track_buffer = buf
    sort_args.max_age      = buf
    if tracker_type in [TRACKER_SORT, TRACKER_BYTE]:
      trkstr = f'{trkstr}_BUF{buf}'
    trkstr = f'{trkstr}_THR{track_thresh}_mindim{mindim}'
    if True:
    #for use_means, measure_methods, quantiles, worryingPersistence, worryingScale in anomaly_confs:
        anomstr = '_'.join(f'{MEASURE_NAMES[measure_method]}_{"mean" if use_mean else "median"}' for measure_method, use_mean in zip(measure_methods, use_means))
        prefix = f'{baseprefix}experiments/quantiles_{anomstr}_N{num_neighs:03}_P{worryingPersistence:02}_S{worryingScale}_Percentile{quantiles[0]}{border_correction_str}{trkstr}_{feat_sim_args_str}_use_distance_limit_small_{args.use_distance_limit_small}/'
        prefix = f'{baseprefix}experiments/quantiles_{anomstr}_N{num_neighs:03}_P{worryingPersistence:02}_S{worryingScale}_Percentile{quantiles[0]}{border_correction_str}{trkstr}_{feat_sim_args_str}_use_distance_limit_small_{args.use_distance_limit_small}/'
        #prefix = f'{baseprefix}experiments/quantiles_{anomstr}_N{num_neighs:03}_P{worryingPersistence:02}_S{worryingScale}_no_border_correction/'
        print(f'FOR PREFIX {prefix}')
        conf = makeconf(args.device, '', '', mainconf['prefixinput'], prefix, show, num_neighs, 0, worryingPersistence, worryingScale, use_means, measure_methods, quantiles, graph_siz)
        conf.ctr.use_distance_limit_small = args.use_distance_limit_small
        conf.ctr.tracker_type = tracker_type
        conf.ctr.byte_args = byte_args
        conf.ctr.sort_args = sort_args
        conf.ctr.mindim    = mindim
        conf.ctr.feat_sim_args = feat_sim_args
        conf.anom.do_border_correction = args.do_border_correction
        conf.inf.add_rain = args.addRain
        conf.inf.add_snowflakes = args.addSnow
        for videoinputname, videoname, frameskip, framespec, num_prev_frames, red_ids, plot_scaled_percentile, legend_loc in mainconf['videospecs']:
          if conf.inf.add_snowflakes:
            videoname += '_snow'
          elif conf.inf.add_rain:
            videoname += '_rain'
          threshold = conf.anom.worryingScale if plot_scaled_percentile else None
          conf.anom.num_prev_frames = num_prev_frames
          refreshIO(conf.io, videoinputname, num_prev_frames, quantiles, videoname, framespec, frameskip)
          if action in ['run', 'both']:
            anomaly_detection(conf)
          if action in ['graph', 'both']:
            d = np.load(conf.io.prefix+conf.io.dataoutput, allow_pickle=True)
            #show_anomaly_values_original(conf.io.show, conf.io.graphsize, conf.io.prefix+conf.io.graphname, d['anomalies_by_frame'], red_ids, [])
            show_anomaly_values_quantiles(conf.io.show, conf.io.graphsize, conf.io.prefix+conf.io.graphname, d['anomalies_dict'].item(), measure_methods, use_means, d['thresholds_by_frame'], threshold, legend_loc)
            print(f"Para {videoname}: {checkTrackLengths(d['anomalies_dict'].item())}\n")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
  for k, v in default_dict.items():
    v_type = type(v)
    if v is None:
      v_type = str
    elif isinstance(v, bool):
      v_type = str2bool
    parser.add_argument(f"--{k}", default=v, type=v_type)

weightpath  = f'{baseprefix}torchpoints/'
modelZoo = {
  'resnet18':         {'input_size': (224, 224), 'last_layer': 'layer4', 'backbone': 'resnet18', 'weightname': f'{weightpath}resnet18-f37072fd.pth'},
  'resnet34':         {'input_size': (224, 224), 'last_layer': 'layer4', 'backbone': 'resnet34', 'weightname': f'{weightpath}resnet34-b627a593.pth'},
  'resnet50':         {'input_size': (224, 224), 'last_layer': 'layer4', 'backbone': 'resnet50', 'weightname': f'{weightpath}resnet50-0676ba61.pth'},
  'resnet101':        {'input_size': (224, 224), 'last_layer': 'layer4', 'backbone': 'resnet101', 'weightname': f'{weightpath}resnet101-63fe2227.pth'},
  'resnet152':        {'input_size': (224, 224), 'last_layer': 'layer4', 'backbone': 'resnet152', 'weightname': f'{weightpath}resnet152-394f9c45.pth'},
  'wide_resnet50_2':  {'input_size': (224, 224), 'last_layer': 'layer4', 'backbone': 'wide_resnet50_2', 'weightname': f'{weightpath}wide_resnet50_2-95faca4d.pth'},
  'wide_resnet101_2': {'input_size': (224, 224), 'last_layer': 'layer4', 'backbone': 'wide_resnet101_2', 'weightname': f'{weightpath}wide_resnet101_2-32ee1156.pth'},
  'vgg11_bn':         {'input_size': (224, 224), 'last_layer': 'features', 'backbone': 'vgg11_bn', 'weightname': f'{weightpath}vgg11_bn-6002323d.pth'},
  'vgg13_bn':         {'input_size': (224, 224), 'last_layer': 'features', 'backbone': 'vgg13_bn', 'weightname': f'{weightpath}vgg13_bn-abd245e5.pth'},
  'vgg16_bn':         {'input_size': (224, 224), 'last_layer': 'features', 'backbone': 'vgg16_bn', 'weightname': f'{weightpath}vgg16_bn-6c64b313.pth'},
  'vgg19_bn':         {'input_size': (224, 224), 'last_layer': 'features', 'backbone': 'vgg19_bn', 'weightname': f'{weightpath}vgg19_bn-c79401a0.pth'},
}

def create_argparser():
  defaults = dict(
    device='0',
    feat_sim_use=True,
    feat_sim_metric='cos', #'L2'
    feat_sim_threshold=0.3,
    feat_sim_just_small_detections=True,
    feat_sim_backbone='resnet18',
    do_border_correction=True,
    use_distance_limit_small=True,
    use_mean=True,
    measure_method='vector',
    quantile=0.95,
    worryingPersistence=60,
    worryingScale=4,
    subdataset_indexes='',
    #subdataset_indexes='[(0,(0,1,2,3)), (1,(0,1,2,3)), (2,(0,1,2)), (3,(0,1,2)), (4,(0,1,2,3,4,5)), (5,(0,1,2,3,4,5))]',
    addSnow=False,
    addRain=False,
    tracker_type='classic',
  )
  parser = argparse.ArgumentParser()
  add_dict_to_argparser(parser, defaults)
  return parser

"""
import os
def renameFiles(path, pattern='i%07d.'):
  for i, p in enumerate(sorted(os.listdir(path))):
    if i==0:
      ext = p.rsplit('.', 1)[1]
      pattern = pattern+ext
    p1 = os.path.join(path, p)
    p2 = os.path.join(path, pattern % i)
    os.rename(p1, p2)
"""

if __name__ == '__main__':
  args = create_argparser().parse_args()
  #mainconf0 = {
  #  'prefixinput': f'{baseprefix}media/originales',
  #  'videospecs': [
  #    ('video1.mp4', 'original_video1', 0, None, 10000, [24], False, None),
  #    #('video2.mp4', 'original_video2',   0, None, 300, [136, 146], True, 0),
  #    #('video3.mp4', 'original_video3',   0, None, 300, [97, 99, 106], True, 0),
  #    #('video4.mp4', 'original_video4',   0, None, 300, [142, 147, 172], True, 0)
  #]}
  mainconf0 = {
    'prefixinput': f'{baseprefix}media/originales/',
    'videospecs': [
      #('video1.mp4', 'original_video1',    0, None, 50, [24]),
      ('video1.mp4', 'original_video1', 0, None, 10000, [24], False, None),
      ('video2.mp4', 'original_video2', 0, None, 10000, [136, 146], True, None),
      ('video3.mp4', 'original_video3', 0, None, 10000, [97, 99, 106], True, None),
      ('video4.mp4', 'original_video4', 0, None, 10000, [142, 147, 172], True, None),
      #('video2.mp4', 'original_video2',   0, None, 300, [136, 146]),
      #('video3.mp4', 'original_video3',   0, None, 300, [97, 99, 106]),
      #('video4.mp4', 'original_video4',   0, None, 300, [142, 147, 172]),
  ]}
  mainconf1 = {
    'prefixinput': f'{baseprefix}media/jpjodoin_urbantracker/',
    'videospecs': [
      #('rene_video.mov',       'jodoin_urbantracker_rene', 0, None, 300, []),
      #('rouen_video.avi',      'jodoin_urbantracker_rouen', 0, None, 300, []),
      #('sherbrooke_video.avi', 'jodoin_urbantracker_sherbrooke', 0, None, 300, []),
      #('stmarc_video.avi',     'jodoin_urbantracker_stmarc', 0, None, 300, []),
      ('rene_video.mov',       'jodoin_urbantracker_rene', 0, None, 10000, [], True, 0),
      ('rouen_video.avi',      'jodoin_urbantracker_rouen', 0, None, 10000, [], True, 0),
      ('sherbrooke_video.avi', 'jodoin_urbantracker_sherbrooke', 0, None, 10000, [], True, 0),
      ('stmarc_video.avi',     'jodoin_urbantracker_stmarc', 0, None, 10000, [], True, 0),
  ]}
  mainconf2 = {
    'prefixinput': f'{baseprefix}media/changedetection/',
    'videospecs': [
      #('highway/input/in%06d.jpg', 'changedetection_highway', 0, None, 300, []),
      #('streetLight/input/in%06d.jpg', 'changedetection_streetLight', 0, None, 300, []),
      #('traffic/input/in%06d.jpg', 'changedetection_traffic', 0, None, 300, []),
      ('highway/input/in%06d.jpg', 'changedetection_highway', 0, None, 10000, [], True, None),
      ('streetLight/input/in%06d.jpg', 'changedetection_streetLight', 0, None, 10000, [], True, None),
      ('traffic/input/in%06d.jpg', 'changedetection_traffic', 0, None, 10000, [], True, None),
  ]}
  mainconf3 = {
    'prefixinput': f'{baseprefix}media/GRAM_RTM_UAH/',
    'videospecs': [
      #('M-30/image%06d.jpg',    'gram_rtm_uah_M30', 0, None, 300, []),
      #('M-30-HD/image%06d.jpg', 'gram_rtm_uah_M30HD', 0, None, 300, [5453]),
      #('Urban1/image%06d.jpg',  'gram_rtm_uah_Urban1', 0, None, 300, []),
      ('M-30/image%06d.jpg',    'gram_rtm_uah_M30', 0, None, 10000, [], True, 0),
      ('M-30-HD/image%06d.jpg', 'gram_rtm_uah_M30HD', 0, None, 10000, [5453], True, 0),
      ('Urban1/image%06d.jpg',  'gram_rtm_uah_Urban1', 0, None, 10000, [], True, 0),
  ]}
  mainconf4 = {
    'prefixinput': f'{baseprefix}media/uni_ulm/',
    'videospecs': [
      #('Sequence1a/KAB_SK_1_undist/i%07d.bmp', 'uni_ulm_Sequence1a_1', 0, None, 300, [11, 389, 531]),
      #('Sequence1a/KAB_SK_4_undist/i%07d.bmp', 'uni_ulm_Sequence1a_4', 0, None, 300, []),
      #('Sequence2/KAB_SK_1_undist/i%07d.bmp', 'uni_ulm_Sequence2_1', 0, None, 300, []),
      #('Sequence2/KAB_SK_4_undist/i%07d.bmp', 'uni_ulm_Sequence2_4', 0, None, 300, []),
      #('Sequence3/KAB_SK_1_undist/i%07d.bmp', 'uni_ulm_Sequence3_1', 0, None, 300, []),
      #('Sequence3/KAB_SK_4_undist/i%07d.bmp', 'uni_ulm_Sequence3_4', 0, None, 300, []),

      ('Sequence1a/KAB_SK_1_undist/i%07d.bmp', 'uni_ulm_Sequence1a_1', 0, None, 10000, [11, 389, 531], True, None),
      ('Sequence1a/KAB_SK_4_undist/i%07d.bmp', 'uni_ulm_Sequence1a_4', 0, None, 10000, [], True, None),
      ('Sequence2/KAB_SK_1_undist/i%07d.bmp', 'uni_ulm_Sequence2_1', 0, None, 10000, [], True, None),
      ('Sequence2/KAB_SK_4_undist/i%07d.bmp', 'uni_ulm_Sequence2_4', 0, None, 10000, [], True, None),
      ('Sequence3/KAB_SK_1_undist/i%07d.bmp', 'uni_ulm_Sequence3_1', 0, None, 10000, [], True, None),
      ('Sequence3/KAB_SK_4_undist/i%07d.bmp', 'uni_ulm_Sequence3_4', 0, None, 10000, [], True, None),
  ]}
  mainconf5 = {
    'prefixinput': f'{baseprefix}media/citinews/',
    'videospecs': [
      ('citinews1.mp4', 'citinews1', 0, None, 10000, [9, 137], True, 0),
      ('citinews2.mp4', 'citinews2', 0, None, 10000, [4], True, 0),
      ('citinews3.mp4', 'citinews3', 0, None, 10000, [], True, 0),
      ('citinews1.stabilized.mp4', 'citinews1_stabilized', 0, None, 10000, [9, 137], True, 0),
      ('citinews2.stabilized.mp4', 'citinews2_stabilized', 0, None, 10000, [4], True, 0),
      ('citinews3.stabilized.mp4', 'citinews3_stabilized', 0, None, 10000, [], True, 0),
  ]}
  mainconf6 = {
    'prefixinput': f'/media/aic21-track4-train-data/',
    'videospecs': (
      #[(f'{n}.mp4', f'nvidia_train_{n:03d}_seg_1', 0, (None,60*30), 10000, [], True, 0) for n in range(1, 16)]+
      #[(f'{n}.mp4', f'nvidia_train_{n:03d}_seg_2', 0, (None,300*30), 10000, [], True, 0) for n in range(1, 16)]+
      #[(f'{n}.mp4', f'nvidia_train_{n:03d}', 0, None, 10000, [], True, 0) for n in range(1, 16)]
      #[(f'clips/{vid:02d}-{seg:02d}.mp4', f'nvidia_train_clips_{vid:02d}_{seg:02d}', 0, None, 10000, [], True, 0) for vid in reversed((1,2,3,5,9,11,13,14)) for seg in reversed(range(15))]
      #[(f'clips/{vid:02d}-{seg:02d}.mp4', f'nvidia_train_clips_{vid:02d}_{seg:02d}', 0, None, 10000, [], True, 0) for vid in (17, 19, 20, 22, 25, 29, 33, 34, 39, 41, 47, 49, 50) for seg in range(15)]
      #[(f'clips/{vid:02d}-{seg:02d}.mp4', f'nvidia_train_clips_{vid:02d}_{seg:02d}', 0, None, 10000, [], True, 0) for vid in (19, 20, 22, 25, 29, 33, 34, 39, 41, 47, 49, 50) for seg in range(15)]
      [(f'clips/{vid:02d}-{seg:02d}.mp4', f'nvidia_train_clips_{vid:02d}_{seg:02d}', 0, None, 10000, [], True, 0) for vid in (1,2,3,5,9,13,14, 17, 20, 22, 25, 33, 39, 41, 50) for seg in range(15)]
  )}
  mainconf7 = {
    'prefixinput': f'/media/aalborguniversity/',
    'videospecs': [
      ('Hadsundvej-1.mkv',    'Hadsundvej_1', 0, None, 10000, [], True, 0),
      ('Hadsundvej-2.mkv',    'Hadsundvej_2', 0, None, 10000, [], True, 0),
      ('Hasserisvej-1.mkv',   'Hasserisvej_1', 0, None, 10000, [], True, 0),
      ('Hasserisvej-2.mkv',   'Hasserisvej_2', 0, None, 10000, [], True, 0),
      ('Hasserisvej-3.mkv',   'Hasserisvej_3', 0, None, 10000, [], True, 0),
      ('Hjorringvej-2.mkv',   'Hjorringvej_2', 0, None, 10000, [], True, 0),
      ('Ostre-3.mkv',         'Ostre_3', 0, None, 10000, [], True, 0),
  ]}

  #ms = [mainconf0, mainconf1, mainconf2, mainconf3, mainconf4, mainconf5]
  ms = [mainconf0, mainconf2, mainconf4, mainconf6]
  #ms = [mainconf6]
  #ms = [mainconf7]
  if False:
    for m in ms:
      for k in range(len(m['videospecs'])):
        m['videospecs'][k] = (m['videospecs'][k][0], m['videospecs'][k][1]+'_skip1', 1, *m['videospecs'][k][3:])
  if args.subdataset_indexes != '':
    subdataset_indexes = eval(args.subdataset_indexes)
    subdataset_indexes_mayor, subdataset_indexes_minors = zip(*subdataset_indexes)
    ms = [m for i,m in enumerate(ms) if i in subdataset_indexes_mayor]
    for m, subdataset_indexes_minor in zip(ms, subdataset_indexes_minors):
      m['videospecs'] = tuple(mini for i,mini in enumerate(m['videospecs']) if i in subdataset_indexes_minor)
  #ms = [mainconf4, mainconf5, mainconf6]
  #ms = [mainconf1, mainconf2, mainconf3]

  #ms = [mainconf1, mainconf3, mainconf5]
  #ms = [mainconf1]#, mainconf4, mainconf5]
  #ms = [mainconf0, mainconf2, mainconf3]


  #ms = [mainconf6]#, mainconf4]
  #ms = [mainconf0]

  #main_run(mainconf, 'run')
  #main_run(mainconf, 'graph')
  #main_run(mainconf3, 'both')
  for m in ms:
    main_run(m, 'both', args)


"""
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 3 --quantile 0.85
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 3 --quantile 0.90
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 3 --quantile 0.95
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 3 --quantile 0.98
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 3 --quantile 0.99
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 4 --quantile 0.85
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 4 --quantile 0.90
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 4 --quantile 0.95
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 4 --quantile 0.98
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 4 --quantile 0.99

python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 5 --quantile 0.85
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 5 --quantile 0.90
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 5 --quantile 0.95
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 5 --quantile 0.98
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 5 --quantile 0.99
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 6 --quantile 0.85
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 6 --quantile 0.90
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 6 --quantile 0.95
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 6 --quantile 0.98
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use False --worryingScale 6 --quantile 0.99



python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use False
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use False
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small False --feat_sim_use False

python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.5

python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.5

python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.5

python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.5

python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.5

python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.5







python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use False
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.5
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.5


python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use False
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.5
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.2


python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use False
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.5
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.2


python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use False
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.5
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.5


quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_resnet18_thresh_0.1_allvehicles_use_distance_limit_small_False
quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_resnet18_thresh_0.1_allvehicles_use_distance_limit_small_True


python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.1







python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet18 --feat_sim_threshold 0.4




SERV3:1

python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.5

SERV1:0
SERV3:0

quantiles_vector_mean_N005_P60_S4_Percentile0.95_with_border_correction_classic_THR0.25_mindim0_resnet101_thresh_0.1_allvehicles_use_distance_limit_small_True
=>
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.1 --subdataset_indexes '[(2,(2,))]

python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.5

SERV1:1
SERV3:1

python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.5

SERV3:0

python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone resnet101 --feat_sim_threshold 0.5

SERV3:0

python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.5

SERV3:1

python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet101 --feat_sim_threshold 0.5











SERV4:0

python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.5
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.5
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.5


SERV4:1


quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_vgg11_bn_thresh_0.1_allvehicles_use_distance_limit_small_False
quantiles_vector_mean_N005_P60_S4_Percentile0.95_no_border_correction_classic_THR0.25_mindim0_vgg11_bn_thresh_0.2_allvehicles_use_distance_limit_small_False
=> REPETIR

python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.1 --subdataset_indexes '(3,(0,1,2)), (4,(0,1,2,3,4,5)), (5,(0,1,2,3,4,5))'
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.2 --subdataset_indexes '(3,(0,1,2)), (4,(0,1,2,3,4,5)), (5,(0,1,2,3,4,5))'


python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.5
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.5
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg11_bn --feat_sim_threshold 0.5

SERV3:0

python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.5
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.5

SERV3:1

python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.5
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.5

SERV4:1

python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.5

SERV4:0

python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.2
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.4
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.5


python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1

SERV3

python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.2

python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.4

SERV1

python anom_traj_detector.py --device 1 --do_border_correction False --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections True  --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.5
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.5

SERV4

python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.1
python anom_traj_detector.py --device 0 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.2

python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.3
python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small True  --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone vgg19_bn --feat_sim_threshold 0.4











python anom_traj_detector.py --device 1 --do_border_correction True  --use_distance_limit_small False --feat_sim_use True --feat_sim_just_small_detections False --feat_sim_backbone resnet18 --feat_sim_threshold 1.0



DEVICE THRESHOLD JUST_SMALL_DETECTIONS BACKBONE

python anom_traj_detector.py 0 0.1 0 resnet18 1
python anom_traj_detector.py 0 0.2 0 resnet18 1
python anom_traj_detector.py 0 0.3 0 resnet18 1

python anom_traj_detector.py 1 0.5 0 resnet18 1
python anom_traj_detector.py 1 0.7 0 resnet18 1

python anom_traj_detector.py 0 0.1 0 resnet18 1
python anom_traj_detector.py 0 0.2 0 resnet18 1
python anom_traj_detector.py 0 0.3 0 resnet18 1

python anom_traj_detector.py 1 0.5 0 resnet18 1
python anom_traj_detector.py 1 0.7 0 resnet18 1


python anom_traj_detector.py 0 0.1 0 resnet101
python anom_traj_detector.py 0 0.2 0 resnet101
python anom_traj_detector.py 0 0.3 0 resnet101
python anom_traj_detector.py 0 0.5 0 resnet101
python anom_traj_detector.py 0 0.7 0 resnet101
python anom_traj_detector.py 0 0.5 0 vgg19_bn
python anom_traj_detector.py 0 0.7 0 vgg19_bn

python anom_traj_detector.py 1 0.1 1 resnet101
python anom_traj_detector.py 1 0.2 1 resnet101
python anom_traj_detector.py 1 0.3 1 resnet101
python anom_traj_detector.py 1 0.5 1 resnet101
python anom_traj_detector.py 1 0.7 1 resnet101
python anom_traj_detector.py 1 0.5 1 vgg19_bn
python anom_traj_detector.py 1 0.7 1 vgg19_bn

python anom_traj_detector.py 0 0.1 0 vgg11_bn
python anom_traj_detector.py 0 0.2 0 vgg11_bn
python anom_traj_detector.py 0 0.3 0 vgg11_bn
python anom_traj_detector.py 0 0.5 0 vgg11_bn
python anom_traj_detector.py 0 0.7 0 vgg11_bn
python anom_traj_detector.py 0 0.1 0 vgg19_bn
python anom_traj_detector.py 0 0.2 0 vgg19_bn
python anom_traj_detector.py 0 0.3 0 vgg19_bn

python anom_traj_detector.py 1 0.1 1 vgg11_bn
python anom_traj_detector.py 1 0.2 1 vgg11_bn
python anom_traj_detector.py 1 0.3 1 vgg11_bn
python anom_traj_detector.py 1 0.5 1 vgg11_bn
python anom_traj_detector.py 1 0.7 1 vgg11_bn
python anom_traj_detector.py 1 0.1 1 vgg19_bn
python anom_traj_detector.py 1 0.2 1 vgg19_bn
python anom_traj_detector.py 1 0.3 1 vgg19_bn

"""

