import numpy as np
from matching_utils import iou

def data_association(dets, trks, threshold=-0.2, algm='greedy'):
    """
    Q1. Assigns detections to tracked object

    dets:       a list of Box3D object
    trks:       a list of Box3D object
    threshold:  only mark a det-trk pair as a match if their iou distance is less than the threshold
    algm:       for extra credit, implement the hungarian algorithm as well

    Returns 3 lists:
        matches, kx2 np array of match indices, i.e. [[det_idx1, trk_idx1], [det_idx2, trk_idx2], ...]
        unmatched_dets, a 1d array of indices of unmatched detections
        unmatched_trks, a 1d array of indices of unmatched trackers
    """
    # Hint: you should use the provided iou(box_a, box_b) function to compute distance/cost between pairs of box3d objects
    # iou() is an implementation of a 3D box IoU

    matches = []
    unmatched_dets = []
    unmatched_trks = []
    # --------------------------- Begin your code here ---------------------------------------------
    
    # Begin with sets of unmatched_dets and unmatched_trks
    unmatched_dets = list(range(len(dets)))
    unmatched_trks = list(range(len(trks)))

    # Continue until either unmatched_dets or unmatched_trks become empty
    while (len(unmatched_dets) is not 0) and (len(unmatched_trks) is not 0):
        # Find best matching pair between dets and trks
        iou_max = -np.inf
        det_i_max = None
        trk_i_max = None
        for det_i in unmatched_dets:
            for trk_i in unmatched_trks:
                iou_max = max(iou_max, iou(dets[det_i], trks[trk_i]))
                if iou_max == iou(dets[det_i], trks[trk_i]):
                    det_i_max = det_i
                    trk_i_max = trk_i
        
        if (iou_max < threshold) or (iou_max == -np.inf):
            # If the found iou_max is smaller than threshold, stop
            break
        else:
            # Otherwise, add the pair to matches, 
            # remove them from unmatched sets, and continue
            matches.append((det_i_max, trk_i_max))
            unmatched_dets.remove(det_i_max)
            unmatched_trks.remove(trk_i_max)
    # --------------------------- End your code here   ---------------------------------------------

    return np.array(matches), np.array(unmatched_dets), np.array(unmatched_trks)