"""View-gated 2D geometric proxies; no diagnoses or automatic posture labels."""

from __future__ import annotations

import numpy as np
from src.posturaai.keypoints import KEYPOINT_NAMES

VIEWS = ('side', 'front', 'rear', 'oblique', 'mixed', 'unknown')
DIRECTIONS = ('left', 'right', 'unknown')
SIDES = ('left', 'right', 'unknown')
CAMERA_MOTIONS = ('fixed', 'moving', 'unknown')
INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def angle_degrees(a, vertex, b):
    u, v = np.asarray(a) - vertex, np.asarray(b) - vertex
    scale = float(np.linalg.norm(u) * np.linalg.norm(v))
    if scale <= 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / scale, -1, 1))))


def measure_pose(points, valid, *, view='unknown', direction='unknown',
                 visible_side='unknown', camera_motion='unknown'):
    if view not in VIEWS or direction not in DIRECTIONS or visible_side not in SIDES or camera_motion not in CAMERA_MOTIONS:
        raise ValueError('Invalid view, direction, side or camera motion')
    points, valid = np.asarray(points, dtype=float), np.asarray(valid, dtype=bool)
    if points.shape != (30, 2) or valid.shape != (30,):
        raise ValueError('Expected 30 points and validity flags')
    valid = valid & np.isfinite(points).all(axis=1)
    features, reasons = {}, {}

    def put(name, names, planes, compute, *, side=None, needs_direction=False, fixed=False):
        ids = [INDEX[n] for n in names]
        reason = None
        if view not in planes:
            reason = 'incompatible_or_unknown_view'
        elif side and visible_side != 'unknown' and side != visible_side:
            reason = 'far_side_excluded'
        elif needs_direction and direction == 'unknown':
            reason = 'running_direction_unknown'
        elif fixed and camera_motion != 'fixed':
            reason = 'requires_fixed_camera'
        elif not valid[ids].all():
            reason = 'missing_or_low_confidence_points'
        value = None if reason else compute(*points[ids])
        if value is None and not reason:
            reason = 'degenerate_geometry'
        features[name] = value
        if reason:
            reasons[name] = reason

    sign = -1 if direction == 'left' else 1
    put('head_trunk_projected_angle_deg', ['TopHead', 'UpTrunk', 'Pelvis'], VIEWS, angle_degrees)
    for side in ('left', 'right'):
        p = side.capitalize()
        for name, names in (
            ('knee_angle_deg', [p+'Hip', p+'Knee', p+'Ankle']),
            ('hip_angle_deg', [p+'Shoulder', p+'Hip', p+'Knee']),
            ('ankle_angle_deg', [p+'Knee', p+'Ankle', p+'FirstMetatarsal']),
            ('elbow_angle_deg', [p+'Shoulder', p+'Elbow', p+'Wrist']),
            ('shoulder_angle_deg', [p+'Hip', p+'Shoulder', p+'Elbow']),
        ):
            # Always preserve projected geometry, including unlabelled/mixed views.
            put(side+'_'+name.replace('_angle_', '_projected_angle_'), names, VIEWS, angle_degrees)
        put(side+'_ankle_forward_offset', [p+'Ankle', p+'Hip', 'Pelvis', 'UpTrunk'], ('side',),
            lambda ankle, hip, pelvis, trunk: sign * float(ankle[0] - hip[0]) / float(np.linalg.norm(trunk-pelvis))
            if np.linalg.norm(trunk-pelvis) > 1e-8 else None, side=side, needs_direction=True)

    put('trunk_image_lean_deg', ['Pelvis', 'UpTrunk'], VIEWS,
        lambda pelvis, trunk: float(np.degrees(np.arctan2(trunk[0]-pelvis[0], pelvis[1]-trunk[1])))
        if np.linalg.norm(trunk-pelvis) > 1e-8 else None)
    put('trunk_forward_lean_deg', ['Pelvis', 'UpTrunk'], ('side',),
        lambda pelvis, trunk: float(np.degrees(np.arctan2(sign*(trunk[0]-pelvis[0]), pelvis[1]-trunk[1])))
        if np.linalg.norm(trunk-pelvis) > 1e-8 else None, needs_direction=True)
    put('trunk_lateral_lean_deg', ['Pelvis', 'UpTrunk'], ('front', 'rear'),
        lambda pelvis, trunk: float(np.degrees(np.arctan2(trunk[0]-pelvis[0], pelvis[1]-trunk[1])))
        if np.linalg.norm(trunk-pelvis) > 1e-8 else None)
    put('pelvis_line_tilt_deg', ['LeftHip', 'RightHip'], ('front', 'rear'),
        lambda left, right: float(np.degrees(np.arctan2(right[1]-left[1], abs(right[0]-left[0]))))
        if abs(right[0]-left[0]) > 1e-8 else None)

    pelvis, trunk = INDEX['Pelvis'], INDEX['UpTrunk']
    scale = float(np.linalg.norm(points[trunk]-points[pelvis])) if valid[[pelvis, trunk]].all() else None
    if scale is not None and scale <= 1e-8:
        scale = None
    normalized = np.full_like(points, np.nan)
    if scale is not None:
        normalized[valid] = (points[valid]-points[pelvis])/scale
    # Pixel pelvis height is useful only for fixed-camera sequences, with caution
    # about perspective/zoom. It is not a calibrated centre-of-mass oscillation.
    put('pelvis_image_y_px', ['Pelvis'], ('side', 'front', 'rear'),
        lambda p: float(p[1]), fixed=True)
    return dict(features=features, unavailable_reasons=reasons,
                normalized_keypoints=[[float(x), float(y)] if np.isfinite([x,y]).all() else None
                                      for x, y in normalized],
                normalization=dict(origin='Pelvis', scale='distance(Pelvis, UpTrunk)', scale_px=scale),
                interpretation=dict(view=view, projected_angles_available_in_all_views=True,
                                    sagittal_interpretation_eligible=view == 'side',
                                    frontal_interpretation_eligible=view in ('front', 'rear'),
                                    visible_side=visible_side, running_direction=direction,
                                    camera_motion=camera_motion), posture_label=None)


class FeatureExtractor:
    """Temporal changes only within contiguous track/segment observations."""

    def __init__(self, **context):
        self.context = context
        self.previous = {}

    def process(self, frame, timestamp, people):
        output = []
        current = {}
        for person in people:
            key = (person['track_id'], person['segment_id'])
            points = np.array([[np.nan if v is None else v for v in row[:2]]
                               for row in person['keypoints']], dtype=float)
            result = measure_pose(points, person['valid'], **self.context)
            features = result['features']
            previous = self.previous.get(key)
            dt = timestamp - previous['timestamp'] if previous else 0
            continuous = previous and previous['frame'] == frame-1 and dt > 0
            # Copy keys: temporal outputs must not recursively generate velocities.
            for name in list(features):
                if not name.endswith('_angle_deg'):
                    continue
                value, before = features[name], previous['features'].get(name) if continuous else None
                speed_name = name.removesuffix('_deg')+'_velocity_deg_s'
                features[speed_name] = (value-before)/dt if value is not None and before is not None else None
                if features[speed_name] is None:
                    result['unavailable_reasons'][speed_name] = 'requires_two_contiguous_valid_measurements'
            for side, ankle in (('left', 'LeftAnkle'), ('right', 'RightAnkle')):
                i = INDEX[ankle]
                now = result['normalized_keypoints'][i]
                before = previous['normalized'][i] if continuous else None
                for axis, dim in (('x', 0), ('y', 1)):
                    name = side+'_ankle_image_relative_velocity_'+axis
                    features[name] = (now[dim]-before[dim])/dt if now is not None and before is not None else None
                    if features[name] is None:
                        result['unavailable_reasons'][name] = 'requires_contiguous_valid_normalized_measurements'
            result.update(track_id=key[0], segment_id=key[1], valid=person['valid'])
            output.append(result)
            current[key] = dict(frame=frame, timestamp=timestamp, features=features.copy(), normalized=result['normalized_keypoints'])
        self.previous = current
        return output


FEATURE_SPEC = {
    'joint_angles': 'Projected image angle acos(dot(A-B,C-B)/(|A-B||C-B|)), degrees [0,180]. Exported in all views; sagittal interpretation only for side view, with occlusion checks.',
    'knee': 'Hip-Knee-Ankle; 180 degrees is a straight projected knee.',
    'hip': 'Shoulder-Hip-Knee, projected segment angle, not clinical hip flexion.',
    'ankle': 'Knee-Ankle-FirstMetatarsal, projected segment angle, not calibrated dorsiflexion.',
    'elbow': 'Shoulder-Elbow-Wrist.',
    'shoulder': 'Hip-Shoulder-Elbow.',
    'head_trunk': 'TopHead-UpTrunk-Pelvis, projected image angle; no clinical head-position threshold.',
    'trunk_forward_lean_deg': 'atan2(direction*(UpTrunk.x-Pelvis.x), Pelvis.y-UpTrunk.y); positive towards running direction.',
    'trunk_lateral_lean_deg': 'Same vertical reference, positive towards screen right; front/rear only.',
    'pelvis_line_tilt_deg': 'atan2(RightHip.y-LeftHip.y, abs(RightHip.x-LeftHip.x)); screen y down; front/rear only. Not phase-specific hip drop.',
    'ankle_forward_offset': 'direction*(Ankle.x-Hip.x)/distance(Pelvis,UpTrunk); side view. No ground-contact or overstriding label.',
    'normalization': 'Subtract Pelvis and divide by distance(Pelvis,UpTrunk). Image x right, y down; no metric length units.',
    'temporal': 'Finite difference / elapsed seconds; reset on missing frames/points, segment or track changes. Angular speed deg/s; relative ankle speed trunk-lengths/s.',
    'pelvis_image_y_px': 'Pixel y, fixed camera only; not COM or calibrated vertical oscillation.',
}
