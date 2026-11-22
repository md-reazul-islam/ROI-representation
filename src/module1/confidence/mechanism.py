from dataclasses import dataclass, asdict
import json
import math

import numpy as np


@dataclass(frozen=True)
class ConfidenceConfig:
    regime: str = 'equal_fixed'
    top_k: int = 3
    temperature: float = 1.0
    validation_split: str = 'validation'
    training_split: str = 'train'
    min_training_rows: int = 10
    ridge: float = .01
    iterations: int = 800
    learning_rate: float = .08
    sam2_enabled: bool = False
    sam2_weight: float = 0.0

    def __post_init__(self):
        if type(self.sam2_enabled) is not bool:
            raise ValueError('sam2_enabled must be boolean')
        if not math.isfinite(self.sam2_weight) or not 0 <= self.sam2_weight <= 1:
            raise ValueError('sam2_weight must be finite and in [0,1]')
        if self.regime not in ('equal_fixed', 'validation_tuned', 'weakly_supervised'):
            raise ValueError('regime must be equal_fixed, validation_tuned, or weakly_supervised')
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError('top_k must be a positive integer')
        for name in ('temperature', 'ridge', 'learning_rate'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('min_training_rows', 'iterations'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class ConfidenceWeights:
    pose: float
    visibility: float
    skin: float
    temporal: float
    motion: float

    def __post_init__(self):
        values = (self.pose, self.visibility, self.skin, self.temporal, self.motion)
        if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
            raise ValueError('Confidence weights must be finite, nonnegative and nonzero')

    def normalized(self):
        total = self.pose + self.visibility + self.skin + self.temporal + self.motion
        return ConfidenceWeights(*(value/total for value in (self.pose, self.visibility, self.skin, self.temporal, self.motion)))

    def as_dict(self):
        return asdict(self.normalized())


class ConfidenceMechanism:
    """ROI confidence and fusion primitives; no HR/RR or physiological label is produced."""

    TERM_NAMES = ('pose', 'visibility', 'skin', 'temporal', 'motion')

    def __init__(self, config=None):
        self.config = config or ConfidenceConfig()
        self.weights = ConfidenceWeights(.2, .2, .2, .2, .2).normalized()
        self.fit_metadata = dict(regime=self.config.regime, training_rows=0,
                                 weak_supervision=True, target_definition='UNAVAILABLE')

    @staticmethod
    def _value(row, key, default=None):
        value = row.get(key, default)
        if value is None or value == '':
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def terms(self, row):
        pose = self._value(row, 'pose_confidence', self._value(row, 'anatomical_confidence'))
        visibility = self._value(row, 'visibility')
        if visibility is None:
            occlusion = self._value(row, 'occlusion_proxy')
            visibility = None if occlusion is None else 1.-occlusion
        skin = self._value(row, 'skin_availability', self._value(row, 'visible_skin_proportion'))
        temporal = self._value(row, 'temporal_consistency')
        motion = self._value(row, 'nuisance_motion', self._value(row, 'motion_magnitude'))
        values = [pose, visibility, skin, temporal, motion]
        imputed = []
        # No predecessor means neutral continuity and no measured nuisance motion.
        if temporal is None:
            temporal, imputed = 1., imputed + ['temporal']
        if motion is None:
            motion, imputed = 0., imputed + ['motion']
        if any(value is None for value in (pose, visibility, skin)):
            return None, imputed + [name for name, value in zip(('pose', 'visibility', 'skin'), values) if value is None]
        values = [pose, visibility, skin, temporal, motion]
        if not all(math.isfinite(value) for value in values):
            return None, imputed
        return tuple(float(np.clip(value, 0., 1.)) for value in values), imputed

    @staticmethod
    def pseudo_target(terms, row):
        """A transparent weak target from Phase 9 metrics, not ground-truth reliability."""
        if terms is None:
            return None
        extra = []
        for key, invert in (('blur_score', False), ('illumination_score', False),
                            ('saturation_score', False), ('roi_area_variation', True),
                            ('consecutive_roi_iou', False), ('swaddling_proxy', True)):
            value = row.get(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                extra.append(1.-value if invert else value)
        return float(np.mean(list(terms[:4]) + [1.-terms[4]] + extra)) if extra else float(np.mean([terms[0], terms[1], terms[2], terms[3], 1.-terms[4]]))

    def _fit(self, rows, source, target_name):
        matrix, target = [], []
        for row in rows:
            terms, _ = self.terms(row)
            y = self.pseudo_target(terms, row)
            if terms is not None and y is not None:
                matrix.append([terms[0], terms[1], terms[2], terms[3], -terms[4]])
                target.append(y)
        if len(matrix) < self.config.min_training_rows:
            raise ValueError(f'{source} requires at least {self.config.min_training_rows} complete ROI-quality rows; found {len(matrix)}')
        x, y = np.asarray(matrix, dtype=float), np.asarray(target, dtype=float)
        weights = np.full(5, .2, dtype=float)
        for _ in range(self.config.iterations):
            gradient = 2.*x.T.dot(x.dot(weights)-y) / len(y) + self.config.ridge*weights
            candidate = np.maximum(0., weights-self.config.learning_rate*gradient)
            total = candidate.sum()
            weights = candidate/total if total > 0 else np.full(5, .2)
        self.weights = ConfidenceWeights(*weights).normalized()
        self.fit_metadata = dict(regime=self.config.regime, training_source=source,
                                 training_rows=len(matrix), weak_supervision=True,
                                 target_definition='mean of normalized positive terms plus inverse nuisance/area/swaddling proxies',
                                 fitted_weights=self.weights.as_dict(), target_name=target_name)
        return self.weights

    def fit(self, rows):
        if self.config.regime == 'equal_fixed':
            self.fit_metadata = dict(regime='equal_fixed', training_rows=0, weak_supervision=False,
                                     target_definition='fixed equal weights', fitted_weights=self.weights.as_dict())
            return self.weights
        if self.config.regime == 'validation_tuned':
            selected = [row for row in rows if row.get('split') == self.config.validation_split]
            return self._fit(selected, 'validation split', 'validation_pseudo_reliability')
        selected = [row for row in rows if row.get('split') == self.config.training_split]
        if len(selected) < self.config.min_training_rows:
            selected = list(rows)
            source = 'all available frames fallback'
        else:
            source = 'training split'
        return self._fit(selected, source, 'weak_pseudo_reliability')

    def score(self, row):
        if row.get('roi_status') == 'UNAVAILABLE':
            return None, None, []
        terms, imputed = self.terms(row)
        if terms is None:
            return None, None, imputed
        w = self.weights.normalized()
        q = w.pose*terms[0] + w.visibility*terms[1] + w.skin*terms[2] + w.temporal*terms[3] - w.motion*terms[4]
        sam2 = self.sam2_term(row)
        if sam2 is not None:
            q += self.config.sam2_weight*sam2
        return float(q), terms, imputed

    def sam2_term(self, row):
        if not self.config.sam2_enabled:
            return None
        support = self._value(row, 'SAM2_mask_support')
        if support is None or not 0 <= support <= 1:
            return None
        quality = self._value(row, 'SAM2_mask_quality')
        if quality is not None and not 0 <= quality <= 1:
            return None
        return support * (quality if quality is not None else 1.)

    def softmax(self, scores):
        valid = [(index, score) for index, score in enumerate(scores) if score is not None and math.isfinite(score)]
        output = [0.]*len(scores)
        if not valid:
            return output
        values = np.asarray([score for _, score in valid], dtype=float)/self.config.temperature
        values -= np.max(values)
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        for (index, _), probability in zip(valid, probabilities):
            output[index] = float(probability)
        return output

    @staticmethod
    def rank(scores):
        order = sorted((index for index, score in enumerate(scores) if score is not None),
                       key=lambda index: (-scores[index], index))
        result = [None]*len(scores)
        for rank, index in enumerate(order, 1):
            result[index] = rank
        return result

    @staticmethod
    def representation(row):
        import json
        fields = ('bbox_normalized', 'centroid_bbox_normalized', 'width', 'height',
                  'pose_confidence', 'visible_skin_proportion', 'blur_score',
                  'illumination_score', 'saturation_score', 'motion_magnitude',
                  'landmark_velocity', 'roi_centroid_displacement', 'roi_area_variation',
                  'temporal_consistency', 'swaddling_proxy')
        vector, missing = [], []
        for field in fields:
            value = row.get(field)
            if field in ('bbox_normalized', 'centroid_bbox_normalized'):
                try:
                    values = json.loads(value) if isinstance(value, str) else value
                except (TypeError, ValueError, json.JSONDecodeError):
                    values = None
                if not isinstance(values, list):
                    missing.append(field); vector.extend([0.]* (4 if field == 'bbox_normalized' else 2))
                else:
                    vector.extend(float(v) if math.isfinite(float(v)) else 0. for v in values)
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = None
                if number is None or not math.isfinite(number):
                    missing.append(field); vector.append(0.)
                else:
                    vector.append(float(number))
        return vector, missing
