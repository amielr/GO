"""TemplateMatching object."""

# pylint: disable=invalid-unary-operand-type

import glob
import os
from typing import List
import numpy as np
import pandas as pd
import cv2 as cv

from skimage.measure import label, regionprops
from mag_utils.scans.horizontal_scan import HorizontalScan
from mag_algorithms.base_algo import Algorithm
from mag_algorithms.template_matching.consts import Backend, DEFAULT_CORR
from mag_algorithms.template_matching.utils import load_template_scan


class TemplateMatching(Algorithm):
    """Template matching get the directory of templates and run on scan all the templates."""

    def __init__(self,
                 distance_between_points: float,
                 threshold: float = 0.85,
                 backend: str = Backend.OPENCV.value,  # TODO: I fucking hate python enums!!!!
                 template_dir: str = './templates',
                 padding_type: str = 'edge',
                 normalizing_method: str = 'min_max'
                 ):
        """
        TemplateMatching.

        Args:
            distance_between_points: the distance between two points in the template in meters.
                                     Note: both the template and the scan should have the same distance between points..
            backend: str, skimage/opencv, which template matching method to use.
            threshold: The threshold to mark template's corr as "true".
            template_dir: str, directory of the template files, default is './templates'.
            padding_type: type of padding for np.pad, if using 'linear_ramp', the edge value is the mean of the scan.
            normalizing_method: 'min_max' for MinMax normalization and 'standardizaton' for
            reducing mean() and diving by std.
        """
        self.template_dir = template_dir
        self.templates = self.load_templates(self.template_dir, distance_between_points)
        self.threshold = threshold
        self.backend = backend
        self.padding_type = padding_type
        self.normalizing_method = normalizing_method

        available_normalizations = ['standardization', 'min_max']
        if normalizing_method not in available_normalizations:
            raise ValueError(f'Given {normalizing_method=} not in available normalizations {available_normalizations}')

    def match_single_template(self, scan, template):
        """
        Match single template.

        Args:
            scan: a 2D array (b).
            template: The template (b).

        Returns:
            Match Correlation.
        """
        if self.normalizing_method == 'min_max':
            template = (template - template.min()) / (template.max() - template.min())
            scan = (scan - scan[~np.isnan(scan)].min()) / (scan[~np.isnan(scan)].max() - scan[~np.isnan(scan)].min())
        elif self.normalizing_method == 'standardization':
            template = (template - template.mean()) / template.std()
            scan = (scan - scan[~np.isnan(scan)].mean()) / scan[~np.isnan(scan)].std()
        else:
            raise ValueError()
        if self.backend == Backend.OPENCV.value:
            pad_width = tuple(
                (width // 2, width // 2 if width % 2 != 0 else ((width // 2) - 1)) for width in template.shape)

            if self.padding_type == 'linear_ramp':
                scan = np.pad(scan, pad_width=pad_width, mode=self.padding_type, end_values=np.mean(scan))
            else:
                scan = np.pad(scan, pad_width=pad_width, mode=self.padding_type)

            # Method is TM_CCOEFF because its the same as skimage's default.
            corr = cv.matchTemplate(scan.astype(np.float32), template.astype(np.float32), cv.TM_CCOEFF_NORMED)
        else:
            raise ValueError(f"Unkown backend: {self.backend}")

        return corr

    def match_templates(self, scan: np.ndarray):
        """
        Run all the templates on a given scan.

        Args:
            scan: a 2D array (b).

        Returns:
            Array of the correlation for each template.
        """
        res = np.ones([len(self.templates), *scan.shape]) * DEFAULT_CORR

        for i, template in enumerate(self.templates):
            if scan.shape[0] > template.b.shape[0] and scan.shape[1] > template.b.shape[1]:
                res[i] = self.match_single_template(scan, template.b)

        return res

    def get_predictions(self, corr, threshold=None) -> List[dict]:
        """
        Get the output of the match_templates method and return a list of predictions.

        prediction: {'pos': (row, col) in pixels,
                    'corr': the correlation value where the prediction is,
                    'd2s': the d2s of the template the got the correlation a the prediction point [m],
                    'template_name': the template's name for the prediction.}

        Args:
            corr: correlation per template per pixel [TxNxM] (T=amount of templates).
            threshold: pass what value should we count the pixel as positive.

        Returns:
            Pred list, with correlation and pos for each template if its corr is above threshold.
        """
        if threshold is None:
            threshold = self.threshold

        preds = []

        for t_corr, template in zip(corr, self.templates):
            mask = t_corr > threshold
            label_image = label(mask)

            for region in regionprops(label_image, t_corr):
                pred_pos = region.coords[t_corr[region.coords[:, 0], region.coords[:, 1]].argmax()]

                preds.append({'pos': tuple(pred_pos),
                              'corr': t_corr[tuple(np.round(pred_pos))],
                              'd2s': template.d2s,
                              'template_name': template.file_name})

        return preds

    def run(self, scan: HorizontalScan, threshold=None):
        """
        Run all templates on scan, return max pos of each cluster (cluster of template matching preds).

        Args:
            scan: HorizontalScan object to give as input to the algorithm.
            threshold: Min corr to make the prediction "true".

        Returns:
            The result* dict for the highest corr from template.
            Or None if there is not result.

        * Result is dict:
            {"x": x[pred["pos"]],
             "y": y[pred["pos"]],
             "d2s": pred["d2s"],
             'score': pred["corr"],
             'name': pred["template_name"]}
        """
        if scan.interpolated_data is None:
            raise ValueError("There is no interpolated data.")

        corrs = self.match_templates(scan.interpolated_data.b)
        corrs[:, ~scan.interpolated_data.mask] = DEFAULT_CORR

        predictions = self.get_predictions(corrs, threshold)

        result = {
            "x": [scan.interpolated_data.x[pred["pos"]] for pred in predictions],
            "y": [scan.interpolated_data.y[pred["pos"]] for pred in predictions],
            "d2s": [pred["d2s"] for pred in predictions],
            'score': [pred["corr"] for pred in predictions],
            'name': [pred["template_name"] for pred in predictions]
        }

        return result

    @staticmethod
    def load_templates(dir_path, distance_between_points=0.1):
        """
        Load all the template from the dir_path and return a list of TemplateScan.

        Args:
            dir_path: Template's dir path.
            distance_between_points: Distance between points.

        Returns:
            List of templates.
        """
        templates = []

        for filepath in glob.glob(os.path.join(dir_path, '*.h5')):
            templates.append(load_template_scan(filepath, distance_between_points))

        if len(templates) == 0:
            raise ValueError(f'No templates found in {dir_path}')

        return templates
