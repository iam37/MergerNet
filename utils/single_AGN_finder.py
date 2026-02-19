import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.modeling import models, fitting

import sys
sys.path.append("../data_preprocessing/training_datasets/")
from dual_AGN_utils import crop_center
def moffat_fit(quasar_filepath):
    try:
        with fits.open(quasar_filepath, memmap = False) as hdul:
            QSO = hdul[1].data
            QSO = crop_center(QSO, 94, 94)
    except OSError:
        print(f"{quasar_filepath} has been compromised or otherwise cannot be opened, moving on")
    center_x = len(QSO) // 2
    center_y = len(QSO) // 2

    moffat_init_single = models.Moffat2D(amplitude=QSO.max(), x_0=center_x, y_0=center_y, gamma=1, alpha=1.5)
    fitter = fitting.LevMarLSQFitter()
    y_data, x_data = np.indices(QSO.shape)
    moffat_init_dual1 = models.Moffat2D(amplitude=QSO.max()/2, x_0=center_x, y_0=center_y, gamma=1, alpha=1.5)
    moffat_init_dual2 = models.Moffat2D(amplitude=QSO.max()/2, x_0=center_x+10, y_0=center_y, gamma=1, alpha=1.5)
    dual_moffat_init = moffat_init_dual1 + moffat_init_dual2
    moffat_fit_single = fitter(moffat_init_single, x_data, y_data, QSO)
    
    # Fit dual AGN model
    moffat_fit_dual = fitter(dual_moffat_init, x_data, y_data, QSO)
    
    # Evaluate the fit quality using residuals
    residuals_single = QSO - moffat_fit_single(x_data, y_data)
    residuals_dual = QSO - moffat_fit_dual(x_data, y_data)
    
    # Sum of squared residuals
    ssr_single = np.sum(residuals_single**2)
    ssr_dual = np.sum(residuals_dual**2)

    if ssr_single < ssr_dual:
        print("Single AGN is a better fit")
        return True
    else:
        print("Dual AGN is a better fit")
        return False

