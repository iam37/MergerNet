import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.table import Table, hstack, unique, vstack
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.cosmology import Planck15
import seaborn as sns
import umap
import statmorph
from tqdm import tqdm
import sep
import glob
import warnings
import os
from joblib import Parallel, delayed
import stpsf
from photutils.segmentation import detect_threshold, detect_sources
from photutils.background import Background2D
from statmorph.utils.image_diagnostics import make_figure
import logging
from photutils.segmentation import SourceFinder

def crop_center(img, cropx, cropy):
    
    #Function from 
    #https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    
    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    #print(startx)
    starty = y // 2 - (cropy // 2) 
    #print(starty)
    return img[starty:starty + cropy, startx:startx + cropx, ...]

def pick_main_label(segm, img_sub, prefer='largest'):
    """Return the label to analyze."""
    if segm is None or segm.nlabels == 0:
        return None
    if prefer == 'largest':
        areas = segm.areas
        return segm.labels[int(np.argmax(areas))]
    # else nearest center
    ys, xs = np.indices(img_sub.shape)
    cy, cx = (img_sub.shape[0]-1)/2, (img_sub.shape[1]-1)/2
    # compute centroid per label quickly using segm properties
    # (coarse: use mean of coordinates within label mask)
    best_label, best_d2 = None, np.inf
    for lab in segm.labels:
        mask = segm.data == lab
        if mask.sum() == 0:
            continue
        ybar = ys[mask].mean()
        xbar = xs[mask].mean()
        d2 = (ybar-cy)**2 + (xbar-cx)**2
        if d2 < best_d2:
            best_label, best_d2 = lab, d2
    return best_label

def calculate_morphology(img, psf, idx):
    img = img.astype(img.dtype.newbyteorder("="))
    #bad = ~np.isfinite(err_map) | (err_map <= 0)
    #masked_img = img.copy()
    #masked_img[bad] = 0.0
    #masked_err_map = err_map.copy()
    #masked_err_map[bad] = 0.0
    coverage_mask = img[np.isnan(img)]
    
    if np.shape(coverage_mask) == np.shape(img):
        print(np.shape(coverage_mask))
        background = Background2D(img, (5,5), coverage_mask = coverage_mask)
    else:
        background = Background2D(img, (5,5))
    img_sub = img - background.background
    npixels = 12
    objects = sep.extract(img_sub, 1.5)
    ny, nx = img_sub.shape
    x_center, y_center = nx // 2, ny // 2

    distances = np.sqrt((objects['x'] - x_center)**2 + (objects['y'] - y_center)**2)
    # central_idx = np.argmin(distances)

    # x_central = objects['x'][central_idx]
    # y_central = objects['y'][central_idx]
    # a_central = objects['a'][central_idx]
    # b_central = objects['b'][central_idx]
    # theta_central = objects['theta'][central_idx]

    # # Compute isophotal flux only for this source
    # flux_iso, fluxerr_iso, flag = sep.sum_ellipse(
    #     img_sub,
    #     x_central,
    #     y_central,
    #     a_central,
    #     b_central,
    #     theta_central,
    #     r=3.0
    # )
    threshold = 1.5*background.background_rms

    #convolved_image = convolve(img_sub, psf)
    convolved_image = convolve(img, psf)
    # segmap = detect_sources(convolved_image, background_img, npixels) # need to have a NIRCAM PSF for proper convolution.
    finder = SourceFinder(npixels=npixels, progress_bar=False, deblend=True)
    segmap = finder(convolved_image, threshold)

    label_main = pick_main_label(segmap, img, prefer='largest')
    if label_main is None:
        print(f"Skipping image {idx}")
        # print(f"[{i}] no valid label")
        # psb_morphologies.append(-1)
        #psb_ids.append(idx)
    gal_morphs = statmorph.SourceMorphology(img, segmap, label_main, verbose=False, psf = psf, cutout_extent = 2.0)
    return gal_morphs, idx
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    with warnings.catch_warnings():
        logging.info("Accessing NIRCam PSF")
        nircam = stpsf.NIRCam()
        nircam.image_mask= None
        nircam.pupil_mask = None
        nircam.filter='F200W'
        nircam.options['source_offset_x'] = 0
        nircam.options['source_offset_y'] = 0
        # Compute PSFs. Save to disk for reuse
        outname = f"psf_NIRcam-F200W.fits"
        psf_array = nircam.calc_psf(outfile=outname)
        ext = 'DET_DIST'
        psf = psf_array[ext].data
        psf = crop_center(psf, 159, 159)
        indexes = []
        morphologies = []
        for image in tqdm(glob.glob(f"../CEERS_Sim_images/F200W/active_merger_low_z/train_data/*.fits")):
            try:
                img = fits.getdata(image, memmap=False)
                idx = os.path.basename(image)
                morph, idx = calculate_morphology(img, psf, idx)
                morphologies.append(morph)
                indexes.append(idx)
            except ValueError as e:
                print(f"[{idx}] ValueError: {e}")
                psb_ids.append(idx)
                continue
            except ZeroDivisionError as e:
                print(f"{idx} ran into: {e}, skipping...")
                continue
            except Exception as e:
                print(f"{e} on image {idx}")
                continue
    morph_data = []
    for idx, morph in zip(indexes, morphologies):
        if morph is None or isinstance(morph, int):
            morph_data.append({'ID': idx, 'Gini': np.nan, 'M20': np.nan, 'Gini_M20_merger': np.nan,'Asymmetry': np.nan, 'Concentration': np.nan})
        else:
            morph_data.append({
                'ID': idx,
                'Gini': morph.gini,
                'M20': morph.m20,
                'Gini_M20_merger': morph.gini_m20_merger,
                'Asymmetry': morph.asymmetry,
                'rms_asymmetry': np.sqrt(morph.rms_asymmetry2),
                # 'isophotal_asymmetry':morph.isophote_asymmetry,
                'Concentration': morph.concentration,
                'Smoothness': morph.smoothness,
                'Sersic_n': morph.sersic_n,
                'Ellipticity_asymmetry': morph.ellipticity_asymmetry,
                'r20': morph.r20
            })
    morph_df = pd.DataFrame(morph_data)
    print(os.getcwd())
    morph_df.to_csv('../CEERS_Sim_images/F200W/active_merger_low_z/morphology.csv', index=False)