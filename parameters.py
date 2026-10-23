"""Estimated metric geometry; alternative hypotheses, not identified ground truth."""
import numpy as np
TABLE_TOP = .75
BODY_CENTER = np.array([0., .10, .965])
# Chosen after the slender-body hypothesis allowed the band to slide off.
# A 0.27 x 0.216 m waist has ~0.766 m circumference, consistent with the .8 m band.
BODY_RADII = np.array([.135, .108, .168])
BACK_WRAP_Y = .215
# Alternative articulated-bear hypothesis. Zero preserves the initial pose;
# 55 degrees during threading brings the hanging arms approximately level.
# This is scripted assistance inferred for the task, not recovered motion.
ARMS_ABDUCTION_DEG = 80.0
