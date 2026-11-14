"""The three count data sets and their design matrices.

  nmes          physician office visits, National Medical Expenditure Survey
                1987-88 (Deb and Trivedi 1997; AER::NMES1988), 4,406 people aged
                66 and over with Medicare; regressors as in Zeileis, Kleiber and
                Jackman (2008)
  biochemists   articles published in the last three years of the PhD by 915
                biochemists (Long 1990; pscl::bioChemists)
  doctorvisits  doctor visits in the past two weeks, Australian Health Survey
                1977-78 (Cameron and Trivedi 1986; AER::DoctorVisits), 5,190
                adults; regressors: every variable in the data set and age squared
"""

import numpy as np

from common import fetch


def nmes():
    """y, X (count regressors), Z (zero regressors of the reduced hurdle and ZINB
    models in the pscl vignette), column names of X."""
    d = fetch("AER", "NMES1988")
    y = d.visits.values.astype(float)
    names = ["(Intercept)", "hosp", "healthpoor", "healthexcellent", "numchron", "gendermale", "school", "privinsyes"]
    X = np.column_stack([np.ones(len(d)), d.hospital, d.health == "poor", d.health == "excellent", d.chronic,
                         d.gender == "male", d.school, d.insurance == "yes"]).astype(float)
    Z = np.column_stack([np.ones(len(d)), d.hospital, d.chronic, d.insurance == "yes", d.school,
                         d.gender == "male"]).astype(float)
    return y, X, Z, names


def biochemists():
    d = fetch("pscl", "bioChemists")
    y = d.art.values.astype(float)
    X = np.column_stack([np.ones(len(d)), d.fem == "Women", d.mar == "Married", d.kid5, d.phd, d.ment]).astype(float)
    return y, X


def doctorvisits():
    d = fetch("AER", "DoctorVisits")
    y = d.visits.values.astype(float)
    X = np.column_stack([np.ones(len(d)), d.gender == "female", d.age, d.age ** 2, d.income, d.illness, d.reduced,
                         d.health, d.private == "yes", d.freepoor == "yes", d.freerepat == "yes", d.nchronic == "yes",
                         d.lchronic == "yes"]).astype(float)
    return y, X
