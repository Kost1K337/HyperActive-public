! ********************************************************
! * Relative permeability data taken from Imagine2 field *
! ********************************************************

TABLE RLPDATA.TWOPHASETABLEDEF Rel_Perm_TaUn_rlp_mat_watoil

! Columns: Sat, kr1, kr2
! Set units locally:
VOLFRAC PUSH
VOLFRAC = FT3/FT3
REAL PUSH
REAL = REALUNIT
REAL PUSH
REAL = REALUNIT

Rel_Perm_TaUn_rlp_mat_watoil PLOT_EDITABLE : VARDICT.ON

Rel_Perm_TaUn_rlp_mat_watoil CLEAR
Rel_Perm_TaUn_rlp_mat_watoil Sat INTERPOL  : VARDICT.RLPINTERPOL
Rel_Perm_TaUn_rlp_mat_watoil kr1 INTERPOL  : VARDICT.RLPINTERPOL
Rel_Perm_TaUn_rlp_mat_watoil kr2 INTERPOL  : VARDICT.RLPINTERPOL
Rel_Perm_TaUn_rlp_mat_watoil,  0.2000,  0.0000e+00,  8.0000e-01
Rel_Perm_TaUn_rlp_mat_watoil,  0.2500,  2.7310e-04,  5.8082e-01
Rel_Perm_TaUn_rlp_mat_watoil,  0.3000,  2.1848e-03,  4.1010e-01
Rel_Perm_TaUn_rlp_mat_watoil,  0.3500,  7.3737e-03,  2.8010e-01
Rel_Perm_TaUn_rlp_mat_watoil,  0.4000,  1.7478e-02,  1.8378e-01
Rel_Perm_TaUn_rlp_mat_watoil,  0.4500,  3.4138e-02,  1.1473e-01
Rel_Perm_TaUn_rlp_mat_watoil,  0.5000,  5.8990e-02,  6.7253e-02
Rel_Perm_TaUn_rlp_mat_watoil,  0.5500,  9.3673e-02,  3.6301e-02
Rel_Perm_TaUn_rlp_mat_watoil,  0.6000,  1.3983e-01,  1.7506e-02
Rel_Perm_TaUn_rlp_mat_watoil,  0.6500,  1.9909e-01,  7.1706e-03
Rel_Perm_TaUn_rlp_mat_watoil,  0.7000,  2.7310e-01,  2.2688e-03
Rel_Perm_TaUn_rlp_mat_watoil,  0.7500,  3.6350e-01,  4.4820e-04
Rel_Perm_TaUn_rlp_mat_watoil,  0.8000,  4.7192e-01,  2.8000e-05
Rel_Perm_TaUn_rlp_mat_watoil,  0.8500,  6.0000e-01,  0.0000e+00
Rel_Perm_TaUn_rlp_mat_watoil,  0.9000,  7.4939e-01,  0.0000e+00
Rel_Perm_TaUn_rlp_mat_watoil,  0.9500,  9.2171e-01,  0.0000e+00
Rel_Perm_TaUn_rlp_mat_watoil,  1.0000,  1.0000e+00,  0.0000e+00

! Restoring units:
VOLFRAC POP
REAL POP
REAL POP
