! *****************
! * Geometry data *
! *****************

GEODATA BEGIN

   ! Datum plane
   ORIGINZ = -4000.0 * M; DIPX = 0.0; DIPY = 0.0;

   ! Gridblock sizes equal in X- and Y-directions

   DELX[1:NX] = 8 * M;
   DELY[1:NY] = 8 * M;
   DELZ[1:NZ] = 4 * M;

   ! Calculate the cornerpoints for a box shaped reservoir
   GEOMODEL BOX

   ! Include absolute permeabilities
   INCLUDE Perm.inc

   ! Include void blocks
   INCLUDE Void_Blocks.inc

   ! Calculate grid block properties
   GEOMODEL CALCULATE

   ! Clear array of diffusion transmissibilities
   DTR CLEAR

GEODATA END

! ***************
! Porosity data *
! ***************

PORDATA BEGIN

   ! Reference pressure
   PREF[1:NX,1:NY,1:NZ,1:NF] = 400 * BAR;

   ! Include porosities
   INCLUDE Phi.inc

   ! Rock compressibility
   CR  [1:NX,1:NY,1:NZ,MAT] = 0 / PSI;

PORDATA END