!      ****************************************************************
!      *                                                              *
!      *  FrontEnd Atlas Library                                      *
!      *                                                              *
!      *  Support: RESupport / EPT-AE                                 *
!      *           Shell International Exploration & Production B.V.  *
!      *           Email   : RESupport                                *
!      *           Phone   : +31 70 4473320                           *
!      *           Fax     : +31 70 4472980, attn RESupport           *
!      *           Internet: resupport@shell.com                      *
!      *           SWW     : http://sww.siep.shell.com/mhr/support/index.html *
!      *                                                              *
!      *  Name: QC_ArraysV.INC                                        *
!      *                                                              *
!      *  Purpose: Same as QC_Array.INC, but for Reduce (Voxels)      *
!      *                                                              *
!      *  MoReS solver utilities component, see MoReS manual section  *
!      *  8.3 for details.                                            *
!      *                                                              *
!      ****************************************************************
!
!

IF ( NOT OBJ_EXISTS("PRJDICT.MAX_NBR_WARN") ) THEN EVAL\
(
   \\ Define local variable to containt the current warn. level
   VARIABLE INTEGER PRJDICT.MAX_NBR_WARN = WARNING_NUMBER;
   VARIABLE INTEGER PRJDICT.MAX_LEV_WARN = WARNING_LEVEL;
   PRJDICT.MAX_NBR_WARN HIDE;
   PRJDICT.MAX_LEV_WARN HIDE;

   \\ open the QC-dictionary and leave it open...
   UTILITIES BEGIN;
)

\\ Include this file only once
IF ( 1 ) THEN EVAL\
(
   ! Temporarily disable warnings
   WARNING_NUMBER = 0;
   WARNING_LEVEL = 1;

\\Proceed without indent...
UTILITIES BEGIN;


FUNCTION INTEGER MOD( INTEGER arg, INTEGER mod )\
(
   RETURN( arg - (INT(arg/mod) * mod) );
);

! Make dummy variables/ranges (if not avialable in this Appl)
IF ( NOT OBJ_EXISTS( "XYZV" ) ) THEN EVAL
(
   RANGE XYZ XYZV;
);

IF ( NOT OBJ_EXISTS( "XYZDV" ) ) THEN EVAL
(
   RANGE XYZ XYZDV;
);


IF ( NOT OBJ_EXISTS( "NX_VOXEL" ) ) THEN EVAL
(
   VARIABLE INTEGER NX_VOXEL NY_VOXEL NZ_VOXEL;
   NX_VOXEL = -1;
   NY_VOXEL = -1;
   NZ_VOXEL = -1;
);

IF ( NOT OBJ_EXISTS( "NC" ) ) THEN EVAL
(
   VARIABLE INTEGER NC = -1;
);

IF ( NOT OBJ_EXISTS( "NP" ) ) THEN EVAL
(
   VARIABLE INTEGER NP = -1;
);




!******************************************************************************
!
! Make integer directions dirX = DIR_X if it exists, -1 if not
!                         dirY = DIR_Y if it exists, -1 if not
!                         dirZ = DIR_Z if it exists, -1 if not
!
!******************************************************************************

FUNCTION INTEGER ArrayMakeDIR( )
(
   VARIABLE INTEGER done = 0;

   ! PRINT("DEBUG:\t starting ArrayMakeDIR...\n" );

   IF( (NOT OBJ_EXISTS( "VARDICT.DIR_X" )) AND
       (NOT OBJ_EXISTS( "GEODATA.DIR_X" )) ) THEN EVAL
   (
      VARIABLE INTEGER VARDICT.DIR_X = -1;
      done += 1;
   );

   IF( (NOT OBJ_EXISTS( "VARDICT.DIR_Y" )) AND
       (NOT OBJ_EXISTS( "GEODATA.DIR_Y" )) ) THEN EVAL
   (
      VARIABLE INTEGER VARDICT.DIR_Y = -1;
      done += 1;
   );

   IF( (NOT OBJ_EXISTS( "VARDICT.DIR_Z" )) AND
       (NOT OBJ_EXISTS( "GEODATA.DIR_Z" )) ) THEN EVAL

   (
      VARIABLE INTEGER VARDICT.DIR_Z = -1;
      done += 1;
   );

   IF( (NOT OBJ_EXISTS( "VARDICT.DIR_F" )) AND
       (NOT OBJ_EXISTS( "GEODATA.DIR_F" )) ) THEN EVAL

   (
      VARIABLE INTEGER VARDICT.DIR_F = -1;
      done += 1;
   );

   IF( (NOT OBJ_EXISTS( "VARDICT.DIR_ZF" )) AND
       (NOT OBJ_EXISTS( "GEODATA.DIR_ZF" )) ) THEN EVAL

   (
      VARIABLE INTEGER VARDICT.DIR_ZF = -1;
      done += 1;
   );

   IF( (NOT OBJ_EXISTS( "VARDICT.DIR_FH" )) AND
       (NOT OBJ_EXISTS( "GEODATA.DIR_FH" )) ) THEN EVAL

   (
      VARIABLE INTEGER VARDICT.DIR_FH = -1;
      done += 1;
   );

   RETURN( done );
);


!*******************************************************************************
!
! Make an integer XYZF-ranged void array, to facilitate XYZF looping
! Make an integer XYZF-ranged with only non-void blocks
! ... put in GEODATA
!
!*******************************************************************************

FUNCTION OBJECT_ID ArrayGetVoid( INTEGER type, INTEGER Refresh->OFF, INTEGER Info->OFF )\
(
   VARIABLE STRING nm;
   VARIABLE OBJECT_ID voidID;

   IF ( type EQ 0 ) THEN
   (
      nm = "GEODATA.VOIDXYZF";
      IF ( NOT OBJ_EXISTS( "GEODATA.VOIDXYZF" ) ) THEN EVAL
      (
         ARRAY INTEGER XYZF GEODATA.VOIDXYZF;
         Refresh = 1;
      );

      IF ( Refresh EQ 1 ) THEN EVAL
      (
         GEODATA.VOIDXYZF CLEAR;

         FOREACH g IN XYZ DO
         (
            IF ( GEODATA.VOIDBLKS[<g>] EQ 1 ) THEN
            (
               GEODATA.VOIDXYZF[<g>, 1]  = 1;
               GEODATA.VOIDXYZF[<g>, NF] = 1;
            );
         );
      );
      EVAL( voidID = GEODATA.VOIDXYZF ID );
   )
   ELSE IF ( type EQ -1 ) THEN EVAL
   (
      \\ XYZ type, default VOID array, should be present
      voidID = GEODATA.VOIDBLKS ID;
      nm = "GEODATA.VOIDBLKS";
   )
   ELSE IF ( type LE -2 ) THEN EVAL
   (
      \\ voxel type; use the MONDATA.PHIV to find void blocks
      nm = "MONDATA.VOIDXYZV";
      IF ( NOT OBJ_EXISTS( "MONDATA.VOIDXYZV" ) ) THEN EVAL
      (
         ARRAY INTEGER XYZV MONDATA.VOIDXYZV;
         Refresh = 1;
         voidID = MONDATA.VOIDXYZV ID;
      );

      IF ( Refresh EQ 1 ) THEN EVAL
      (
         MONDATA.VOIDXYZV CLEAR;

         FOREACH g IN XYZV DO
         (
            IF ( MONDATA.PHIV[<g>] LE TINY ) THEN
            (
               MONDATA.VOIDXYZV[<g>]  = 1;
            );
         );
      );
      EVAL( voidID = MONDATA.VOIDXYZV ID );
   );


   IF ( Info GT 0 ) THEN
   (
      PRINT("INFO:\tComputed void array in ", STRING, nm, "\n" );
   );

   RETURN( voidID );
);



!*******************************************************************************
!
! Restrict an input XYZF range to either MAT or FRC
!
!*******************************************************************************

FUNCTION VOID ArrayRestrictMatFrc( OBJECT_ID FullRange, OBJECT_ID PartRange,
                                   INTEGER frc )
(
   ARRAY INTEGER XYZF tmpMap;
   RANGE XYZF tmpRng;

   IF ( EXISTS( FullRange ) AND EXISTS( PartRange ) ) THEN EVAL
   (
      IF ( NF EQ 1 ) THEN
      (
         tmpRng = XYZF_ACTIVE;
         tmpRng INVERT;
         $PartRange = $FullRange - tmpRng;

         tmpRng CLEAR;
      )
      ELSE
      (
         $PartRange CLEAR;
         $PartRange INVERT;

         tmpMap XYZF = 0;
         FOREACH g IN $FullRange DO
         (
            tmpMap[<g>] = 1;
         );

         IF ( frc EQ MAT ) THEN
         (
            FOREACH g IN XYZF DO
            (
               IF ( MOD( RANGE_POS( XYZF, [<g>] ), NF ) EQ 1 ) THEN
               (
                  IF ( tmpMap[<g>] EQ 1 AND GEODATA.VOIDXYZF[<g>] EQ 0 ) THEN
                  (
                     $PartRange += [<g>];
                  );
               );
            );
         );
      
         IF ( frc EQ NF ) THEN
         (
            FOREACH g IN XYZF DO
            (
               IF ( MOD( RANGE_POS( XYZF, [<g>] ), NF ) EQ 0 ) THEN
               (
                  IF ( tmpMap[<g>] EQ 1 AND GEODATA.VOIDXYZF[<g>] EQ 0 ) THEN
                  (
                     $PartRange += [<g>];
                  );
               );
            );
         );
         tmpMap CLEAR;

         ! PRINT("DEBUG:\t frac index = ", INTEGER, frc,"\n" );
         ! $PartRange CONTENTS;
      );
   )
   ELSE
   (
      ERROR(1, "Illegal ranges in conversion..." );
   );
);

!*******************************************************************************
!
! Compute array dimensions from named range
!
!*******************************************************************************

FUNCTION INTEGER ArraySubDim( OBJECT_ID array, OBJECT_ID tblID,
                              OBJECT_ID Dim1, OBJECT_ID Dim2,
                              OBJECT_ID Dim3, OBJECT_ID Dim4 )
(
   VARIABLE OBJECT_ID rngID;
   VARIABLE STRING cmp zeros range str str1 cmd;
   VARIABLE INTEGER err indx len extra;

   IF ( NOT IS_ARRAY( array ) ) THEN
   (
      ERROR(1, "ArrayStat: Input must be an array\n" );
   );

   EVAL
   (
      \\ get the quantity/unit of this array
      rngID = OBJ_GETRANGE( array );
      range = SPRINT( NAME, rngID );

      \\ check if the range is XYZ, XYZF or XYZV
      str = STRSTR( range, "XYZ" );

      err = 10;
      IF ( STRLEN( str ) GT 0 ) THEN
      (
         \\ if none of the ones below fit, it is an XYZ.. range
         err = -1;

         \\ is it a XYZF
         str1  = STRSTR( range, "XYZF" );
         IF ( STRLEN( str1 )  GT 0 ) THEN
         (
            err = 0;
         );

         \\ is it a voxel range?
         str1  = "XYZV";
         str1  = STRSTR( range, "XYZV" );
         IF ( STRLEN( str1 )  GT 0 ) THEN
         (
            err = -2;
            str = str1;
         );

         \\ Or, finally, is it a voxel D-range?
         str1  = "XYZDV";
         str1  = STRSTR( range, "XYZDV" );
         IF ( STRLEN( str1 )  GT 0 ) THEN
         (
            err = -3;
            str = str1;
         );
      );

      \\ classification:
      \\ XYZF => err = 0
      \\ XYZ  => err = -1
      \\ XYZV => err = -2
      \\ XYZDV => err = -3


      ! PRINT("DEBUG:\t err = ", INTEGER, err,"\n" );
      ! PRINT("DEBUG:\t range     = ", STRING, range,"\n" );
      ! PRINT("DEBUG:\t range bit = ", STRING, str,"\n" );

      \\ find the sub-dimensions
      \\ replace RD by R
      \\ DIFD -> DCP

      \\ always start from empty table
      $tblID CLEAR;

      \\ make sure all DIR's are present
      ArrayMakeDIR();

      \\ any extra (non-standard) dimensions appended?
      extra = 0;

      IF ( $Dim1 LT 0 AND err LE 0 ) THEN
      (
         IF ( EXISTS( rngID ) ) THEN
         (
            indx = 0;
            len = STRLEN( range );

            IF ( STRIDX( range, "XYZFCP" ) EQ 1 OR
                 STRIDX( range, "XYZCP" )  EQ 1 OR
                 STRIDX( range, "XYZVCP" ) EQ 1 ) THEN
            (
               $Dim1 = NC;
               $Dim2 = NP;

               WHILE ( indx+=1; indx LE NC ) DO
               (
                  ! PRINT("DEBUG:\t idx = ", INTEGER, indx,"\n" );
                  cmd = SPRINT( FULLNAME, tblID,", \q", C, indx,"\q\n" );
                  EXEC_STRING( cmd );
               );
  
               indx -= 1;
               WHILE ( indx+=1; indx LE NC + NP ) DO
               (
                  ! PRINT("DEBUG:\t idx = ", INTEGER, indx,"\n" );
                  cmd = SPRINT( FULLNAME, tblID,", \q", P, indx - NC,"\q\n" );
                  EXEC_STRING( cmd );
               );

               IF ( err EQ 0 AND len GT 6 ) THEN EVAL
               (
                  $Dim3 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*NF*NC*NP);
               )
               ELSE IF ( err EQ -1 AND len GT 5 ) THEN EVAL
               (
                  $Dim3 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*NC*NP);
               );
               extra = $Dim3;
            )
            ELSE IF ( STRIDX( range, "XYZFC" ) EQ 1 OR
                      STRIDX( range, "XYZC"  ) EQ 1 OR
                      STRIDX( range, "XYZVC" ) EQ 1 ) THEN
            (
               $Dim1 = NC;
               WHILE ( indx+=1; indx LE NC ) DO
               (
                  cmd = SPRINT( FULLNAME, tblID,", \q",C, indx,"\q\n" );
                  EXEC_STRING( cmd );
               );
   
               IF ( err EQ 0 AND len GT 5 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*NF*NC);
               )
               ELSE IF ( err EQ -1 AND len GT 4 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*NC);
               ) 
               ELSE IF ( err EQ -2 AND len GT 5 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX_VOXEL*NY_VOXEL*NZ_VOXEL*NC);
               );
               extra = $Dim2;
            )
            ELSE IF ( STRIDX( range, "XYZFP" ) EQ 1 OR
                      STRIDX( range, "XYZP"  ) EQ 1 OR
                      STRIDX( range, "XYZVP" ) EQ 1 ) THEN
            (
               $Dim1 = NP;
               WHILE ( indx+=1; indx LE NP ) DO
               (
                  cmd = SPRINT( FULLNAME, tblID,", \q",P, indx,"\q\n" );
                  EXEC_STRING( cmd );
               );
   
               IF ( err EQ 0 AND len GT 5 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*NF*NP);
               )
               ELSE IF ( err EQ -1 AND len GT 4 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*NP);
               ) 
               ELSE IF ( err EQ -2 AND len GT 5 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX_VOXEL*NY_VOXEL*NZ_VOXEL*NP);
               );
               extra = $Dim2;
            )
            ELSE IF ( STRIDX( range, "XYZFD" ) EQ 1 OR
                      STRIDX( range, "XYZD"  ) EQ 1 OR
                      STRIDX( range, "XYZDV" ) EQ 1 ) THEN
            (
               $Dim1 = ND;

               WHILE ( indx+=1; indx LE ND ) DO
               (
                  IF ( indx EQ DIR_X ) THEN
                  (
                     cmd = SPRINT( FULLNAME, tblID,", \qDIR_X\q\n" );
                  )
                  ELSE IF ( indx EQ DIR_Y ) THEN
                  (
                     cmd = SPRINT( FULLNAME, tblID,", \qDIR_Y\q\n" );
                  )
                  ELSE IF ( indx EQ DIR_Z ) THEN
                  (
                     cmd = SPRINT( FULLNAME, tblID,", \qDIR_Z\q\n" );
                  )
                  ELSE IF ( indx EQ DIR_F ) THEN
                  (
                     cmd = SPRINT( FULLNAME, tblID,", \qDIR_F\q\n" );
                  )
                  ELSE IF ( indx EQ DIR_ZF ) THEN
                  (
                     cmd = SPRINT( FULLNAME, tblID,", \qDIR_ZF\q\n" );
                  )
                  ELSE IF ( indx EQ DIR_FH ) THEN
                  (
                     cmd = SPRINT( FULLNAME, tblID,", \qDIR_FH\q\n" );
                  );
                  EXEC_STRING( cmd );
               );
   
               IF ( err EQ 0 AND len GT 5 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*NF*ND);
               )
               ELSE IF ( err EQ -1 AND len GT 4 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*ND);
               )
               ELSE IF ( err EQ -3 AND len GT 5 ) THEN EVAL
               (
                  $Dim2 = RANGE_NBR_ELEMENTS( $range ) / (NX_VOXEL*NY_VOXEL*NZ_VOXEL*ND);
               );
               extra = $Dim2
            )
            ELSE
            (
               \\ range that starts wit XYZ or XYZF or XYZV with more to go
               \\ Tricky if this works (may have too many dims)
               IF ( err EQ 0 AND len GT 4 ) THEN
               (
                  $Dim1 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ*NF);
                  extra = $Dim1
               )
               ELSE IF ( err EQ -1 AND len GT 3 ) THEN
               (
                  $Dim1 = RANGE_NBR_ELEMENTS( $range ) / (NX*NY*NZ);
                  extra = $Dim1
               )
               ELSE IF ( err EQ -2 AND len GT 4 ) THEN
               (
                  $Dim1 = RANGE_NBR_ELEMENTS( $range ) / (NX_VOXEL*NY_VOXEL*NZ_VOXEL);
                  extra = $Dim1
               )
               ELSE IF ( (err EQ  0 AND len EQ 4) OR
                         (err EQ -1 AND len EQ 3) OR
                         (err EQ -2 AND len EQ 4) OR
                         (err EQ -3 AND len EQ 5)  ) THEN
               (
                  \\ this is fine, simple range, no extra dims
                  err *= 1;
               )
               ELSE
               (
                  \\ this is too complex: error
                  err = 4;
               );
            );

            indx = 0;
            WHILE ( indx+=1; indx LE extra ) DO
            (
               cmd = SPRINT( FULLNAME, tblID,",\qC",INTEGER, indx,"\q\n" );
               EXEC_STRING( cmd );
            );
         )
         ELSE
         (
            err = 4;
         );
      )
      ELSE EVAL
      (
         indx = 0;
         WHILE ( indx+=1; indx LE $Dim1 ) DO
         (
            cmd = SPRINT( FULLNAME, tblID,", \q",INTEGER, indx,"\q\n" );
            EXEC_STRING( cmd );
         );
         indx = 0;
         WHILE ( indx+=1; indx LE $Dim2 ) DO
         (
            cmd = SPRINT( FULLNAME, tblID,", \q",INTEGER, indx,"\q\n" );
            EXEC_STRING( cmd );
         );
         indx = 0;
         WHILE ( indx+=1; indx LE $Dim3 ) DO
         (
            cmd = SPRINT( FULLNAME, tblID,", \q",INTEGER, indx,"\q\n" );
            EXEC_STRING( cmd );
         );
         indx = 0;
         WHILE ( indx+=1; indx LE $Dim4 ) DO
         (
            cmd = SPRINT( FULLNAME, tblID,", \q",INTEGER, indx,"\q\n" );
            EXEC_STRING( cmd );
         );
      );

      $tblID, "";
      $tblID, "";
      $tblID, "";
      $tblID, "";
   );

   \\DEBUG
   ! EVAL
  !  (
  !     $tblID CONTENTS;
  !     PRINT( "err   = ", INTEGER, err,"\n");
  !     IF ( err EQ 0  ) THEN PRINT(" range type = XYZF\n");
  !     IF ( err EQ -1 ) THEN PRINT(" range type = XYZ\n");
  !     IF ( err EQ -2 ) THEN PRINT(" range type = XYZV\n");
  !     IF ( err EQ -3 ) THEN PRINT(" range type = XYZDV\n");
  !     PRINT( "Dim 1 = ", INTEGER, $Dim1,"\n");
  !     PRINT( "Dim 2 = ", INTEGER, $Dim2,"\n");
  !     PRINT( "Dim 3 = ", INTEGER, $Dim3,"\n");
  !     PRINT( "Dim 4 = ", INTEGER, $Dim4,"\n");
  ! );


   RETURN( err );
);
   

FUNCTION VOID ArrayStatXYZF( OBJECT_ID array, OBJECT_ID cmpTbl,
                             OBJECT_ID Region, OBJECT_ID Weight,
                             INTEGER Dim1, INTEGER Dim2,
                             INTEGER Dim3, INTEGER Dim4, INTEGER Info  )
(
   VARIABLE REAL minz min max avg std val tot;
   VARIABLE REAL minz1 min1 max1 avg1 std1;
   VARIABLE REAL minz2 min2 max2 avg2 std2;
   VARIABLE INTEGER i1 i2 i3 i4 err NDim1 NDim2 NDim3 NDim4;
   VARIABLE OBJECT_ID cmpCol rngID;
   VARIABLE STRING qwgt quant minzBlk1 minzBlk2 minBlk1 minBlk2 maxBlk1 maxBlk2;
   VARIABLE STRING st1 st2 st3 st4 cmpStr wgt cmp str;

   RANGE XYZF theRange;
   RANGE XYZF minzBlk minBlk maxBlk;

   ! PRINT("DEBUG:\tXYZF range: Dim1 = ", INTEGER,Dim1,"Dim2 = ", INTEGER, Dim2,
   !       "Dim3 = ", INTEGER, Dim3,"\n");
   EVAL
   (
      cmpCol = TABLE_COLUMN_ID( cmpTbl, 1 );
      quant = OBJ_GETUNIT( array );

      IF ( Dim1 LE 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = " ";
         str = " ";
         NDim1 = 1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1 ";
         str = ", i1 ";
         NDim1 = Dim1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2 ";
         str = ", i1, i2 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 GT 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2, i3 ";
         str = ", i1, i2, i3 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = Dim3; NDim4 = 1;
      )
      ELSE
      (
         cmp = ", i1, i2, i3, i4 ";
         str = ", i1, i2, i3, i4 ";
      );

      \\ is there a weight to be used in the averaging?
      IF ( Weight GT 1 ) THEN
      (
         IF ( NOT IS_ARRAY( Weight ) ) THEN
         (
            ERROR(1, "ArrayStat: Weight must be an array\n" );
         );

         qwgt  = OBJ_GETUNIT( Weight );
         rngID = OBJ_GETRANGE( Weight );

         IF ( STRCMP( SPRINT( NAME, rngID ), "XYZF" ) NE 0 ) THEN
         (
            ERROR(1, "ArrayStat: Weight must have an XYZF range\n" );
         );
            
         wgt   = SPRINT( "(", FULLNAME, Weight,"[<g>] / (",STRING, qwgt,"))" );
      )
      ELSE
      (
         wgt = "1.0";
      );
 
      EVAL
      (
         i4 = 0;
         WHILE ( i4+=1; i4 LE NDim4 ) DO
         (
            i3 = 0;
            WHILE ( i3+=1; i3 LE NDim3 ) DO
            (
               i2 = 0;
               WHILE ( i2+=1; i2 LE NDim2 ) DO
               (
                  i1 = 0;
                  WHILE ( i1+=1; i1 LE NDim1 ) DO
                  (
                     F = 0;
                     WHILE( F+=1; F LE NF ) DO
                     (
                        min     = HUGE; 
                        minz    = HUGE; 
                        max     =-HUGE; 
                        avg     = 0.0; 
                        std     = 0.0; 
                        minBlk  = [1,1,1,MAT];
                        minzBlk = [1,1,1,MAT];
                        maxBlk  = [1,1,1,MAT];

                        \\ restrict the active range to MAT or FRC
                        ArrayRestrictMatFrc( Region, theRange ID, F );

                        tot    = 0.0; 
                        FOREACH g IN $theRange DO
                        (
                           tot += $wgt;
                           val =  $array[ <g> $cmp ] / ( $quant );

                           IF( val LT minz ) THEN
                           (
                              IF ( val GT 1.E-20 ) THEN
                              (
                                 minzBlk = [<g>];
                                 minz    = val;
                              )
                              ELSE IF ( val LT min ) THEN
                              (
                                 minBlk = [<g>];
                                 min    = val;
                              );
                           );
                           IF( val GT max ) THEN
                           (
                               maxBlk = [<g>];
                               max    = val;
                           );
                           avg += $wgt * val;
                           std += $wgt * val*val;
                        );
                        IF ( min GT minz ) THEN
                        (
                           min = minz;
                           minBlk = minzBlk;
                        );

                        avg /= (TINY + tot);
                        std /= (TINY + tot);
                        std = MAX(0, std - avg^2)^0.5;

                        IF ( Info GT 0 ) THEN
                        (
                           \\ get the component names in a string
                           st1 = COLUMN_STRINGVALUE( cmpCol, i1 );
                           st2 = COLUMN_STRINGVALUE( cmpCol, NDim1 + i2 );
                           st3 = COLUMN_STRINGVALUE( cmpCol, NDim2 + NDim1 + i3 );
                           st4 = COLUMN_STRINGVALUE( cmpCol, NDim3 + NDim2 + NDim1 + i4 );
         
                           cmpStr = "";
                           IF ( STRLEN( st1 ) GT 0 ) THEN cmpStr = cmpStr + st1;
                           IF ( STRLEN( st2 ) GT 0 ) THEN cmpStr = cmpStr + "," + st2;
                           IF ( STRLEN( st3 ) GT 0 ) THEN cmpStr = cmpStr + "," + st3;
                           IF ( STRLEN( st4 ) GT 0 ) THEN cmpStr = cmpStr + "," + st4;

                           IF ( STRLEN( cmpStr ) GT 0 ) THEN
                           (
                              IF ( F EQ 1 ) THEN
                              (
                                 PRINT( " \t", NAME, array, 
                                  "[",STRING, cmpStr,",MAT]\n\tAvg: ", REAL, avg);
                              )
                              ELSE
                              (
                                 PRINT( " \t", NAME, array, 
                                  "[",STRING, cmpStr,",FRC]:\n\tAvg: ", REAL, avg);
                              );
                           )
                           ELSE
                           (
                              IF ( F EQ 1 ) THEN
                              (
                                 PRINT( " \t", NAME, array, "[MAT]:\n\tAvg: ", REAL, avg);
                              )
                              ELSE
                              (
                                 PRINT( " \t", NAME, array, "[FRC]:\n\tAvg: ", REAL, avg);
                              );
                           );

                           PRINT(" (", REAL, std, ") * ", STRING, quant );
                           IF ( Weight GT 1 ) THEN
                           (
                              PRINT(", weighted by: ", FULLNAME, Weight,"\n" );
                           )
                           ELSE
                           (
                              PRINT(", non-weighted\n" );
                           );
                           PRINT("\tMin: ", REAL, min );
                           PRINT(" ", RANGE_INDEX, XYZF minBlk );
                           IF ( minz GT min AND minz LT HUGE ) THEN
                           (
                              PRINT("; non-zero: ", REAL, minz );
                              PRINT(" ",RANGE_INDEX, XYZF minzBlk );
                           );
                           PRINT("; max: ", REAL, max );
                           PRINT(" ",RANGE_INDEX, XYZF maxBlk,".\n\n");
                        );

                        IF ( F EQ 1 ) THEN
                        (
                           minz1 = minz; min1 = min; max1 = max; avg1 = avg; std1 = std;
                           minz2 = minz; min2 = min; max2 = max; avg2 = avg; std2 = std;
                           minzBlk1 = SPRINT(RANGE_INDEX, XYZF minzBlk);
                           minBlk1 = SPRINT(RANGE_INDEX, XYZF minBlk);
                           maxBlk1 = SPRINT(RANGE_INDEX, XYZF maxBlk);
                        );
                        IF ( F EQ 2 ) THEN
                        (
                           minz2 = minz; min2 = min; max2 = max; avg2 = avg; std2 = std;
                           minzBlk2 = SPRINT(RANGE_INDEX, XYZF minzBlk);
                           minBlk2 = SPRINT(RANGE_INDEX, XYZF minBlk);
                           maxBlk2 = SPRINT(RANGE_INDEX, XYZF maxBlk);
                        );
                     );
                  );
               );
            );
         );
      );
   );
);


!*******************************************************************************
!
! Compute statistics of input array (min, max, st. dev.)
!
!*******************************************************************************
FUNCTION VOID ArrayStatXYZ( OBJECT_ID array, OBJECT_ID cmpTbl, OBJECT_ID Region,
                            OBJECT_ID Weight, INTEGER Dim1, INTEGER Dim2,
                            INTEGER Dim3, INTEGER Dim4, INTEGER Info  )
(
   VARIABLE REAL minz min max avg std val tot;
   VARIABLE REAL minz1 min1 max1 avg1 std1;
   VARIABLE REAL minz2 min2 max2 avg2 std2;
   VARIABLE INTEGER i1 i2 i3 i4 err NDim1 NDim2 NDim3 NDim4;
   VARIABLE OBJECT_ID cmpCol rngID;
   VARIABLE STRING qwgt quant minzBlk1 minzBlk2 minBlk1 minBlk2 maxBlk1 maxBlk2 cmp str;
   VARIABLE STRING st1 st2 st3 st4 cmpStr wgt;

   RANGE XYZ minzBlk minBlk maxBlk;

   ! PRINT("DEBUG:\tXYZ range: Dim1 = ", INTEGER,Dim1,"Dim2 = ", INTEGER, Dim2,
   !       "Dim3 = ", INTEGER, Dim3,"\n");
   EVAL
   (
      cmpCol = TABLE_COLUMN_ID( cmpTbl, 1 );
      quant = OBJ_GETUNIT( array );

      IF ( Dim1 LE 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = " ";
         str = " ";
         NDim1 = 1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1 ";
         str = ", i1 ";
         NDim1 = Dim1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2 ";
         str = ", i1, i2 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 GT 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2, i3 ";
         str = ", i1, i2, i3 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = Dim3; NDim4 = 1;
      )
      ELSE
      (
         cmp = ", i1, i2, i3, i4 ";
         str = ", i1, i2, i3, i4 ";
      );

      \\ is there a weight to be used in the averaging?
      IF ( Weight GT 1 ) THEN
      (
         IF ( NOT IS_ARRAY( Weight ) ) THEN
         (
            ERROR(1, "ArrayStat: Weight must be an array\n" );
         );

         qwgt  = OBJ_GETUNIT( Weight );
         rngID = OBJ_GETRANGE( Weight );

         IF ( STRLEN( STRSTR( SPRINT( NAME, rngID ), "XYZ" ) ) EQ 0 ) THEN
         (
            ERROR(1, "ArrayStat: Weight must have an XYZ, XYZF or XYZV range\n" );
         );
            
         IF ( STRCMP( SPRINT( NAME, rngID ), "XYZF" ) EQ 0 ) THEN
         (
            wgt = SPRINT( "(", FULLNAME, Weight,"[<g>,MAT] / (",STRING, qwgt,"))" );
         )
         ELSE
         (
            wgt = SPRINT( "(", FULLNAME, Weight,"[<g>] / (",STRING, qwgt,"))" );
         );
      )
      ELSE
      (
         wgt = "1.0";
      );
 
      EVAL
      (
         i4 = 0;
         WHILE ( i4+=1; i4 LE NDim4 ) DO
         (
            i3 = 0;
            WHILE ( i3+=1; i3 LE NDim3 ) DO
            (
               i2 = 0;
               WHILE ( i2+=1; i2 LE NDim2 ) DO
               (
                  i1 = 0;
                  WHILE ( i1+=1; i1 LE NDim1 ) DO
                  (
                     min     = HUGE; 
                     minz    = HUGE; 
                     max     =-HUGE; 
                     avg     = 0.0; 
                     std     = 0.0; 
                     minBlk  = [1,1,1];
                     minzBlk = [1,1,1];
                     maxBlk  = [1,1,1];

                     tot    = 0.0; 
                     FOREACH g IN $Region DO
                     (
                        tot += $wgt;
                        val =  $array[ <g> $cmp ] / ( $quant );

                        IF( val LT minz ) THEN
                        (
                           IF ( val GT 1.E-20 ) THEN
                           (
                              minzBlk = [<g>];
                              minz    = val;
                           )
                           ELSE IF ( val LT min ) THEN
                           (
                              minBlk = [<g>];
                              min    = val;
                           );
                        );
                        IF( val GT max ) THEN
                        (
                            maxBlk = [<g>];
                            max    = val;
                        );
                        avg += $wgt * val;
                        std += $wgt * val*val;
                     );
                     IF ( min GT minz ) THEN
                     (
                        min = minz;
                        minBlk = minzBlk;
                     );

                     avg /= (TINY + tot);
                     std /= (TINY + tot);
                     std = MAX(0, std - avg^2)^0.5;

                     IF ( Info GT 0 ) THEN
                     (
                        \\ get the component names in a string
                        st1 = COLUMN_STRINGVALUE( cmpCol, i1 );
                        st2 = COLUMN_STRINGVALUE( cmpCol, NDim1 + i2 );
                        st3 = COLUMN_STRINGVALUE( cmpCol, NDim2 + NDim1 + i3 );
                        st4 = COLUMN_STRINGVALUE( cmpCol, NDim3 + NDim2 + NDim1 + i4 );
      
                        cmpStr = "";
                        IF ( STRLEN( st1 ) GT 0 ) THEN cmpStr = cmpStr + st1;
                        IF ( STRLEN( st2 ) GT 0 ) THEN cmpStr = cmpStr + "," + st2;
                        IF ( STRLEN( st3 ) GT 0 ) THEN cmpStr = cmpStr + "," + st3;
                        IF ( STRLEN( st4 ) GT 0 ) THEN cmpStr = cmpStr + "," + st4;

                        IF ( STRLEN( cmpStr ) GT 0 ) THEN
                        (
                           PRINT( " \t", NAME, array, 
                               "[",STRING, cmpStr,"]\n\tAvg: ", REAL, avg);
                        )
                        ELSE
                        (
                           PRINT( " \t", NAME, array, ":\n\tAvg: ", REAL, avg);
                        );

                        PRINT(" (", REAL, std, ") * ", STRING, quant );
                        IF ( Weight GT 1 ) THEN
                        (
                           PRINT(", weighted by: ", FULLNAME, Weight,"\n" );
                        )
                        ELSE
                        (
                           PRINT(", non-weighted\n" );
                        );
                        PRINT("\tMin: ", REAL, min );
                        PRINT(" ", RANGE_INDEX, XYZ minBlk );
                        IF ( minz GT min AND minz LT HUGE ) THEN
                        (
                           PRINT("; non-zero: ", REAL, minz );
                           PRINT(" ",RANGE_INDEX, XYZ minzBlk );
                        );
                        PRINT("; max: ", REAL, max );
                        PRINT(" ",RANGE_INDEX, XYZ maxBlk,".\n\n");
                     );

                     minz1 = minz; min1 = min; max1 = max; avg1 = avg; std1 = std;
                     minz2 = minz; min2 = min; max2 = max; avg2 = avg; std2 = std;
                     minzBlk1 = SPRINT(RANGE_INDEX, XYZ minzBlk);
                     minBlk1 = SPRINT(RANGE_INDEX, XYZ minBlk);
                     maxBlk1 = SPRINT(RANGE_INDEX, XYZ maxBlk);
                  );
               );
            );
         );
      );
   );
);



!*******************************************************************************
!
! Compute statistics of input array (min, max, st. dev.)
!
!*******************************************************************************
FUNCTION VOID ArrayStatXYZV( OBJECT_ID array, OBJECT_ID cmpTbl, OBJECT_ID Region,
                            OBJECT_ID Weight, INTEGER Dim1, INTEGER Dim2,
                            INTEGER Dim3, INTEGER Dim4, INTEGER Info  )
(
   VARIABLE REAL minz min max avg std val tot;
   VARIABLE REAL minz1 min1 max1 avg1 std1;
   VARIABLE REAL minz2 min2 max2 avg2 std2;
   VARIABLE INTEGER i1 i2 i3 i4 err NDim1 NDim2 NDim3 NDim4;
   VARIABLE OBJECT_ID cmpCol rngID;
   VARIABLE STRING qwgt quant minzBlk1 minzBlk2 minBlk1 minBlk2 maxBlk1 maxBlk2 cmp str;
   VARIABLE STRING st1 st2 st3 st4 cmpStr wgt;

   RANGE XYZV minzBlk minBlk maxBlk;

   ! PRINT("DEBUG:\tXYZV range: Dim1 = ", INTEGER,Dim1,"Dim2 = ", INTEGER, Dim2,
   !       "Dim3 = ", INTEGER, Dim3,"\n");
   EVAL
   (
      cmpCol = TABLE_COLUMN_ID( cmpTbl, 1 );
      quant = OBJ_GETUNIT( array );

      IF ( Dim1 LE 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = " ";
         str = " ";
         NDim1 = 1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1 ";
         str = ", i1 ";
         NDim1 = Dim1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2 ";
         str = ", i1, i2 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 GT 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2, i3 ";
         str = ", i1, i2, i3 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = Dim3; NDim4 = 1;
      )
      ELSE
      (
         cmp = ", i1, i2, i3, i4 ";
         str = ", i1, i2, i3, i4 ";
      );

      \\ is there a weight to be used in the averaging?
      IF ( Weight GT 1 ) THEN
      (
         IF ( NOT IS_ARRAY( Weight ) ) THEN
         (
            ERROR(1, "ArrayStat: Weight must be an array\n" );
         );

         qwgt  = OBJ_GETUNIT( Weight );
         rngID = OBJ_GETRANGE( Weight );

         IF ( SPRINT( NAME, rngID ) NE "XYZ" ) THEN
         (
            ERROR(1, "ArrayStat: Weight must have an XYZV range\n" );
         );
            
         wgt = SPRINT( "(", FULLNAME, Weight,"[<g>] / (",STRING, qwgt,"))" );
      )
      ELSE
      (
         wgt = "1.0";
      );
 
      EVAL
      (
         i4 = 0;
         WHILE ( i4+=1; i4 LE NDim4 ) DO
         (
            i3 = 0;
            WHILE ( i3+=1; i3 LE NDim3 ) DO
            (
               i2 = 0;
               WHILE ( i2+=1; i2 LE NDim2 ) DO
               (
                  i1 = 0;
                  WHILE ( i1+=1; i1 LE NDim1 ) DO
                  (
                     min     = HUGE; 
                     minz    = HUGE; 
                     max     =-HUGE; 
                     avg     = 0.0; 
                     std     = 0.0; 
                     minBlk  = [1,1,1];
                     minzBlk = [1,1,1];
                     maxBlk  = [1,1,1];

                     tot    = 0.0; 
                     FOREACH g IN $Region DO
                     (
                        tot += $wgt;
                        val =  $array[ <g> $cmp ] / ( $quant );

                        IF( val LT minz ) THEN
                        (
                           IF ( val GT 1.E-20 ) THEN
                           (
                              minzBlk = [<g>];
                              minz    = val;
                           )
                           ELSE IF ( val LT min ) THEN
                           (
                              minBlk = [<g>];
                              min    = val;
                           );
                        );
                        IF( val GT max ) THEN
                        (
                            maxBlk = [<g>];
                            max    = val;
                        );
                        avg += $wgt * val;
                        std += $wgt * val*val;
                     );
                     IF ( min GT minz ) THEN
                     (
                        min = minz;
                        minBlk = minzBlk;
                     );

                     avg /= (TINY + tot);
                     std /= (TINY + tot);
                     std = MAX(0, std - avg^2)^0.5;

                     IF ( Info GT 0 ) THEN
                     (
                        \\ get the component names in a string
                        st1 = COLUMN_STRINGVALUE( cmpCol, i1 );
                        st2 = COLUMN_STRINGVALUE( cmpCol, NDim1 + i2 );
                        st3 = COLUMN_STRINGVALUE( cmpCol, NDim2 + NDim1 + i3 );
                        st4 = COLUMN_STRINGVALUE( cmpCol, NDim3 + NDim2 + NDim1 + i4 );
      
                        cmpStr = "";
                        IF ( STRLEN( st1 ) GT 0 ) THEN cmpStr = cmpStr + st1;
                        IF ( STRLEN( st2 ) GT 0 ) THEN cmpStr = cmpStr + "," + st2;
                        IF ( STRLEN( st3 ) GT 0 ) THEN cmpStr = cmpStr + "," + st3;
                        IF ( STRLEN( st4 ) GT 0 ) THEN cmpStr = cmpStr + "," + st4;

                        IF ( STRLEN( cmpStr ) GT 0 ) THEN
                        (
                           PRINT( " \t", NAME, array, 
                               "[",STRING, cmpStr,"]\n\tAvg: ", REAL, avg);
                        )
                        ELSE
                        (
                           PRINT( " \t", NAME, array, ":\n\tAvg: ", REAL, avg);
                        );

                        PRINT(" (", REAL, std, ") * ", STRING, quant );
                        IF ( Weight GT 1 ) THEN
                        (
                           PRINT(", weighted by: ", FULLNAME, Weight,"\n" );
                        )
                        ELSE
                        (
                           PRINT(", non-weighted\n" );
                        );
                        PRINT("\tMin: ", REAL, min );
                        PRINT(" ", RANGE_INDEX, XYZV minBlk );
                        IF ( minz GT min AND minz LT HUGE ) THEN
                        (
                           PRINT("; non-zero: ", REAL, minz );
                           PRINT(" ",RANGE_INDEX, XYZV minzBlk );
                        );
                        PRINT("; max: ", REAL, max );
                        PRINT(" ",RANGE_INDEX, XYZV maxBlk,".\n\n");
                     );

                     minz1 = minz; min1 = min; max1 = max; avg1 = avg; std1 = std;
                     minz2 = minz; min2 = min; max2 = max; avg2 = avg; std2 = std;
                     minzBlk1 = SPRINT(RANGE_INDEX, XYZV minzBlk);
                     minBlk1 = SPRINT(RANGE_INDEX, XYZV minBlk);
                     maxBlk1 = SPRINT(RANGE_INDEX, XYZV maxBlk);
                  );
               );
            );
         );
      );
   );
);


!*******************************************************************************
!
! Compute statistics of input array (min, max, st. dev.)
!
!*******************************************************************************
FUNCTION VOID ArrayStat( OBJECT_ID array,
                         OBJECT_ID Region->(NIL ID),
                         OBJECT_ID Weight->0,
                         INTEGER Dim1->-1, INTEGER Dim2->-1,
                         INTEGER Dim3->-1, INTEGER Dim4->-1,
                         INTEGER Info->ON  )
(
   VARIABLE REAL minz min max avg std val tot;
   VARIABLE REAL minz1 min1 max1 avg1 std1;
   VARIABLE REAL minz2 min2 max2 avg2 std2;
   VARIABLE INTEGER i1 i2 i3 i4 err NDim1 NDim2 NDim3 NDim4;
   VARIABLE OBJECT_ID cmpCol rngID voidID;
   VARIABLE STRING minzBlk1 minzBlk2 minBlk1 minBlk2 maxBlk1 maxBlk2;
   TABLE (STRING cmp) cmpTbl;
   RANGE XYZ xyzRange;
   RANGE XYZV xyzvRange;

   IF ( NOT IS_ARRAY( array ) ) THEN
   (
      ERROR(1, "ArrayStat: Input must be an array\n" );
   );

   \\ get the dimesions from the range (if named)
   err = ArraySubDim( array, cmpTbl ID, Dim1 ID, Dim2 ID, Dim3 ID, Dim4 ID );

   voidID = ArrayGetVoid( err );

   IF ( Weight EQ 0 ) THEN
   (
      IF ( err EQ 0 OR err EQ -1 ) THEN EVAL
      (
         Weight = GEODATA.GBV ID;
      );
   );

   IF ( err EQ -1 ) THEN
   (
      IF ( Region EQ NIL ID ) THEN EVAL
      (
         Region = XYZ ID;
      );
 
      EVAL
      (
         xyzRange CLEAR;
         xyzRange INVERT;
         FOREACH g IN $Region DO
         (
            IF ( $voidID[<g>] EQ 0 ) THEN
            (
               xyzRange += [<g>];
            );
         );

         IF ( RANGE_NBR_ELEMENTS( xyzRange ) EQ 0 ) THEN
         (
            ERROR(0, "No (active) blocks in the selected range...\n");
            RETURN;
         );
      );

      ArrayStatXYZ( array, cmpTbl ID, xyzRange ID, Weight, Dim1, Dim2, Dim3, Dim4, Info );

      minz1 = ArrayStatXYZ::minz1;
      minz2 = ArrayStatXYZ::minz2;
      min1  = ArrayStatXYZ::min1;
      min2  = ArrayStatXYZ::min2;
      max1  = ArrayStatXYZ::max1;
      max2  = ArrayStatXYZ::max2;
      avg1  = ArrayStatXYZ::avg1;
      avg2  = ArrayStatXYZ::avg2;
      std1  = ArrayStatXYZ::std1;
      std2  = ArrayStatXYZ::std2;

      minzBlk1 = ArrayStatXYZ::minzBlk1;
      minBlk1  = ArrayStatXYZ::minBlk1;
      maxBlk1  = ArrayStatXYZ::maxBlk1;
      minzBlk2 = ArrayStatXYZ::minzBlk2;
      minBlk2  = ArrayStatXYZ::minBlk2;
      maxBlk2  = ArrayStatXYZ::maxBlk2;
   )

   ELSE IF ( err EQ 0 ) THEN
   (
      IF ( Region EQ NIL ID ) THEN EVAL
      (
         Region = XYZF ID;
      );

      EVAL
      (
         IF ( RANGE_NBR_ELEMENTS( $Region ) EQ 0 ) THEN
         (
            ERROR(0, "No (active) blocks in the selected range...\n");
            RETURN;
         );
      );
      ArrayStatXYZF( array, cmpTbl ID, Region, Weight, Dim1, Dim2, Dim3, Dim4, Info );
      minz1 = ArrayStatXYZF::minz1;
      minz2 = ArrayStatXYZF::minz2;
      min1  = ArrayStatXYZF::min1;
      min2  = ArrayStatXYZF::min2;
      max1  = ArrayStatXYZF::max1;
      max2  = ArrayStatXYZF::max2;
      avg1  = ArrayStatXYZF::avg1;
      avg2  = ArrayStatXYZF::avg2;
      std1  = ArrayStatXYZF::std1;
      std2  = ArrayStatXYZF::std2;

      minzBlk1 = ArrayStatXYZF::minzBlk1;
      minBlk1  = ArrayStatXYZF::minBlk1;
      maxBlk1  = ArrayStatXYZF::maxBlk1;
      minzBlk2 = ArrayStatXYZF::minzBlk2;
      minBlk2  = ArrayStatXYZF::minBlk2;
      maxBlk2  = ArrayStatXYZF::maxBlk2;
   )
   ELSE IF ( err LT -1 ) THEN
   (
      \\ voxel range: no void masking
      IF ( Region EQ NIL ID ) THEN EVAL
      (
         Region = XYZV ID;
      );

      EVAL
      (
         xyzvRange CLEAR;
         xyzvRange INVERT;
         FOREACH g IN $Region DO
         (
            IF ( $voidID[<g>] EQ 0 ) THEN
            (
               xyzvRange += [<g>];
            );
         );

         IF ( RANGE_NBR_ELEMENTS( xyzvRange ) EQ 0 ) THEN
         (
            ERROR(0, "No (active) blocks in the selected range...\n");
            RETURN;
         );
      );

      ArrayStatXYZV( array, cmpTbl ID, xyzvRange, Weight, Dim1, Dim2, Dim3, Dim4, Info );
      minz1 = ArrayStatXYZV::minz1;
      minz2 = ArrayStatXYZV::minz2;
      min1  = ArrayStatXYZV::min1;
      min2  = ArrayStatXYZV::min2;
      max1  = ArrayStatXYZV::max1;
      max2  = ArrayStatXYZV::max2;
      avg1  = ArrayStatXYZV::avg1;
      avg2  = ArrayStatXYZV::avg2;
      std1  = ArrayStatXYZV::std1;
      std2  = ArrayStatXYZV::std2;

      minzBlk1 = ArrayStatXYZV::minzBlk1;
      minBlk1  = ArrayStatXYZV::minBlk1;
      maxBlk1  = ArrayStatXYZV::maxBlk1;
      minzBlk2 = ArrayStatXYZV::minzBlk2;
      minBlk2  = ArrayStatXYZV::minBlk2;
      maxBlk2  = ArrayStatXYZV::maxBlk2;
   )
   ELSE IF ( err EQ 3 ) THEN
   (
      ERROR(0, "Only XYZ, XYZV or XYZF arrays can be handled...\n");
   )
   ELSE IF ( err EQ 4 ) THEN
   (
      ERROR(0, "Unknown range, must provide array dimensions using Dim1->dim1 etc.\n");
   )
   ELSE IF ( err EQ 1 ) THEN
   (
      ERROR(0, "No active blocks, no results...\n");
   )
   ELSE IF ( err EQ 10 ) THEN
   (
      ERROR(0, "Only XYZ or XYZF named ranges can be done...\n");
   )
   ELSE
   (
      ERROR(0, "Unknown error... abort\n" ); 
   );
);


!*******************************************************************************
!
! Make a table for the array-histogram, based on sub-dimension
!
!*******************************************************************************

FUNCTION OBJECT_ID ArrayGetHistoTable( OBJECT_ID array, OBJECT_ID cmpTblID,
                                       INTEGER Dim1, INTEGER Dim2,
                                       INTEGER Dim3, INTEGER Dim4, INTEGER NF )
(
   VARIABLE STRING cmp head st1 st2 st3 st4 tblDef dnm tnm quant tblAtr;
   VARIABLE OBJECT_ID tblID cmpCol;
   VARIABLE INTEGER i1 i2 i3 i4 noF;

   \\ get the quantity/unit of this array
   quant = SPRINT( NAME, OBJ_GETQUANTITY( array ) );
   tblID = NIL ID;

   noF = 0;
   IF ( NF EQ 0 ) THEN
   (
      NF = 1;
      noF = 1;
   );

   \\ the array name make the table def unique...
   dnm = SPRINT( "UTILITIES.",NAME, array ) + "_HISTODEF";

   cmpCol = TABLE_COLUMN_ID( cmpTblID, 1 );

   tblDef = "TABLEDEF ( " + quant + " Property";
   IF ( Dim1 LE 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
   (
      cmp = "freq_MF";
      Dim1  = 1; Dim2 = 1; Dim3 = 1; Dim4 = 1;
   )
   ELSE IF ( Dim1 GT 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
   (
      cmp = "freq_xx_MF";
      Dim2  = 1; Dim3 = 1; Dim4 = 1;
   )
   ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
   (
      cmp = "freq_xx_yy_MF";
      Dim3  = 1; Dim4 = 1;
   )
   ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 GT 0 AND Dim4 LE 0 ) THEN
   (
      cmp = "freq_xx_yy_zz_MF";
      Dim4  = 1;
   )
   ELSE
   (
      cmp = "freq_xx_yy_zz_uu_MF";
   );
   tblAtr = "";

   IF ( NOT OBJ_EXISTS( dnm ) ) THEN EVAL
   (
      F = 0;
      WHILE( F+=1; F LE NF ) DO
      (
         i4 = 0;
         WHILE ( i4+=1; i4 LE Dim4 ) DO
         (
            i3 = 0;
            WHILE ( i3+=1; i3 LE Dim3 ) DO
            (
               i2 = 0;
               WHILE ( i2+=1; i2 LE Dim2 ) DO
               (
                  i1 = 0;
                  WHILE ( i1+=1; i1 LE Dim1 ) DO
                  (
                     IF ( F EQ MAT AND noF EQ 0 ) THEN
                     (
                        head = STRREP( cmp, "MF", "MAT", 1 );
                     )
                     ELSE IF ( noF EQ 0 ) THEN
                     (
                        head = STRREP( cmp, "MF", "FRC", 1 );
                     )
                     ELSE
                     (
                        head = STRREP( cmp, "_MF", "", 1 );
                     );


                     \\ get the component names in a string
                     st1 = COLUMN_STRINGVALUE( cmpCol, i1 );
                     st2 = COLUMN_STRINGVALUE( cmpCol, Dim1 + i2 );
                     st3 = COLUMN_STRINGVALUE( cmpCol, Dim2 + Dim1 + i3 );
                     st4 = COLUMN_STRINGVALUE( cmpCol, Dim3 + Dim2 + Dim1 + i4 );
                     head = STRREP( head, "xx", st1, 1 );
                     head = STRREP( head, "yy", st2, 1 );
                     head = STRREP( head, "zz", st3, 1 );
                     head = STRREP( head, "uu", st4, 1 );

                     tblDef = tblDef + ", REAL " + head;

                     tblAtr = tblAtr + dnm + " " + head + " AVERAGE = AVG_STEP;\n";
                  );
               );
            );
         );
      );

      \\ and do the creation of the table type
      tblDef = tblDef + ") " + dnm;

      IF ( NOT OBJ_EXISTS( dnm ) ) THEN 
      (
         EXEC_STRING( tblDef );
         EXEC_STRING( tblAtr );
      );
   );

   \\ the default table name
   tnm = SPRINT( FULLNAME, array ) + "_histo";

   IF ( OBJ_EXISTS( tnm ) ) THEN EVAL
   (
      $tnm CLEAR;
      tblID = $tnm ID;
   )
   ELSE EVAL
   (
      TABLE $dnm $tnm;
      tblID = $tnm ID;
   );

   RETURN( tblID );
);



!*******************************************************************************
!
! Make a table to contain array data
!
!*******************************************************************************

FUNCTION OBJECT_ID ArrayGetTable( OBJECT_ID array, OBJECT_ID cmpTblID,
                                  INTEGER Dim1, INTEGER Dim2,
                                  INTEGER Dim3, INTEGER Dim4, INTEGER NF )
(
   VARIABLE STRING cmp head st1 st2 st3 st4 tblDef dnm tnm quant tblAtr;
   VARIABLE OBJECT_ID tblID cmpCol;
   VARIABLE INTEGER i1 i2 i3 i4 noF;
   VARIABLE REAL    wgt tot;

   \\ get the quantity/unit of this array
   quant = SPRINT( NAME, OBJ_GETQUANTITY( array ) );
   tblID = NIL ID;

   noF = 0;
   IF ( NF EQ 0 ) THEN
   (
      NF = 1;
      noF = 1;
   );

   \\ the array name makes the table def unique...
   tnm = SPRINT( NAME, array );
   dnm = "UTILITIES." + tnm + "_TABLEDEF";

   cmpCol = TABLE_COLUMN_ID( cmpTblID, 1 );

   ! PRINT("DEBUG:\t NF = ", INTEGER, NF,"\n" );
   ! PRINT("DEBUG:\tnoF = ", INTEGER, noF,"\n" );
   ! PRINT("DEBUG:\t dims: ",INTEGER, Dim1, ", ", INTEGER, Dim2,"\n" );
   ! EVAL( $cmpTblID CONTENTS; );

   tblDef = "TABLEDEF ( INTEGER blockIndex, ";

   IF ( Dim1 LE 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
   (
      cmp = tnm + "_MF";
      Dim1  = 1; Dim2 = 1; Dim3 = 1; Dim4 = 1;
   )
   ELSE IF ( Dim1 GT 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
   (
      cmp = tnm + "_xx_MF";
      Dim2  = 1; Dim3 = 1; Dim4 = 1;
   )
   ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
   (
      cmp = tnm + "_xx_yy_MF";
      Dim3  = 1; Dim4 = 1;
   )
   ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 GT 0 AND Dim4 LE 0 ) THEN
   (
      cmp = tnm + "_xx_yy_zz_MF";
      Dim4  = 1;
   )
   ELSE
   (
      cmp = tnm + "_xx_yy_zz_uu_MF";
   );
   tblAtr = "";

   IF ( NOT OBJ_EXISTS( dnm ) ) THEN EVAL
   (
      F = 0;
      WHILE( F+=1; F LE NF ) DO
      (
         i4 = 0;
         WHILE ( i4+=1; i4 LE Dim4 ) DO
         (
            i3 = 0;
            WHILE ( i3+=1; i3 LE Dim3 ) DO
            (
               i2 = 0;
               WHILE ( i2+=1; i2 LE Dim2 ) DO
               (
                  i1 = 0;
                  WHILE ( i1+=1; i1 LE Dim1 ) DO
                  (
                     IF ( F EQ MAT AND noF EQ 0 ) THEN
                     (
                        head = STRREP( cmp, "MF", "MAT", 1 );
                     )
                     ELSE IF ( noF EQ 0 ) THEN
                     (
                        head = STRREP( cmp, "MF", "FRC", 1 );
                     )
                     ELSE
                     (
                        head = STRREP( cmp, "_MF", "", 1 );
                     );

                     \\ get the component names in a string
                     st1  = COLUMN_STRINGVALUE( cmpCol, i1 );
                     st2  = COLUMN_STRINGVALUE( cmpCol, Dim1 + i2 );
                     st3  = COLUMN_STRINGVALUE( cmpCol, Dim2 + Dim1 + i3 );
                     st4  = COLUMN_STRINGVALUE( cmpCol, Dim3 + Dim2 + Dim1 + i4 );
                     head = STRREP( head, "xx", st1, 1 );
                     head = STRREP( head, "yy", st2, 1 );
                     head = STRREP( head, "zz", st3, 1 );
                     head = STRREP( head, "uu", st4, 1 );

                     IF ( F * i1 * i2 * i3 * i4 EQ 1 ) THEN
                     (
                        tblDef = tblDef +  quant + " " + head;
                     )
                     ELSE
                     (
                        tblDef = tblDef + ", " + quant + " " + head;
                     );
                     tblAtr = tblAtr + dnm + " " + head + " AVERAGE = AVG_STEP;\n";
                  );
               );
            );
         );
      );

      \\ and do the creation of the table type
      tblDef = tblDef + ") " + dnm;

      IF ( NOT OBJ_EXISTS( dnm ) ) THEN 
      (
         EXEC_STRING( tblDef );
         EXEC_STRING( tblAtr );
      );
   );

   \\ the default table name
   tnm = SPRINT( FULLNAME, array ) + "_table";

   IF ( OBJ_EXISTS( tnm ) ) THEN EVAL
   (
      $tnm CLEAR;
      tblID = $tnm ID;
   )
   ELSE EVAL
   (
      TABLE $dnm $tnm;
      tblID = $tnm ID;
   );

   RETURN( tblID );
);

!*******************************************************************************
!
! Make a table to contain array data
!
!*******************************************************************************

FUNCTION OBJECT_ID ArrayGetCorrTable( OBJECT_ID arr1ID, OBJECT_ID arr2ID,
                                      OBJECT_ID cmpT1ID, OBJECT_ID cmpT2ID,
                                      INTEGER A1, INTEGER A2, INTEGER A3,
                                      INTEGER B1, INTEGER B2, INTEGER B3,
                                      INTEGER NF )
(
   VARIABLE STRING cmp head st1 st2 st3 tblDef dnm tnm quant;
   VARIABLE OBJECT_ID tblID cmpCol;
   VARIABLE INTEGER i1 i2 i3 Dim1 Dim2 Dim3 noF;

   tblID = NIL ID;

   \\ the array names make the table def unique...
   tnm = SPRINT( NAME, arr1ID, "vs", NAME, arr2ID );
   dnm = "UTILITIES." + tnm + "_DEF";

   tblDef = "TABLEDEF ( ";

   noF = 0;
   IF ( NF EQ 0 ) THEN
   (
      NF = 1;
      noF = 1;
   );

   IF ( NOT OBJ_EXISTS( dnm ) ) THEN EVAL
   (
      I = 0;
      WHILE ( I+=1; I LE 2 ) DO
      (
         IF ( I EQ 1 ) THEN
         (
            \\ get the quantity/unit of the first array
            quant = SPRINT( NAME, OBJ_GETQUANTITY( arr1ID ) );

            cmpCol = TABLE_COLUMN_ID( cmpT1ID, 1 );
            tnm = SPRINT( NAME, arr1ID );
            Dim1 = A1; Dim2 = A2; Dim3 = A3;
         )
         ELSE
         (
            \\ get the quantity/unit of the first array
            quant = SPRINT( NAME, OBJ_GETQUANTITY( arr2ID ) );

            cmpCol = TABLE_COLUMN_ID( cmpT2ID, 1 );
            tnm = SPRINT( NAME, arr2ID );
            Dim1 = B1; Dim2 = B2; Dim3 = B3;
         );
         \\ tnm = tnm + " value";

         IF ( Dim1 LE 0 AND Dim2 LE 0 AND Dim3 LE 0 ) THEN
         (
            cmp = tnm + "_MF";
            Dim1  = 1; Dim2 = 1; Dim3 = 1;
         )
         ELSE IF ( Dim1 GT 0 AND Dim2 LE 0 AND Dim3 LE 0 ) THEN
         (
            cmp = tnm + "_xx_MF";
            Dim2  = 1; Dim3 = 1;)
         ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 LE 0 ) THEN
         (
            cmp = tnm + "_xx_yy_MF";
            Dim3  = 1;
         )
         ELSE
         (
            cmp = tnm + "_xx_yy_zz_MF";
         );

         F = 0;
         WHILE( F+=1; F LE NF ) DO
         (
            i3 = 0;
            WHILE ( i3+=1; i3 LE Dim3 ) DO
            (
               i2 = 0;
               WHILE ( i2+=1; i2 LE Dim2 ) DO
               (
                  i1 = 0;
                  WHILE ( i1+=1; i1 LE Dim1 ) DO
                  (
                     IF ( F EQ MAT AND noF EQ 0 ) THEN
                     (
                        head = STRREP( cmp, "MF", "MAT", 1 );
                     )
                     ELSE IF ( noF EQ 0 ) THEN
                     (
                        head = STRREP( cmp, "MF", "FRC", 1 );
                     )
                     ELSE
                     (
                        head = STRREP( cmp, "_MF", "", 1 );
                     );

                     \\ get the component names in a string
                     st1 = COLUMN_STRINGVALUE( cmpCol, i1 );
                     st2 = COLUMN_STRINGVALUE( cmpCol, Dim1 + i2 );
                     st3 = COLUMN_STRINGVALUE( cmpCol, Dim2 + Dim1 + i3 );
                     head = STRREP( head, "xx", st1, 1 );
                     head = STRREP( head, "yy", st2, 1 );
                     head = STRREP( head, "zz", st3, 1 );

                     IF ( I * F * i1 * i2 * i3 EQ 1 ) THEN
                     (
                        tblDef = tblDef + quant + " " + head;
                     )
                     ELSE
                     (
                        tblDef = tblDef + ", " + quant + " " + head;
                     );
                  );
               );
            );
         );
      );

      \\ and do the creation of the table type
      tblDef = tblDef + ") " + dnm;

      IF ( NOT OBJ_EXISTS( dnm ) ) THEN 
      (
         EXEC_STRING( tblDef );
      );
   );

   \\ the default table name
   tnm = SPRINT( FULLNAME, arr1ID ) + SPRINT( "vs", NAME, arr2ID );

   IF ( OBJ_EXISTS( tnm ) ) THEN EVAL
   (
      $tnm CLEAR;
      tblID = $tnm ID;
   )
   ELSE EVAL
   (
      TABLE $dnm $tnm;
      tblID = $tnm ID;
   );

   RETURN( tblID );
);


!*******************************************************************************
!
! Put array data into a table
!
!*******************************************************************************

FUNCTION VOID ArrayToTable( OBJECT_ID array,
                            OBJECT_ID Region->(NIL ID), INTEGER MaxRow->100000,
                            INTEGER Dim1->-1, INTEGER Dim2->-1,
                            INTEGER Dim3->-1, INTEGER Dim4->-1, INTEGER Info->ON )
(
   VARIABLE REAL min max step done val;
   VARIABLE INTEGER NR NS NDim1 NDim2 NDim3 NDim4 nf fMode;
   VARIABLE INTEGER i1 i2 i3 i4 count err row col nCol nBlocks;
   VARIABLE STRING cmp zeros range str colNm;
   VARIABLE OBJECT_ID histoID colID tblID  theRange voidID;
   RANGE XYZF xyzfRange;
   RANGE XYZV xyzvRange;
   RANGE XYZ  xyzRange;

   TABLE (STRING cmp) cmpTbl;

   IF ( NOT IS_ARRAY( array ) ) THEN
   (
      ERROR(1, "ArrayStat: Input must be an array\n" );
   );

   \\ get the dimesions from the range (if named)
   err = ArraySubDim( array, cmpTbl ID, Dim1 ID, Dim2 ID, Dim3 ID, Dim4 ID );



   IF ( err EQ 0 ) THEN
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZF_ACTIVE ID;
      fMode = NF;
      nf    = NF;
      theRange = xyzfRange ID;
   )
   ELSE IF ( err EQ -1 ) THEN
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZ ID;
      fMode = 0;
      nf    = 1;
      theRange = xyzRange ID;
   )
   ELSE
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZV ID;
      fMode = 0;
      nf = 1;
      theRange = xyzvRange ID;
   );

   voidID = ArrayGetVoid( err );

   EVAL
   (
      $theRange CLEAR;
      $theRange INVERT;
      FOREACH g IN $Region DO
      (
         IF ( $voidID[<g>] EQ 0 ) THEN
         (
            $theRange += [<g>];
         );
      );

      nBlocks = RANGE_NBR_ELEMENTS( $theRange );
      IF ( nBlocks EQ 0 ) THEN
      (
         err = 1;
      );
   );

   IF ( err LE 0 ) THEN EVAL
   (
      \\ find a proper number of rows, skip array data if too many blocks
      step =  1 + nBlocks / MaxRow;
 
      \\ get the output table and its columns
      tblID = ArrayGetTable( array, cmpTbl ID, Dim1, Dim2, Dim3, Dim4, fMode );

      IF ( Dim1 LE 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = " ";
         NDim1 = 1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1 ";
         NDim1 = Dim1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 GT 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2, i3 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = Dim3; NDim4 = 1;
      )
      ELSE
      (
         cmp = ", i1, i2, i3, i4 ";
      );

      EVAL
      (
         col  = 1;
         done = 0;
         F = 0;
         WHILE( F+=1; F LE nf ) DO
         (
            IF ( fMode GT 0 ) THEN
            (
               \\ restrict the active range to MAT or FRC
               ArrayRestrictMatFrc( Region, theRange, F );
            );


            i4 = 0;
            WHILE ( i4+=1; i4 LE NDim4 ) DO
            (
               i3 = 0;
               WHILE ( i3+=1; i3 LE NDim3 ) DO
               (
                  i2 = 0;
                  WHILE ( i2+=1; i2 LE NDim2 ) DO
                  (
                     i1 = 0;
                     WHILE ( i1+=1; i1 LE NDim1 ) DO
                     (
                        \\ get the columns for this F, and set of components
                        col += 1;
                        colNm = COLUMN_HEADER( TABLE_COLUMN_ID( tblID, col ) );

                        count = 0;
                        EVAL
                        (
                           row = 0;
                           $tblID $colNm CLEAR;
                           FOREACH g IN $theRange DO
                           (
                              IF ( MOD( count, step ) EQ 0 ) THEN
                              (
                                 row += 1;
                                 $tblID blockIndex[row] = count + 1;
                                 $tblID $colNm[row] = $array[ <g> $cmp ];
                              );
                              count += 1;
                           );
                        );
                     );
                  );
               );
            );
         );
      );
   )
   ELSE
   (
      IF ( err EQ 1 ) THEN
      (
         ERROR(0, "No active blocks, no results...\n");
         RETURN;
      )
      ELSE IF ( err EQ 4 ) THEN
      (
         ERROR(0, "Unknown range, must provide array dimensions using Dim1->dim1 etc.\n");
         RETURN;
      )
      ELSE
      (
         ERROR(0, "Unknown error... abort\n" ); 
         RETURN;
      );
   );

   IF ( Info GT 0 ) THEN
   (
      PRINT("INFO:\tArray data put into table ", FULLNAME, tblID, "\n");
      PRINT(" \t",INTEGER, MIN(nBlocks, MaxRow),
            " grid blocks are sampled, skipping ", INTEGER, step-1,"\n" );
      IF ( Region NE NIL ID ) THEN
      (
         PRINT("\tBlocks are restricted to range ", FULLNAME, Region,"\n");
      );
   );

   xyzfRange CLEAR;
   xyzRange CLEAR;
   xyzvRange CLEAR;

);



!*******************************************************************************
!
! Put array data for two arrays into a table, to plot correlations
!
!*******************************************************************************
FUNCTION VOID ArrayCorrelate( OBJECT_ID array1, OBJECT_ID array2,
                              OBJECT_ID Region->(NIL ID), INTEGER MaxRow->100000,
                              INTEGER Info->ON )
(
   VARIABLE REAL min max step done val;
   VARIABLE INTEGER A1 A2 A3 A4 B1 B2 B3 B4;
   VARIABLE INTEGER NR NS NDim1 NDim2 NDim3 NDim4 nf fMode;
   VARIABLE INTEGER i1 i2 i3 i4 count err err1 err2 row col nCol nBlocks;
   VARIABLE STRING cmp zeros range str colNm;
   VARIABLE OBJECT_ID histoID colID tblID array theRange voidID;
   RANGE XYZF xyzfRange;
   RANGE XYZV xyzvRange;
   RANGE XYZ  xyzRange;
  
   TABLE (STRING cmp) cmpTbl1 cmpTbl2;

   IF ( NOT IS_ARRAY( array1 ) ) THEN
   (
      ERROR(1, "ArrayCorrelate: Input must be an array\n" );
   );

   IF ( NOT IS_ARRAY( array2 ) ) THEN
   (
      ERROR(1, "ArrayCorrelate: Input must be an array\n" );
   );

   \\ get the dimesions from the range (if named)
   A1 = -1; A2 = -1; A3 = -1; A4 = -1;
   B1 = -1; B2 = -1; B3 = -1; B4 = -1;

   err1 = ArraySubDim( array1, cmpTbl1 ID, A1 ID, A2 ID, A3 ID, A4 ID );
   err2 = ArraySubDim( array2, cmpTbl2 ID, B1 ID, B2 ID, B3 ID, B4 ID );
   err = MAX(err1, err2 );
   
   voidID = ArrayGetVoid( err );

   IF ( err1 EQ 0 AND err2 EQ 0 ) THEN
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZF_ACTIVE ID;
      fMode = NF;
      nf    = NF;
      theRange = xyzfRange ID;
   )
   ELSE IF ( err1 EQ -1 AND err2 EQ -1 ) THEN
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZ ID;
      fMode = 0;
      nf = 1;
      theRange = xyzRange ID;
   )
   ELSE IF ( err1 LT -1 AND err2 LT -1 ) THEN EVAL
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZV ID;
      fMode = 0;
      nf = 1;
      theRange = xyzvRange ID;
   )
   ELSE
   (
      ERROR( 1, "Invalid ranges..." );
   );

   EVAL
   (
      $theRange CLEAR;
      $theRange INVERT;
      FOREACH g IN $Region DO
      (
         IF ( $voidID[<g>] EQ 0 ) THEN
         (
            $theRange += [<g>];
         );
      );

      nBlocks = RANGE_NBR_ELEMENTS( $theRange );
      IF ( nBlocks EQ 0 ) THEN
      (
         err = 1;
      );
   );

   IF ( err LE 0 ) THEN EVAL
   (
      \\ find a proper number of rows, skip array data if too many blocks
      step =  1 + nBlocks / MaxRow;
 
      \\ get the output table and its columns
      tblID = ArrayGetCorrTable( array1, array2, cmpTbl1 ID, cmpTbl2 ID,
                                 A1, A2, A3, B1, B2, B3, fMode );

      col = 0;
      I = 0;
      WHILE ( I+=1; I LE 2  ) DO EVAL
      (
         IF ( I EQ 1 ) THEN
         (
            array = array1;

            IF ( A1 LE 0 AND A2 LE 0 AND A3 LE 0 AND A4 LE 0 ) THEN
            (
               cmp = " ";
               NDim1 = 1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
            )
            ELSE IF ( A1 GT 0 AND A2 LE 0 AND A3 LE 0 AND A4 LE 0 ) THEN
            (
               cmp = ", i1 ";
               NDim1 = A1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
            )
            ELSE IF ( A1 GT 0 AND A2 GT 0 AND A3 LE 0 AND A4 LE 0 ) THEN
            (
               cmp = ", i1, i2 ";
               NDim1 = A1; NDim2 = A2; NDim3 = 1; NDim4 = 1;
            )
            ELSE IF ( A1 GT 0 AND A2 GT 0 AND A3 GT 0 AND A4 LE 0 ) THEN
            (
               cmp = ", i1, i2, i3 ";
               NDim1 = A1; NDim2 = A2; NDim3 = A3; NDim4 = 1;
            )
            ELSE
            (
               cmp = ", i1, i2, i3, i4 ";
            );
         )
         ELSE
         (
            array = array2;

            IF ( B1 LE 0 AND B2 LE 0 AND B3 LE 0 AND B4 LE 0 ) THEN
            (
               cmp = " ";
               NDim1 = 1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
            )
            ELSE IF ( B1 GT 0 AND B2 LE 0 AND B3 LE 0 AND B4 LE 0 ) THEN
            (
               cmp = ", i1 ";
               NDim1 = B1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
            )
            ELSE IF ( B1 GT 0 AND B2 GT 0 AND B3 LE 0 AND B4 LE 0 ) THEN
            (
               cmp = ", i1, i2 ";
               NDim1 = B1; NDim2 = B2; NDim3 = 1; NDim4 = 1;
            )
            ELSE IF ( B1 GT 0 AND B2 GT 0 AND B3 GT 0 AND B4 LE 0 ) THEN
            (
               cmp = ", i1, i2, i3 ";
               NDim1 = B1; NDim2 = B2; NDim3 = B3; NDim4 = 1;
            )
            ELSE
            (
               cmp = ", i1, i2, i3, i4 ";
            );
         );

         done = 0;
         F = 0;
         WHILE( F+=1; F LE nf ) DO
         (
            IF ( fMode GT 0 ) THEN
            (
               \\ restrict the active range to MAT or FRC
               ArrayRestrictMatFrc( Region, theRange, F );
            );

            i4 = 0;
            WHILE ( i4+=1; i4 LE NDim4 ) DO
            (
               i3 = 0;
               WHILE ( i3+=1; i3 LE NDim3 ) DO
               (
                  i2 = 0;
                  WHILE ( i2+=1; i2 LE NDim2 ) DO
                  (
                     i1 = 0;
                     WHILE ( i1+=1; i1 LE NDim1 ) DO
                     (
                        \\ get the columns for this F, and set of components
                        col += 1;
                        colNm = COLUMN_HEADER( TABLE_COLUMN_ID( tblID, col ) );

                        count = 0;
                        EVAL
                        (
                           row = 0;
                           $tblID $colNm CLEAR;
                           FOREACH g IN $theRange DO
                           (
                              IF ( MOD( count, step ) EQ 0 ) THEN
                              (
                                 row += 1;
                                 $tblID $colNm[row] = $array[ <g> $cmp ];
                              );
                              count += 1;
                           );
                        );
                     );
                  );
               );
            );
         );
      );
   )
   ELSE
   (
      IF ( err EQ 1 ) THEN
      (
         ERROR(0, "No active blocks, no results...\n");
         RETURN;
      )
      ELSE IF ( err EQ 4 ) THEN
      (
         ERROR(0, "Unknown range, must provide array dimensions using Dim1->dim1 etc.\n");
         RETURN;
      )
      ELSE
      (
         ERROR(0, "Unknown error...\n" );
         RETURN;
      );
   );

   IF ( Info GT 0 ) THEN EVAL
   (
      PRINT("INFO:\tCorrelation data is put in table ", FULLNAME, tblID, "\n");
      PRINT(" \t",INTEGER, MIN( nBlocks, MaxRow),
             " grid blocks are sampled, skipping ", INTEGER, step -1 ,"\n" );
      IF ( Region NE NIL ID ) THEN
      (
         PRINT("\tBlocks are restricted to range ", FULLNAME, Region,"\n");
      );

      \\ make 1 plot (first column of the A array, with all columns of B )
      colNm = COLUMN_HEADER( TABLE_COLUMN_ID( tblID, 1 ) );
      str = "NAME = correlation XCOLUMN = " + colNm;

      i2 = MAX(1, A1) * MAX(1, A2) * MAX(1, A3);

      \\ do not make more than 12 curves (limit is 30, CMP: NC=12 has too many)
      i3 = MIN( 12, MAX(1, B1) * MAX(1, B2) * MAX(1, B3) );
      i1 = i2;
      WHILE ( i1 += 1; i1 LE i2 + i3 ) DO
      (
         str = str + " YCOLUMN=" + COLUMN_HEADER( TABLE_COLUMN_ID( tblID, i1 ));
      );
      
      EVAL( $tblID PLOT $str );

      i1 = i2;
      WHILE ( i1 += 1; i1 LE i2 + i3 ) DO
      (
         str = SPRINT( FULLNAME, tblID, " correlation DRAW CURVE ",
                       STRING, COLUMN_HEADER( TABLE_COLUMN_ID( tblID, i1 )) );
         EXEC_STRING( str + " LINE = OFF" );
         EXEC_STRING( str + " MARKER = ON" );
         EXEC_STRING( str + " MARKER_TYPE = DIAMOND_FILL" );
      );
   );

   xyzfRange CLEAR;
   xyzRange CLEAR;
   xyzvRange CLEAR;

);


!*******************************************************************************
!
! Compute Histogram of input array
!
!*******************************************************************************

FUNCTION VOID ArrayHisto( OBJECT_ID array,
                          OBJECT_ID Region->(NIL ID),
                          OBJECT_ID Weight->0,
                          INTEGER Dim1->-1, INTEGER Dim2->-1,
                          INTEGER Dim3->-1, INTEGER Dim4->-1,
                          INTEGER Nbin->20, REAL Min->-HUGE, REAL Max->HUGE,
                          REAL Scale->-1, REAL Lower->0.1, REAL Upper->0.1,
                          INTEGER Info->ON )
(
   VARIABLE REAL min max step done val tot;
   VARIABLE INTEGER NR NS NDim1 NDim2 NDim3 NDim4 nf fMode;
   VARIABLE INTEGER i1 i2 i3 i4 count err indx col nCol nBlocks;
   VARIABLE STRING quant cmp zeros range str;
   VARIABLE OBJECT_ID histoID colID clxID rngID theRange voidID;
   RANGE XYZF xyzfRange;
   RANGE XYZV xyzvRange;
   RANGE XYZ  xyzRange;
   VARIABLE STRING qwgt wgt colNm;
   TABLE (STRING cmp) cmpTbl;


   IF ( NOT IS_ARRAY( array ) ) THEN
   (
      ERROR(1, "ArrayStat: Input must be an array\n" );
   );

   \\ get the quantity/unit of this array
   quant = OBJ_GETUNIT( array );
   rngID = OBJ_GETRANGE( array );

   \\ get the dimesions from the range (if named)
   err = ArraySubDim( array, cmpTbl ID, Dim1 ID, Dim2 ID, Dim3 ID, Dim4 ID );

   IF ( Weight EQ 0 ) THEN
   (
      IF ( err EQ 0 OR err EQ -1 ) THEN EVAL
      (
         Weight = GEODATA.GBV ID;
      );
   );

   IF ( err EQ 0 ) THEN
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZF_ACTIVE ID;
      fMode = NF;
      nf    = NF;
      theRange = xyzfRange ID;
   )
   ELSE IF ( err EQ -1 ) THEN
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZ ID;
      fMode = 0;
      nf = 1;
      theRange = xyzRange ID;
   )
   ELSE
   (
      IF ( Region EQ NIL ID ) THEN Region = XYZV ID;
      fMode = 0;
      nf = 1;
      theRange = xyzvRange ID;
   );

   voidID = ArrayGetVoid( err );

   EVAL
   (
      $theRange CLEAR;
      $theRange INVERT;
      FOREACH g IN $Region DO
      (
         IF ( $voidID[<g>] EQ 0 ) THEN
         (
            $theRange += [<g>];
         );
      );

      nBlocks = RANGE_NBR_ELEMENTS( $theRange );
      IF ( nBlocks EQ 0 ) THEN
      (
         err = 1;
      );

      IF ( Scale LE 0 AND  Min GT -0.9*HUGE AND Max LT 0.9 * HUGE ) THEN
      (
         Scale = Max - Min;
      )
      ELSE IF ( Scale LE 0 ) THEN
      (
         \\ try to find scale from global values

         IF ( EXISTS( rngID ) ) THEN EVAL
         (
            max = ($array MAX $rngID) / ($quant);
            min = ($array MIN $rngID) / ($quant);
            Scale = max - min;
         );
      );

      IF ( Scale LE 0 ) THEN
      (
         err = 2;
      );
   );

   IF ( err LE 0 ) THEN EVAL
   (
      \\ find a proper number of bins
      \\ want at least 5 values, on average, per bin
      IF ( nBlocks / Nbin LT 5 ) THEN
      (
         Nbin = MAX( 3, nBlocks / 5 );
      );
 
      \\ get the histogram table and its columns
      histoID = ArrayGetHistoTable( array, cmpTbl ID, Dim1, Dim2, Dim3, Dim4, fMode );

      \\ first column is the step value
      col   = 1;
      clxID = TABLE_COLUMN_ID( histoID, 1);
      nCol  = TABLE_NBOF_COLUMNS( histoID );

      IF ( Dim1 LE 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = " ";
         NDim1 = 1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 LE 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1 ";
         NDim1 = Dim1; NDim2 = 1; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 LE 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = 1; NDim4 = 1;
      )
      ELSE IF ( Dim1 GT 0 AND Dim2 GT 0 AND Dim3 GT 0 AND Dim4 LE 0 ) THEN
      (
         cmp = ", i1, i2, i3 ";
         NDim1 = Dim1; NDim2 = Dim2; NDim3 = Dim3; NDim4 = 1;
      )
      ELSE
      (
         cmp = ", i1, i2, i3, i4 ";
      );

      EVAL
      (
         \\ make a string with a zero entry for each histogram column
         zeros = "";
         i1 = 0;
         WHILE( i1 += 1; i1 LE NDim1 * NDim2 * NDim3 * NDim4 * nf ) DO
         (
            zeros = zeros + ", 0.";
         );

         \\ normal grid array, XYZF type, no multiplicity
         done = 0;
         F = 0;
         WHILE( F+=1; F LE nf ) DO
         (
            IF ( fMode GT 0 ) THEN
            (
               \\ restrict the active range to MAT or FRC
               ArrayRestrictMatFrc( Region, theRange, F );
            );

            \\ is there a weight to be used in the averaging?
            IF ( Weight GT 1 ) THEN
            (
               IF ( NOT IS_ARRAY( Weight ) ) THEN
               (
                  ERROR(1, "ArrayHisto: Weight must be an array\n" );
               );

               qwgt  = OBJ_GETUNIT( Weight );
               rngID = OBJ_GETRANGE( Weight );

               IF ( SPRINT( NAME, rngID ) EQ "XYZF" AND err EQ -1 ) THEN
               (  
                  wgt = SPRINT( "(", FULLNAME, Weight,"[<g>,MAT] / (",STRING, qwgt,"))" );
               )
               ELSE IF ( SPRINT( NAME, rngID ) EQ "XYZ" OR
                         SPRINT( NAME, rngID ) EQ "XYZF" OR
                         SPRINT( NAME, rngID ) EQ "XYZV" ) THEN
               ( 
                  wgt   = SPRINT( "(", FULLNAME, Weight,"[<g>] / (",STRING, qwgt,"))" );
               )
               ELSE
               (
                  ERROR(1, "ArrayHisto: Weight must have an XYZ, XYZF or XYZV range\n" );
               );

               tot = 0.0;
               EVAL
               (
                  FOREACH g IN $theRange DO
                  (
                     tot += $wgt;
                  );
               );
               IF ( SPRINT( NAME, rngID ) EQ "XYZF" AND err EQ -1 ) THEN
               (  
                  wgt = SPRINT( "(", FULLNAME, Weight,"[<g>,MAT] / (",REAL, tot," *(",STRING, qwgt,")))" );
               )
               ELSE IF ( SPRINT( NAME, rngID ) EQ "XYZ" OR
                         SPRINT( NAME, rngID ) EQ "XYZF" OR
                         SPRINT( NAME, rngID ) EQ "XYZV" ) THEN
               ( 
                  wgt   = SPRINT( "(", FULLNAME, Weight,"[<g>] / (",REAL, tot," *(",STRING, qwgt,")))" );
               )
            )
            ELSE
            (
               wgt = "1";
            );

            i4 = 0;
            WHILE ( i4+=1; i4 LE NDim4 ) DO EVAL
            (
               i3 = 0;
               WHILE ( i3+=1; i3 LE NDim3 ) DO
               (
                  i2 = 0;
                  WHILE ( i2+=1; i2 LE NDim2 ) DO
                  (
                     i1 = 0;
                     WHILE ( i1+=1; i1 LE NDim1 ) DO
                     (
                        min  = HUGE; 
                        max  =-HUGE; 
                        count = 0;

                        \\ get the columns for this F, and set of components
                        col += 1;
                        colID = TABLE_COLUMN_ID( histoID, col );
   
                        IF ( Min GT -0.9*HUGE AND Max LT 0.9 * HUGE ) THEN
                        (
                           min = Min;
                           max = Max;
                        )
                        ELSE
                        (
                           FOREACH g IN $theRange DO
                           (
                              val = $array[ <g> $cmp ] / ( $quant );
                              min = MIN( min, val );
                              max = MAX( max, val );
                           );
                        );

                        IF ( Min GT -0.9*HUGE ) THEN
                        (
                           min = Min;
                        );

                        IF ( Max LT 0.9 * HUGE ) THEN
                        (
                           max = Max;
                        );
            
                        \\ get the bin-step size
                        step = (max - min);
                        IF ( step LT TINY) THEN ( step = Scale );
                        max += Upper * step;
                        IF ( min GE 0) THEN
                        (
                           min = MAX(0, min - Lower * step);
                        )
                        ELSE
                        (
                           min -= Lower * step;
                        );
                        step = (max - min) / Nbin;

                        IF ( done EQ 0 ) THEN
                        (
                           \\ fill the table with start vales 
                           done = 1;
                           indx = -1;
                           WHILE ( indx += 1; indx LE Nbin+1 ) DO
                           (
                              $histoID, (min + indx * step) * ($quant) $zeros;
                           );
                        );
            
                        \\ and fill the bins
                        FOREACH g IN $theRange DO
                        (
                           \\ compute the row index
                           val  = $array[ <g> $cmp ] / ( $quant );
                           indx = MIN( Nbin+2, MAX(1,  1 + INT( (val - min) / step )));
                         
                           COLUMN_SETVALUE( colID, indx, COLUMN_VALUE( colID, indx ) + $wgt );
                        );
                     );
                  );
               );
            );
         );
      );
   )
   ELSE
   (
      IF ( err EQ 1 ) THEN
      (
         ERROR(0, "No active blocks, no results...\n" );
         RETURN;
      )
      ELSE IF ( err EQ 2 ) THEN
      (
         str = SPRINT( "All array values the same ", NAME, array, " = ",
                  REAL, min, "*", STRING, quant," \n");
         ERROR(0, str );
         RETURN;
      )
      ELSE IF ( err EQ 4 ) THEN
      (
         ERROR(0, "Unknown range, must provide array dimensions using Dim1->dim1 etc.\n");
         RETURN;
      )
      ELSE
      (
         ERROR(0, "Unknown error... abort\n" ); 
         RETURN;
      );
   );


   IF ( Info GT 0 ) THEN
   (
      PRINT("INFO:\tThe histogram name is ", FULLNAME, histoID, ".\n");
      PRINT(" \tAll ", INTEGER, nBlocks, " active grid blocks are sampled.\n" );
      IF ( Region NE NIL ID ) THEN
      (
         PRINT("\tBlocks are restricted to range ", FULLNAME, Region,".\n");
      );

      \\ make 1 plot (firt column of the A array, with 1 columns of B )
      colNm = COLUMN_HEADER( TABLE_COLUMN_ID( histoID, 1 ) );
      str = "BAR = " + colNm;

      i1 = MAX(1, Dim1) * MAX(1, Dim2) * MAX(1, Dim3);
      str = str + ", " + COLUMN_HEADER( TABLE_COLUMN_ID( histoID, i1+1 ) );
      
      EVAL( $histoID BARPLOT $str );
   );

   xyzfRange CLEAR;
   xyzRange CLEAR;
   xyzvRange CLEAR;

);

UTILITIES END;

! ============================================================================
!  Restore the warning limit
   WARNING_NUMBER = PRJDICT.MAX_NBR_WARN;
   WARNING_LEVEL = PRJDICT.MAX_LEV_WARN;
! ============================================================================
);  \\Closing the "include only once" IF
