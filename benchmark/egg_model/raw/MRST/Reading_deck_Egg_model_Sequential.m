%%%
%%%           IMPLEMENTATION OF EGG MODEL IN MRST 
%%%
%%%  Implementation of Gijs Egg reservoir model in MRST 
%%%
%%%  Created:       June 2013
%%%  Last update:   Nov 2013
%%%  Owner:         M.Mohsin Siraj (TU/e) & R.M. Fonseca (TUD)

%
% This file is an implementation of Gijs van Essen Egg reservoir model in
% MRST and will be used as reservoir model for further research.

clc,clear all, 
close all

%% Set up gravity and add deckformat.
% gravity reset
% gravity on

mrstModule add deckformat

%% Initialize the model properties (Grids, paraemters etc.) of EGGs Model

% We read in the deck file, convert the units to SI and then use the 
% resulting deck variable to create grid, rock, fluid and well 
% configurations.

%%%
%%%     Add directly the Eclipse deck here!!!
%%%
fn    = fullfile('D:', 'PHD', 'Egg_Model','Standard_Egg_Model', 'MRST', ...
    'Egg_Model_ECL.DATA');

if ~exist(fn, 'file'),
   error('Egg model data is not available.')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% 
%%%     Reading the input deck
%%%
deck  = readEclipseDeck(fn);                     % Reading deck
deck  = convertDeckUnits(deck);                 % Converting units
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%
%%%     Defining fluid properties
%%%

% fluid = initEclipseFluid(deck);       % Parameters are defined manually.

% As we are using the incompressible solver, so define the fluid
% properties here!!! 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gijs van Essen	SPE 124332 Model properties (Report parameters)
fluid     = initCoreyFluid('mu' ,  [   1,   5]*centi*poise     , ...
                            'rho',  [1000, 900]*kilogram/meter^3, ...
                            'n'  ,  [   3,   4]                 , ...
                            'sr' ,  [ 0.2,   0.1]                 , ...
                            'kwm',  [   0.749,0.8]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%
%%%     Reading grid structure
%%%
G     = initEclipseGrid(deck);   
G     = removeCells(G, ~deck.GRID.ACTNUM);
G     = computeGeometry(G);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
%%%     Reading rock properties (permeability and porosity fields)
%%%
                        
rock  = initEclipseRock(deck);                  
% Remove any rock data corresponding to inactive cells.
rock  = compressRock(rock, G.cells.indexMap);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
actnum        = deck.GRID.ACTNUM; 


%% 
%                     Introduce wells

% The wells are introduced here, 8 injection wells and 4 production wells.
% The placement is similar to the one, introduced in Eggs model.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nz = G.cartDims(3);

I = [ 5, 30,  2, 27, 50, 8, 32, 57];
J = [57, 53, 35, 29, 35, 9,  2,  6];

waterInj=79.5*meter^3/day;
R = waterInj;
W = [];
diameter=0.2;
% diameter=0.1546;
for i = 1 : numel(I),
   W = verticalWell(W, G, rock, I(i), J(i), 1:nz, 'Type', 'rate', ...
                    'Val', R, 'Radius', diameter/2, ...
                    'name', ['I$_{', int2str(i), '}$'],'InnerProduct', 'ip_tpf');
end

% Set vertical producers, completed in the upper 14 layers
I = [16, 35, 23, 43];
J = [43, 40, 16, 18];

for i = 1 : numel(I),
   W = verticalWell(W, G, rock, I(i), J(i), 1:nz, 'Type', 'bhp', ...
                    'Val', 395*barsa(), 'Radius', diameter/2, ...
                    'name', ['P$_{', int2str(i), '}$'],'InnerProduct', 'ip_tpf');
end


% StoreProduction uses wells instead of W
wells = W;

%% Initialize and construct the linear system
S            = computeMimeticIP(G, rock,'Type', 'tpfa','InnerProduct', 'ip_tpf','Verbose', true); 

% Pressure initialization
% po(1) = 400*barsa;
% 
% for i = 2:7
%     po(i) = po(i-1) + 0.35316*barsa;
% end
% 
% Po = repmat(po(1),3600,1);
% for i =2:7
%     Po = [Po;repmat(po(i),3600,1)];
% end
rSol         = initResSol(G, 400*barsa,0.1 );
rSol.wellSol = initWellSol(W, 400*barsa()); 
initpress    = convertTo(rSol.pressure(1:G.cells.num), barsa);
   
rSol = solveIncompFlow(rSol, G, S, fluid,'wells',W,'Solver','tpfa');


%%  Incompressible flow %%%%%%%%%%%%%%%%%%%%

% Transport loop
% We solve the two-phase system using a sequential splitting in which the
% pressure and fluxes are computed by solving the flow equation and then
% held fixed as the saturation is advanced according to the transport
% equation. This procedure is repeated for a given number of time steps
% (here we use 15 equally spaced time steps). The error introduced by this
% splitting of flow and transport can be reduced by iterating each time
% step until e.g., the residual is below a certain user-prescribed
% threshold (this is not done herein).

T      = 3600*day();%10*year();%
dT     = 30*day(); %2*year;%
dTplot = 1*year();  %100*day();% % plot only every 2nd year



%% Start the main loop
t  = 0; plotNo = 1; hi = 'Implicit: ';

figure,
while t < T,
   rSol = implicitTransport(rSol, G, dT, rock, fluid, 'wells', W);
%    rSol = explicitTransport(rSol, G, dT, rock, fluid, 'wells', W);
 
   % Check for inconsistent saturations
   s = rSol.s(:,1);
   assert(max(s) < 1+eps && min(s) > -eps);

   % Update solution of pressure equation.  
   rSol = solveIncompFlow(rSol, G, S, fluid,'wells',W,'Solver','tpfa');

   %---------------------------------------------------------------------------
   % store production data time series (rates in m^3/s, pressures in Pa)
   %---------------------------------------------------------------------------
   StoreProduction; % Code from Olwijn Leeuwenburgh (TNO)

   % Increase time and continue if we do not want to plot saturations
   t = t + dT;
   if ( t < plotNo*dTplot && t <T), continue, end

   % Plot saturation
   heading = [num2str(convertTo(t,year)),  ' year'];
   r = 0.01;
   
%  Normal plotting
   subplot(2,3,plotNo)
   plotCellData(G, 1-rSol.s(:,1),find( actnum(G(1).cells.indexMap)));
   view(60,70), axis equal off, %title([hi heading])
%    plotWell(G, W, 'height', 10, 'color', 'k');
   

   plotNo = plotNo+1;
end

