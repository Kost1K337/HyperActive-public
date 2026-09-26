%% MODIFIED BY MOHSIN
    %---------------------------------------------------------------------------
    % store production data time series (rates in m^3/s, pressures in Pa)
    %---------------------------------------------------------------------------             
    if ~exist('wellData','var')
      for i = 1 : length(wells)
         wellData(i).bhp  = 0;
         wellData(i).oil  = 0;         % oil rate
         wellData(i).wat  = 0;         % water rate
       % wellData(i).icv  = [];        % total liquid rate for each connection
      end
    end
    for i = 1 : length(wells)
      % bottom-hole pressure
      wellData(i).bhp = [wellData(i).bhp; rSol.wellSol(i).pressure];
      % totla liquid rate
      rate = 0;
      for k=1:length(rSol.wellSol(i).flux)
      % wellData(i).icv = [wellData(i).icv; rSol.wellSol(i).flux(k)]; 
        rate = rate + rSol.wellSol(i).flux(k);
      end

      % compute fractional flow assuming horizontal flow and no 
      % capillary pressure: fw = qw/qt = 1/(1+(muw/krw)*(kro/muo))
      if rate < 0 % producers only
          s = mean(rSol.s(wells(i).cells));  % *** currently only valid for 2D grids ***
          kr = fluid.relperm(s); % [krw kro]
          mu = fluid.properties(s); % [muw muo]
          mo = bsxfun(@rdivide, kr, mu); % [krw/muw kro/muo]
          fw = 1/(1 + mo(2)/mo(1));
          if isnan(fw), fw = 0; end
          
          
          wellData(i).wat = [wellData(i).wat; fw*rate*86400];
          wellData(i).oil = [wellData(i).oil; (1-fw)*rate*86400];
      else
          wellData(i).wat = [wellData(i).wat; rate*86400];
          wellData(i).oil = [wellData(i).oil; 0];
      end
    end