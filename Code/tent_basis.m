function B = tent_basis(y, ctrs)
    if numel(ctrs)==1
        dc = ctrs;
        dscale1 = dc;
        dscale2 = dc;
    else
        dc = diff(ctrs);
        if ctrs(1)==0
            dscale1 = [dc(1) dc];
        else
            dscale1 = [ctrs(1) dc];
        end
        
        dscale2 = [dc dc(end)];
    end
    
    ydiff = y(:) - ctrs;
    yd = (ydiff .* (ydiff<0) ./ dscale1) + (ydiff .* (ydiff>0) ./ dscale2);
    B = max(1-abs(yd), 0);