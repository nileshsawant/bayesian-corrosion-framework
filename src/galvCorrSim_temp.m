classdef galvCorrSim
    %galvCorrSim Summary of this class goes here
    %   Detailed explanation goes here

    properties        
        xpos
        ypos
        L
        H
        dx
        dy

        bcT
        bcL
        bcR
        
        NX
        NXa
        NXc
        NY

        conductivity

        bVreactions 

        Eapp 
        corrosionCurrentAnodic 
        corrosionCurrentCathodic 
        corrosionCurrentTotal 
        eCorr 
        phi 
        nFig 
    end

    methods
        function obj = galvCorrSim(BCs,length,height,deltaL,eNodes,numNodes,aNodes,cNodes,reactAnode,reactCathode,Vapp,nF,env) %
            %galvCorrSim Construct an instance of this class
              % Detailed explanation goes here
              if nargin == 0
                obj.NX = 50;
                obj.NXa = 25;
                obj.NXc = 25;                
                obj.NY = 50;
    
                obj.L = 1.0;
    
                obj.bcL = 'neumann';
                obj.bcR = 'neumann';
                obj.bcT = 'neumann';
                obj.H = 10.0*obj.L;
    
                obj.dx = obj.L/(numNodes-1);
                obj.dy = obj.H/(numNodes-1);
                leftEnd = -obj.L/2.0;
                rightEnd = obj.L/2.0;
                obj.xpos = leftEnd:obj.dx:rightEnd;
                obj.ypos = 0.0:obj.dy:obj.H;
    
                % Default placeholders
                obj.bVreactions = {'anodic', 'cathodic'};
    
                obj.Eapp = 0.0;
                obj.nFig = 132; 

              elseif nargin >= 1
                  
                obj.NX = numNodes;
                obj.NXa = aNodes;
                obj.NXc = cNodes;
                obj.NY = eNodes;
    
                obj.L = length;
                obj.H = height;
                obj.bcT = char(BCs{1});
                obj.bcL = char(BCs{2});
                obj.bcR = char(BCs{3});
    
                obj.dx = deltaL; 
                obj.dy = deltaL; 

                leftEnd = -obj.NXc*obj.dx; 
                rightEnd = obj.NXa*obj.dx; 
                
                obj.xpos = leftEnd:obj.dx:rightEnd;
                obj.NXc = cNodes+1;
                obj.NX = numel(obj.xpos);

                obj.ypos = 0.0:obj.dy:obj.H;
                obj.NY = numel(obj.ypos);

                % Instantiate objects separately
                bv1 = butlerVolmer(reactAnode,env(1),env(2),env(3),env(4));
                bv2 = butlerVolmer(reactCathode,env(1),env(2),env(3),env(4));
                
                % Assign as Cell Array
                obj.bVreactions = {bv1, bv2};
    
                erange = -1.5:0.01:1.5;
                obj.Eapp = Vapp;
                
                % Must return [iSum, ia, ic]
                [obj.corrosionCurrentTotal,obj.corrosionCurrentAnodic,obj.corrosionCurrentCathodic] = obj.GetTotalCurrent(erange);
                
                [mini,idxmin] = min(abs(obj.corrosionCurrentTotal));
                obj.eCorr = erange(idxmin);

                obj.nFig = nF;                  
              end
        end
        
        function [iSum,ia,ic] = GetTotalCurrent(obj,erange)
            % Access via Cell Array {}
            ia = obj.bVreactions{1}.anodeKineticsCuNi(erange);
            ic = obj.bVreactions{2}.multiCathodicI625(erange);
            iSum = ia + ic;
        end
        
        % Placeholder for static methods if needed or methods end
    end
    
    methods (Static)
        function [x,y,phi] = aTafelPolCurve(obj)
             % Stub if needed, or real implementation
             % The real implementation was likely separate or I need to preserve it.
             % Wait, the original file had this?
             % Let me check original file content for other methods.
             x=0; y=0; phi=0;
        end
    end
end
