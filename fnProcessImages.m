function returnImageSeg = fnProcessImages(opt, thOptions, rgbImg, depImg, imgNormals, allInfo, ShowImages, MethodType)
   clear regImg labelImg bdry;
    % Display images
    if (ShowImages)
        subplot(2, 3, 1); imshow(rgbImg); title('Color Image');
        subplot(2, 3, 2); imshow(depImg, []); title('Depth Image');
        subplot(2, 3, 3); imshow(imgNormals); title('Image Normals');
    end

    szh(1) = uint16(size(rgbImg(1:opt.sc:end,1:opt.sc:end,:),1));
    szh(2) = uint16(size(rgbImg(1:opt.sc:end,1:opt.sc:end,:),2));

    % accumulate features
    combFeat = getCombinedFeatures(allInfo, opt.sc);

    % Generate oversegmentation
    if(strcmp(MethodType, 'rb_jcsa_rm') | strcmp(MethodType, 'rb_jcsa'))
        display('Applying JCSA Clustering ...');
        label = uint8(fusionBD_Color_3D_Axis(combFeat, opt.kMax, [1 1 1], opt));
    elseif(strcmp(MethodType, 'rb_jcsd_rm') | strcmp(MethodType, 'rb_jcsd'))
        display('Applying JCSD Clustering ...');
        label = uint8(fusionBD_Color_3D_Normal(combFeat, opt.kMax, [1 1 1], opt));
    else
        display('Choose your method name properly');
    end

    labelImg = reshape(label, szh(1), szh(2));

    % post-processing
    display('Applying Post-processing ...');
    bdry = seg2bdry_2(labelImg);
    regImg = bwlabeln(~bdry);
    %rgbImg = rgbImg(1:2:end, 1:2:end, :); comentei

    [edges, tmp, neighbors, tmpLabImg] = seg2fragments(double(regImg), rgbImg, 10);

    display('Applying Region-Megring method ...');
    if(strcmp(MethodType, 'rb_jcsa_rm'))
        % Compute parameters of the distributions
        parNormal = getKappaWMM(combFeat(:,7:9), tmpLabImg(:));
        parColor = getParamsDivergenceGMM(combFeat(:,1:3), tmpLabImg(:));

        % Set initial parameters for the neighborhood regions
        rgnb.nbsegs = getRegionNeighbors(tmpLabImg);
        rgnb.Div_N = ones(length(rgnb.nbsegs))*5000;
        rgnb.Div_C = ones(length(rgnb.nbsegs))*5000;
        rgnb.KappaMerged = ones(length(rgnb.nbsegs)) * -20;

        % Update parameters for the neighborhood regions
        for i=1:length(rgnb.nbsegs)
            adjs = rgnb.nbsegs{i};
            adjEdges = cell(length(adjs),1);

            for j = 1:length(adjs)
                commonEdge = intersect(neighbors.segment_fragments{i}, neighbors.segment_fragments{adjs(j)});
                commIndx = [];
                for k=1:length(commonEdge)
                    commIndx = [commIndx; fliplr(floor(edges{commonEdge(k)}))];
                end
                rgnb.adjEdges{i,adjs(j)} = commIndx;

                % Compute kappa after it is merged with the neighbor cluster
                rgnb.KappaMerged(i,adjs(j)) = mergeKappaWMM(parNormal, [i,adjs(j)]);

                % Compute BD_normal among the neighbors
                rgnb.Div_N(i,adjs(j)) = parNormal.DLNF(i) - parNormal.DLNF(adjs(j)) - ( (parNormal.eta(i,:) - parNormal.eta(adjs(j),:)) * parNormal.theta_cl(adjs(j),:)');

                % Compute BD_color among the neighbors
                rgnb.Div_C(i,adjs(j)) = compute_Div_C(parColor, i, adjs(j));
            end
        end

        % Apply RM method
        img = getRAGMergeSegments(combFeat, tmpLabImg, thOptions, parNormal, parColor, rgnb, allInfo, opt.sc, MethodType);
    elseif(strcmp(MethodType, 'rb_jcsd_rm'))
        % Compute parameters of the distributions
        parNormal = getParamsVMFBD(combFeat(:,7:9), tmpLabImg(:));
        parColor = getParamsDivergenceGMM(combFeat(:,1:3), tmpLabImg(:));

        % Set initial parameters for the neighborhood regions
        rgnb.nbsegs = getRegionNeighbors(tmpLabImg);
        rgnb.Div_N = ones(length(rgnb.nbsegs))*5000;
        rgnb.Div_C = ones(length(rgnb.nbsegs))*5000;
        rgnb.KappaMerged = ones(length(rgnb.nbsegs)) * -20;

        % Update parameters for the neighborhood regions
        for i=1:length(rgnb.nbsegs)
            adjs = rgnb.nbsegs{i};
            adjEdges = cell(length(adjs),1);

            for j = 1:length(adjs)
                commonEdge = intersect(neighbors.segment_fragments{i}, neighbors.segment_fragments{adjs(j)});
                commIndx = [];
                for k=1:length(commonEdge)
                    commIndx = [commIndx; fliplr(floor(edges{commonEdge(k)}))];
                end
                rgnb.adjEdges{i,adjs(j)} = commIndx;

                % Compute kappa after it is merged with the neighbor cluster
                rgnb.KappaMerged(i,adjs(j)) = mergeKappaVMFMM(parNormal, [i,adjs(j)]);

                % Compute BD_normal among the neighbors
                rgnb.Div_N(i,adjs(j)) = parNormal.DLNF(i) - parNormal.DLNF(adjs(j)) - ( (parNormal.eta(i,:) - parNormal.eta(adjs(j),:)) * parNormal.theta_cl(adjs(j),:)');

                % Compute BD_color among the neighbors
                rgnb.Div_C(i,adjs(j)) = compute_Div_C(parColor, i, adjs(j));
            end
        end

        % Apply RM method
        img = getRAGMergeSegments(combFeat, tmpLabImg, thOptions, parNormal, parColor, rgnb, allInfo, opt.sc, MethodType);
    else
        img = tmpLabImg;
    end


returnImageSeg = img;
end

