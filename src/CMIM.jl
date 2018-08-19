module CMIM

export cmim

function jointProbability(x::AbstractArray{Bool}, y::AbstractArray{Bool})
    
    counts = zeros(Int, 2, 2)
    
    for ii in 1:length(x)
        counts[x[ii]+1, y[ii]+1] += 1
    end
    
    counts ./ length(x)
    
end


function jointProbability(x::AbstractArray{Bool}, y::AbstractArray{Bool}, z::AbstractArray{Bool})
    
    counts = zeros(Int, 2, 2, 2)
    
    for ii in 1:length(x)
        counts[x[ii]+1, y[ii]+1, z[ii]+1] += 1
    end
    
    counts ./ length(x)
    
end


function mutualInformation(x::AbstractArray{Bool}, y::AbstractArray{Bool})
    
    pxy = jointProbability(x, y)
    
    mutualInformation = 0.0
    
    for xx in 1:2, yy in 1:2 
        if pxy[xx,yy] > 0.0
            px = sum(pxy[xx,:])
            py = sum(pxy[:,yy])
            mutualInformation += pxy[xx,yy] * log(pxy[xx,yy]/px/py)
        end
    end
    
    mutualInformation
    
end


function conditionalEntropy(x::AbstractArray{Bool}, y::AbstractArray{Bool})
    
    pxy = jointProbability(x, y)
    
    conditionalEtropy = 0.0
    
    for xx in 1:2, yy in 1:2 
        if pxy[xx,yy] > 0.0
            py = sum(pxy[:,yy])
            conditionalEtropy -= pxy[xx,yy] * log(pxy[xx,yy]/py)
        end
    end
    
    conditionalEtropy
    
end


function conditionalEntropy(x::AbstractArray{Bool}, y::AbstractArray{Bool}, z::AbstractArray{Bool})
    
    pxyz = jointProbability(x, y, z)
    
    conditionalEntropy = 0.0
    
    for xx in 1:2, yy in 1:2, zz in 1:2
        if pxyz[xx,yy,zz] > 0.0
            pyz = sum(pxyz[:,yy,zz])
            conditionalEntropy -= pxyz[xx,yy,zz] * log(pxyz[xx,yy,zz]/pyz)
        end
    end
    
    conditionalEntropy
    
end


function conditionalMutualInformation(x::AbstractArray{Bool}, y::AbstractArray{Bool}, z::AbstractArray{Bool})
   
    conditionalEntropy(x, z) - conditionalEntropy(x, y, z)
#     conditionalEntropy(y, z) - conditionalEntropy(y, x, z)
    
end


function conditionalMutualInformation(cacheConditionalEntropy::Vector, X::AbstractArray{Bool,2}, x::Int, y::AbstractArray{Bool,1}, z::Int)
   
    if cacheConditionalEntropy[z] < 0.0
        cacheConditionalEntropy[z] = conditionalEntropy(y, X[:,z])
    end
        
    cacheConditionalEntropy[z] - conditionalEntropy(y, X[:,x], X[:,z])
    
end


function cmim(k::Int, X::AbstractArray{Bool,2}, y::AbstractArray{Bool,1})
    
    @assert k <= size(X, 2)
    @assert size(X, 1) == length(y)
    
    outputFeatures = Vector{Int}(k)
    outputScores = Vector{Float64}(k)
    
    selectedFeatures = falses(size(X, 2))
    lastUsedFeature = ones(Int, size(X, 2))
    
    cacheConditionalEntropy = -ones(size(X, 2))
    
    classMI = [mutualInformation(X[:,ff], y) for ff in 1:size(X, 2)]

    for kk in 1:k
        score = -1.0
        for ff in 1:size(X, 2)
            if !selectedFeatures[ff]
            
                while (classMI[ff] > score) && (lastUsedFeature[ff] < kk)
                    classMI[ff] = min(
                        classMI[ff], 
#                         conditionalMutualInformation(X[:,ff], y, X[:,outputFeatures[lastUsedFeature[ff]]])
                        conditionalMutualInformation(cacheConditionalEntropy, X, ff, y, outputFeatures[lastUsedFeature[ff]])
                    )
                    lastUsedFeature[ff] += 1
                end

                if classMI[ff] > score
                    score = classMI[ff]
                    outputFeatures[kk] = ff
                    outputScores[kk] = classMI[ff]
                end
                
            end
        end
        selectedFeatures[outputFeatures[kk]] = true
    end

    collect(zip(outputFeatures, outputScores))
    
end

end # module
